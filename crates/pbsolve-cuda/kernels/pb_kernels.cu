// pbsolve CUDA kernels, compiled at runtime with NVRTC (no nvcc at build time).
//
// Memory layout is identical to the CPU backend: idx = i + nx*(j + ny*k).
// Every kernel is a template instantiated for double (suffix _f64) and float
// (suffix _f32, mixed-precision multigrid). Expressions are written in
// exactly the same order as the generic CPU kernels and compiled with
// --fmad=false, so they round identically on both platforms; only the order
// of the dot-product reduction differs.

#define BLOCK 256

template <typename T>
__device__ void apply_impl(
    const int nx, const int ny, const int nz,
    const T cx, const T cy, const T cz,
    const T* __restrict__ ex, const T* __restrict__ ey, const T* __restrict__ ez,
    const T* __restrict__ kvol, const T* __restrict__ x, T* __restrict__ y)
{
    const long long n = (long long)nx * ny * nz;
    const long long sy = nx;
    const long long sz = (long long)nx * ny;
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x; c < n; c += stride) {
        const int i = (int)(c % nx);
        const int j = (int)((c / nx) % ny);
        const int k = (int)(c / sz);
        if (i == 0 || j == 0 || k == 0 || i == nx - 1 || j == ny - 1 || k == nz - 1) {
            y[c] = T(0);
            continue;
        }
        const T xc = x[c];
        y[c] = cx * (ex[c - 1] * (xc - x[c - 1]) + ex[c] * (xc - x[c + 1]))
             + cy * (ey[c - sy] * (xc - x[c - sy]) + ey[c] * (xc - x[c + sy]))
             + cz * (ez[c - sz] * (xc - x[c - sz]) + ez[c] * (xc - x[c + sz]))
             + kvol[c] * xc;
    }
}

// 27-point operator (multigrid coarse levels), coefficients stored per offset:
// coef[o*n + c], o = (dx+1) + 3*(dy+1) + 9*(dz+1). Same summation order as
// apply27 on the CPU.
template <typename T>
__device__ void apply27_impl(
    const int nx, const int ny, const int nz,
    const T* __restrict__ coef, const T* __restrict__ x, T* __restrict__ y)
{
    const long long n = (long long)nx * ny * nz;
    const long long sy = nx;
    const long long sz = (long long)nx * ny;
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x; c < n; c += stride) {
        const int i = (int)(c % nx);
        const int j = (int)((c / nx) % ny);
        const int k = (int)(c / sz);
        if (i == 0 || j == 0 || k == 0 || i == nx - 1 || j == ny - 1 || k == nz - 1) {
            y[c] = T(0);
            continue;
        }
        T acc = T(0);
        int o = 0;
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    acc += coef[o * n + c] * x[c + dx + dy * sy + dz * sz];
                    ++o;
                }
            }
        }
        y[c] = acc;
    }
}

// Multigrid transfer weights along one axis (see multigrid.rs).
template <typename T>
__device__ int restrict_axis(const int c, const int coarsen, const int n, int* idx, T* w)
{
    if (!coarsen) {
        idx[0] = c;
        w[0] = T(1);
        return 1;
    }
    int m = 0;
    const int f0 = 2 * c - 1, f1 = 2 * c, f2 = 2 * c + 1;
    if (f0 > 0 && f0 + 1 < n) { idx[m] = f0; w[m] = T(0.5); ++m; }
    if (f1 > 0 && f1 + 1 < n) { idx[m] = f1; w[m] = T(1); ++m; }
    if (f2 > 0 && f2 + 1 < n) { idx[m] = f2; w[m] = T(0.5); ++m; }
    return m;
}

template <typename T>
__device__ int prolong_axis(const int f, const int coarsen, int* idx, T* w)
{
    if (!coarsen) {
        idx[0] = f;
        w[0] = T(1);
        return 1;
    }
    if (f % 2 == 0) {
        idx[0] = f / 2;
        w[0] = T(1);
        return 1;
    }
    idx[0] = (f - 1) / 2;
    idx[1] = (f + 1) / 2;
    w[0] = T(0.5);
    w[1] = T(0.5);
    return 2;
}

// coarse = P^T fine (interior coarse nodes; 0 on the coarse boundary)
template <typename T>
__device__ void restrict_impl(
    const int nx, const int ny, const int nz,
    const int mx, const int my, const int mz,
    const int cx, const int cy, const int cz,
    const T* __restrict__ fine, T* __restrict__ coarse)
{
    const long long n = (long long)mx * my * mz;
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x; c < n; c += stride) {
        const int ic = (int)(c % mx);
        const int jc = (int)((c / mx) % my);
        const int kc = (int)(c / ((long long)mx * my));
        if (ic == 0 || jc == 0 || kc == 0 || ic == mx - 1 || jc == my - 1 || kc == mz - 1) {
            coarse[c] = T(0);
            continue;
        }
        int fx[3], fy[3], fz[3];
        T wx[3], wy[3], wz[3];
        const int ax = restrict_axis<T>(ic, cx, nx, fx, wx);
        const int ay = restrict_axis<T>(jc, cy, ny, fy, wy);
        const int az = restrict_axis<T>(kc, cz, nz, fz, wz);
        T acc = T(0);
        for (int q = 0; q < az; ++q) {
            for (int b = 0; b < ay; ++b) {
                for (int a = 0; a < ax; ++a) {
                    const T w = wx[a] * wy[b] * wz[q];
                    acc += w * fine[fx[a] + (long long)nx * (fy[b] + (long long)ny * fz[q])];
                }
            }
        }
        coarse[c] = acc;
    }
}

// fine += P coarse (interior fine nodes)
template <typename T>
__device__ void prolong_add_impl(
    const int nx, const int ny, const int nz,
    const int mx, const int my, const int mz,
    const int cx, const int cy, const int cz,
    const T* __restrict__ coarse, T* __restrict__ fine)
{
    const long long n = (long long)nx * ny * nz;
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x; c < n; c += stride) {
        const int i = (int)(c % nx);
        const int j = (int)((c / nx) % ny);
        const int k = (int)(c / ((long long)nx * ny));
        if (i == 0 || j == 0 || k == 0 || i == nx - 1 || j == ny - 1 || k == nz - 1) {
            continue;
        }
        int gx[2], gy[2], gz[2];
        T wx[2], wy[2], wz[2];
        const int ax = prolong_axis<T>(i, cx, gx, wx);
        const int ay = prolong_axis<T>(j, cy, gy, wy);
        const int az = prolong_axis<T>(k, cz, gz, wz);
        T acc = T(0);
        for (int q = 0; q < az; ++q) {
            for (int b = 0; b < ay; ++b) {
                for (int a = 0; a < ax; ++a) {
                    const T w = wx[a] * wy[b] * wz[q];
                    acc += w * coarse[gx[a] + (long long)mx * (gy[b] + (long long)my * gz[q])];
                }
            }
        }
        fine[c] += acc;
    }
}

// Per-block partial sums of a.b in double. The host adds the partials in
// block order, which makes the reduction deterministic (no atomics).
// Requires blockDim.x == BLOCK.
template <typename T>
__device__ void dot_partial_impl(
    const long long n, const T* __restrict__ a, const T* __restrict__ b, double* __restrict__ partial)
{
    __shared__ double s[BLOCK];
    const long long stride = (long long)blockDim.x * gridDim.x;
    double acc = 0.0;
    for (long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x; c < n; c += stride) {
        acc += (double)a[c] * (double)b[c];
    }
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int w = BLOCK / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) {
            s[threadIdx.x] += s[threadIdx.x + w];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = s[0];
    }
}

// (a.b, a.c) in one pass: partial[block] and partial[gridDim.x + block], each
// reduced exactly like dot_partial_impl, so the sums equal two separate dots.
template <typename T>
__device__ void dot2_partial_impl(
    const long long n, const T* __restrict__ a, const T* __restrict__ b, const T* __restrict__ c,
    double* __restrict__ partial)
{
    __shared__ double s1[BLOCK];
    __shared__ double s2[BLOCK];
    const long long stride = (long long)blockDim.x * gridDim.x;
    double acc1 = 0.0, acc2 = 0.0;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const double ai = (double)a[i];
        acc1 += ai * (double)b[i];
        acc2 += ai * (double)c[i];
    }
    s1[threadIdx.x] = acc1;
    s2[threadIdx.x] = acc2;
    __syncthreads();
    for (int w = BLOCK / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) {
            s1[threadIdx.x] += s1[threadIdx.x + w];
            s2[threadIdx.x] += s2[threadIdx.x + w];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = s1[0];
        partial[gridDim.x + blockIdx.x] = s2[0];
    }
}

#define GRID_STRIDE_LOOP(c, n) \
    for (long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x; c < (n); c += (long long)blockDim.x * gridDim.x)

#define PB_KERNELS(T, S)                                                                                  \
    extern "C" __global__ void pb_apply_##S(                                                             \
        const int nx, const int ny, const int nz, const T cx, const T cy, const T cz,                    \
        const T* __restrict__ ex, const T* __restrict__ ey, const T* __restrict__ ez,                    \
        const T* __restrict__ kvol, const T* __restrict__ x, T* __restrict__ y)                          \
    { apply_impl<T>(nx, ny, nz, cx, cy, cz, ex, ey, ez, kvol, x, y); }                                   \
    extern "C" __global__ void pb_apply27_##S(                                                           \
        const int nx, const int ny, const int nz,                                                        \
        const T* __restrict__ coef, const T* __restrict__ x, T* __restrict__ y)                          \
    { apply27_impl<T>(nx, ny, nz, coef, x, y); }                                                         \
    /* y = (alpha * d) * x   (Jacobi preconditioner, smoother scaling) */                                \
    extern "C" __global__ void pb_scale_##S(const long long n, const T alpha,                            \
        const T* __restrict__ d, const T* __restrict__ x, T* __restrict__ y)                             \
    { GRID_STRIDE_LOOP(c, n) { y[c] = (alpha * d[c]) * x[c]; } }                                          \
    /* y = alpha * x + beta * y */                                                                       \
    extern "C" __global__ void pb_axpby_##S(const long long n, const T alpha,                            \
        const T* __restrict__ x, const T beta, T* __restrict__ y)                                        \
    { GRID_STRIDE_LOOP(c, n) { y[c] = alpha * x[c] + beta * y[c]; } }                                     \
    /* y += alpha * x */                                                                                 \
    extern "C" __global__ void pb_axpy_##S(const long long n, const T alpha,                             \
        const T* __restrict__ x, T* __restrict__ y)                                                      \
    { GRID_STRIDE_LOOP(c, n) { y[c] += alpha * x[c]; } }                                                  \
    /* y = x + beta * y */                                                                               \
    extern "C" __global__ void pb_xpay_##S(const long long n, const T beta,                              \
        const T* __restrict__ x, T* __restrict__ y)                                                      \
    { GRID_STRIDE_LOOP(c, n) { y[c] = x[c] + beta * y[c]; } }                                             \
    extern "C" __global__ void pb_dot_partial_##S(const long long n,                                     \
        const T* __restrict__ a, const T* __restrict__ b, double* __restrict__ partial)                  \
    { dot_partial_impl<T>(n, a, b, partial); }                                                           \
    extern "C" __global__ void pb_dot2_partial_##S(const long long n, const T* __restrict__ a,           \
        const T* __restrict__ b, const T* __restrict__ c, double* __restrict__ partial)                  \
    { dot2_partial_impl<T>(n, a, b, c, partial); }                                                       \
    extern "C" __global__ void pb_restrict_##S(                                                          \
        const int nx, const int ny, const int nz, const int mx, const int my, const int mz,              \
        const int cx, const int cy, const int cz, const T* __restrict__ fine, T* __restrict__ coarse)    \
    { restrict_impl<T>(nx, ny, nz, mx, my, mz, cx, cy, cz, fine, coarse); }                              \
    extern "C" __global__ void pb_prolong_add_##S(                                                       \
        const int nx, const int ny, const int nz, const int mx, const int my, const int mz,              \
        const int cx, const int cy, const int cz, const T* __restrict__ coarse, T* __restrict__ fine)    \
    { prolong_add_impl<T>(nx, ny, nz, mx, my, mz, cx, cy, cz, coarse, fine); }

PB_KERNELS(double, f64)
PB_KERNELS(float, f32)

// Precision conversions (round to nearest / exact widening / copy).
#define PB_CONVERT(S, D, SN, DN)                                                                         \
    extern "C" __global__ void pb_convert_##SN##_##DN(const long long n,                                 \
        const S* __restrict__ x, D* __restrict__ y)                                                      \
    { GRID_STRIDE_LOOP(c, n) { y[c] = (D)x[c]; } }

PB_CONVERT(double, float, f64, f32)
PB_CONVERT(float, double, f32, f64)
PB_CONVERT(float, float, f32, f32)

// ---------------------------------------------------------------- multigrid setup (double only)
//
// Galerkin coarse operators Pᵀ A P, their symmetrisation, inverse diagonals
// and Gershgorin bounds, as `galerkin` and `gershgorin_lmax` in multigrid.rs
// compute them. Each coarse row is accumulated by one thread in exactly the
// CPU's loop order (fine rows, then stencil entries, then interpolation
// targets), so with --fmad=false the operators are bit-identical to the host.
// Coarsening flags are kx, ky, kz; the 7-point coefficients are cx, cy, cz.

__device__ __forceinline__ bool interior3(const int i, const int j, const int k, const int nx, const int ny, const int nz)
{
    return i > 0 && j > 0 && k > 0 && i + 1 < nx && j + 1 < ny && k + 1 < nz;
}

// Adds fine-matrix entry `val` (offset dx, dy, dz of fine row fi, fj, fk,
// restriction weight wf) to coarse row (ic, jc, kc). The coarse offsets are
// in {-1, 0, 1}³ by construction.
__device__ __forceinline__ void galerkin_entry(
    double* acc, const double wf, const double val, const int dx, const int dy, const int dz,
    const int fi, const int fj, const int fk, const int nx, const int ny, const int nz,
    const int mx, const int my, const int mz, const int kx, const int ky, const int kz,
    const int ic, const int jc, const int kc)
{
    if (val == 0.0) {
        return;
    }
    const int gi = fi + dx, gj = fj + dy, gk = fk + dz;
    if (!interior3(gi, gj, gk, nx, ny, nz)) {
        return;
    }
    int px[2], py[2], pz[2];
    double wx[2], wy[2], wz[2];
    const int ax = prolong_axis<double>(gi, kx, px, wx);
    const int ay = prolong_axis<double>(gj, ky, py, wy);
    const int az = prolong_axis<double>(gk, kz, pz, wz);
    for (int r = 0; r < az; ++r) {
        for (int q = 0; q < ay; ++q) {
            for (int p = 0; p < ax; ++p) {
                if (!interior3(px[p], py[q], pz[r], mx, my, mz)) {
                    continue;
                }
                const double w = wx[p] * wy[q] * wz[r];
                const int o = (px[p] - ic + 1) + 3 * ((py[q] - jc + 1) + 3 * (pz[r] - kc + 1));
                acc[o] += wf * val * w;
            }
        }
    }
}

// Unsymmetrised Galerkin rows of a 7-point fine operator: rows[o*n_c + c].
extern "C" __global__ void pb_galerkin7(
    const int nx, const int ny, const int nz, const int mx, const int my, const int mz,
    const int kx, const int ky, const int kz, const double cx, const double cy, const double cz,
    const double* __restrict__ ex, const double* __restrict__ ey, const double* __restrict__ ez,
    const double* __restrict__ kvol, double* __restrict__ rows)
{
    const long long nc = (long long)mx * my * mz;
    const long long sy = nx;
    const long long sz = (long long)nx * ny;
    GRID_STRIDE_LOOP(c, nc) {
        const int ic = (int)(c % mx);
        const int jc = (int)((c / mx) % my);
        const int kc = (int)(c / ((long long)mx * my));
        double acc[27];
        for (int o = 0; o < 27; ++o) {
            acc[o] = 0.0;
        }
        if (interior3(ic, jc, kc, mx, my, mz)) {
            int fx[3], fy[3], fz[3];
            double wx[3], wy[3], wz[3];
            const int ax = restrict_axis<double>(ic, kx, nx, fx, wx);
            const int ay = restrict_axis<double>(jc, ky, ny, fy, wy);
            const int az = restrict_axis<double>(kc, kz, nz, fz, wz);
            for (int q = 0; q < az; ++q) {
                for (int b = 0; b < ay; ++b) {
                    for (int a = 0; a < ax; ++a) {
                        const double wf = wx[a] * wy[b] * wz[q];
                        const int fi = fx[a], fj = fy[b], fk = fz[q];
                        const long long f = fi + (long long)nx * (fj + (long long)ny * fk);
                        const double exm = ex[f - 1], exp = ex[f];
                        const double eym = ey[f - sy], eyp = ey[f];
                        const double ezm = ez[f - sz], ezp = ez[f];
                        const double diag = cx * (exm + exp) + cy * (eym + eyp) + cz * (ezm + ezp) + kvol[f];
                        // Entry order of LevelMatrix::for_each_entry (Seven).
                        galerkin_entry(acc, wf, -cz * ezm, 0, 0, -1, fi, fj, fk, nx, ny, nz, mx, my, mz, kx, ky, kz, ic, jc, kc);
                        galerkin_entry(acc, wf, -cy * eym, 0, -1, 0, fi, fj, fk, nx, ny, nz, mx, my, mz, kx, ky, kz, ic, jc, kc);
                        galerkin_entry(acc, wf, -cx * exm, -1, 0, 0, fi, fj, fk, nx, ny, nz, mx, my, mz, kx, ky, kz, ic, jc, kc);
                        galerkin_entry(acc, wf, diag, 0, 0, 0, fi, fj, fk, nx, ny, nz, mx, my, mz, kx, ky, kz, ic, jc, kc);
                        galerkin_entry(acc, wf, -cx * exp, 1, 0, 0, fi, fj, fk, nx, ny, nz, mx, my, mz, kx, ky, kz, ic, jc, kc);
                        galerkin_entry(acc, wf, -cy * eyp, 0, 1, 0, fi, fj, fk, nx, ny, nz, mx, my, mz, kx, ky, kz, ic, jc, kc);
                        galerkin_entry(acc, wf, -cz * ezp, 0, 0, 1, fi, fj, fk, nx, ny, nz, mx, my, mz, kx, ky, kz, ic, jc, kc);
                    }
                }
            }
        }
        for (int o = 0; o < 27; ++o) {
            rows[o * nc + c] = acc[o];
        }
    }
}

// Unsymmetrised Galerkin rows of a 27-point fine operator (coef[o*n_f + f]).
extern "C" __global__ void pb_galerkin27(
    const int nx, const int ny, const int nz, const int mx, const int my, const int mz,
    const int kx, const int ky, const int kz, const double* __restrict__ fine, double* __restrict__ rows)
{
    const long long nc = (long long)mx * my * mz;
    const long long nf = (long long)nx * ny * nz;
    GRID_STRIDE_LOOP(c, nc) {
        const int ic = (int)(c % mx);
        const int jc = (int)((c / mx) % my);
        const int kc = (int)(c / ((long long)mx * my));
        double acc[27];
        for (int o = 0; o < 27; ++o) {
            acc[o] = 0.0;
        }
        if (interior3(ic, jc, kc, mx, my, mz)) {
            int fx[3], fy[3], fz[3];
            double wx[3], wy[3], wz[3];
            const int ax = restrict_axis<double>(ic, kx, nx, fx, wx);
            const int ay = restrict_axis<double>(jc, ky, ny, fy, wy);
            const int az = restrict_axis<double>(kc, kz, nz, fz, wz);
            for (int q = 0; q < az; ++q) {
                for (int b = 0; b < ay; ++b) {
                    for (int a = 0; a < ax; ++a) {
                        const double wf = wx[a] * wy[b] * wz[q];
                        const int fi = fx[a], fj = fy[b], fk = fz[q];
                        const long long f = fi + (long long)nx * (fj + (long long)ny * fk);
                        for (int o = 0; o < 27; ++o) {
                            galerkin_entry(acc, wf, fine[o * nf + f], o % 3 - 1, (o / 3) % 3 - 1, o / 9 - 1,
                                           fi, fj, fk, nx, ny, nz, mx, my, mz, kx, ky, kz, ic, jc, kc);
                        }
                    }
                }
            }
        }
        for (int o = 0; o < 27; ++o) {
            rows[o * nc + c] = acc[o];
        }
    }
}

// In-place symmetrisation of the off-diagonal entries: entries (c, o) and
// (c + offset(o), 26 - o) both become 0.5 (a + b), and rows whose diagonal is
// 0 (boundary rows) become 0, as on the CPU. One thread per pair (o < 13);
// a pair whose row c lies on the grid boundary is already exactly 0.
extern "C" __global__ void pb_symmetrise27(const int mx, const int my, const int mz, double* __restrict__ coef)
{
    const long long n = (long long)mx * my * mz;
    const long long sy = mx;
    const long long sz = (long long)mx * my;
    GRID_STRIDE_LOOP(t, 13 * n) {
        const int o = (int)(t / n);
        const long long c = t % n;
        const int i = (int)(c % mx);
        const int j = (int)((c / mx) % my);
        const int k = (int)(c / sz);
        if (!interior3(i, j, k, mx, my, mz)) {
            coef[o * n + c] = 0.0;
            continue;
        }
        const long long p = c + (o % 3 - 1) + ((o / 3) % 3 - 1) * sy + (o / 9 - 1) * sz;
        const double v = 0.5 * (coef[o * n + c] + coef[(26 - o) * n + p]);
        coef[o * n + c] = coef[13 * n + c] != 0.0 ? v : 0.0;
        coef[(26 - o) * n + p] = coef[13 * n + p] != 0.0 ? v : 0.0;
    }
}

// Diagonal entries (pair with themselves) and the inverse diagonal; `bad` is
// set if a non-zero diagonal is NaN or negative (the host reports the first).
extern "C" __global__ void pb_diagonal27(const long long n, double* __restrict__ coef, double* __restrict__ inv,
                                         int* __restrict__ bad)
{
    GRID_STRIDE_LOOP(c, n) {
        const double d = coef[13 * n + c];
        const double s = d != 0.0 ? 0.5 * (d + d) : 0.0;
        coef[13 * n + c] = s;
        if (s != 0.0) {
            if (isnan(s) || s < 0.0) {
                *bad = 1;
            } else {
                inv[c] = 1.0 / s;
            }
        }
    }
}

// Block maxima of Σ_o |a_o| / a_diag over interior rows (identity 0), for the
// Gershgorin bound λmax(D⁻¹A). Requires blockDim.x == BLOCK.
__device__ __forceinline__ void block_max(double v, double* __restrict__ partial)
{
    __shared__ double s[BLOCK];
    s[threadIdx.x] = v;
    __syncthreads();
    for (int w = BLOCK / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) {
            s[threadIdx.x] = fmax(s[threadIdx.x], s[threadIdx.x + w]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = s[0];
    }
}

extern "C" __global__ void pb_gershgorin7(
    const int nx, const int ny, const int nz, const double cx, const double cy, const double cz,
    const double* __restrict__ ex, const double* __restrict__ ey, const double* __restrict__ ez,
    const double* __restrict__ kvol, double* __restrict__ partial)
{
    const long long n = (long long)nx * ny * nz;
    const long long sy = nx;
    const long long sz = (long long)nx * ny;
    double m = 0.0;
    GRID_STRIDE_LOOP(c, n) {
        const int i = (int)(c % nx);
        const int j = (int)((c / nx) % ny);
        const int k = (int)(c / sz);
        if (!interior3(i, j, k, nx, ny, nz)) {
            continue;
        }
        const double exm = ex[c - 1], exp = ex[c];
        const double eym = ey[c - sy], eyp = ey[c];
        const double ezm = ez[c - sz], ezp = ez[c];
        const double diag = cx * (exm + exp) + cy * (eym + eyp) + cz * (ezm + ezp) + kvol[c];
        double sum = 0.0;
        sum += fabs(-cz * ezm);
        sum += fabs(-cy * eym);
        sum += fabs(-cx * exm);
        sum += fabs(diag);
        sum += fabs(-cx * exp);
        sum += fabs(-cy * eyp);
        sum += fabs(-cz * ezp);
        m = fmax(m, sum / diag);
    }
    block_max(m, partial);
}

extern "C" __global__ void pb_gershgorin27(
    const int nx, const int ny, const int nz, const double* __restrict__ coef, double* __restrict__ partial)
{
    const long long n = (long long)nx * ny * nz;
    double m = 0.0;
    GRID_STRIDE_LOOP(c, n) {
        const int i = (int)(c % nx);
        const int j = (int)((c / nx) % ny);
        const int k = (int)(c / ((long long)nx * ny));
        if (!interior3(i, j, k, nx, ny, nz)) {
            continue;
        }
        double sum = 0.0;
        for (int o = 0; o < 27; ++o) {
            sum += fabs(coef[o * n + c]);
        }
        m = fmax(m, sum / coef[13 * n + c]);
    }
    block_max(m, partial);
}

// ---------------------------------------------------------------- assembly (double only)
//
// Coefficient maps, the 7-point operator and the right-hand side, as
// `mark_map` (surface.rs), `StencilOperator::new` and `assemble_rhs`
// (operator.rs) compute them on the host, expression for expression.

extern "C" __global__ void pb_fill(const long long n, const double value, double* __restrict__ v)
{
    GRID_STRIDE_LOOP(c, n) { v[c] = value; }
}

// Sets `value` at every node within (<=) one of `count` spheres (x, y, z, r)
// on the grid whose node (0, 0, 0) lies at (minx, miny, minz): APBS
// markSphere with the node test of `mark_map`. One warp per sphere
// (MARK_WARPS per block); a node inside several spheres receives the same
// value from each, so the order of the writes does not matter. The squared
// axis distances dx², dy², dz² are computed once per sphere (same
// expressions as the host) and kept in shared memory; boxes longer than
// MARK_AXIS nodes along an axis compute them per node.
#define MARK_WARPS 4
#define MARK_AXIS 64
extern "C" __global__ void pb_mark_spheres(
    const int nx, const int ny, const int nz, const double hx, const double hy, const double hz,
    const double minx, const double miny, const double minz, const long long count,
    const double* __restrict__ spheres, const double value, double* __restrict__ map)
{
    __shared__ double terms[MARK_WARPS][3][MARK_AXIS];
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    double* const tx = terms[warp][0];
    double* const ty = terms[warp][1];
    double* const tz = terms[warp][2];
    for (long long s = (long long)blockIdx.x * MARK_WARPS + warp; s < count; s += (long long)gridDim.x * MARK_WARPS) {
        const double posx = spheres[4 * s] - minx;
        const double posy = spheres[4 * s + 1] - miny;
        const double posz = spheres[4 * s + 2] - minz;
        const double radius = spheres[4 * s + 3];
        // `mark_range` of the three axes, one bound per lane (lanes 0-2 the
        // lower, 3-5 the upper bounds of x, y, z), shared through shuffles.
        double bound = 0.0;
        if (lane < 6) {
            const int a = lane % 3;
            const double pos = a == 0 ? posx : (a == 1 ? posy : posz);
            const double h = a == 0 ? hx : (a == 1 ? hy : hz);
            const int n = a == 0 ? nx : (a == 1 ? ny : nz);
            const double range = radius + 0.5 * h;
            bound = lane < 3 ? fmax(ceil((pos - range) / h), 0.0) : fmin(floor((pos + range) / h), (double)(n - 1));
        }
        const double lx = __shfl_sync(0xffffffffu, bound, 0);
        const double ly = __shfl_sync(0xffffffffu, bound, 1);
        const double lz = __shfl_sync(0xffffffffu, bound, 2);
        const double ux = __shfl_sync(0xffffffffu, bound, 3);
        const double uy = __shfl_sync(0xffffffffu, bound, 4);
        const double uz = __shfl_sync(0xffffffffu, bound, 5);
        if (ux >= lx && uy >= ly && uz >= lz) {
            const int i0 = (int)lx, i1 = (int)ux;
            const int j0 = (int)ly, j1 = (int)uy;
            const int k0 = (int)lz, k1 = (int)uz;
            const double rtot2 = radius * radius;
            const int ni = i1 - i0 + 1;
            const int nj = j1 - j0 + 1;
            const int nk = k1 - k0 + 1;
            const bool cached = ni <= MARK_AXIS && nj <= MARK_AXIS && nk <= MARK_AXIS;
            if (cached) {
                for (int t = lane; t < ni; t += 32) {
                    const double dx = posx - hx * (double)(i0 + t);
                    tx[t] = dx * dx;
                }
                for (int t = lane; t < nj; t += 32) {
                    const double dy = posy - hy * (double)(j0 + t);
                    ty[t] = dy * dy;
                }
                for (int t = lane; t < nk; t += 32) {
                    const double dz = posz - hz * (double)(k0 + t);
                    tz[t] = dz * dz;
                }
            }
            __syncwarp();
            const int rows = nj * nk;
            for (int r = lane; r < rows; r += 32) {
                const int j = j0 + r % nj;
                const int k = k0 + r / nj;
                double dy2, dz2;
                if (cached) {
                    dy2 = ty[j - j0];
                    dz2 = tz[k - k0];
                } else {
                    const double dy = posy - hy * (double)j;
                    dy2 = dy * dy;
                    const double dz = posz - hz * (double)k;
                    dz2 = dz * dz;
                }
                // The host tests (dz² + dy²) + dx² <= r² after skipping nodes
                // with dx² + dy² > r². Rounding is monotone and dx², dz² >= 0, so
                // a node that skip rejects also fails the test, and so does every
                // node of a row with dz² + dy² > r²: testing only the sum gives
                // the same marks.
                const double dzy = dz2 + dy2;
                if (dzy > rtot2) {
                    continue;
                }
                double* const row = map + (long long)nx * (j + (long long)ny * k);
                for (int i = i0; i <= i1; ++i) {
                    double dx2;
                    if (cached) {
                        dx2 = tx[i - i0];
                    } else {
                        const double dx = posx - hx * (double)i;
                        dx2 = dx * dx;
                    }
                    if (dzy + dx2 <= rtot2) {
                        row[i] = value;
                    }
                }
            }
            // The next sphere overwrites the terms only after every lane is done.
            __syncwarp();
        }
    }
}

// kvol = kappa_sq * vol (in place) and the inverse diagonal (0 on boundary nodes).
extern "C" __global__ void pb_operator7(
    const int nx, const int ny, const int nz, const double cx, const double cy, const double cz, const double vol,
    const double* __restrict__ ex, const double* __restrict__ ey, const double* __restrict__ ez,
    double* __restrict__ kvol, double* __restrict__ inv)
{
    const long long n = (long long)nx * ny * nz;
    const long long sy = nx;
    const long long sz = (long long)nx * ny;
    GRID_STRIDE_LOOP(c, n) {
        const double kv = kvol[c] * vol;
        kvol[c] = kv;
        const int i = (int)(c % nx);
        const int j = (int)((c / nx) % ny);
        const int k = (int)(c / sz);
        if (!interior3(i, j, k, nx, ny, nz)) {
            inv[c] = 0.0;
            continue;
        }
        const double diag = cx * (ex[c - 1] + ex[c]) + cy * (ey[c - sy] + ey[c]) + cz * (ez[c - sz] + ez[c]) + kv;
        inv[c] = 1.0 / diag;
    }
}

// y = (d == 0) ? 0 : src * q - y   (y = A g on entry: the right-hand side)
extern "C" __global__ void pb_source_rhs(const long long n, const double src, const double* __restrict__ q,
                                         const double* __restrict__ d, double* __restrict__ y)
{
    GRID_STRIDE_LOOP(c, n) { y[c] = d[c] == 0.0 ? 0.0 : src * q[c] - y[c]; }
}

// out[idx[m]] = values[m] (distinct indices)
extern "C" __global__ void pb_scatter(const long long m, const unsigned int* __restrict__ idx,
                                      const double* __restrict__ values, double* __restrict__ out)
{
    GRID_STRIDE_LOOP(c, m) { out[idx[c]] = values[c]; }
}
