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
