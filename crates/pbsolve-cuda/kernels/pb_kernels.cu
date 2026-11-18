// pbsolve CUDA kernels, compiled at runtime with NVRTC (no nvcc at build time).
//
// Memory layout is identical to the CPU backend: idx = i + nx*(j + ny*k).
// Every kernel is a template instantiated for double (suffix _f64) and float
// (suffix _f32, mixed-precision multigrid). Expressions are written in
// exactly the same order as the generic CPU kernels and compiled with
// --fmad=false, so they round identically on both platforms; only the order
// of the dot-product reduction differs.

#define BLOCK 256

// Periodic axes `per` (bit 0: x, bit 1: y; grid.rs `Grid::periodic`): on such
// an axis of n nodes node n - 1 is the image of node 0, so nodes 0 .. n - 2
// are unknowns and the neighbours across the seam wrap around (axis_unknown,
// axis_prev, axis_next in grid.rs). With per = 0 the indices are the plain ones.
__device__ __forceinline__ bool axis_unknown(const int i, const int n, const int p)
{
    return i + 1 < n && (i > 0 || p);
}

__device__ __forceinline__ int axis_prev(const int i, const int n, const int p)
{
    return (p && i == 0) ? n - 2 : i - 1;
}

__device__ __forceinline__ int axis_next(const int i, const int n, const int p)
{
    return (p && i + 2 == n) ? 0 : i + 1;
}

__device__ __forceinline__ int axis_step(const int i, const int d, const int n, const int p)
{
    return d < 0 ? axis_prev(i, n, p) : (d > 0 ? axis_next(i, n, p) : i);
}

__device__ __forceinline__ bool unknown3(const int i, const int j, const int k, const int nx, const int ny,
                                         const int nz, const int per)
{
    return axis_unknown(i, nx, per & 1) && axis_unknown(j, ny, (per >> 1) & 1) && axis_unknown(k, nz, 0);
}

// Interior node of a grid without periodic axes (the nonlinear kernels).
__device__ __forceinline__ bool interior3(const int i, const int j, const int k, const int nx, const int ny, const int nz)
{
    return i > 0 && j > 0 && k > 0 && i + 1 < nx && j + 1 < ny && k + 1 < nz;
}

// The node of unwrapped coordinate f on an axis of n nodes if it is an
// unknown, else -1 (multigrid.rs `wrapped`).
__device__ __forceinline__ int wrapped(const int f, const int n, const int p)
{
    if (p) {
        const int u = n - 1;
        return ((f % u) + u) % u;
    }
    return (f > 0 && f + 1 < n) ? f : -1;
}

// (A x)_c of the 7-point operator, 0 on boundary nodes: the value pb_apply
// writes, also used by the fused residuals. The -a neighbour m of c is also
// the index of the -a face (operator.rs neighbours7, stencil7_at).
template <typename T>
__device__ __forceinline__ T stencil7_at(
    const long long c, const int nx, const int ny, const int nz, const int per,
    const T cx, const T cy, const T cz,
    const T* __restrict__ ex, const T* __restrict__ ey, const T* __restrict__ ez,
    const T* __restrict__ kvol, const T* __restrict__ x)
{
    const long long sz = (long long)nx * ny;
    const int i = (int)(c % nx);
    const int j = (int)((c / nx) % ny);
    const int k = (int)(c / sz);
    if (!unknown3(i, j, k, nx, ny, nz, per)) {
        return T(0);
    }
    const long long row = c - i;
    const long long plane = row - (long long)j * nx;
    const long long xm = row + axis_prev(i, nx, per & 1);
    const long long xp = row + axis_next(i, nx, per & 1);
    const long long ym = plane + (long long)axis_prev(j, ny, (per >> 1) & 1) * nx + i;
    const long long yp = plane + (long long)axis_next(j, ny, (per >> 1) & 1) * nx + i;
    const T xc = x[c];
    return cx * (ex[xm] * (xc - x[xm]) + ex[c] * (xc - x[xp]))
         + cy * (ey[ym] * (xc - x[ym]) + ey[c] * (xc - x[yp]))
         + cz * (ez[c - sz] * (xc - x[c - sz]) + ez[c] * (xc - x[c + sz]))
         + kvol[c] * xc;
}

template <typename T>
__device__ void apply_impl(
    const int nx, const int ny, const int nz, const int per,
    const T cx, const T cy, const T cz,
    const T* __restrict__ ex, const T* __restrict__ ey, const T* __restrict__ ez,
    const T* __restrict__ kvol, const T* __restrict__ x, T* __restrict__ y)
{
    const long long n = (long long)nx * ny * nz;
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x; c < n; c += stride) {
        y[c] = stencil7_at<T>(c, nx, ny, nz, per, cx, cy, cz, ex, ey, ez, kvol, x);
    }
}

// (A x)_c of a 27-point operator (multigrid coarse levels), 0 on boundary
// nodes. Coefficients are stored per offset: coef[o*n + c],
// o = (dx+1) + 3*(dy+1) + 9*(dz+1). Same summation order as apply27 on the CPU.
template <typename T>
__device__ __forceinline__ T stencil27_at(
    const long long c, const int nx, const int ny, const int nz, const int per,
    const T* __restrict__ coef, const T* __restrict__ x)
{
    const long long n = (long long)nx * ny * nz;
    const long long sy = nx;
    const long long sz = (long long)nx * ny;
    const int i = (int)(c % nx);
    const int j = (int)((c / nx) % ny);
    const int k = (int)(c / sz);
    if (!unknown3(i, j, k, nx, ny, nz, per)) {
        return T(0);
    }
    const int px = per & 1, py = (per >> 1) & 1;
    // Next to a seam the neighbours wrap (multigrid.rs on_seam, neighbour27).
    const bool seam = (px && (i == 0 || i + 2 == nx)) || (py && (j == 0 || j + 2 == ny));
    T acc = T(0);
    int o = 0;
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const long long q = seam ? axis_step(i, dx, nx, px) +
                                               (long long)nx * (axis_step(j, dy, ny, py) + (long long)ny * (k + dz))
                                         : c + dx + dy * sy + dz * sz;
                acc += coef[o * n + c] * x[q];
                ++o;
            }
        }
    }
    return acc;
}

template <typename T>
__device__ void apply27_impl(
    const int nx, const int ny, const int nz, const int per,
    const T* __restrict__ coef, const T* __restrict__ x, T* __restrict__ y)
{
    const long long n = (long long)nx * ny * nz;
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x; c < n; c += stride) {
        y[c] = stencil27_at<T>(c, nx, ny, nz, per, coef, x);
    }
}

// Multigrid transfer weights along one axis (see multigrid.rs): the fine
// unknowns of coarse unknown c, their weights and unwrapped coordinates.
template <typename T>
__device__ int restrict_axis(const int c, const int coarsen, const int n, const int p, int* idx, T* w, int* at)
{
    if (!coarsen) {
        idx[0] = c;
        w[0] = T(1);
        at[0] = c;
        return 1;
    }
    int m = 0;
    for (int s = -1; s <= 1; ++s) {
        const int f = 2 * c + s;
        const int g = wrapped(f, n, p);
        if (g >= 0) {
            idx[m] = g;
            w[m] = s == 0 ? T(1) : T(0.5);
            at[m] = f;
            ++m;
        }
    }
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

// coarse = P^T fine (coarse unknowns; 0 on the other coarse nodes)
template <typename T>
__device__ void restrict_impl(
    const int nx, const int ny, const int nz,
    const int mx, const int my, const int mz,
    const int cx, const int cy, const int cz, const int per,
    const T* __restrict__ fine, T* __restrict__ coarse)
{
    const long long n = (long long)mx * my * mz;
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x; c < n; c += stride) {
        const int ic = (int)(c % mx);
        const int jc = (int)((c / mx) % my);
        const int kc = (int)(c / ((long long)mx * my));
        if (!unknown3(ic, jc, kc, mx, my, mz, per)) {
            coarse[c] = T(0);
            continue;
        }
        int fx[3], fy[3], fz[3], ux[3], uy[3], uz[3];
        T wx[3], wy[3], wz[3];
        const int ax = restrict_axis<T>(ic, cx, nx, per & 1, fx, wx, ux);
        const int ay = restrict_axis<T>(jc, cy, ny, (per >> 1) & 1, fy, wy, uy);
        const int az = restrict_axis<T>(kc, cz, nz, 0, fz, wz, uz);
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

// fine += P coarse (fine unknowns); the last coarse node of a periodic axis
// is read as its image 0
template <typename T>
__device__ void prolong_add_impl(
    const int nx, const int ny, const int nz,
    const int mx, const int my, const int mz,
    const int cx, const int cy, const int cz, const int per,
    const T* __restrict__ coarse, T* __restrict__ fine)
{
    const long long n = (long long)nx * ny * nz;
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x; c < n; c += stride) {
        const int i = (int)(c % nx);
        const int j = (int)((c / nx) % ny);
        const int k = (int)(c / ((long long)nx * ny));
        if (!unknown3(i, j, k, nx, ny, nz, per)) {
            continue;
        }
        int gx[2], gy[2], gz[2];
        T wx[2], wy[2], wz[2];
        const int ax = prolong_axis<T>(i, cx, gx, wx);
        const int ay = prolong_axis<T>(j, cy, gy, wy);
        const int az = prolong_axis<T>(k, cz, gz, wz);
        for (int a = 0; a < ax; ++a) {
            if ((per & 1) && gx[a] == mx - 1) { gx[a] = 0; }
        }
        for (int b = 0; b < ay; ++b) {
            if (((per >> 1) & 1) && gy[b] == my - 1) { gy[b] = 0; }
        }
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

// The sum of the block's per-thread values `acc` (tree order), stored by
// thread 0 at *out; `s` holds BLOCK shared doubles. Every dot product sums
// its blocks this way. Requires blockDim.x == BLOCK.
__device__ __forceinline__ void block_sum(const double acc, double* s, double* out)
{
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int w = BLOCK / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) {
            s[threadIdx.x] += s[threadIdx.x + w];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *out = s[0];
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
    block_sum(acc, s, partial + blockIdx.x);
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
        const int nx, const int ny, const int nz, const int per, const T cx, const T cy, const T cz,     \
        const T* __restrict__ ex, const T* __restrict__ ey, const T* __restrict__ ez,                    \
        const T* __restrict__ kvol, const T* __restrict__ x, T* __restrict__ y)                          \
    { apply_impl<T>(nx, ny, nz, per, cx, cy, cz, ex, ey, ez, kvol, x, y); }                              \
    extern "C" __global__ void pb_apply27_##S(                                                           \
        const int nx, const int ny, const int nz, const int per,                                         \
        const T* __restrict__ coef, const T* __restrict__ x, T* __restrict__ y)                          \
    { apply27_impl<T>(nx, ny, nz, per, coef, x, y); }                                                    \
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
        const int cx, const int cy, const int cz, const int per,                                         \
        const T* __restrict__ fine, T* __restrict__ coarse)                                              \
    { restrict_impl<T>(nx, ny, nz, mx, my, mz, cx, cy, cz, per, fine, coarse); }                         \
    extern "C" __global__ void pb_prolong_add_##S(                                                       \
        const int nx, const int ny, const int nz, const int mx, const int my, const int mz,              \
        const int cx, const int cy, const int cz, const int per,                                         \
        const T* __restrict__ coarse, T* __restrict__ fine)                                              \
    { prolong_add_impl<T>(nx, ny, nz, mx, my, mz, cx, cy, cz, per, coarse, fine); }                     \
    /* Fused multigrid smoother steps (Backend::residual, chebyshev_first,                             \
       chebyshev_next): each element as the separate kernels compute it.                                \
       r = b + (-1) * (A x)   (apply, copy, axpy) */                                                       \
    extern "C" __global__ void pb_residual_##S(                                                          \
        const int nx, const int ny, const int nz, const int per, const T cx, const T cy, const T cz,     \
        const T* __restrict__ ex, const T* __restrict__ ey, const T* __restrict__ ez,                    \
        const T* __restrict__ kvol, const T* __restrict__ x, const T* __restrict__ b,                    \
        T* __restrict__ r)                                                                               \
    { const long long n = (long long)nx * ny * nz;                                                       \
      GRID_STRIDE_LOOP(c, n) {                                                                           \
          r[c] = b[c] + T(-1) * stencil7_at<T>(c, nx, ny, nz, per, cx, cy, cz, ex, ey, ez, kvol, x); } } \
    /* r = r + (-1) * (A x)   (apply, axpy) */                                                             \
    extern "C" __global__ void pb_residual_update_##S(                                                   \
        const int nx, const int ny, const int nz, const int per, const T cx, const T cy, const T cz,     \
        const T* __restrict__ ex, const T* __restrict__ ey, const T* __restrict__ ez,                    \
        const T* __restrict__ kvol, const T* __restrict__ x, T* __restrict__ r)                          \
    { const long long n = (long long)nx * ny * nz;                                                       \
      GRID_STRIDE_LOOP(c, n) {                                                                           \
          r[c] = r[c] + T(-1) * stencil7_at<T>(c, nx, ny, nz, per, cx, cy, cz, ex, ey, ez, kvol, x); } } \
    extern "C" __global__ void pb_residual27_##S(                                                        \
        const int nx, const int ny, const int nz, const int per, const T* __restrict__ coef,             \
        const T* __restrict__ x, const T* __restrict__ b, T* __restrict__ r)                             \
    { const long long n = (long long)nx * ny * nz;                                                       \
      GRID_STRIDE_LOOP(c, n) { r[c] = b[c] + T(-1) * stencil27_at<T>(c, nx, ny, nz, per, coef, x); } }    \
    extern "C" __global__ void pb_residual27_update_##S(                                                 \
        const int nx, const int ny, const int nz, const int per, const T* __restrict__ coef,             \
        const T* __restrict__ x, T* __restrict__ r)                                                      \
    { const long long n = (long long)nx * ny * nz;                                                       \
      GRID_STRIDE_LOOP(c, n) { r[c] = r[c] + T(-1) * stencil27_at<T>(c, nx, ny, nz, per, coef, x); } }    \
    /* d = (s * w) * r; x = d (overwrite) or x = x + 1 * d   (scale, copy or axpy) */                      \
    extern "C" __global__ void pb_cheb_first_##S(const long long n, const T s,                          \
        const T* __restrict__ w, const T* __restrict__ r, T* __restrict__ d, T* __restrict__ x,          \
        const int overwrite)                                                                             \
    { GRID_STRIDE_LOOP(c, n) {                                                                           \
          const T dc = (s * w[c]) * r[c];                                                                \
          d[c] = dc;                                                                                     \
          x[c] = overwrite ? dc : x[c] + T(1) * dc; } }                                                  \
    /* t = (c1 * w) * r; d = 1 * t + c2 * d; x = x + 1 * d   (scale, axpby, axpy) */                       \
    extern "C" __global__ void pb_cheb_next_##S(const long long n, const T c1,                          \
        const T* __restrict__ w, const T* __restrict__ r, const T c2, T* __restrict__ d,                 \
        T* __restrict__ x)                                                                               \
    { GRID_STRIDE_LOOP(c, n) {                                                                           \
          const T t = (c1 * w[c]) * r[c];                                                                \
          const T dc = T(1) * t + c2 * d[c];                                                             \
          d[c] = dc;                                                                                     \
          x[c] = x[c] + T(1) * dc; } }                                                                   \
    /* Fused CG steps (Backend::cg_update, promote_dot): each element as the                             \
       separate kernels compute it, each dot product summed like pb_dot_partial                          \
       (same grid on the host, grid-stride order, block_sum).                                            \
       x += a p; r += na q (na = -a); partial = r.r and, with z, r.z (second                             \
       half); with r32, r32 = (float)r   (axpy, axpy, dot or dot2, demote) */                            \
    extern "C" __global__ void pb_cg_update_##S(const long long n, const T a, const T na,                \
        const T* __restrict__ p, const T* __restrict__ q, T* __restrict__ x, T* __restrict__ r,          \
        const T* __restrict__ z, float* __restrict__ r32, double* __restrict__ partial)                  \
    { __shared__ double s1[BLOCK];                                                                       \
      __shared__ double s2[BLOCK];                                                                       \
      double rr = 0.0, rz = 0.0;                                                                         \
      GRID_STRIDE_LOOP(c, n) {                                                                           \
          x[c] += a * p[c];                                                                              \
          const T rc = r[c] + na * q[c];                                                                 \
          r[c] = rc;                                                                                     \
          const double rd = (double)rc;                                                                  \
          rr += rd * rd;                                                                                 \
          if (z) { rz += rd * (double)z[c]; }                                                            \
          if (r32) { r32[c] = (float)rc; } }                                                             \
      block_sum(rr, s1, partial + blockIdx.x);                                                           \
      if (z) { block_sum(rz, s2, partial + gridDim.x + blockIdx.x); } }                                  \
    /* z = (T)z32; partial = r.z   (promote, dot) */                                                     \
    extern "C" __global__ void pb_promote_dot_##S(const long long n, const float* __restrict__ z32,     \
        const T* __restrict__ r, T* __restrict__ z, double* __restrict__ partial)                        \
    { __shared__ double s[BLOCK];                                                                        \
      double acc = 0.0;                                                                                  \
      GRID_STRIDE_LOOP(c, n) {                                                                           \
          const T zc = (T)z32[c];                                                                        \
          z[c] = zc;                                                                                     \
          acc += (double)r[c] * (double)zc; }                                                            \
      block_sum(acc, s, partial + blockIdx.x); }

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
// Coarsening flags are kx, ky, kz; the 7-point coefficients are cx, cy, cz;
// per are the periodic axes. Coordinates are followed unwrapped and wrapped
// only to address nodes, as `galerkin` does.

// Adds fine-matrix entry `val` (offset dx, dy, dz of the fine row at
// unwrapped coordinates fi, fj, fk, restriction weight wf) to coarse row
// (ic, jc, kc). The coarse offsets are in {-1, 0, 1}³ by construction
// (prolong_axis gives the floor halves of negative coordinates too).
__device__ __forceinline__ void galerkin_entry(
    double* acc, const double wf, const double val, const int dx, const int dy, const int dz,
    const int fi, const int fj, const int fk, const int nx, const int ny, const int nz,
    const int mx, const int my, const int mz, const int kx, const int ky, const int kz,
    const int ic, const int jc, const int kc, const int per)
{
    if (val == 0.0) {
        return;
    }
    const int pxa = per & 1, pya = (per >> 1) & 1;
    const int gi = fi + dx, gj = fj + dy, gk = fk + dz;
    if (wrapped(gi, nx, pxa) < 0 || wrapped(gj, ny, pya) < 0 || wrapped(gk, nz, 0) < 0) {
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
                if (wrapped(px[p], mx, pxa) < 0 || wrapped(py[q], my, pya) < 0 || wrapped(pz[r], mz, 0) < 0) {
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
    const int kx, const int ky, const int kz, const int per, const double cx, const double cy, const double cz,
    const double* __restrict__ ex, const double* __restrict__ ey, const double* __restrict__ ez,
    const double* __restrict__ kvol, double* __restrict__ rows)
{
    const long long nc = (long long)mx * my * mz;
    const long long sz = (long long)nx * ny;
    const int pxa = per & 1, pya = (per >> 1) & 1;
    GRID_STRIDE_LOOP(c, nc) {
        const int ic = (int)(c % mx);
        const int jc = (int)((c / mx) % my);
        const int kc = (int)(c / ((long long)mx * my));
        double acc[27];
        for (int o = 0; o < 27; ++o) {
            acc[o] = 0.0;
        }
        if (unknown3(ic, jc, kc, mx, my, mz, per)) {
            int fx[3], fy[3], fz[3], ux[3], uy[3], uz[3];
            double wx[3], wy[3], wz[3];
            const int ax = restrict_axis<double>(ic, kx, nx, pxa, fx, wx, ux);
            const int ay = restrict_axis<double>(jc, ky, ny, pya, fy, wy, uy);
            const int az = restrict_axis<double>(kc, kz, nz, 0, fz, wz, uz);
            for (int q = 0; q < az; ++q) {
                for (int b = 0; b < ay; ++b) {
                    for (int a = 0; a < ax; ++a) {
                        const double wf = wx[a] * wy[b] * wz[q];
                        const int fi = ux[a], fj = uy[b], fk = uz[q];
                        const long long row = (long long)nx * (fy[b] + (long long)ny * fz[q]);
                        const long long f = fx[a] + row;
                        const long long plane = (long long)nx * ny * fz[q];
                        // The -a faces of node f: those of its -a neighbours.
                        const double exm = ex[row + axis_prev(fx[a], nx, pxa)], exp = ex[f];
                        const double eym = ey[plane + (long long)axis_prev(fy[b], ny, pya) * nx + fx[a]], eyp = ey[f];
                        const double ezm = ez[f - sz], ezp = ez[f];
                        const double diag = cx * (exm + exp) + cy * (eym + eyp) + cz * (ezm + ezp) + kvol[f];
                        // Entry order of LevelMatrix::for_each_entry (Seven).
                        galerkin_entry(acc, wf, -cz * ezm, 0, 0, -1, fi, fj, fk, nx, ny, nz, mx, my, mz, kx, ky, kz, ic, jc, kc, per);
                        galerkin_entry(acc, wf, -cy * eym, 0, -1, 0, fi, fj, fk, nx, ny, nz, mx, my, mz, kx, ky, kz, ic, jc, kc, per);
                        galerkin_entry(acc, wf, -cx * exm, -1, 0, 0, fi, fj, fk, nx, ny, nz, mx, my, mz, kx, ky, kz, ic, jc, kc, per);
                        galerkin_entry(acc, wf, diag, 0, 0, 0, fi, fj, fk, nx, ny, nz, mx, my, mz, kx, ky, kz, ic, jc, kc, per);
                        galerkin_entry(acc, wf, -cx * exp, 1, 0, 0, fi, fj, fk, nx, ny, nz, mx, my, mz, kx, ky, kz, ic, jc, kc, per);
                        galerkin_entry(acc, wf, -cy * eyp, 0, 1, 0, fi, fj, fk, nx, ny, nz, mx, my, mz, kx, ky, kz, ic, jc, kc, per);
                        galerkin_entry(acc, wf, -cz * ezp, 0, 0, 1, fi, fj, fk, nx, ny, nz, mx, my, mz, kx, ky, kz, ic, jc, kc, per);
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
    const int kx, const int ky, const int kz, const int per, const double* __restrict__ fine,
    double* __restrict__ rows)
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
        if (unknown3(ic, jc, kc, mx, my, mz, per)) {
            int fx[3], fy[3], fz[3], ux[3], uy[3], uz[3];
            double wx[3], wy[3], wz[3];
            const int ax = restrict_axis<double>(ic, kx, nx, per & 1, fx, wx, ux);
            const int ay = restrict_axis<double>(jc, ky, ny, (per >> 1) & 1, fy, wy, uy);
            const int az = restrict_axis<double>(kc, kz, nz, 0, fz, wz, uz);
            for (int q = 0; q < az; ++q) {
                for (int b = 0; b < ay; ++b) {
                    for (int a = 0; a < ax; ++a) {
                        const double wf = wx[a] * wy[b] * wz[q];
                        const long long f = fx[a] + (long long)nx * (fy[b] + (long long)ny * fz[q]);
                        for (int o = 0; o < 27; ++o) {
                            galerkin_entry(acc, wf, fine[o * nf + f], o % 3 - 1, (o / 3) % 3 - 1, o / 9 - 1,
                                           ux[a], uy[b], uz[q], nx, ny, nz, mx, my, mz, kx, ky, kz, ic, jc, kc, per);
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
extern "C" __global__ void pb_symmetrise27(const int mx, const int my, const int mz, const int per,
                                           double* __restrict__ coef)
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
        if (!unknown3(i, j, k, mx, my, mz, per)) {
            coef[o * n + c] = 0.0;
            continue;
        }
        // The partner of entry o of c (multigrid.rs neighbour27 with periodic axes).
        const int dx = o % 3 - 1, dy = (o / 3) % 3 - 1, dz = o / 9 - 1;
        const long long p = per ? axis_step(i, dx, mx, per & 1) +
                                      sy * (axis_step(j, dy, my, (per >> 1) & 1) + (long long)my * (k + dz))
                                : c + dx + dy * sy + dz * sz;
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
    const int nx, const int ny, const int nz, const int per, const double cx, const double cy, const double cz,
    const double* __restrict__ ex, const double* __restrict__ ey, const double* __restrict__ ez,
    const double* __restrict__ kvol, double* __restrict__ partial)
{
    const long long n = (long long)nx * ny * nz;
    const long long sz = (long long)nx * ny;
    double m = 0.0;
    GRID_STRIDE_LOOP(c, n) {
        const int i = (int)(c % nx);
        const int j = (int)((c / nx) % ny);
        const int k = (int)(c / sz);
        if (!unknown3(i, j, k, nx, ny, nz, per)) {
            continue;
        }
        const long long row = c - i;
        const double exm = ex[row + axis_prev(i, nx, per & 1)], exp = ex[c];
        const double eym = ey[row - (long long)j * nx + (long long)axis_prev(j, ny, (per >> 1) & 1) * nx + i], eyp = ey[c];
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
    const int nx, const int ny, const int nz, const int per, const double* __restrict__ coef,
    double* __restrict__ partial)
{
    const long long n = (long long)nx * ny * nz;
    double m = 0.0;
    GRID_STRIDE_LOOP(c, n) {
        const int i = (int)(c % nx);
        const int j = (int)((c / nx) % ny);
        const int k = (int)(c / ((long long)nx * ny));
        if (!unknown3(i, j, k, nx, ny, nz, per)) {
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
    const int nx, const int ny, const int nz, const int per, const double cx, const double cy, const double cz,
    const double vol, const double* __restrict__ ex, const double* __restrict__ ey,
    const double* __restrict__ ez, double* __restrict__ kvol, double* __restrict__ inv)
{
    const long long n = (long long)nx * ny * nz;
    const long long sz = (long long)nx * ny;
    GRID_STRIDE_LOOP(c, n) {
        const double kv = kvol[c] * vol;
        kvol[c] = kv;
        const int i = (int)(c % nx);
        const int j = (int)((c / nx) % ny);
        const int k = (int)(c / sz);
        if (!unknown3(i, j, k, nx, ny, nz, per)) {
            inv[c] = 0.0;
            continue;
        }
        const long long row = c - i;
        const long long xm = row + axis_prev(i, nx, per & 1);
        const long long ym = row - (long long)j * nx + (long long)axis_prev(j, ny, (per >> 1) & 1) * nx + i;
        const double diag = cx * (ex[xm] + ex[c]) + cy * (ey[ym] + ey[c]) + cz * (ez[c - sz] + ez[c]) + kv;
        inv[c] = 1.0 / diag;
    }
}

// y = (d == 0) ? 0 : src * q - y   (y = A g on entry: the right-hand side)
extern "C" __global__ void pb_source_rhs(const long long n, const double src, const double* __restrict__ q,
                                         const double* __restrict__ d, double* __restrict__ y)
{
    GRID_STRIDE_LOOP(c, n) { y[c] = d[c] == 0.0 ? 0.0 : src * q[c] - y[c]; }
}

// Ion terms of the nonlinear PBE (pbsolve-core nonlinear.rs, ion_sums,
// slope_at and residual_at, same expressions): ions = m triples (q_j,
// a_j = c_j / 2I, phi_j = lattice volume fraction, 0 for point ions),
// Boltzmann factors exp(-q u) with the exponent clamped to +-85 as in APBS.
__device__ __forceinline__ double pb_boltzmann(const double x) { return exp(fmin(fmax(x, -85.0), 85.0)); }

// s1 = sum a q e, s2 = sum a q^2 e, p1 = sum phi q e, d = 1 - sum phi + sum phi e
__device__ __forceinline__ void pb_ion_sums(const int m, const double* __restrict__ ions, const double u, double* s1,
                                            double* s2, double* p1, double* d)
{
    double a1 = 0.0, a2 = 0.0, b1 = 0.0, phi0 = 0.0, occupied = 0.0;
    for (int j = 0; j < m; ++j) {
        const double q = ions[3 * j], a = ions[3 * j + 1], phi = ions[3 * j + 2];
        const double e = pb_boltzmann(-q * u);
        a1 += a * q * e;
        a2 += a * q * q * e;
        b1 += phi * q * e;
        phi0 += phi;
        occupied += phi * e;
    }
    *s1 = a1;
    *s2 = a2;
    *p1 = b1;
    *d = 1.0 - phi0 + occupied;
}

// slope = kbar2 * (s2 / d - s1 * p1 / d^2)   (0 where kbar2 = 0)
extern "C" __global__ void pb_boltzmann_slope(const long long n, const int m, const double* __restrict__ ions,
                                              const double* __restrict__ u, const double* __restrict__ k2,
                                              double* __restrict__ slope)
{
    GRID_STRIDE_LOOP(c, n)
    {
        const double k = k2[c];
        double s = 0.0;
        if (k > 0.0) {
            double s1, s2, p1, d;
            pb_ion_sums(m, ions, u[c], &s1, &s2, &p1, &d);
            s = k * (s2 / d - s1 * p1 / (d * d));
        }
        slope[c] = s;
    }
}

// y = 0 on the boundary, else y + vol * (N(u) - slope * u) - f, N(u) = -kbar2 * s1 / d
// (y = J u on entry, J assembled with kappa^2 = slope: the nonlinear residual)
extern "C" __global__ void pb_boltzmann_residual(const long long n, const int m, const double* __restrict__ ions,
                                                 const double vol, const int nx, const int ny, const int nz,
                                                 const double* __restrict__ u, const double* __restrict__ k2,
                                                 const double* __restrict__ slope, const double* __restrict__ f,
                                                 double* __restrict__ y)
{
    GRID_STRIDE_LOOP(c, n)
    {
        const int i = (int)(c % nx), j = (int)((c / nx) % ny), k3 = (int)(c / ((long long)nx * ny));
        if (!interior3(i, j, k3, nx, ny, nz)) {
            y[c] = 0.0;
            continue;
        }
        const double k = k2[c];
        double nn = 0.0;
        if (k > 0.0) {
            double s1, s2, p1, d;
            pb_ion_sums(m, ions, u[c], &s1, &s2, &p1, &d);
            nn = -s1 * k / d;
        }
        y[c] = y[c] + vol * (nn - slope[c] * u[c]) - f[c];
    }
}

// out[idx[m]] = values[m] (distinct indices)
extern "C" __global__ void pb_scatter(const long long m, const unsigned int* __restrict__ idx,
                                      const double* __restrict__ values, double* __restrict__ out)
{
    GRID_STRIDE_LOOP(c, m) { out[idx[c]] = values[c]; }
}

// out[m] = x[idx[m]]
extern "C" __global__ void pb_gather(const long long m, const unsigned int* __restrict__ idx,
                                     const double* __restrict__ x, double* __restrict__ out)
{
    GRID_STRIDE_LOOP(c, m) { out[c] = x[idx[c]]; }
}
