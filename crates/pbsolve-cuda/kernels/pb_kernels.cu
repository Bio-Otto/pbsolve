// pbsolve CUDA kernels, compiled at runtime with NVRTC (no nvcc at build time).
//
// Memory layout is identical to the CPU backend: idx = i + nx*(j + ny*k).
// The stencil expression is written in exactly the same order as
// StencilOperator::apply on the CPU and is compiled with --fmad=false, so the
// operator application is bit-identical between platforms; only the order of
// the dot-product reduction differs.

#define BLOCK 256

extern "C" __global__ void pb_apply(
    const int nx, const int ny, const int nz,
    const double cx, const double cy, const double cz,
    const double* __restrict__ ex,
    const double* __restrict__ ey,
    const double* __restrict__ ez,
    const double* __restrict__ kvol,
    const double* __restrict__ x,
    double* __restrict__ y)
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
            y[c] = 0.0;
            continue;
        }
        const double xc = x[c];
        y[c] = cx * (ex[c - 1] * (xc - x[c - 1]) + ex[c] * (xc - x[c + 1]))
             + cy * (ey[c - sy] * (xc - x[c - sy]) + ey[c] * (xc - x[c + sy]))
             + cz * (ez[c - sz] * (xc - x[c - sz]) + ez[c] * (xc - x[c + sz]))
             + kvol[c] * xc;
    }
}

// 27-point operator (multigrid coarse levels), coefficients stored per offset:
// coef[o*n + c], o = (dx+1) + 3*(dy+1) + 9*(dz+1). Same summation order as
// Stencil27::apply on the CPU.
extern "C" __global__ void pb_apply27(
    const int nx, const int ny, const int nz,
    const double* __restrict__ coef,
    const double* __restrict__ x,
    double* __restrict__ y)
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
            y[c] = 0.0;
            continue;
        }
        double acc = 0.0;
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

// y = (alpha * d) * x   (Jacobi preconditioner, smoother scaling)
extern "C" __global__ void pb_scale(
    const long long n, const double alpha,
    const double* __restrict__ d,
    const double* __restrict__ x,
    double* __restrict__ y)
{
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x; c < n; c += stride) {
        y[c] = (alpha * d[c]) * x[c];
    }
}

// y = alpha * x + beta * y
extern "C" __global__ void pb_axpby(
    const long long n, const double alpha,
    const double* __restrict__ x,
    const double beta,
    double* __restrict__ y)
{
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x; c < n; c += stride) {
        y[c] = alpha * x[c] + beta * y[c];
    }
}

// Multigrid transfer weights along one axis (see multigrid.rs).
__device__ int restrict_axis(const int c, const int coarsen, const int n, int* idx, double* w)
{
    if (!coarsen) {
        idx[0] = c;
        w[0] = 1.0;
        return 1;
    }
    int m = 0;
    const int f0 = 2 * c - 1, f1 = 2 * c, f2 = 2 * c + 1;
    if (f0 > 0 && f0 + 1 < n) { idx[m] = f0; w[m] = 0.5; ++m; }
    if (f1 > 0 && f1 + 1 < n) { idx[m] = f1; w[m] = 1.0; ++m; }
    if (f2 > 0 && f2 + 1 < n) { idx[m] = f2; w[m] = 0.5; ++m; }
    return m;
}

__device__ int prolong_axis(const int f, const int coarsen, int* idx, double* w)
{
    if (!coarsen) {
        idx[0] = f;
        w[0] = 1.0;
        return 1;
    }
    if (f % 2 == 0) {
        idx[0] = f / 2;
        w[0] = 1.0;
        return 1;
    }
    idx[0] = (f - 1) / 2;
    idx[1] = (f + 1) / 2;
    w[0] = 0.5;
    w[1] = 0.5;
    return 2;
}

// coarse = P^T fine (interior coarse nodes; 0 on the coarse boundary)
extern "C" __global__ void pb_restrict(
    const int nx, const int ny, const int nz,
    const int mx, const int my, const int mz,
    const int cx, const int cy, const int cz,
    const double* __restrict__ fine,
    double* __restrict__ coarse)
{
    const long long n = (long long)mx * my * mz;
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x; c < n; c += stride) {
        const int ic = (int)(c % mx);
        const int jc = (int)((c / mx) % my);
        const int kc = (int)(c / ((long long)mx * my));
        if (ic == 0 || jc == 0 || kc == 0 || ic == mx - 1 || jc == my - 1 || kc == mz - 1) {
            coarse[c] = 0.0;
            continue;
        }
        int fx[3], fy[3], fz[3];
        double wx[3], wy[3], wz[3];
        const int ax = restrict_axis(ic, cx, nx, fx, wx);
        const int ay = restrict_axis(jc, cy, ny, fy, wy);
        const int az = restrict_axis(kc, cz, nz, fz, wz);
        double acc = 0.0;
        for (int q = 0; q < az; ++q) {
            for (int b = 0; b < ay; ++b) {
                for (int a = 0; a < ax; ++a) {
                    const double w = wx[a] * wy[b] * wz[q];
                    acc += w * fine[fx[a] + (long long)nx * (fy[b] + (long long)ny * fz[q])];
                }
            }
        }
        coarse[c] = acc;
    }
}

// fine += P coarse (interior fine nodes)
extern "C" __global__ void pb_prolong_add(
    const int nx, const int ny, const int nz,
    const int mx, const int my, const int mz,
    const int cx, const int cy, const int cz,
    const double* __restrict__ coarse,
    double* __restrict__ fine)
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
        double wx[2], wy[2], wz[2];
        const int ax = prolong_axis(i, cx, gx, wx);
        const int ay = prolong_axis(j, cy, gy, wy);
        const int az = prolong_axis(k, cz, gz, wz);
        double acc = 0.0;
        for (int q = 0; q < az; ++q) {
            for (int b = 0; b < ay; ++b) {
                for (int a = 0; a < ax; ++a) {
                    const double w = wx[a] * wy[b] * wz[q];
                    acc += w * coarse[gx[a] + (long long)mx * (gy[b] + (long long)my * gz[q])];
                }
            }
        }
        fine[c] += acc;
    }
}

// y += alpha * x
extern "C" __global__ void pb_axpy(
    const long long n, const double alpha,
    const double* __restrict__ x,
    double* __restrict__ y)
{
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x; c < n; c += stride) {
        y[c] += alpha * x[c];
    }
}

// y = x + beta * y
extern "C" __global__ void pb_xpay(
    const long long n, const double beta,
    const double* __restrict__ x,
    double* __restrict__ y)
{
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x; c < n; c += stride) {
        y[c] = x[c] + beta * y[c];
    }
}

// Per-block partial sums of a.b. The host adds the partials in block order,
// which makes the reduction deterministic (no atomics). Requires blockDim.x == BLOCK.
extern "C" __global__ void pb_dot_partial(
    const long long n,
    const double* __restrict__ a,
    const double* __restrict__ b,
    double* __restrict__ partial)
{
    __shared__ double s[BLOCK];
    const long long stride = (long long)blockDim.x * gridDim.x;
    double acc = 0.0;
    for (long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x; c < n; c += stride) {
        acc += a[c] * b[c];
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
