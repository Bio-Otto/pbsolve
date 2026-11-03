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

// z = r * inv_diag  (inv_diag is 0 on boundary nodes)
extern "C" __global__ void pb_jacobi(
    const long long n,
    const double* __restrict__ inv_diag,
    const double* __restrict__ r,
    double* __restrict__ z)
{
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x; c < n; c += stride) {
        z[c] = r[c] * inv_diag[c];
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
