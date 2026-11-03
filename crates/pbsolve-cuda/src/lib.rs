//! CUDA backend for pbsolve.
//!
//! * The NVIDIA driver (`libcuda`) and NVRTC are loaded at runtime, so the
//!   same binary works on machines without a GPU: [`is_available`] simply
//!   returns `false` and callers fall back to the CPU backend.
//! * Kernels are compiled with NVRTC on first use, for the GPU actually
//!   present; no nvcc and no fat binaries are needed at build time.
//! * Without the `cuda-12` / `cuda-13` feature this crate compiles to a stub.

#[cfg(all(feature = "cuda-12", feature = "cuda-13"))]
compile_error!("enable only one of the `cuda-12` / `cuda-13` features");

#[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
mod imp;
#[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
pub use imp::{device_count, is_available, unavailable_reason, CudaBackend, CudaOperator};

/// True if this build contains the CUDA backend.
pub const fn compiled_with_cuda() -> bool {
    cfg!(any(feature = "cuda-12", feature = "cuda-13"))
}

/// CUDA major version the bindings were built for (0 if none).
pub const fn cuda_major() -> u32 {
    if cfg!(feature = "cuda-13") {
        13
    } else if cfg!(feature = "cuda-12") {
        12
    } else {
        0
    }
}

#[cfg(not(any(feature = "cuda-12", feature = "cuda-13")))]
mod stub {
    pub fn is_available() -> bool {
        false
    }
    pub fn device_count() -> usize {
        0
    }
    pub fn unavailable_reason() -> Option<String> {
        Some("this build of pbsolve was compiled without CUDA support (CPU variant)".into())
    }
}
#[cfg(not(any(feature = "cuda-12", feature = "cuda-13")))]
pub use stub::{device_count, is_available, unavailable_reason};
