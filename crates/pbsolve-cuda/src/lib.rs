//! CUDA backend for pbsolve.
//!
//! * The NVIDIA driver (`libcuda`) and NVRTC are loaded at runtime, so the
//!   same binary works on machines without a GPU: [`is_available`] simply
//!   returns `false` and callers fall back to the CPU backend.
//! * Kernels are compiled with NVRTC on first use, to a CUBIN for the GPU
//!   actually present (PTX only for GPUs newer than the NVRTC); no nvcc and
//!   no fat binaries are needed at build time. NVRTC is loaded by its
//!   versioned name (`libnvrtc.so.<major>`), so the library the Python
//!   package preloads wins over a system toolkit; `PBSOLVE_NVRTC` overrides
//!   the choice and `PBSOLVE_CUDA_CODE=ptx|cubin` forces the code type.
//! * Without the `cuda-12` / `cuda-13` feature this crate compiles to a stub.

#[cfg(all(feature = "cuda-12", feature = "cuda-13"))]
compile_error!("enable only one of the `cuda-12` / `cuda-13` features");

#[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
mod imp;
#[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
mod nvrtc;
#[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
pub use imp::{
    device_count, driver_version, is_available, nvrtc_info, unavailable_reason, CudaBackend, CudaCoarse, CudaOperator,
    CudaReal, CudaSingle, Dev, NvrtcInfo, PoolPeak,
};
#[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
pub use nvrtc::CodeType;

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
        Some(NO_CUDA.into())
    }
    pub fn driver_version() -> Option<(i32, i32)> {
        None
    }
    /// See the CUDA build's `NvrtcInfo`.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct NvrtcInfo {
        pub name: String,
        pub path: Option<String>,
        pub version: (i32, i32),
        pub archs: Vec<i32>,
    }
    pub fn nvrtc_info() -> Result<NvrtcInfo, String> {
        Err(NO_CUDA.into())
    }
    const NO_CUDA: &str = "this build of pbsolve was compiled without CUDA support (CPU variant)";
}
#[cfg(not(any(feature = "cuda-12", feature = "cuda-13")))]
pub use stub::{device_count, driver_version, is_available, nvrtc_info, unavailable_reason, NvrtcInfo};
