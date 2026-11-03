use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use pbsolve_core::operator::StencilOperator;
use pbsolve_core::{Backend, PbError, Result};
use std::sync::{Arc, Mutex};

const KERNELS: &str = include_str!("../kernels/pb_kernels.cu");
const BLOCK: u32 = 256;
const MAX_REDUCTION_BLOCKS: u32 = 1024;
const MAX_GRID_BLOCKS: u32 = 1 << 20;

fn cuda_err<E: std::fmt::Debug>(what: &str) -> impl FnOnce(E) -> PbError + '_ {
    move |e| PbError::Backend(format!("CUDA {what}: {e:?}"))
}

/// Probes the driver, NVRTC and device count without ever panicking.
fn probe() -> std::result::Result<usize, String> {
    let outcome = std::panic::catch_unwind(|| {
        // SAFETY: these only try to dlopen the libraries.
        if !unsafe { cudarc::driver::sys::is_culib_present() } {
            return Err("NVIDIA driver library (libcuda) not found".to_string());
        }
        if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
            return Err("NVRTC library not found (conda: install cuda-nvrtc)".to_string());
        }
        match CudaContext::device_count() {
            Ok(n) if n > 0 => Ok(n as usize),
            Ok(_) => Err("no CUDA device present".to_string()),
            Err(e) => Err(format!("cuInit/cuDeviceGetCount failed: {e:?}")),
        }
    });
    outcome.unwrap_or_else(|_| Err("CUDA initialisation panicked".to_string()))
}

pub fn is_available() -> bool {
    probe().is_ok()
}

pub fn device_count() -> usize {
    probe().unwrap_or(0)
}

pub fn unavailable_reason() -> Option<String> {
    probe().err()
}

pub struct CudaBackend {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    f_apply: CudaFunction,
    f_jacobi: CudaFunction,
    f_axpy: CudaFunction,
    f_xpay: CudaFunction,
    f_dot: CudaFunction,
    partials: Mutex<CudaSlice<f64>>,
    name: String,
}

pub struct CudaOperator {
    dims: [i32; 3],
    coef: [f64; 3],
    n: usize,
    eps_x: CudaSlice<f64>,
    eps_y: CudaSlice<f64>,
    eps_z: CudaSlice<f64>,
    kvol: CudaSlice<f64>,
    inv_diag: CudaSlice<f64>,
}

impl CudaBackend {
    /// Creates a context on device `ordinal` and JIT-compiles the kernels.
    pub fn new(ordinal: usize) -> Result<Self> {
        if let Err(reason) = probe() {
            return Err(PbError::Backend(format!("CUDA unavailable: {reason}")));
        }
        let ctx = CudaContext::new(ordinal).map_err(cuda_err("context creation"))?;
        let stream = ctx.default_stream();
        let opts = CompileOptions {
            // Keep a*b+c unfused so the stencil matches the CPU bit-for-bit.
            fmad: Some(false),
            name: Some("pb_kernels.cu".into()),
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(KERNELS, opts).map_err(cuda_err("NVRTC compilation"))?;
        let module = ctx.load_module(ptx).map_err(cuda_err("module load"))?;
        let load = |name: &str| module.load_function(name).map_err(cuda_err("function lookup"));
        let (major, minor) = ctx.compute_capability().map_err(cuda_err("device query"))?;
        let dev = ctx.name().map_err(cuda_err("device query"))?;
        let partials = stream
            .alloc_zeros::<f64>(MAX_REDUCTION_BLOCKS as usize)
            .map_err(cuda_err("allocation"))?;
        Ok(Self {
            f_apply: load("pb_apply")?,
            f_jacobi: load("pb_jacobi")?,
            f_axpy: load("pb_axpy")?,
            f_xpay: load("pb_xpay")?,
            f_dot: load("pb_dot_partial")?,
            partials: Mutex::new(partials),
            name: format!("{dev} (cc {major}.{minor}, device {ordinal})"),
            _ctx: ctx,
            stream,
        })
    }

    fn grid_for(n: usize, cap: u32) -> LaunchConfig {
        let blocks = n.div_ceil(BLOCK as usize).clamp(1, cap as usize) as u32;
        LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

impl Backend for CudaBackend {
    type Vector = CudaSlice<f64>;
    type Operator = CudaOperator;

    fn platform(&self) -> &'static str {
        "CUDA"
    }

    fn device_name(&self) -> String {
        self.name.clone()
    }

    fn prepare(&self, op: Arc<StencilOperator>) -> Result<Self::Operator> {
        let up = |v: &[f64]| self.stream.clone_htod(v).map_err(cuda_err("upload"));
        Ok(CudaOperator {
            dims: op.dims.map(|d| d as i32),
            coef: op.coef,
            n: op.len(),
            eps_x: up(&op.eps_x)?,
            eps_y: up(&op.eps_y)?,
            eps_z: up(&op.eps_z)?,
            kvol: up(&op.kvol)?,
            inv_diag: up(&op.inv_diag)?,
        })
    }

    fn upload(&self, host: &[f64]) -> Result<Self::Vector> {
        self.stream.clone_htod(host).map_err(cuda_err("upload"))
    }

    fn zeros(&self, n: usize) -> Result<Self::Vector> {
        self.stream.alloc_zeros::<f64>(n).map_err(cuda_err("allocation"))
    }

    fn download(&self, v: &Self::Vector) -> Result<Vec<f64>> {
        let out = self.stream.clone_dtoh(v).map_err(cuda_err("download"))?;
        self.stream.synchronize().map_err(cuda_err("synchronize"))?;
        Ok(out)
    }

    fn apply(&self, op: &Self::Operator, x: &Self::Vector, y: &mut Self::Vector) -> Result<()> {
        let [nx, ny, nz] = op.dims;
        let [cx, cy, cz] = op.coef;
        let mut b = self.stream.launch_builder(&self.f_apply);
        b.arg(&nx).arg(&ny).arg(&nz).arg(&cx).arg(&cy).arg(&cz);
        b.arg(&op.eps_x)
            .arg(&op.eps_y)
            .arg(&op.eps_z)
            .arg(&op.kvol)
            .arg(x)
            .arg(y);
        // SAFETY: argument list matches the kernel signature; buffers have length n.
        unsafe { b.launch(Self::grid_for(op.n, MAX_GRID_BLOCKS)) }.map_err(cuda_err("pb_apply"))?;
        Ok(())
    }

    fn precondition(&self, op: &Self::Operator, r: &Self::Vector, z: &mut Self::Vector) -> Result<()> {
        let n = op.n as i64;
        let mut b = self.stream.launch_builder(&self.f_jacobi);
        b.arg(&n).arg(&op.inv_diag).arg(r).arg(z);
        unsafe { b.launch(Self::grid_for(op.n, MAX_GRID_BLOCKS)) }.map_err(cuda_err("pb_jacobi"))?;
        Ok(())
    }

    fn dot(&self, a: &Self::Vector, bvec: &Self::Vector) -> Result<f64> {
        let n = a.len();
        let cfg = Self::grid_for(n, MAX_REDUCTION_BLOCKS);
        let blocks = cfg.grid_dim.0 as usize;
        let mut partials = self
            .partials
            .lock()
            .map_err(|_| PbError::Backend("poisoned lock".into()))?;
        let n64 = n as i64;
        {
            let mut b = self.stream.launch_builder(&self.f_dot);
            b.arg(&n64).arg(a).arg(bvec).arg(&mut *partials);
            unsafe { b.launch(cfg) }.map_err(cuda_err("pb_dot_partial"))?;
        }
        let host = self.stream.clone_dtoh(&*partials).map_err(cuda_err("download"))?;
        self.stream.synchronize().map_err(cuda_err("synchronize"))?;
        Ok(host[..blocks].iter().sum())
    }

    fn axpy(&self, alpha: f64, x: &Self::Vector, y: &mut Self::Vector) -> Result<()> {
        let n = x.len() as i64;
        let mut b = self.stream.launch_builder(&self.f_axpy);
        b.arg(&n).arg(&alpha).arg(x).arg(y);
        unsafe { b.launch(Self::grid_for(x.len(), MAX_GRID_BLOCKS)) }.map_err(cuda_err("pb_axpy"))?;
        Ok(())
    }

    fn xpay(&self, x: &Self::Vector, beta: f64, y: &mut Self::Vector) -> Result<()> {
        let n = x.len() as i64;
        let mut b = self.stream.launch_builder(&self.f_xpay);
        b.arg(&n).arg(&beta).arg(x).arg(y);
        unsafe { b.launch(Self::grid_for(x.len(), MAX_GRID_BLOCKS)) }.map_err(cuda_err("pb_xpay"))?;
        Ok(())
    }
}
