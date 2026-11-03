use crate::nvrtc::{self, CodeType, Target};
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use pbsolve_core::multigrid::{Stencil27, Transfer};
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
        nvrtc::get()?;
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

/// The NVRTC library in use (loaded on first call).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NvrtcInfo {
    /// Name or path it was opened with.
    pub name: String,
    /// Resolved file (Linux only).
    pub path: Option<String>,
    /// NVRTC (CUDA) version, major and minor.
    pub version: (i32, i32),
    /// GPU architectures it can target (10·major + minor).
    pub archs: Vec<i32>,
}

/// Loads NVRTC if needed and describes it; `Err` explains why it is unusable.
pub fn nvrtc_info() -> std::result::Result<NvrtcInfo, String> {
    nvrtc::get().map(|n| NvrtcInfo {
        name: n.name.clone(),
        path: n.path.clone(),
        version: n.version,
        archs: n.archs.clone(),
    })
}

/// CUDA version supported by the driver (e.g. (13, 2)).
pub fn driver_version() -> Option<(i32, i32)> {
    if probe().is_err() {
        return None;
    }
    let mut v = 0;
    // SAFETY: the driver library is present (probe succeeded).
    let status = unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut v) };
    (status == cudarc::driver::sys::CUresult::CUDA_SUCCESS).then_some((v / 1000, (v % 1000) / 10))
}

pub struct CudaBackend {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    f_apply: CudaFunction,
    f_apply27: CudaFunction,
    f_scale: CudaFunction,
    f_axpy: CudaFunction,
    f_xpay: CudaFunction,
    f_axpby: CudaFunction,
    f_dot: CudaFunction,
    f_restrict: CudaFunction,
    f_prolong: CudaFunction,
    partials: Mutex<CudaSlice<f64>>,
    name: String,
    target: Target,
}

pub struct CudaOperator {
    dims: [i32; 3],
    coef: [f64; 3],
    n: usize,
    eps_x: CudaSlice<f64>,
    eps_y: CudaSlice<f64>,
    eps_z: CudaSlice<f64>,
    kvol: CudaSlice<f64>,
}

/// Device copy of a multigrid coarse-level (27-point) operator.
pub struct CudaCoarse {
    dims: [i32; 3],
    n: usize,
    coef: CudaSlice<f64>,
}

impl CudaBackend {
    /// Creates a context on device `ordinal` and compiles the kernels
    /// ([`CodeType::Auto`] unless `PBSOLVE_CUDA_CODE` says otherwise).
    pub fn new(ordinal: usize) -> Result<Self> {
        let code = CodeType::parse(std::env::var(nvrtc::ENV_CODE).ok().as_deref()).map_err(PbError::Backend)?;
        Self::with_code(ordinal, code)
    }

    /// Like [`new`](Self::new) with an explicit code type.
    pub fn with_code(ordinal: usize, code: CodeType) -> Result<Self> {
        if let Err(reason) = probe() {
            return Err(PbError::Backend(format!("CUDA unavailable: {reason}")));
        }
        let ctx = CudaContext::new(ordinal).map_err(cuda_err("context creation"))?;
        // SAFETY: every buffer of this backend is allocated, used and freed on
        // its own stream only, so cudarc's cross-stream events are not needed.
        unsafe { ctx.disable_event_tracking() };
        // A stream of its own (not the legacy default stream, which serialises
        // with every other stream), so several backends run concurrently.
        let stream = ctx.new_stream().map_err(cuda_err("stream creation"))?;
        let (major, minor) = ctx.compute_capability().map_err(cuda_err("device query"))?;
        let dev = ctx.name().map_err(cuda_err("device query"))?;
        let compiler = nvrtc::get().map_err(PbError::Backend)?;
        let target = nvrtc::select_target(10 * major + minor, &compiler.archs, code).map_err(PbError::Backend)?;
        // Keep a*b+c unfused so the stencil matches the CPU bit-for-bit.
        let image = compiler
            .compile(KERNELS, "pb_kernels.cu", &["--fmad=false"], target)
            .map_err(|e| PbError::Backend(format!("CUDA NVRTC compilation: {e}")))?;
        let module = ctx.load_module(Ptx::from_binary(image)).map_err(|e| {
            let (nv, nm) = compiler.version;
            let driver = driver_version().map_or("unknown".into(), |(a, b)| format!("{a}.{b}"));
            PbError::Backend(format!(
                "CUDA module load ({target} from NVRTC {nv}.{nm}, driver CUDA {driver}): {e:?}"
            ))
        })?;
        let load = |name: &str| module.load_function(name).map_err(cuda_err("function lookup"));
        let partials = stream
            .alloc_zeros::<f64>(MAX_REDUCTION_BLOCKS as usize)
            .map_err(cuda_err("allocation"))?;
        Ok(Self {
            f_apply: load("pb_apply")?,
            f_apply27: load("pb_apply27")?,
            f_scale: load("pb_scale")?,
            f_axpy: load("pb_axpy")?,
            f_xpay: load("pb_xpay")?,
            f_axpby: load("pb_axpby")?,
            f_dot: load("pb_dot_partial")?,
            f_restrict: load("pb_restrict")?,
            f_prolong: load("pb_prolong_add")?,
            partials: Mutex::new(partials),
            name: format!("{dev} (cc {major}.{minor}, device {ordinal})"),
            target,
            _ctx: ctx,
            stream,
        })
    }

    /// The code the kernels were built as, e.g. "CUBIN sm_75".
    pub fn code_target(&self) -> String {
        self.target.to_string()
    }

    fn grid_for(n: usize, cap: u32) -> LaunchConfig {
        let blocks = n.div_ceil(BLOCK as usize).clamp(1, cap as usize) as u32;
        LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    fn transfer_args(t: &Transfer) -> [i32; 9] {
        [
            t.fine[0] as i32,
            t.fine[1] as i32,
            t.fine[2] as i32,
            t.coarse[0] as i32,
            t.coarse[1] as i32,
            t.coarse[2] as i32,
            t.coarsen[0] as i32,
            t.coarsen[1] as i32,
            t.coarsen[2] as i32,
        ]
    }
}

impl Backend for CudaBackend {
    type Vector = CudaSlice<f64>;
    type Operator = CudaOperator;
    type Coarse = CudaCoarse;

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
        })
    }

    fn prepare_coarse(&self, op: Arc<Stencil27>) -> Result<Self::Coarse> {
        Ok(CudaCoarse {
            dims: op.dims.map(|d| d as i32),
            n: op.len(),
            coef: self.stream.clone_htod(&op.coef).map_err(cuda_err("upload"))?,
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

    fn read(&self, v: &Self::Vector, host: &mut [f64]) -> Result<()> {
        self.stream.memcpy_dtoh(v, host).map_err(cuda_err("download"))?;
        self.stream.synchronize().map_err(cuda_err("synchronize"))
    }

    fn write(&self, host: &[f64], v: &mut Self::Vector) -> Result<()> {
        self.stream.memcpy_htod(host, v).map_err(cuda_err("upload"))
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

    fn apply_coarse(&self, op: &Self::Coarse, x: &Self::Vector, y: &mut Self::Vector) -> Result<()> {
        let [nx, ny, nz] = op.dims;
        let mut b = self.stream.launch_builder(&self.f_apply27);
        b.arg(&nx).arg(&ny).arg(&nz).arg(&op.coef).arg(x).arg(y);
        // SAFETY: argument list matches the kernel signature; coef has 27 n values.
        unsafe { b.launch(Self::grid_for(op.n, MAX_GRID_BLOCKS)) }.map_err(cuda_err("pb_apply27"))?;
        Ok(())
    }

    fn scale(&self, alpha: f64, d: &Self::Vector, x: &Self::Vector, y: &mut Self::Vector) -> Result<()> {
        let n = x.len() as i64;
        let mut b = self.stream.launch_builder(&self.f_scale);
        b.arg(&n).arg(&alpha).arg(d).arg(x).arg(y);
        unsafe { b.launch(Self::grid_for(x.len(), MAX_GRID_BLOCKS)) }.map_err(cuda_err("pb_scale"))?;
        Ok(())
    }

    fn axpby(&self, alpha: f64, x: &Self::Vector, beta: f64, y: &mut Self::Vector) -> Result<()> {
        let n = x.len() as i64;
        let mut b = self.stream.launch_builder(&self.f_axpby);
        b.arg(&n).arg(&alpha).arg(x).arg(&beta).arg(y);
        unsafe { b.launch(Self::grid_for(x.len(), MAX_GRID_BLOCKS)) }.map_err(cuda_err("pb_axpby"))?;
        Ok(())
    }

    fn copy(&self, x: &Self::Vector, y: &mut Self::Vector) -> Result<()> {
        self.stream.memcpy_dtod(x, y).map_err(cuda_err("copy"))
    }

    fn restrict(&self, t: &Transfer, fine: &Self::Vector, coarse: &mut Self::Vector) -> Result<()> {
        let a = Self::transfer_args(t);
        let mut b = self.stream.launch_builder(&self.f_restrict);
        for v in &a {
            b.arg(v);
        }
        b.arg(fine).arg(coarse);
        let n = t.coarse.iter().product();
        unsafe { b.launch(Self::grid_for(n, MAX_GRID_BLOCKS)) }.map_err(cuda_err("pb_restrict"))?;
        Ok(())
    }

    fn prolong_add(&self, t: &Transfer, coarse: &Self::Vector, fine: &mut Self::Vector) -> Result<()> {
        let a = Self::transfer_args(t);
        let mut b = self.stream.launch_builder(&self.f_prolong);
        for v in &a {
            b.arg(v);
        }
        b.arg(coarse).arg(fine);
        let n = t.fine.iter().product();
        unsafe { b.launch(Self::grid_for(n, MAX_GRID_BLOCKS)) }.map_err(cuda_err("pb_prolong_add"))?;
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
