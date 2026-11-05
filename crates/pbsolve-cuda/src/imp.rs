use crate::nvrtc::{self, CodeType, Target};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
    ValidAsZeroBits,
};
use cudarc::nvrtc::Ptx;
use pbsolve_core::multigrid::{Stencil27, Transfer};
use pbsolve_core::operator::StencilOperator;
use pbsolve_core::real::{convert, Real};
use pbsolve_core::{Backend, PbError, Result};
use std::marker::PhantomData;
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

/// Precisions the CUDA kernels are instantiated for.
pub trait CudaReal: Real + DeviceRepr + ValidAsZeroBits + Unpin {
    /// Host f64 data → device vector of this precision.
    fn upload(stream: &Arc<CudaStream>, host: &[f64]) -> Result<CudaSlice<Self>>;
    /// Device vector → host f64 buffer of the same length.
    fn read(stream: &Arc<CudaStream>, v: &CudaSlice<Self>, host: &mut [f64]) -> Result<()>;
    /// Host f64 buffer → existing device vector.
    fn write(stream: &Arc<CudaStream>, host: &[f64], v: &mut CudaSlice<Self>) -> Result<()>;
}

impl CudaReal for f64 {
    fn upload(stream: &Arc<CudaStream>, host: &[f64]) -> Result<CudaSlice<f64>> {
        stream.clone_htod(host).map_err(cuda_err("upload"))
    }
    fn read(stream: &Arc<CudaStream>, v: &CudaSlice<f64>, host: &mut [f64]) -> Result<()> {
        stream.memcpy_dtoh(v, host).map_err(cuda_err("download"))?;
        stream.synchronize().map_err(cuda_err("synchronize"))
    }
    fn write(stream: &Arc<CudaStream>, host: &[f64], v: &mut CudaSlice<f64>) -> Result<()> {
        stream.memcpy_htod(host, v).map_err(cuda_err("upload"))
    }
}

impl CudaReal for f32 {
    fn upload(stream: &Arc<CudaStream>, host: &[f64]) -> Result<CudaSlice<f32>> {
        stream
            .clone_htod(&convert::<f64, f32>(host))
            .map_err(cuda_err("upload"))
    }
    fn read(stream: &Arc<CudaStream>, v: &CudaSlice<f32>, host: &mut [f64]) -> Result<()> {
        let mut tmp = vec![0.0f32; host.len()];
        stream.memcpy_dtoh(v, &mut tmp).map_err(cuda_err("download"))?;
        stream.synchronize().map_err(cuda_err("synchronize"))?;
        host.iter_mut().zip(&tmp).for_each(|(h, t)| *h = f64::from(*t));
        Ok(())
    }
    fn write(stream: &Arc<CudaStream>, host: &[f64], v: &mut CudaSlice<f32>) -> Result<()> {
        // A copy from pageable memory returns once the source is staged, so
        // the temporary may be dropped right away (as for f64).
        let tmp = convert::<f64, f32>(host);
        stream.memcpy_htod(&tmp, v).map_err(cuda_err("upload"))
    }
}

/// Context, stream and compiled module, shared by a backend and its twin.
struct Shared {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    name: String,
    target: Target,
}

/// Kernels of one precision.
struct Kernels {
    apply: CudaFunction,
    apply27: CudaFunction,
    scale: CudaFunction,
    axpy: CudaFunction,
    xpay: CudaFunction,
    axpby: CudaFunction,
    dot: CudaFunction,
    dot2: CudaFunction,
    restrict: CudaFunction,
    prolong: CudaFunction,
    demote: CudaFunction,
    promote: CudaFunction,
}

impl Kernels {
    fn load(module: &Arc<CudaModule>, s: &str) -> Result<Self> {
        let f = |name: String| module.load_function(&name).map_err(cuda_err("function lookup"));
        Ok(Self {
            apply: f(format!("pb_apply_{s}"))?,
            apply27: f(format!("pb_apply27_{s}"))?,
            scale: f(format!("pb_scale_{s}"))?,
            axpy: f(format!("pb_axpy_{s}"))?,
            xpay: f(format!("pb_xpay_{s}"))?,
            axpby: f(format!("pb_axpby_{s}"))?,
            dot: f(format!("pb_dot_partial_{s}"))?,
            dot2: f(format!("pb_dot2_partial_{s}"))?,
            restrict: f(format!("pb_restrict_{s}"))?,
            prolong: f(format!("pb_prolong_add_{s}"))?,
            demote: f(format!("pb_convert_{s}_f32"))?,
            promote: f(format!("pb_convert_f32_{s}"))?,
        })
    }
}

/// CUDA backend computing in `T` on one device and stream.
pub struct Dev<T: CudaReal> {
    shared: Arc<Shared>,
    k: Kernels,
    partials: Mutex<CudaSlice<f64>>,
    _t: PhantomData<fn() -> T>,
}

/// The f64 CUDA backend.
pub type CudaBackend = Dev<f64>;
/// Its f32 twin (mixed-precision multigrid); same context, stream and module.
pub type CudaSingle = Dev<f32>;

/// Device copy of the fine-grid 7-point operator.
pub struct CudaOperator<T> {
    dims: [i32; 3],
    coef: [T; 3],
    n: usize,
    eps_x: CudaSlice<T>,
    eps_y: CudaSlice<T>,
    eps_z: CudaSlice<T>,
    kvol: CudaSlice<T>,
}

/// Device copy of a multigrid coarse-level (27-point) operator.
pub struct CudaCoarse<T> {
    dims: [i32; 3],
    n: usize,
    coef: CudaSlice<T>,
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
        Dev::from_shared(Arc::new(Shared {
            _ctx: ctx,
            stream,
            module,
            name: format!("{dev} (cc {major}.{minor}, device {ordinal})"),
            target,
        }))
    }
}

impl<T: CudaReal> Dev<T> {
    fn from_shared(shared: Arc<Shared>) -> Result<Self> {
        let k = Kernels::load(&shared.module, T::NAME)?;
        let partials = shared
            .stream
            .alloc_zeros::<f64>(2 * MAX_REDUCTION_BLOCKS as usize)
            .map_err(cuda_err("allocation"))?;
        Ok(Self {
            shared,
            k,
            partials: Mutex::new(partials),
            _t: PhantomData,
        })
    }

    fn stream(&self) -> &Arc<CudaStream> {
        &self.shared.stream
    }

    /// The code the kernels were built as, e.g. "CUBIN sm_75".
    pub fn code_target(&self) -> String {
        self.shared.target.to_string()
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

    /// Element-wise kernel `f(n, x, y)` over vectors of possibly different type.
    fn convert<S: DeviceRepr, D: DeviceRepr>(
        &self,
        f: &CudaFunction,
        x: &CudaSlice<S>,
        y: &mut CudaSlice<D>,
    ) -> Result<()> {
        let n = x.len() as i64;
        let mut b = self.stream().launch_builder(f);
        b.arg(&n).arg(x).arg(y);
        // SAFETY: pb_convert_* (n, const S*, D*); both buffers have length n.
        unsafe { b.launch(Self::grid_for(x.len(), MAX_GRID_BLOCKS)) }.map_err(cuda_err("pb_convert"))?;
        Ok(())
    }
}

impl<T: CudaReal> Backend for Dev<T> {
    type Vector = CudaSlice<T>;
    type Operator = CudaOperator<T>;
    type Coarse = CudaCoarse<T>;
    type Single = CudaSingle;

    fn platform(&self) -> &'static str {
        "CUDA"
    }

    fn device_name(&self) -> String {
        self.shared.name.clone()
    }

    fn single(&self) -> Result<CudaSingle> {
        Dev::from_shared(self.shared.clone())
    }

    fn demote(&self, x: &Self::Vector, y: &mut CudaSlice<f32>) -> Result<()> {
        self.convert(&self.k.demote, x, y)
    }

    fn promote(&self, x: &CudaSlice<f32>, y: &mut Self::Vector) -> Result<()> {
        self.convert(&self.k.promote, x, y)
    }

    fn prepare(&self, op: Arc<StencilOperator>) -> Result<Self::Operator> {
        let up = |v: &[f64]| T::upload(self.stream(), v);
        Ok(CudaOperator {
            dims: op.dims.map(|d| d as i32),
            coef: op.coef.map(T::from_f64),
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
            coef: T::upload(self.stream(), &op.coef)?,
        })
    }

    fn upload(&self, host: &[f64]) -> Result<Self::Vector> {
        T::upload(self.stream(), host)
    }

    fn zeros(&self, n: usize) -> Result<Self::Vector> {
        self.stream().alloc_zeros::<T>(n).map_err(cuda_err("allocation"))
    }

    fn download(&self, v: &Self::Vector) -> Result<Vec<f64>> {
        let mut out = vec![0.0; v.len()];
        T::read(self.stream(), v, &mut out)?;
        Ok(out)
    }

    fn read(&self, v: &Self::Vector, host: &mut [f64]) -> Result<()> {
        T::read(self.stream(), v, host)
    }

    fn write(&self, host: &[f64], v: &mut Self::Vector) -> Result<()> {
        T::write(self.stream(), host, v)
    }

    fn apply(&self, op: &Self::Operator, x: &Self::Vector, y: &mut Self::Vector) -> Result<()> {
        let [nx, ny, nz] = op.dims;
        let [cx, cy, cz] = op.coef;
        let mut b = self.stream().launch_builder(&self.k.apply);
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
        let mut b = self.stream().launch_builder(&self.k.apply27);
        b.arg(&nx).arg(&ny).arg(&nz).arg(&op.coef).arg(x).arg(y);
        // SAFETY: argument list matches the kernel signature; coef has 27 n values.
        unsafe { b.launch(Self::grid_for(op.n, MAX_GRID_BLOCKS)) }.map_err(cuda_err("pb_apply27"))?;
        Ok(())
    }

    fn scale(&self, alpha: f64, d: &Self::Vector, x: &Self::Vector, y: &mut Self::Vector) -> Result<()> {
        let n = x.len() as i64;
        let alpha = T::from_f64(alpha);
        let mut b = self.stream().launch_builder(&self.k.scale);
        b.arg(&n).arg(&alpha).arg(d).arg(x).arg(y);
        // SAFETY: argument list matches the kernel signature.
        unsafe { b.launch(Self::grid_for(x.len(), MAX_GRID_BLOCKS)) }.map_err(cuda_err("pb_scale"))?;
        Ok(())
    }

    fn axpby(&self, alpha: f64, x: &Self::Vector, beta: f64, y: &mut Self::Vector) -> Result<()> {
        let n = x.len() as i64;
        let (alpha, beta) = (T::from_f64(alpha), T::from_f64(beta));
        let mut b = self.stream().launch_builder(&self.k.axpby);
        b.arg(&n).arg(&alpha).arg(x).arg(&beta).arg(y);
        // SAFETY: argument list matches the kernel signature.
        unsafe { b.launch(Self::grid_for(x.len(), MAX_GRID_BLOCKS)) }.map_err(cuda_err("pb_axpby"))?;
        Ok(())
    }

    fn copy(&self, x: &Self::Vector, y: &mut Self::Vector) -> Result<()> {
        self.stream().memcpy_dtod(x, y).map_err(cuda_err("copy"))
    }

    fn restrict(&self, t: &Transfer, fine: &Self::Vector, coarse: &mut Self::Vector) -> Result<()> {
        let a = Self::transfer_args(t);
        let mut b = self.stream().launch_builder(&self.k.restrict);
        for v in &a {
            b.arg(v);
        }
        b.arg(fine).arg(coarse);
        let n = t.coarse.iter().product();
        // SAFETY: argument list matches the kernel signature.
        unsafe { b.launch(Self::grid_for(n, MAX_GRID_BLOCKS)) }.map_err(cuda_err("pb_restrict"))?;
        Ok(())
    }

    fn prolong_add(&self, t: &Transfer, coarse: &Self::Vector, fine: &mut Self::Vector) -> Result<()> {
        let a = Self::transfer_args(t);
        let mut b = self.stream().launch_builder(&self.k.prolong);
        for v in &a {
            b.arg(v);
        }
        b.arg(coarse).arg(fine);
        let n = t.fine.iter().product();
        // SAFETY: argument list matches the kernel signature.
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
            let mut b = self.stream().launch_builder(&self.k.dot);
            b.arg(&n64).arg(a).arg(bvec).arg(&mut *partials);
            // SAFETY: argument list matches the kernel signature; partials holds MAX_REDUCTION_BLOCKS values.
            unsafe { b.launch(cfg) }.map_err(cuda_err("pb_dot_partial"))?;
        }
        let host = self.stream().clone_dtoh(&*partials).map_err(cuda_err("download"))?;
        self.stream().synchronize().map_err(cuda_err("synchronize"))?;
        Ok(host[..blocks].iter().sum())
    }

    fn dot2(&self, a: &Self::Vector, bvec: &Self::Vector, cvec: &Self::Vector) -> Result<(f64, f64)> {
        let n = a.len();
        let cfg = Self::grid_for(n, MAX_REDUCTION_BLOCKS);
        let blocks = cfg.grid_dim.0 as usize;
        let mut partials = self
            .partials
            .lock()
            .map_err(|_| PbError::Backend("poisoned lock".into()))?;
        let n64 = n as i64;
        {
            let mut b = self.stream().launch_builder(&self.k.dot2);
            b.arg(&n64).arg(a).arg(bvec).arg(cvec).arg(&mut *partials);
            // SAFETY: argument list matches the kernel signature; partials holds 2 MAX_REDUCTION_BLOCKS values.
            unsafe { b.launch(cfg) }.map_err(cuda_err("pb_dot2_partial"))?;
        }
        let host = self.stream().clone_dtoh(&*partials).map_err(cuda_err("download"))?;
        self.stream().synchronize().map_err(cuda_err("synchronize"))?;
        Ok((host[..blocks].iter().sum(), host[blocks..2 * blocks].iter().sum()))
    }

    fn axpy(&self, alpha: f64, x: &Self::Vector, y: &mut Self::Vector) -> Result<()> {
        let n = x.len() as i64;
        let alpha = T::from_f64(alpha);
        let mut b = self.stream().launch_builder(&self.k.axpy);
        b.arg(&n).arg(&alpha).arg(x).arg(y);
        // SAFETY: argument list matches the kernel signature.
        unsafe { b.launch(Self::grid_for(x.len(), MAX_GRID_BLOCKS)) }.map_err(cuda_err("pb_axpy"))?;
        Ok(())
    }

    fn xpay(&self, x: &Self::Vector, beta: f64, y: &mut Self::Vector) -> Result<()> {
        let n = x.len() as i64;
        let beta = T::from_f64(beta);
        let mut b = self.stream().launch_builder(&self.k.xpay);
        b.arg(&n).arg(&beta).arg(x).arg(y);
        // SAFETY: argument list matches the kernel signature.
        unsafe { b.launch(Self::grid_for(x.len(), MAX_GRID_BLOCKS)) }.map_err(cuda_err("pb_xpay"))?;
        Ok(())
    }
}
