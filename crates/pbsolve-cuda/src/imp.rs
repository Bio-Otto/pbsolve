use crate::nvrtc::{self, CodeType, Target};
use cudarc::driver::{
    sys, CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
    ValidAsZeroBits,
};
use cudarc::nvrtc::Ptx;
use pbsolve_core::backend::{check_gather, check_scatter};
use pbsolve_core::multigrid::{LevelOp, Stencil27, Transfer};
use pbsolve_core::operator::StencilOperator;
use pbsolve_core::real::{convert, zeros_touched, Real};
use pbsolve_core::surface::Sphere;
use pbsolve_core::{Backend, Grid, PbError, Result};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

const KERNELS: &str = include_str!("../kernels/pb_kernels.cu");
const BLOCK: u32 = 256;
const MAX_REDUCTION_BLOCKS: u32 = 1024;
const MAX_GRID_BLOCKS: u32 = 1 << 20;
/// Spheres (one warp each) per block of `pb_mark_spheres` (MARK_WARPS there).
const MARK_WARPS: usize = 4;

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
    ctx: Arc<CudaContext>,
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
    residual: CudaFunction,
    residual_update: CudaFunction,
    residual27: CudaFunction,
    residual27_update: CudaFunction,
    cheb_first: CudaFunction,
    cheb_next: CudaFunction,
    cg_update: CudaFunction,
    promote_dot: CudaFunction,
}

/// Multigrid setup and assembly kernels (f64 only).
struct Setup {
    galerkin7: CudaFunction,
    galerkin27: CudaFunction,
    symmetrise: CudaFunction,
    diagonal: CudaFunction,
    gershgorin7: CudaFunction,
    gershgorin27: CudaFunction,
    fill: CudaFunction,
    mark: CudaFunction,
    operator7: CudaFunction,
    source_rhs: CudaFunction,
    scatter: CudaFunction,
    gather: CudaFunction,
    boltzmann_slope: CudaFunction,
    boltzmann_residual: CudaFunction,
}

impl Setup {
    fn load(module: &Arc<CudaModule>) -> Result<Self> {
        let f = |name: &str| module.load_function(name).map_err(cuda_err("function lookup"));
        Ok(Self {
            galerkin7: f("pb_galerkin7")?,
            galerkin27: f("pb_galerkin27")?,
            symmetrise: f("pb_symmetrise27")?,
            diagonal: f("pb_diagonal27")?,
            gershgorin7: f("pb_gershgorin7")?,
            gershgorin27: f("pb_gershgorin27")?,
            fill: f("pb_fill")?,
            mark: f("pb_mark_spheres")?,
            operator7: f("pb_operator7")?,
            source_rhs: f("pb_source_rhs")?,
            scatter: f("pb_scatter")?,
            gather: f("pb_gather")?,
            boltzmann_slope: f("pb_boltzmann_slope")?,
            boltzmann_residual: f("pb_boltzmann_residual")?,
        })
    }
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
            residual: f(format!("pb_residual_{s}"))?,
            residual_update: f(format!("pb_residual_update_{s}"))?,
            residual27: f(format!("pb_residual27_{s}"))?,
            residual27_update: f(format!("pb_residual27_update_{s}"))?,
            cheb_first: f(format!("pb_cheb_first_{s}"))?,
            cheb_next: f(format!("pb_cheb_next_{s}"))?,
            cg_update: f(format!("pb_cg_update_{s}"))?,
            promote_dot: f(format!("pb_promote_dot_{s}"))?,
        })
    }
}

/// CUDA backend computing in `T` on one device and stream.
pub struct Dev<T: CudaReal> {
    shared: Arc<Shared>,
    k: Kernels,
    /// Present on the f64 backend only.
    setup: Option<Setup>,
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
            ctx,
            stream,
            module,
            name: format!("{dev} (cc {major}.{minor}, device {ordinal})"),
            target,
        }))
    }
}

/// The most memory this process held in the device's memory pool at once
/// since the last [`CudaBackend::reset_pool_peak`] (bytes, all backends of
/// the process together).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolPeak {
    /// Allocated buffers.
    pub used: u64,
    /// Device memory the pool had reserved for them (at least `used`).
    pub reserved: u64,
}

/// The device's memory pool, which every backend of the process on that
/// device allocates from (cudarc allocates with `cuMemAllocAsync`).
fn device_pool(ctx: &CudaContext) -> Result<sys::CUmemoryPool> {
    ctx.bind_to_thread().map_err(cuda_err("memory pool query"))?;
    let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
    // SAFETY: `pool` is a valid out-pointer and the device belongs to the live context.
    unsafe { sys::cuDeviceGetMemPool(&mut pool, ctx.cu_device()) }
        .result()
        .map_err(cuda_err("memory pool query"))?;
    Ok(pool)
}

/// A cuuint64_t attribute of a pool (memory amounts, release threshold).
fn pool_attribute(pool: sys::CUmemoryPool, attr: sys::CUmemPool_attribute) -> Result<u64> {
    let mut v: u64 = 0;
    // SAFETY: every attribute read here is a cuuint64_t value.
    unsafe { sys::cuMemPoolGetAttribute(pool, attr, (&mut v as *mut u64).cast()) }
        .result()
        .map_err(cuda_err("memory pool query"))?;
    Ok(v)
}

fn set_pool_attribute(pool: sys::CUmemoryPool, attr: sys::CUmemPool_attribute, mut v: u64) -> Result<()> {
    // SAFETY: every attribute written here is a cuuint64_t value.
    unsafe { sys::cuMemPoolSetAttribute(pool, attr, (&mut v as *mut u64).cast()) }
        .result()
        .map_err(cuda_err("memory pool setting"))
}

const RELEASE_THRESHOLD: sys::CUmemPool_attribute = sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD;

/// Memory that calculations in progress retain in one device's pool
/// ([`Backend::retain_memory`]).
struct Retention {
    device: sys::CUdevice,
    /// The pool's release threshold before the first retention, restored
    /// after the last one.
    base: u64,
    /// Bytes retained by the calculations in progress.
    bytes: u64,
}

/// One entry per device with calculations in progress. The pool releases
/// the memory it holds beyond its release threshold at every
/// synchronisation (threshold 0 by default, so the buffers of a focusing
/// level are returned to the driver and reserved again for the next one);
/// while calculations run, the threshold is raised by the bytes they retain.
static RETAINED: Mutex<Vec<Retention>> = Mutex::new(Vec::new());

impl<T: CudaReal> Dev<T> {
    fn mem_pool(&self) -> Result<sys::CUmemoryPool> {
        device_pool(&self.shared.ctx)
    }

    fn retain(&self, bytes: u64) -> Result<()> {
        let pool = self.mem_pool()?;
        let device = self.shared.ctx.cu_device();
        let mut all = RETAINED.lock().unwrap_or_else(|e| e.into_inner());
        let i = match all.iter().position(|r| r.device == device) {
            Some(i) => i,
            None => {
                let base = pool_attribute(pool, RELEASE_THRESHOLD)?;
                all.push(Retention { device, base, bytes: 0 });
                all.len() - 1
            }
        };
        let retained = all[i].bytes.saturating_add(bytes);
        if let Err(e) = set_pool_attribute(pool, RELEASE_THRESHOLD, all[i].base.saturating_add(retained)) {
            if all[i].bytes == 0 {
                all.swap_remove(i);
            }
            return Err(e);
        }
        all[i].bytes = retained;
        Ok(())
    }

    fn release(&self, bytes: u64) -> Result<()> {
        // Buffers freed on this stream can be returned once their frees have run.
        self.stream().synchronize().map_err(cuda_err("synchronize"))?;
        let pool = self.mem_pool()?;
        let device = self.shared.ctx.cu_device();
        let mut all = RETAINED.lock().unwrap_or_else(|e| e.into_inner());
        let Some(i) = all.iter().position(|r| r.device == device) else {
            return Ok(());
        };
        all[i].bytes = all[i].bytes.saturating_sub(bytes);
        let keep = all[i].base.saturating_add(all[i].bytes);
        if all[i].bytes == 0 {
            all.swap_remove(i);
        }
        set_pool_attribute(pool, RELEASE_THRESHOLD, keep)?;
        // Return the unused memory now rather than at some later synchronisation.
        // SAFETY: a valid pool; the driver keeps memory still in use.
        unsafe { sys::cuMemPoolTrimTo(pool, usize::try_from(keep).unwrap_or(usize::MAX)) }
            .result()
            .map_err(cuda_err("memory pool trim"))
    }
}

impl CudaBackend {
    /// Free and total memory of the device in bytes (`cuMemGetInfo`): memory
    /// of other processes and of this one counts as taken.
    pub fn memory_info(&self) -> Result<(u64, u64)> {
        let (free, total) = self.shared.ctx.mem_get_info().map_err(cuda_err("memory query"))?;
        Ok((free as u64, total as u64))
    }

    /// Peak memory of the device's pool since the last reset (the stream is
    /// synchronised first, so queued work is included).
    pub fn pool_peak(&self) -> Result<PoolPeak> {
        self.stream().synchronize().map_err(cuda_err("synchronisation"))?;
        let pool = self.mem_pool()?;
        Ok(PoolPeak {
            used: pool_attribute(pool, sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH)?,
            reserved: pool_attribute(pool, sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH)?,
        })
    }

    /// Memory of the device's pool now, like [`pool_peak`](Self::pool_peak)
    /// (the stream is synchronised first, which also lets the pool release
    /// memory beyond its threshold).
    pub fn pool_usage(&self) -> Result<PoolPeak> {
        self.stream().synchronize().map_err(cuda_err("synchronisation"))?;
        let pool = self.mem_pool()?;
        Ok(PoolPeak {
            used: pool_attribute(pool, sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT)?,
            reserved: pool_attribute(pool, sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT)?,
        })
    }

    /// Memory the device's pool keeps reserved when it is not in use
    /// (bytes): its release threshold, raised by the calculations in
    /// progress ([`Backend::retain_memory`]).
    pub fn pool_release_threshold(&self) -> Result<u64> {
        pool_attribute(self.mem_pool()?, RELEASE_THRESHOLD)
    }

    /// Restarts [`pool_peak`](Self::pool_peak) from the current usage.
    pub fn reset_pool_peak(&self) -> Result<()> {
        self.stream().synchronize().map_err(cuda_err("synchronisation"))?;
        let pool = self.mem_pool()?;
        for attr in [
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,
        ] {
            // The high watermarks may only be reset to zero.
            set_pool_attribute(pool, attr, 0)?;
        }
        Ok(())
    }
}

impl<T: CudaReal> Dev<T> {
    fn from_shared(shared: Arc<Shared>) -> Result<Self> {
        let k = Kernels::load(&shared.module, T::NAME)?;
        let setup = if T::NAME == "f64" {
            Some(Setup::load(&shared.module)?)
        } else {
            None
        };
        let partials = shared
            .stream
            .alloc_zeros::<f64>(2 * MAX_REDUCTION_BLOCKS as usize)
            .map_err(cuda_err("allocation"))?;
        Ok(Self {
            shared,
            k,
            setup,
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

    fn setup(&self) -> Result<&Setup> {
        self.setup
            .as_ref()
            .ok_or_else(|| PbError::Backend("multigrid setup needs the f64 backend".into()))
    }

    fn assembly(&self) -> Result<&Setup> {
        self.setup
            .as_ref()
            .ok_or_else(|| PbError::Backend("assembly needs the f64 backend".into()))
    }

    /// The `K` sums of a reduction kernel's per-block partials (sum k of
    /// block b at `partials[k·blocks + b]`), each added in block order: the
    /// result of every dot product.
    fn block_sums<const K: usize>(&self, partials: &CudaSlice<f64>, blocks: usize) -> Result<[f64; K]> {
        let host = self.stream().clone_dtoh(partials).map_err(cuda_err("download"))?;
        self.stream().synchronize().map_err(cuda_err("synchronize"))?;
        Ok(std::array::from_fn(|k| host[k * blocks..(k + 1) * blocks].iter().sum()))
    }

    fn lock_partials(&self) -> Result<std::sync::MutexGuard<'_, CudaSlice<f64>>> {
        self.partials
            .lock()
            .map_err(|_| PbError::Backend("poisoned lock".into()))
    }

    /// A device buffer of `n` values that the caller overwrites completely.
    fn uninit(&self, n: usize) -> Result<CudaSlice<T>> {
        // SAFETY: every caller launches a kernel that writes all n values
        // before the buffer is read.
        unsafe { self.stream().alloc::<T>(n) }.map_err(cuda_err("allocation"))
    }

    /// A vector of this precision rounded to f32 (a new device buffer).
    fn to_f32(&self, x: &CudaSlice<T>) -> Result<CudaSlice<f32>> {
        let mut y = self
            .stream()
            .alloc_zeros::<f32>(x.len())
            .map_err(cuda_err("allocation"))?;
        self.convert(&self.k.demote, x, &mut y)?;
        Ok(y)
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
        let mut out = zeros_touched(v.len());
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
        let mut partials = self.lock_partials()?;
        let n64 = n as i64;
        {
            let mut b = self.stream().launch_builder(&self.k.dot);
            b.arg(&n64).arg(a).arg(bvec).arg(&mut *partials);
            // SAFETY: argument list matches the kernel signature; partials holds MAX_REDUCTION_BLOCKS values.
            unsafe { b.launch(cfg) }.map_err(cuda_err("pb_dot_partial"))?;
        }
        let [ab] = self.block_sums(&partials, blocks)?;
        Ok(ab)
    }

    fn dot2(&self, a: &Self::Vector, bvec: &Self::Vector, cvec: &Self::Vector) -> Result<(f64, f64)> {
        let n = a.len();
        let cfg = Self::grid_for(n, MAX_REDUCTION_BLOCKS);
        let blocks = cfg.grid_dim.0 as usize;
        let mut partials = self.lock_partials()?;
        let n64 = n as i64;
        {
            let mut b = self.stream().launch_builder(&self.k.dot2);
            b.arg(&n64).arg(a).arg(bvec).arg(cvec).arg(&mut *partials);
            // SAFETY: argument list matches the kernel signature; partials holds 2 MAX_REDUCTION_BLOCKS values.
            unsafe { b.launch(cfg) }.map_err(cuda_err("pb_dot2_partial"))?;
        }
        let [ab, ac] = self.block_sums(&partials, blocks)?;
        Ok((ab, ac))
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

    // The smoother steps fused into one kernel each (pb_residual*,
    // pb_cheb_first, pb_cheb_next), with the expressions of the separate
    // kernels, so the results equal the step-by-step defaults bit for bit.
    // The scratch vector `t` is not used.

    fn residual(
        &self,
        op: LevelOp<'_, Self>,
        x: &Self::Vector,
        b: Option<&Self::Vector>,
        _t: &mut Self::Vector,
        r: &mut Self::Vector,
    ) -> Result<()> {
        let (n, name) = match (op, b) {
            (LevelOp::Fine(op), b) => {
                let [nx, ny, nz] = op.dims;
                let [cx, cy, cz] = op.coef;
                let f = if b.is_some() {
                    &self.k.residual
                } else {
                    &self.k.residual_update
                };
                let mut l = self.stream().launch_builder(f);
                l.arg(&nx).arg(&ny).arg(&nz).arg(&cx).arg(&cy).arg(&cz);
                l.arg(&op.eps_x).arg(&op.eps_y).arg(&op.eps_z).arg(&op.kvol).arg(x);
                if let Some(b) = b {
                    l.arg(b);
                }
                l.arg(&mut *r);
                // SAFETY: argument list matches pb_residual(_update); buffers have length n.
                unsafe { l.launch(Self::grid_for(op.n, MAX_GRID_BLOCKS)) }.map_err(cuda_err("pb_residual"))?;
                (op.n, "pb_residual")
            }
            (LevelOp::Coarse(op), b) => {
                let [nx, ny, nz] = op.dims;
                let f = if b.is_some() {
                    &self.k.residual27
                } else {
                    &self.k.residual27_update
                };
                let mut l = self.stream().launch_builder(f);
                l.arg(&nx).arg(&ny).arg(&nz).arg(&op.coef).arg(x);
                if let Some(b) = b {
                    l.arg(b);
                }
                l.arg(&mut *r);
                // SAFETY: argument list matches pb_residual27(_update); coef has 27 n values.
                unsafe { l.launch(Self::grid_for(op.n, MAX_GRID_BLOCKS)) }.map_err(cuda_err("pb_residual27"))?;
                (op.n, "pb_residual27")
            }
        };
        debug_assert!(x.len() == n && r.len() == n, "{name}: vector lengths");
        Ok(())
    }

    fn chebyshev_first(
        &self,
        s: f64,
        w: &Self::Vector,
        r: &Self::Vector,
        d: &mut Self::Vector,
        x: &mut Self::Vector,
        overwrite: bool,
    ) -> Result<()> {
        let n = r.len() as i64;
        let s = T::from_f64(s);
        let overwrite = i32::from(overwrite);
        let mut b = self.stream().launch_builder(&self.k.cheb_first);
        b.arg(&n).arg(&s).arg(w).arg(r).arg(d).arg(x).arg(&overwrite);
        // SAFETY: argument list matches the kernel signature.
        unsafe { b.launch(Self::grid_for(r.len(), MAX_GRID_BLOCKS)) }.map_err(cuda_err("pb_cheb_first"))?;
        Ok(())
    }

    fn chebyshev_next(
        &self,
        c1: f64,
        w: &Self::Vector,
        r: &Self::Vector,
        c2: f64,
        _t: &mut Self::Vector,
        d: &mut Self::Vector,
        x: &mut Self::Vector,
    ) -> Result<()> {
        let n = r.len() as i64;
        let (c1, c2) = (T::from_f64(c1), T::from_f64(c2));
        let mut b = self.stream().launch_builder(&self.k.cheb_next);
        b.arg(&n).arg(&c1).arg(w).arg(r).arg(&c2).arg(d).arg(x);
        // SAFETY: argument list matches the kernel signature.
        unsafe { b.launch(Self::grid_for(r.len(), MAX_GRID_BLOCKS)) }.map_err(cuda_err("pb_cheb_next"))?;
        Ok(())
    }

    // Two CG steps fused into one kernel each (pb_cg_update, pb_promote_dot),
    // with the expressions of the separate kernels and the grid of `dot`
    // (each thread adds the same elements in the same order), so the results
    // equal the step-by-step defaults bit for bit. `apply_dot` keeps its
    // default, `apply` and `dot`: on the dot's grid of at most 1024 blocks,
    // which run in waves, the stencil no longer finds the neighbours one
    // plane away in the L2 cache, and A p alone takes 1.4–1.5× longer than
    // with one thread per node; the fused kernel was slower than the pair.

    fn cg_update(
        &self,
        alpha: f64,
        p: &Self::Vector,
        q: &Self::Vector,
        x: &mut Self::Vector,
        r: &mut Self::Vector,
        z: Option<&Self::Vector>,
        rounded: Option<&mut CudaSlice<f32>>,
    ) -> Result<(f64, f64)> {
        let n = r.len();
        let lengths_match = [p.len(), q.len(), x.len()].iter().all(|&m| m == n)
            && z.is_none_or(|z| z.len() == n)
            && rounded.as_ref().is_none_or(|v| v.len() == n);
        if !lengths_match {
            return Err(PbError::Backend("cg_update: vectors of different lengths".into()));
        }
        let (a, minus_a) = (T::from_f64(alpha), T::from_f64(-alpha));
        let cfg = Self::grid_for(n, MAX_REDUCTION_BLOCKS);
        let blocks = cfg.grid_dim.0 as usize;
        let n64 = n as i64;
        // A null device pointer for an absent `z` or `rounded`.
        let null: u64 = 0;
        let mut partials = self.lock_partials()?;
        {
            let mut b = self.stream().launch_builder(&self.k.cg_update);
            b.arg(&n64)
                .arg(&a)
                .arg(&minus_a)
                .arg(p)
                .arg(q)
                .arg(&mut *x)
                .arg(&mut *r);
            match z {
                Some(z) => b.arg(z),
                None => b.arg(&null),
            };
            match rounded {
                Some(v) => b.arg(v),
                None => b.arg(&null),
            };
            b.arg(&mut *partials);
            // SAFETY: argument list matches pb_cg_update (z and r32 may be null); vectors have
            // length n, partials 2 MAX_REDUCTION_BLOCKS values.
            unsafe { b.launch(cfg) }.map_err(cuda_err("pb_cg_update"))?;
        }
        if z.is_some() {
            let [rr, rz] = self.block_sums(&partials, blocks)?;
            Ok((rr, rz))
        } else {
            let [rr] = self.block_sums(&partials, blocks)?;
            Ok((rr, 0.0))
        }
    }

    fn promote_dot(&self, z32: &CudaSlice<f32>, r: &Self::Vector, z: &mut Self::Vector) -> Result<f64> {
        let n = r.len();
        if z32.len() != n || z.len() != n {
            return Err(PbError::Backend("promote_dot: vector lengths".into()));
        }
        let cfg = Self::grid_for(n, MAX_REDUCTION_BLOCKS);
        let blocks = cfg.grid_dim.0 as usize;
        let n64 = n as i64;
        let mut partials = self.lock_partials()?;
        {
            let mut b = self.stream().launch_builder(&self.k.promote_dot);
            b.arg(&n64).arg(z32).arg(r).arg(&mut *z).arg(&mut *partials);
            // SAFETY: argument list matches pb_promote_dot; vectors have length n.
            unsafe { b.launch(cfg) }.map_err(cuda_err("pb_promote_dot"))?;
        }
        let [rz] = self.block_sums(&partials, blocks)?;
        Ok(rz)
    }

    fn galerkin(&self, t: &Transfer, fine: LevelOp<'_, Self>) -> Result<(Self::Coarse, Self::Vector)> {
        let setup = self.setup()?;
        let a = Self::transfer_args(t);
        let n: usize = t.coarse.iter().product();
        let mut coef = self.stream().alloc_zeros::<T>(27 * n).map_err(cuda_err("allocation"))?;
        let cfg = Self::grid_for(n, MAX_GRID_BLOCKS);
        match fine {
            LevelOp::Fine(op) => {
                let mut b = self.stream().launch_builder(&setup.galerkin7);
                for v in &a {
                    b.arg(v);
                }
                let [cx, cy, cz] = &op.coef;
                b.arg(cx).arg(cy).arg(cz);
                b.arg(&op.eps_x)
                    .arg(&op.eps_y)
                    .arg(&op.eps_z)
                    .arg(&op.kvol)
                    .arg(&mut coef);
                // SAFETY: argument list matches pb_galerkin7; coef holds 27 n values.
                unsafe { b.launch(cfg) }.map_err(cuda_err("pb_galerkin7"))?;
            }
            LevelOp::Coarse(op) => {
                let mut b = self.stream().launch_builder(&setup.galerkin27);
                for v in &a {
                    b.arg(v);
                }
                b.arg(&op.coef).arg(&mut coef);
                // SAFETY: argument list matches pb_galerkin27; coef holds 27 n values.
                unsafe { b.launch(cfg) }.map_err(cuda_err("pb_galerkin27"))?;
            }
        }
        let [mx, my, mz] = [a[3], a[4], a[5]];
        {
            let mut b = self.stream().launch_builder(&setup.symmetrise);
            b.arg(&mx).arg(&my).arg(&mz).arg(&mut coef);
            // SAFETY: argument list matches pb_symmetrise27 (13 n pairs).
            unsafe { b.launch(Self::grid_for(13 * n, MAX_GRID_BLOCKS)) }.map_err(cuda_err("pb_symmetrise27"))?;
        }
        let mut inv_diag = self.zeros(n)?;
        let mut bad = self.stream().alloc_zeros::<i32>(1).map_err(cuda_err("allocation"))?;
        {
            let n64 = n as i64;
            let mut b = self.stream().launch_builder(&setup.diagonal);
            b.arg(&n64).arg(&mut coef).arg(&mut inv_diag).arg(&mut bad);
            // SAFETY: argument list matches pb_diagonal27.
            unsafe { b.launch(cfg) }.map_err(cuda_err("pb_diagonal27"))?;
        }
        let flag = self.stream().clone_dtoh(&bad).map_err(cuda_err("download"))?;
        self.stream().synchronize().map_err(cuda_err("synchronize"))?;
        let coarse = CudaCoarse {
            dims: t.coarse.map(|d| d as i32),
            n,
            coef,
        };
        if flag[0] != 0 {
            // Report the first bad diagonal in node order, as the host does.
            let host = self.coarse_to_host(&coarse).err();
            return Err(host.unwrap_or_else(|| PbError::Backend("multigrid: non-positive Galerkin diagonal".into())));
        }
        Ok((coarse, inv_diag))
    }

    fn gershgorin_lmax(&self, level: LevelOp<'_, Self>) -> Result<f64> {
        let setup = self.setup()?;
        let n = match &level {
            LevelOp::Fine(op) => op.n,
            LevelOp::Coarse(op) => op.n,
        };
        let cfg = Self::grid_for(n, MAX_REDUCTION_BLOCKS);
        let blocks = cfg.grid_dim.0 as usize;
        let mut partials = self
            .partials
            .lock()
            .map_err(|_| PbError::Backend("poisoned lock".into()))?;
        match level {
            LevelOp::Fine(op) => {
                let [nx, ny, nz] = op.dims;
                let [cx, cy, cz] = op.coef;
                let mut b = self.stream().launch_builder(&setup.gershgorin7);
                b.arg(&nx).arg(&ny).arg(&nz).arg(&cx).arg(&cy).arg(&cz);
                b.arg(&op.eps_x)
                    .arg(&op.eps_y)
                    .arg(&op.eps_z)
                    .arg(&op.kvol)
                    .arg(&mut *partials);
                // SAFETY: argument list matches pb_gershgorin7; partials holds MAX_REDUCTION_BLOCKS values.
                unsafe { b.launch(cfg) }.map_err(cuda_err("pb_gershgorin7"))?;
            }
            LevelOp::Coarse(op) => {
                let [nx, ny, nz] = op.dims;
                let mut b = self.stream().launch_builder(&setup.gershgorin27);
                b.arg(&nx).arg(&ny).arg(&nz).arg(&op.coef).arg(&mut *partials);
                // SAFETY: argument list matches pb_gershgorin27; partials holds MAX_REDUCTION_BLOCKS values.
                unsafe { b.launch(cfg) }.map_err(cuda_err("pb_gershgorin27"))?;
            }
        }
        let host = self.stream().clone_dtoh(&*partials).map_err(cuda_err("download"))?;
        self.stream().synchronize().map_err(cuda_err("synchronize"))?;
        // Per-block maxima (identity 0); max is exact in any order.
        Ok(host[..blocks].iter().fold(0.0, |m, &v| f64::max(m, v)))
    }

    fn coarse_to_host(&self, op: &Self::Coarse) -> Result<Stencil27> {
        let mut coef = vec![0.0; op.coef.len()];
        T::read(self.stream(), &op.coef, &mut coef)?;
        Stencil27::from_coef(op.dims.map(|d| d as usize), coef)
    }

    fn demote_operator(&self, op: &Self::Operator) -> Result<CudaOperator<f32>> {
        Ok(CudaOperator {
            dims: op.dims,
            coef: op.coef.map(|c| f32::from_f64(c.to_f64())),
            n: op.n,
            eps_x: self.to_f32(&op.eps_x)?,
            eps_y: self.to_f32(&op.eps_y)?,
            eps_z: self.to_f32(&op.eps_z)?,
            kvol: self.to_f32(&op.kvol)?,
        })
    }

    fn demote_coarse(&self, op: &Self::Coarse) -> Result<CudaCoarse<f32>> {
        Ok(CudaCoarse {
            dims: op.dims,
            n: op.n,
            coef: self.to_f32(&op.coef)?,
        })
    }

    fn mark_maps(
        &self,
        grid: &Grid,
        shifts: &[[f64; 3]],
        fill: f64,
        passes: &[(&[Sphere], f64)],
    ) -> Result<Vec<Self::Vector>> {
        let k = self.assembly()?;
        let n = grid.len();
        let [nx, ny, nz] = grid.dims.map(|d| d as i32);
        let [hx, hy, hz] = grid.spacing;
        // Every pass is uploaded once for all shifts.
        let mut uploaded = Vec::with_capacity(passes.len());
        for (spheres, value) in passes {
            if spheres.is_empty() {
                continue;
            }
            let flat: Vec<f64> = spheres
                .iter()
                .flat_map(|s| [s.center[0], s.center[1], s.center[2], s.radius])
                .collect();
            let buf = self.stream().clone_htod(&flat).map_err(cuda_err("upload"))?;
            uploaded.push((buf, spheres.len(), *value));
        }
        let mut maps = Vec::with_capacity(shifts.len());
        for shift in shifts {
            let mut map = self.uninit(n)?;
            {
                let n64 = n as i64;
                let mut b = self.stream().launch_builder(&k.fill);
                b.arg(&n64).arg(&fill).arg(&mut map);
                // SAFETY: argument list matches pb_fill; map holds n values.
                unsafe { b.launch(Self::grid_for(n, MAX_GRID_BLOCKS)) }.map_err(cuda_err("pb_fill"))?;
            }
            let [minx, miny, minz] = [0, 1, 2].map(|a| grid.origin[a] + shift[a]);
            for (buf, count, value) in &uploaded {
                let count64 = *count as i64;
                let mut b = self.stream().launch_builder(&k.mark);
                b.arg(&nx).arg(&ny).arg(&nz).arg(&hx).arg(&hy).arg(&hz);
                b.arg(&minx).arg(&miny).arg(&minz).arg(&count64).arg(buf).arg(value);
                b.arg(&mut map);
                let cfg = LaunchConfig {
                    grid_dim: (count.div_ceil(MARK_WARPS).min(MAX_GRID_BLOCKS as usize) as u32, 1, 1),
                    block_dim: (32 * MARK_WARPS as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                // SAFETY: argument list matches pb_mark_spheres; buf holds 4 count values, map n.
                unsafe { b.launch(cfg) }.map_err(cuda_err("pb_mark_spheres"))?;
            }
            maps.push(map);
        }
        Ok(maps)
    }

    fn scatter(&self, n: usize, idx: &[usize], values: &[f64]) -> Result<Self::Vector> {
        let k = self.assembly()?;
        check_scatter(n, idx, values)?;
        if n > u32::MAX as usize {
            return Err(PbError::Backend(format!("scatter: {n} values exceed 32-bit indices")));
        }
        let mut out = self.zeros(n)?;
        if idx.is_empty() {
            return Ok(out);
        }
        let idx32: Vec<u32> = idx.iter().map(|&c| c as u32).collect();
        let idx_dev = self.stream().clone_htod(&idx32).map_err(cuda_err("upload"))?;
        let val_dev = self.stream().clone_htod(values).map_err(cuda_err("upload"))?;
        let m = idx.len() as i64;
        let mut b = self.stream().launch_builder(&k.scatter);
        b.arg(&m).arg(&idx_dev).arg(&val_dev).arg(&mut out);
        // SAFETY: argument list matches pb_scatter; every index is below n (checked).
        unsafe { b.launch(Self::grid_for(idx.len(), MAX_GRID_BLOCKS)) }.map_err(cuda_err("pb_scatter"))?;
        Ok(out)
    }

    fn assemble_operator(&self, grid: &Grid, maps: [Self::Vector; 4]) -> Result<(Self::Operator, Self::Vector)> {
        let k = self.assembly()?;
        let n = grid.len();
        if let Some(m) = maps.iter().find(|m| m.len() != n) {
            return Err(PbError::Backend(format!(
                "a map has {} values, grid has {n} nodes",
                m.len()
            )));
        }
        let [eps_x, eps_y, eps_z, mut kvol] = maps;
        let coef = StencilOperator::face_coefficients(grid);
        let vol = grid.cell_volume();
        let dims = grid.dims.map(|d| d as i32);
        let mut inv_diag = self.uninit(n)?;
        {
            let [nx, ny, nz] = dims;
            let [cx, cy, cz] = coef;
            let mut b = self.stream().launch_builder(&k.operator7);
            b.arg(&nx).arg(&ny).arg(&nz).arg(&cx).arg(&cy).arg(&cz).arg(&vol);
            b.arg(&eps_x).arg(&eps_y).arg(&eps_z).arg(&mut kvol).arg(&mut inv_diag);
            // SAFETY: argument list matches pb_operator7; every buffer holds n values.
            unsafe { b.launch(Self::grid_for(n, MAX_GRID_BLOCKS)) }.map_err(cuda_err("pb_operator7"))?;
        }
        let op = CudaOperator {
            dims,
            coef: coef.map(T::from_f64),
            n,
            eps_x,
            eps_y,
            eps_z,
            kvol,
        };
        Ok((op, inv_diag))
    }

    fn source_rhs(&self, src: f64, q: &Self::Vector, d: &Self::Vector, y: &mut Self::Vector) -> Result<()> {
        let k = self.assembly()?;
        let n = y.len() as i64;
        let mut b = self.stream().launch_builder(&k.source_rhs);
        b.arg(&n).arg(&src).arg(q).arg(d).arg(y);
        // SAFETY: argument list matches pb_source_rhs; the vectors have the same length.
        unsafe { b.launch(Self::grid_for(n as usize, MAX_GRID_BLOCKS)) }.map_err(cuda_err("pb_source_rhs"))?;
        Ok(())
    }

    fn boltzmann_slope(
        &self,
        ions: &[[f64; 2]],
        u: &Self::Vector,
        kappa_sq: &Self::Vector,
        slope: &mut Self::Vector,
    ) -> Result<()> {
        let k = self.assembly()?;
        let n = slope.len() as i64;
        let m = ions.len() as i32;
        let flat: Vec<f64> = ions.iter().flatten().copied().collect();
        let ions_dev = self.stream().clone_htod(&flat).map_err(cuda_err("upload"))?;
        let mut b = self.stream().launch_builder(&k.boltzmann_slope);
        b.arg(&n).arg(&m).arg(&ions_dev).arg(u).arg(kappa_sq).arg(slope);
        // SAFETY: argument list matches pb_boltzmann_slope; the vectors have the same length.
        unsafe { b.launch(Self::grid_for(n as usize, MAX_GRID_BLOCKS)) }.map_err(cuda_err("pb_boltzmann_slope"))?;
        Ok(())
    }

    fn boltzmann_residual(
        &self,
        ions: &[[f64; 2]],
        vol: f64,
        dims: [usize; 3],
        u: &Self::Vector,
        kappa_sq: &Self::Vector,
        slope: &Self::Vector,
        f: &Self::Vector,
        y: &mut Self::Vector,
    ) -> Result<()> {
        let k = self.assembly()?;
        let n = y.len() as i64;
        let m = ions.len() as i32;
        let [nx, ny, nz] = dims.map(|d| d as i32);
        let flat: Vec<f64> = ions.iter().flatten().copied().collect();
        let ions_dev = self.stream().clone_htod(&flat).map_err(cuda_err("upload"))?;
        let mut b = self.stream().launch_builder(&k.boltzmann_residual);
        b.arg(&n).arg(&m).arg(&ions_dev).arg(&vol).arg(&nx).arg(&ny).arg(&nz);
        b.arg(u).arg(kappa_sq).arg(slope).arg(f).arg(y);
        // SAFETY: argument list matches pb_boltzmann_residual; the vectors have the same length.
        unsafe { b.launch(Self::grid_for(n as usize, MAX_GRID_BLOCKS)) }.map_err(cuda_err("pb_boltzmann_residual"))?;
        Ok(())
    }

    fn operator_to_host(&self, op: &Self::Operator) -> Result<StencilOperator> {
        self.assembly()?;
        let read = |v: &CudaSlice<T>| -> Result<Vec<f64>> {
            let mut host = vec![0.0; op.n];
            T::read(self.stream(), v, &mut host)?;
            Ok(host)
        };
        Ok(StencilOperator::from_parts(
            op.dims.map(|d| d as usize),
            op.coef.map(|c| c.to_f64()),
            read(&op.eps_x)?,
            read(&op.eps_y)?,
            read(&op.eps_z)?,
            read(&op.kvol)?,
        ))
    }

    fn gather(&self, v: &Self::Vector, idx: &[usize]) -> Result<Vec<f64>> {
        let k = self.assembly()?;
        check_gather(v.len(), idx)?;
        if v.len() > u32::MAX as usize {
            return Err(PbError::Backend(format!(
                "gather: {} values exceed 32-bit indices",
                v.len()
            )));
        }
        if idx.is_empty() {
            return Ok(Vec::new());
        }
        let idx32: Vec<u32> = idx.iter().map(|&c| c as u32).collect();
        let idx_dev = self.stream().clone_htod(&idx32).map_err(cuda_err("upload"))?;
        let mut out = self.uninit(idx.len())?;
        let m = idx.len() as i64;
        let mut b = self.stream().launch_builder(&k.gather);
        b.arg(&m).arg(&idx_dev).arg(v).arg(&mut out);
        // SAFETY: argument list matches pb_gather; every index is below v.len()
        // (checked) and the kernel writes all m values of `out`.
        unsafe { b.launch(Self::grid_for(idx.len(), MAX_GRID_BLOCKS)) }.map_err(cuda_err("pb_gather"))?;
        self.download(&out)
    }

    fn synchronize(&self) -> Result<()> {
        self.stream().synchronize().map_err(cuda_err("synchronize"))
    }

    fn runs_on_device(&self) -> bool {
        true
    }

    fn retain_memory(&self, bytes: u64) -> Result<()> {
        self.retain(bytes)
    }

    fn release_memory(&self, bytes: u64) -> Result<()> {
        self.release(bytes)
    }
}
