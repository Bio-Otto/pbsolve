//! Runtime binding to NVRTC with an explicit library choice and CUBIN output.
//!
//! Two problems made the GPU path fragile on machines with more than one CUDA
//! installation:
//!
//! * cudarc's loader tries the unversioned `libnvrtc.so` first. That name only
//!   exists in full toolkits, so a system toolkit (e.g. `/usr/local/cuda`) won
//!   over the NVRTC shipped with the Python environment.
//! * PTX produced by an NVRTC newer than the driver is rejected
//!   (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`), whereas a CUBIN for the device's
//!   architecture loads on any driver of the same major version (CUDA
//!   minor-version compatibility).
//!
//! This module therefore loads `$PBSOLVE_NVRTC` if set, otherwise the
//! versioned name of the CUDA major version pbsolve was built for (which
//! resolves to a library the Python package preloaded, as the dynamic loader
//! matches already-loaded objects by soname), and compiles to CUBIN. PTX is
//! used only for devices newer than the NVRTC, where the driver must be newer
//! than the NVRTC anyway.

use libloading::Library;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::sync::{Mutex, OnceLock};

/// Environment variable naming the NVRTC library to load (a path or a name).
pub(crate) const ENV_LIBRARY: &str = "PBSOLVE_NVRTC";
/// Environment variable forcing the code type: `ptx` or `cubin` (diagnostics).
pub(crate) const ENV_CODE: &str = "PBSOLVE_CUDA_CODE";

type Program = *mut c_void;
type Status = c_int;

/// What NVRTC emits and the driver loads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    /// Machine code for `sm_XY` (XY = 10·major + minor).
    Cubin(i32),
    /// PTX for `compute_XY`, JIT-compiled by the driver.
    Ptx(i32),
}

impl Target {
    pub(crate) fn option(self) -> String {
        match self {
            Target::Cubin(a) => format!("--gpu-architecture=sm_{a}"),
            Target::Ptx(a) => format!("--gpu-architecture=compute_{a}"),
        }
    }
}

impl std::fmt::Display for Target {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Target::Cubin(a) => write!(f, "CUBIN sm_{a}"),
            Target::Ptx(a) => write!(f, "PTX compute_{a}"),
        }
    }
}

/// Which code to build the kernels as.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CodeType {
    /// CUBIN for the device if NVRTC supports it, PTX otherwise.
    #[default]
    Auto,
    /// PTX, JIT-compiled by the driver (needs a driver at least as new as NVRTC).
    Ptx,
    /// CUBIN for the device (fails if NVRTC does not know the architecture).
    Cubin,
}

impl CodeType {
    /// Parses the value of `PBSOLVE_CUDA_CODE` (unset or empty = `Auto`).
    pub(crate) fn parse(value: Option<&str>) -> Result<Self, String> {
        match value.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
            None | Some("") | Some("auto") => Ok(Self::Auto),
            Some("ptx") => Ok(Self::Ptx),
            Some("cubin") => Ok(Self::Cubin),
            Some(other) => Err(format!("{ENV_CODE}={other}: expected 'auto', 'cubin' or 'ptx'")),
        }
    }
}

/// Chooses the code for a device of architecture `device` (10·major + minor)
/// given the architectures the loaded NVRTC supports (empty if unknown).
///
/// * supported → CUBIN for the device;
/// * device newer than every supported architecture → PTX for the newest
///   supported one, which the (necessarily newer) driver JIT-compiles;
/// * device older than every supported architecture → error.
pub(crate) fn select_target(device: i32, supported: &[i32], pref: CodeType) -> Result<Target, String> {
    let newest_below = supported.iter().copied().filter(|&a| a <= device).max();
    match pref {
        CodeType::Ptx => {
            return match (supported.is_empty(), newest_below) {
                (true, _) => Ok(Target::Ptx(device)),
                (false, Some(a)) => Ok(Target::Ptx(a)),
                (false, None) => Err(too_old(device, supported)),
            }
        }
        CodeType::Cubin => {
            return if supported.is_empty() || supported.contains(&device) {
                Ok(Target::Cubin(device))
            } else {
                Err(format!(
                    "NVRTC cannot emit a CUBIN for sm_{device} (supported: {supported:?})"
                ))
            }
        }
        CodeType::Auto => {}
    }
    if supported.is_empty() || supported.contains(&device) {
        return Ok(Target::Cubin(device));
    }
    match newest_below {
        Some(a) => Ok(Target::Ptx(a)),
        None => Err(too_old(device, supported)),
    }
}

fn too_old(device: i32, supported: &[i32]) -> String {
    let oldest = supported.iter().min().copied().unwrap_or(0);
    format!(
        "GPU architecture sm_{device} is older than the oldest one this NVRTC supports (sm_{oldest}); \
         use an NVRTC from an older CUDA release or the CPU platform"
    )
}

/// Library names tried in order when [`ENV_LIBRARY`] is not set.
pub(crate) fn candidates(cuda_major: u32) -> Vec<String> {
    if cfg!(windows) {
        vec![
            format!("nvrtc64_{cuda_major}0_0.dll"),
            format!("nvrtc64_{cuda_major}0.dll"),
            format!("nvrtc64_{cuda_major}.dll"),
            "nvrtc64.dll".into(),
            "nvrtc.dll".into(),
        ]
    } else if cfg!(target_os = "macos") {
        vec!["libnvrtc.dylib".into()]
    } else {
        // Versioned first: it matches the soname of a preloaded library and is
        // the name runtime-only installations (conda, pip) provide.
        vec![format!("libnvrtc.so.{cuda_major}"), "libnvrtc.so".into()]
    }
}

type CreateFn = unsafe extern "C" fn(
    *mut Program,
    *const c_char,
    *const c_char,
    c_int,
    *const *const c_char,
    *const *const c_char,
) -> Status;
type CompileFn = unsafe extern "C" fn(Program, c_int, *const *const c_char) -> Status;
type DestroyFn = unsafe extern "C" fn(*mut Program) -> Status;
type SizeFn = unsafe extern "C" fn(Program, *mut usize) -> Status;
type BytesFn = unsafe extern "C" fn(Program, *mut c_char) -> Status;
type ErrorStringFn = unsafe extern "C" fn(Status) -> *const c_char;
type VersionFn = unsafe extern "C" fn(*mut c_int, *mut c_int) -> Status;
type NumArchsFn = unsafe extern "C" fn(*mut c_int) -> Status;
type ArchsFn = unsafe extern "C" fn(*mut c_int) -> Status;

/// A loaded NVRTC library.
pub(crate) struct Nvrtc {
    /// Name or path the library was opened with.
    pub name: String,
    /// File the dynamic loader resolved it to (Linux only).
    pub path: Option<String>,
    pub version: (i32, i32),
    /// Architectures NVRTC can target, as 10·major + minor.
    pub archs: Vec<i32>,
    create: CreateFn,
    compile: CompileFn,
    destroy: DestroyFn,
    log_size: SizeFn,
    log: BytesFn,
    cubin_size: SizeFn,
    cubin: BytesFn,
    ptx_size: SizeFn,
    ptx: BytesFn,
    error_string: ErrorStringFn,
    // Keeps the function pointers above valid; never dropped (static).
    _lib: Library,
}

static NVRTC: OnceLock<Result<Nvrtc, String>> = OnceLock::new();
/// Serialises compilations and caches the image per (architecture, target).
static IMAGES: Mutex<Vec<(Target, Vec<u8>)>> = Mutex::new(Vec::new());

/// The process-wide NVRTC, loaded on first use.
pub(crate) fn get() -> Result<&'static Nvrtc, String> {
    NVRTC
        .get_or_init(|| std::panic::catch_unwind(load).unwrap_or_else(|_| Err("loading NVRTC panicked".into())))
        .as_ref()
        .map_err(Clone::clone)
}

fn load() -> Result<Nvrtc, String> {
    let names = match std::env::var(ENV_LIBRARY) {
        Ok(v) if !v.trim().is_empty() => vec![v.trim().to_string()],
        _ => candidates(crate::cuda_major()),
    };
    let mut errors = Vec::new();
    for name in &names {
        // SAFETY: loading NVRTC runs no initialisation code with preconditions.
        match unsafe { Library::new(name) } {
            Ok(lib) => return Nvrtc::from_library(lib, name.clone()),
            Err(e) => errors.push(format!("{name}: {e}")),
        }
    }
    let hint = if std::env::var_os(ENV_LIBRARY).is_some() {
        format!("check {ENV_LIBRARY}")
    } else {
        "install it with `conda install cuda-nvrtc` or `pip install nvidia-cuda-nvrtc`, \
         or point PBSOLVE_NVRTC at libnvrtc"
            .to_string()
    };
    Err(format!("NVRTC library not found ({hint}); tried {}", errors.join("; ")))
}

impl Nvrtc {
    fn from_library(lib: Library, name: String) -> Result<Self, String> {
        // SAFETY: the signatures follow nvrtc.h, stable since CUDA 11.2.
        unsafe {
            macro_rules! sym {
                ($t:ty, $s:literal) => {
                    *lib.get::<$t>(concat!($s, "\0").as_bytes())
                        .map_err(|e| format!("{name} lacks {}: {e}", $s))?
                };
            }
            let version_fn: VersionFn = sym!(VersionFn, "nvrtcVersion");
            let num_archs: NumArchsFn = sym!(NumArchsFn, "nvrtcGetNumSupportedArchs");
            let archs_fn: ArchsFn = sym!(ArchsFn, "nvrtcGetSupportedArchs");
            let create: CreateFn = sym!(CreateFn, "nvrtcCreateProgram");
            let (mut major, mut minor) = (0, 0);
            if version_fn(&mut major, &mut minor) != 0 {
                return Err(format!("{name}: nvrtcVersion failed"));
            }
            let mut n = 0;
            let mut archs = Vec::new();
            if num_archs(&mut n) == 0 && n > 0 {
                archs = vec![0; n as usize];
                if archs_fn(archs.as_mut_ptr()) != 0 {
                    archs.clear();
                }
            }
            let compile = sym!(CompileFn, "nvrtcCompileProgram");
            let destroy = sym!(DestroyFn, "nvrtcDestroyProgram");
            let log_size = sym!(SizeFn, "nvrtcGetProgramLogSize");
            let log = sym!(BytesFn, "nvrtcGetProgramLog");
            let cubin_size = sym!(SizeFn, "nvrtcGetCUBINSize");
            let cubin = sym!(BytesFn, "nvrtcGetCUBIN");
            let ptx_size = sym!(SizeFn, "nvrtcGetPTXSize");
            let ptx = sym!(BytesFn, "nvrtcGetPTX");
            let error_string = sym!(ErrorStringFn, "nvrtcGetErrorString");
            Ok(Self {
                path: library_path(create as *const c_void),
                name,
                version: (major, minor),
                archs,
                create,
                compile,
                destroy,
                log_size,
                log,
                cubin_size,
                cubin,
                ptx_size,
                ptx,
                error_string,
                _lib: lib,
            })
        }
    }

    fn check(&self, status: Status, what: &str) -> Result<(), String> {
        if status == 0 {
            return Ok(());
        }
        // SAFETY: nvrtcGetErrorString returns a static string for any code.
        let msg = unsafe { CStr::from_ptr((self.error_string)(status)) };
        Err(format!("{what}: {}", msg.to_string_lossy()))
    }

    /// Compiles `src` for `target`; returns the module image (a CUBIN, or
    /// NUL-terminated PTX) ready for `cuModuleLoadData`. Images are cached
    /// per target for the lifetime of the process.
    pub(crate) fn compile(&self, src: &str, name: &str, options: &[&str], target: Target) -> Result<Vec<u8>, String> {
        let mut cache = IMAGES.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, image)) = cache.iter().find(|(t, _)| *t == target) {
            return Ok(image.clone());
        }
        let image = self.compile_uncached(src, name, options, target)?;
        cache.push((target, image.clone()));
        Ok(image)
    }

    fn compile_uncached(&self, src: &str, name: &str, options: &[&str], target: Target) -> Result<Vec<u8>, String> {
        let src = CString::new(src).map_err(|_| "kernel source contains NUL".to_string())?;
        let cname = CString::new(name).map_err(|_| "kernel name contains NUL".to_string())?;
        let mut opts: Vec<CString> = options
            .iter()
            .map(|o| CString::new(*o).map_err(|_| "option contains NUL".to_string()))
            .collect::<Result<_, _>>()?;
        opts.push(CString::new(target.option()).expect("no NUL"));
        let opt_ptrs: Vec<*const c_char> = opts.iter().map(|o| o.as_ptr()).collect();

        let mut prog: Program = std::ptr::null_mut();
        // SAFETY: all pointers are valid NUL-terminated strings that outlive
        // the program; the program is destroyed on every path below.
        unsafe {
            let status = (self.create)(
                &mut prog,
                src.as_ptr(),
                cname.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            );
            self.check(status, "nvrtcCreateProgram")?;
            let result = self.build(prog, &opt_ptrs, target);
            (self.destroy)(&mut prog);
            result
        }
    }

    /// SAFETY: `prog` must be a live program.
    unsafe fn build(&self, prog: Program, opts: &[*const c_char], target: Target) -> Result<Vec<u8>, String> {
        let status = (self.compile)(prog, opts.len() as c_int, opts.as_ptr());
        if status != 0 {
            let log = self.bytes(prog, self.log_size, self.log).unwrap_or_default();
            let log = String::from_utf8_lossy(&log);
            return Err(format!(
                "{} for {target}\n{}",
                self.check(status, "nvrtcCompileProgram").unwrap_err(),
                log.trim_end_matches('\0').trim()
            ));
        }
        match target {
            Target::Cubin(_) => self.bytes(prog, self.cubin_size, self.cubin),
            Target::Ptx(_) => self.bytes(prog, self.ptx_size, self.ptx),
        }
    }

    /// SAFETY: `prog` must be a live program.
    unsafe fn bytes(&self, prog: Program, size: SizeFn, get: BytesFn) -> Result<Vec<u8>, String> {
        let mut n = 0usize;
        self.check(size(prog, &mut n), "NVRTC size query")?;
        let mut buf = vec![0u8; n];
        self.check(get(prog, buf.as_mut_ptr().cast()), "NVRTC output")?;
        Ok(buf)
    }
}

#[cfg(target_os = "linux")]
fn library_path(symbol: *const c_void) -> Option<String> {
    #[repr(C)]
    struct DlInfo {
        fname: *const c_char,
        fbase: *mut c_void,
        sname: *const c_char,
        saddr: *mut c_void,
    }
    extern "C" {
        fn dladdr(addr: *const c_void, info: *mut DlInfo) -> c_int;
    }
    let mut info = DlInfo {
        fname: std::ptr::null(),
        fbase: std::ptr::null_mut(),
        sname: std::ptr::null(),
        saddr: std::ptr::null_mut(),
    };
    // SAFETY: dladdr only inspects the loader's tables.
    if unsafe { dladdr(symbol, &mut info) } == 0 || info.fname.is_null() {
        return None;
    }
    // SAFETY: dladdr returned a valid C string owned by the loader.
    let path = unsafe { CStr::from_ptr(info.fname) }.to_string_lossy().into_owned();
    Some(std::fs::canonicalize(&path).map_or(path, |p| p.display().to_string()))
}

#[cfg(not(target_os = "linux"))]
fn library_path(_symbol: *const c_void) -> Option<String> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const NVRTC_13: [i32; 8] = [75, 80, 86, 87, 89, 90, 100, 120];

    #[test]
    fn supported_devices_get_a_cubin() {
        for d in NVRTC_13 {
            assert_eq!(select_target(d, &NVRTC_13, CodeType::Auto), Ok(Target::Cubin(d)));
        }
    }

    #[test]
    fn devices_newer_than_nvrtc_get_ptx_of_the_newest_older_architecture() {
        assert_eq!(select_target(121, &NVRTC_13, CodeType::Auto), Ok(Target::Ptx(120)));
        assert_eq!(select_target(88, &NVRTC_13, CodeType::Auto), Ok(Target::Ptx(87)));
        assert_eq!(
            select_target(130, &[50, 60, 70, 75, 80, 86, 89, 90], CodeType::Auto),
            Ok(Target::Ptx(90))
        );
    }

    #[test]
    fn devices_older_than_nvrtc_are_rejected_with_a_reason() {
        let err = select_target(61, &NVRTC_13, CodeType::Auto).unwrap_err();
        assert!(err.contains("sm_61") && err.contains("sm_75"), "{err}");
        assert!(select_target(61, &NVRTC_13, CodeType::Ptx).is_err());
    }

    #[test]
    fn unknown_architecture_lists_fall_back_to_a_device_cubin() {
        assert_eq!(select_target(75, &[], CodeType::Auto), Ok(Target::Cubin(75)));
        assert_eq!(select_target(75, &[], CodeType::Ptx), Ok(Target::Ptx(75)));
    }

    #[test]
    fn preferences_force_the_code_type() {
        assert_eq!(select_target(75, &NVRTC_13, CodeType::Ptx), Ok(Target::Ptx(75)));
        assert_eq!(select_target(121, &NVRTC_13, CodeType::Ptx), Ok(Target::Ptx(120)));
        assert_eq!(select_target(86, &NVRTC_13, CodeType::Cubin), Ok(Target::Cubin(86)));
        assert!(select_target(121, &NVRTC_13, CodeType::Cubin).is_err());
    }

    #[test]
    fn preference_parsing() {
        assert_eq!(CodeType::parse(None), Ok(CodeType::Auto));
        assert_eq!(CodeType::parse(Some("")), Ok(CodeType::Auto));
        assert_eq!(CodeType::parse(Some(" PTX ")), Ok(CodeType::Ptx));
        assert_eq!(CodeType::parse(Some("cubin")), Ok(CodeType::Cubin));
        assert!(CodeType::parse(Some("sass")).is_err());
    }

    #[test]
    fn target_options_and_names() {
        assert_eq!(Target::Cubin(75).option(), "--gpu-architecture=sm_75");
        assert_eq!(Target::Ptx(120).option(), "--gpu-architecture=compute_120");
        assert_eq!(Target::Cubin(86).to_string(), "CUBIN sm_86");
        assert_eq!(Target::Ptx(90).to_string(), "PTX compute_90");
    }

    #[test]
    fn versioned_library_name_comes_first() {
        let c = candidates(13);
        if cfg!(target_os = "linux") {
            assert_eq!(c, ["libnvrtc.so.13", "libnvrtc.so"]);
        } else if cfg!(windows) {
            assert_eq!(c[0], "nvrtc64_130_0.dll");
        }
        assert_eq!(candidates(12)[0].contains("12"), !cfg!(target_os = "macos"));
    }
}
