//! How the pipeline uses its backend, seen through a backend that records
//! the calls around the CPU backend: calculations retain memory from start to
//! end (`Backend::retain_memory` / `release_memory` through `MemoryHold`),
//! and a backend that claims to run on a device (`Backend::runs_on_device`)
//! makes the pipeline build host inputs (the solvated surface and outermost
//! boundary) in the background and set up the outermost level before its
//! boundary is ready, without changing any result.

use pbsolve_core::backend::MemoryHold;
use pbsolve_core::batch::{polar_solvation_batch, worker_device_bytes, Job};
use pbsolve_core::multigrid::{LevelOp, Stencil27, Transfer};
use pbsolve_core::operator::StencilOperator;
use pbsolve_core::pb::{polar_solvation, solve_focused, Medium, PolarSolvation};
use pbsolve_core::surface::Sphere;
use pbsolve_core::{
    Atom, Backend, BoundaryCondition, CpuBackend, CpuSingle, Grid, GridSpec, PbParams, Preconditioner, Result,
};
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Event {
    Retain(u64),
    Release(u64),
    /// Any other backend call (consecutive ones are merged).
    Work,
}

#[derive(Default)]
struct Recorder {
    cpu: CpuBackend,
    events: Mutex<Vec<Event>>,
    /// The name of every backend call, in order.
    calls: Mutex<Vec<&'static str>>,
    device: bool,
}

impl Recorder {
    fn on_device() -> Self {
        Self {
            device: true,
            ..Self::default()
        }
    }

    fn record(&self, e: Event) {
        let mut events = self.events.lock().unwrap();
        if !(e == Event::Work && events.last() == Some(&Event::Work)) {
            events.push(e);
        }
    }

    fn take(&self) -> Vec<Event> {
        std::mem::take(&mut *self.events.lock().unwrap())
    }

    fn call(&self, name: &'static str) {
        self.record(Event::Work);
        self.calls.lock().unwrap().push(name);
    }

    fn take_calls(&self) -> Vec<&'static str> {
        std::mem::take(&mut *self.calls.lock().unwrap())
    }
}

fn lower<'a>(op: LevelOp<'a, Recorder>) -> LevelOp<'a, CpuBackend> {
    match op {
        LevelOp::Fine(o) => LevelOp::Fine(o),
        LevelOp::Coarse(c) => LevelOp::Coarse(c),
    }
}

macro_rules! delegate {
    ($(fn $name:ident(&self $(, $a:ident: $t:ty)*) -> $r:ty;)*) => {$(
        fn $name(&self $(, $a: $t)*) -> $r {
            self.call(stringify!($name));
            self.cpu.$name($($a),*)
        }
    )*};
}

impl Backend for Recorder {
    type Vector = <CpuBackend as Backend>::Vector;
    type Operator = <CpuBackend as Backend>::Operator;
    type Coarse = <CpuBackend as Backend>::Coarse;
    type Single = CpuSingle;

    delegate! {
        fn platform(&self) -> &'static str;
        fn device_name(&self) -> String;
        fn single(&self) -> Result<CpuSingle>;
        fn demote(&self, x: &Self::Vector, y: &mut <CpuSingle as Backend>::Vector) -> Result<()>;
        fn promote(&self, x: &<CpuSingle as Backend>::Vector, y: &mut Self::Vector) -> Result<()>;
        fn prepare(&self, op: Arc<StencilOperator>) -> Result<Self::Operator>;
        fn prepare_coarse(&self, op: Arc<Stencil27>) -> Result<Self::Coarse>;
        fn upload(&self, host: &[f64]) -> Result<Self::Vector>;
        fn upload_owned(&self, host: Vec<f64>) -> Result<Self::Vector>;
        fn zeros(&self, n: usize) -> Result<Self::Vector>;
        fn download(&self, v: &Self::Vector) -> Result<Vec<f64>>;
        fn read(&self, v: &Self::Vector, host: &mut [f64]) -> Result<()>;
        fn write(&self, host: &[f64], v: &mut Self::Vector) -> Result<()>;
        fn apply(&self, op: &Self::Operator, x: &Self::Vector, y: &mut Self::Vector) -> Result<()>;
        fn apply_coarse(&self, op: &Self::Coarse, x: &Self::Vector, y: &mut Self::Vector) -> Result<()>;
        fn dot(&self, a: &Self::Vector, b: &Self::Vector) -> Result<f64>;
        fn dot2(&self, a: &Self::Vector, b: &Self::Vector, c: &Self::Vector) -> Result<(f64, f64)>;
        fn axpy(&self, alpha: f64, x: &Self::Vector, y: &mut Self::Vector) -> Result<()>;
        fn xpay(&self, x: &Self::Vector, beta: f64, y: &mut Self::Vector) -> Result<()>;
        fn axpby(&self, alpha: f64, x: &Self::Vector, beta: f64, y: &mut Self::Vector) -> Result<()>;
        fn scale(&self, alpha: f64, d: &Self::Vector, x: &Self::Vector, y: &mut Self::Vector) -> Result<()>;
        fn copy(&self, x: &Self::Vector, y: &mut Self::Vector) -> Result<()>;
        fn restrict(&self, t: &Transfer, fine: &Self::Vector, coarse: &mut Self::Vector) -> Result<()>;
        fn prolong_add(&self, t: &Transfer, coarse: &Self::Vector, fine: &mut Self::Vector) -> Result<()>;
        fn coarse_to_host(&self, op: &Self::Coarse) -> Result<Stencil27>;
        fn demote_operator(&self, op: &Self::Operator) -> Result<<CpuSingle as Backend>::Operator>;
        fn demote_coarse(&self, op: &Self::Coarse) -> Result<<CpuSingle as Backend>::Coarse>;
        fn mark_maps(&self, grid: &Grid, shifts: &[[f64; 3]], fill: f64, passes: &[(&[Sphere], f64)]) -> Result<Vec<Self::Vector>>;
        fn scatter(&self, n: usize, idx: &[usize], values: &[f64]) -> Result<Self::Vector>;
        fn assemble_operator(&self, grid: &Grid, maps: [Self::Vector; 4]) -> Result<(Self::Operator, Self::Vector)>;
        fn source_rhs(&self, src: f64, q: &Self::Vector, d: &Self::Vector, y: &mut Self::Vector) -> Result<()>;
        fn operator_to_host(&self, op: &Self::Operator) -> Result<StencilOperator>;
        fn gather(&self, v: &Self::Vector, idx: &[usize]) -> Result<Vec<f64>>;
        fn synchronize(&self) -> Result<()>;
    }

    fn galerkin(&self, t: &Transfer, fine: LevelOp<'_, Self>) -> Result<(Self::Coarse, Self::Vector)> {
        self.call("galerkin");
        self.cpu.galerkin(t, lower(fine))
    }

    fn gershgorin_lmax(&self, level: LevelOp<'_, Self>) -> Result<f64> {
        self.call("gershgorin_lmax");
        self.cpu.gershgorin_lmax(lower(level))
    }

    fn runs_on_device(&self) -> bool {
        self.device
    }

    fn retain_memory(&self, bytes: u64) -> Result<()> {
        self.record(Event::Retain(bytes));
        self.cpu.retain_memory(bytes)
    }

    fn release_memory(&self, bytes: u64) -> Result<()> {
        self.record(Event::Release(bytes));
        self.cpu.release_memory(bytes)
    }
}

fn atoms() -> Vec<Atom> {
    let mut s: u64 = 7;
    let mut rnd = move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..40)
        .map(|_| Atom {
            position: [rnd(), rnd(), rnd()].map(|v| 8.0 * (v - 0.5)),
            charge: rnd() - 0.5,
            radius: 1.2 + 0.7 * rnd(),
        })
        .collect()
}

/// Two focusing levels (coarse 1.6 × fine) at a spacing of 0.5 Å.
fn spec(dims: [usize; 3]) -> GridSpec {
    let fine = dims.map(|n| 0.5 * (n - 1) as f64);
    GridSpec {
        center: [0.0; 3],
        dims,
        coarse_length: fine.map(|l| 1.6 * l),
        fine_length: fine,
    }
}

fn params(pc: Preconditioner) -> PbParams {
    PbParams {
        pdie: 2.0,
        preconditioner: pc,
        bcfl: BoundaryCondition::parse("sdh").unwrap(),
        ..PbParams::default()
    }
    .with_salt(0.15, 2.0)
}

/// One hold of `bytes` around all the work.
fn held(events: &[Event], bytes: u64) -> bool {
    events == [Event::Retain(bytes), Event::Work, Event::Release(bytes)]
}

#[test]
fn a_calculation_retains_memory_from_its_first_to_its_last_backend_call() {
    for rec in [Recorder::default(), Recorder::on_device()] {
        calculations_hold_memory(&rec);
    }
}

fn calculations_hold_memory(rec: &Recorder) {
    let atoms = atoms();
    for (dims, pc) in [
        ([33, 29, 37], Preconditioner::MultigridF32),
        ([33, 29, 37], Preconditioner::Jacobi),
        ([41, 41, 33], Preconditioner::Multigrid),
    ] {
        let bytes = worker_device_bytes([dims], pc);
        let p = polar_solvation(rec, &atoms, &params(pc), &spec(dims)).unwrap();
        // Both states (four levels) under one hold.
        assert_eq!(p.stats_solvated.len() + p.stats_reference.len(), 4);
        let events = rec.take();
        assert!(held(&events, bytes), "{dims:?} {pc:?}: {events:?}");
        // Recording changes nothing.
        let cpu = polar_solvation(&CpuBackend::default(), &atoms, &params(pc), &spec(dims)).unwrap();
        assert_eq!(p.energy_kt.to_bits(), cpu.energy_kt.to_bits());

        let f = solve_focused(rec, &spec(dims), &atoms, &params(pc), Medium::Solvated).unwrap();
        let events = rec.take();
        assert!(held(&events, bytes), "{dims:?} {pc:?}: {events:?}");
        let cpu = solve_focused(
            &CpuBackend::default(),
            &spec(dims),
            &atoms,
            &params(pc),
            Medium::Solvated,
        )
        .unwrap();
        assert!(f
            .potential
            .iter()
            .zip(&cpu.potential)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
    }
}

#[test]
fn a_failing_calculation_releases_its_memory() {
    let dims = [33, 29, 37];
    let pc = Preconditioner::Multigrid;
    // Inside the coarse grid, outside the fine one: the fine level fails
    // (in the reference state, while the solvated inputs are being built in
    // the background on a device).
    let mut atoms = atoms();
    atoms.push(Atom {
        position: [10.0, 0.0, 0.0],
        charge: 1.0,
        radius: 1.5,
    });
    for rec in [Recorder::default(), Recorder::on_device()] {
        let err = polar_solvation(&rec, &atoms, &params(pc), &spec(dims)).unwrap_err();
        assert!(err.to_string().contains("outside the interior"), "{err}");
        let events = rec.take();
        assert!(held(&events, worker_device_bytes([dims], pc)), "{events:?}");
    }
}

#[test]
fn every_calculation_of_a_batch_holds_memory_on_its_worker() {
    let workers = [Recorder::default(), Recorder::on_device()];
    let refs: Vec<&Recorder> = workers.iter().collect();
    let atoms = atoms();
    let pc = Preconditioner::MultigridF32;
    let specs = [
        spec([33, 29, 37]),
        spec([41, 41, 33]),
        spec([33, 29, 37]),
        spec([29, 33, 33]),
    ];
    let jobs: Vec<Job> = specs.iter().map(|spec| Job { atoms: &atoms, spec }).collect();
    let results = polar_solvation_batch(&refs, &params(pc), &jobs);
    let mut holds: Vec<u64> = Vec::new();
    for w in &workers {
        let events = w.take();
        for one in events.chunks(3) {
            let Event::Retain(bytes) = one[0] else {
                panic!("{events:?}")
            };
            assert!(held(one, bytes), "{events:?}");
            holds.push(bytes);
        }
    }
    let mut expected: Vec<u64> = specs.iter().map(|s| worker_device_bytes([s.dims], pc)).collect();
    holds.sort_unstable();
    expected.sort_unstable();
    assert_eq!(holds, expected, "one hold per calculation");
    for (r, spec) in results.into_iter().zip(&specs) {
        let alone = polar_solvation(&CpuBackend::default(), &atoms, &params(pc), spec).unwrap();
        assert_eq!(r.unwrap().energy_kt.to_bits(), alone.energy_kt.to_bits());
    }
}

#[test]
fn a_memory_hold_ends_exactly_once() {
    let rec = Recorder::default();
    let a = MemoryHold::new(&rec, 5).unwrap();
    let b = MemoryHold::new(&rec, 7).unwrap();
    b.release().unwrap();
    drop(a);
    assert_eq!(
        rec.take(),
        [Event::Retain(5), Event::Retain(7), Event::Release(7), Event::Release(5)]
    );
}

/// Everything of a result but the wall times, bit for bit.
fn numerics(p: &PolarSolvation) -> Vec<u64> {
    let mut v = vec![p.energy_kt.to_bits(), p.solvated_kt.to_bits(), p.reference_kt.to_bits()];
    for s in p.stats_solvated.iter().chain(&p.stats_reference) {
        v.extend([s.iterations as u64, s.relative_residual.to_bits(), s.levels as u64]);
    }
    v
}

/// On a device the solvated state's surface and outermost boundary are built
/// in a pool of their own while the device works: every boundary
/// condition, and a molecule without charges, gives the results of the CPU
/// backend bit for bit, with one thread or several, and the stage times stay
/// non-negative and within the wall time.
#[test]
fn building_inputs_in_the_background_changes_no_result() {
    let atoms = atoms();
    let dims = [33, 29, 37];
    // Without charges b = 0 on every level (no iterations, one level reported).
    let neutral: Vec<Atom> = atoms.iter().map(|a| Atom { charge: 0.0, ..*a }).collect();
    for (bcfl, atoms) in [
        ("mdh", &atoms),
        ("mdh-fast", &atoms),
        ("sdh", &atoms),
        ("zero", &atoms),
        ("mdh", &neutral),
    ] {
        let params = PbParams {
            bcfl: BoundaryCondition::parse(bcfl).unwrap(),
            ..params(Preconditioner::MultigridF32)
        };
        let cpu = polar_solvation(&CpuBackend::default(), atoms, &params, &spec(dims)).unwrap();
        for threads in [1, 3] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            let rec = Recorder::on_device();
            let t = std::time::Instant::now();
            let p = pool
                .install(|| polar_solvation(&rec, atoms, &params, &spec(dims)))
                .unwrap();
            let wall = t.elapsed().as_secs_f64();
            assert_eq!(numerics(&p), numerics(&cpu), "{bcfl}, {threads} threads");
            if atoms[0].charge == 0.0 {
                let stats = p.stats_solvated.iter().chain(&p.stats_reference);
                assert!(stats.into_iter().all(|s| s.iterations == 0 && s.levels == 1));
            }
            let t = p.timings;
            let stages = [t.maps, t.boundary, t.assembly, t.setup, t.solve, t.energy];
            assert!(stages.iter().all(|s| s.is_finite() && *s >= 0.0), "{bcfl}: {t:?}");
            assert!(t.total() <= wall, "{bcfl}: stages {} s, wall {wall} s", t.total());
        }
    }
}

/// The calls of each level, split at its operator assembly.
fn levels_of(calls: &[&'static str]) -> Vec<Vec<&'static str>> {
    let mut levels: Vec<Vec<&'static str>> = Vec::new();
    for &c in calls {
        if c == "assemble_operator" {
            levels.push(Vec::new());
        }
        if let Some(level) = levels.last_mut() {
            level.push(c);
        }
    }
    levels
}

/// Whether a level sets up its multigrid before it builds its right-hand side.
fn setup_first(level: &[&'static str]) -> bool {
    let at = |name| level.iter().position(|&c| c == name).unwrap();
    at("galerkin") < at("source_rhs")
}

/// On a device the outermost level of the solvated state (and of a single
/// `solve_focused` state) assembles its operator and sets up its multigrid
/// before it builds the right-hand side, the first step that needs the
/// boundary still being built in the background. The reference state, whose
/// boundary is built first, the focusing levels, whose boundary is ready,
/// and every level on the CPU build the right-hand side first.
#[test]
fn on_a_device_the_outermost_setup_runs_before_its_boundary_is_needed() {
    let atoms = atoms();
    let dims = [33, 29, 37];
    let params = params(Preconditioner::MultigridF32);
    for (rec, outer_first) in [(Recorder::on_device(), true), (Recorder::default(), false)] {
        polar_solvation(&rec, &atoms, &params, &spec(dims)).unwrap();
        let levels = levels_of(&rec.take_calls());
        let order: Vec<bool> = levels.iter().map(|l| setup_first(l)).collect();
        // Two states (reference first), two levels each.
        assert_eq!(order, [false, false, outer_first, false], "device {}", rec.device);
        solve_focused(&rec, &spec(dims), &atoms, &params, Medium::Solvated).unwrap();
        let order: Vec<bool> = levels_of(&rec.take_calls()).iter().map(|l| setup_first(l)).collect();
        assert_eq!(order, [outer_first, false], "device {}", rec.device);
    }
}
