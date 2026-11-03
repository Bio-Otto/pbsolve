//! Concurrent polar-solvation calculations (trajectory frames, MM-PBSA
//! species).
//!
//! A single calculation alternates between host work (surface maps, boundary
//! values, multigrid setup) and solver work that is limited by memory
//! bandwidth. Running several calculations at once overlaps these phases. On
//! a GPU, one worker prepares the next system on the CPU while another one's
//! kernels run. On the CPU, it uses the cores that one solve leaves idle.
//!
//! Each worker owns one backend. A CUDA backend has its own stream, so kernels
//! of different workers run concurrently. Each worker also gets its own
//! thread pool with an equal share of the threads: one large pool shared by
//! all workers splits every operation as finely as for a single calculation
//! and gave no gain on the CPU. Calculations are independent and
//! deterministic (the CPU results do not depend on the thread count), so
//! every result is bit-for-bit identical to a sequential [`polar_solvation`]
//! call, whatever the number of workers or the order in which they pick up
//! the jobs.

use crate::autogrid::GridSpec;
use crate::backend::Backend;
use crate::error::Result;
use crate::molecule::Atom;
use crate::params::PbParams;
use crate::pb::{polar_solvation, PolarSolvation};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// One calculation of a batch: the atoms and the grid to solve them on.
#[derive(Clone, Copy, Debug)]
pub struct Job<'a> {
    pub atoms: &'a [Atom],
    pub spec: &'a GridSpec,
}

/// Runs `jobs` on `backends.len()` concurrent workers (one backend each) and
/// returns the results in job order.
///
/// Workers take the next pending job as soon as they are free. A failed job
/// does not stop the others. With one backend this is a plain loop.
///
/// # Panics
/// If `backends` is empty.
pub fn polar_solvation_batch<B: Backend + Sync>(
    backends: &[&B],
    params: &PbParams,
    jobs: &[Job<'_>],
) -> Vec<Result<PolarSolvation>> {
    assert!(!backends.is_empty(), "polar_solvation_batch needs at least one backend");
    let workers = backends.len().min(jobs.len());
    if workers <= 1 {
        return jobs
            .iter()
            .map(|j| polar_solvation(backends[0], j.atoms, params, j.spec))
            .collect();
    }
    let threads = (rayon::current_num_threads() / workers).max(1);
    let next = AtomicUsize::new(0);
    let slots: Vec<Mutex<Option<Result<PolarSolvation>>>> = jobs.iter().map(|_| Mutex::new(None)).collect();
    std::thread::scope(|scope| {
        for be in &backends[..workers] {
            let (next, slots) = (&next, &slots);
            scope.spawn(move || {
                let work = || loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(job) = jobs.get(i) else { break };
                    let result = polar_solvation(*be, job.atoms, params, job.spec);
                    *slots[i].lock().unwrap_or_else(|e| e.into_inner()) = Some(result);
                };
                // Fall back to the global pool if a private one cannot be built.
                match rayon::ThreadPoolBuilder::new().num_threads(threads).build() {
                    Ok(pool) => pool.install(work),
                    Err(_) => work(),
                }
            });
        }
    });
    slots
        .into_iter()
        .map(|slot| {
            slot.into_inner()
                .unwrap_or_else(|e| e.into_inner())
                .expect("every job is taken by a worker")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuBackend;
    use crate::error::PbError;

    fn molecule(seed: u64, n: usize) -> Vec<Atom> {
        let mut s = seed;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| Atom {
                position: [rnd(), rnd(), rnd()].map(|v| 10.0 * (v - 0.5)),
                charge: rnd() - 0.5,
                radius: 1.3 + 0.6 * rnd(),
            })
            .collect()
    }

    fn params() -> PbParams {
        PbParams {
            pdie: 2.0,
            ..PbParams::default()
        }
        .with_salt(0.15, 2.0)
    }

    #[test]
    fn batch_results_equal_sequential_results_bit_for_bit() {
        let systems: Vec<Vec<Atom>> = (0..7).map(|k| molecule(100 + k, 20 + 5 * k as usize)).collect();
        let specs: Vec<GridSpec> = systems
            .iter()
            .map(|a| GridSpec::auto(a, 0.6, 1.7, 8.0).unwrap())
            .collect();
        let jobs: Vec<Job> = systems
            .iter()
            .zip(&specs)
            .map(|(atoms, spec)| Job { atoms, spec })
            .collect();
        let p = params();
        let cpu = [CpuBackend::new(), CpuBackend::new(), CpuBackend::new()];
        let sequential: Vec<PolarSolvation> = jobs
            .iter()
            .map(|j| polar_solvation(&cpu[0], j.atoms, &p, j.spec).unwrap())
            .collect();
        for workers in 1..=3 {
            let bes: Vec<&CpuBackend> = cpu.iter().take(workers).collect();
            let batch = polar_solvation_batch(&bes, &p, &jobs);
            assert_eq!(batch.len(), jobs.len());
            for (b, s) in batch.iter().zip(&sequential) {
                let b = b.as_ref().unwrap();
                assert_eq!(b.energy_kt.to_bits(), s.energy_kt.to_bits(), "{workers} workers");
                assert_eq!(b.solvated_kt.to_bits(), s.solvated_kt.to_bits());
                let its = |r: &PolarSolvation| r.stats_solvated.iter().map(|x| x.iterations).collect::<Vec<_>>();
                assert_eq!(its(b), its(s));
            }
        }
    }

    #[test]
    fn a_failing_job_does_not_stop_the_others() {
        let good = molecule(7, 15);
        let spec = GridSpec::auto(&good, 0.6, 1.7, 8.0).unwrap();
        // An atom outside the grid cannot be spread onto it.
        let mut bad = good.clone();
        bad[0].position = [1.0e4, 0.0, 0.0];
        let jobs = [
            Job {
                atoms: &good,
                spec: &spec,
            },
            Job {
                atoms: &bad,
                spec: &spec,
            },
            Job {
                atoms: &good,
                spec: &spec,
            },
        ];
        let cpu = [CpuBackend::new(), CpuBackend::new()];
        let out = polar_solvation_batch(&[&cpu[0], &cpu[1]], &params(), &jobs);
        assert!(out[0].is_ok() && out[2].is_ok());
        assert!(
            matches!(out[1], Err(PbError::InvalidInput(_))),
            "{:?}",
            out[1].as_ref().err()
        );
        assert_eq!(
            out[0].as_ref().unwrap().energy_kt.to_bits(),
            out[2].as_ref().unwrap().energy_kt.to_bits()
        );
    }

    #[test]
    fn more_workers_than_jobs_and_empty_batches() {
        let atoms = molecule(3, 10);
        let spec = GridSpec::auto(&atoms, 0.6, 1.7, 8.0).unwrap();
        let cpu = [
            CpuBackend::new(),
            CpuBackend::new(),
            CpuBackend::new(),
            CpuBackend::new(),
        ];
        let bes: Vec<&CpuBackend> = cpu.iter().collect();
        let p = params();
        assert!(polar_solvation_batch(&bes, &p, &[]).is_empty());
        let one = polar_solvation_batch(
            &bes,
            &p,
            &[Job {
                atoms: &atoms,
                spec: &spec,
            }],
        );
        assert_eq!(one.len(), 1);
        assert!(one[0].is_ok());
    }
}
