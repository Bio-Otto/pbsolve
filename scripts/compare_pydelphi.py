#!/usr/bin/env python3
"""Cross-check of the Gaussian dielectric (surface="gauss") with pyDelPhi (ROADMAP R4).

    python scripts/compare_pydelphi.py [--pqr FILE] [--dims 65] [--spacing 0.5] [--salt 0 0.15]
                                       [--pydelphi PATH] [--workdir DIR] [--threads 4]

pyDelPhi (Panday, Zhao & Alexov 2026; AGPL-3.0) runs as a separate program,
``pydelphi-static`` (or ``--pydelphi``); only its output files are read, and
nothing of it is imported. Both programs get one shared grid (pyDelPhi
``grid_size``/``scale``/``acent`` = a single-level :class:`pbsolve.GridSpec`),
σ, ε_in = 1, ε_out = 80, 298.15 K, the ion radius and the contour ε = 20
(pyDelPhi ``gaussiancutoff``: vacuum beyond the contour in the reference run,
which with pdie = 1 is pbsolve's contour reference).

The report has three parts:

* maps: pbsolve's face dielectrics against pyDelPhi's ``out(eps, point=mid)``,
  first as they are, then with pyDelPhi's definitions applied to the same
  formula: every atom's Gaussian is cut at max(3σR_max, R_max + r_ion) +
  √3/2·h, at a face also from the node below; faces of density below 0.02
  get the outer ε (80, or 1 in the vacuum run, whose contour is drawn on the
  density); and the six faces around a node that differ by less than 2 % are
  replaced by their mean (even, then odd i+j+k). The definitions were found
  from pyDelPhi's outputs and checked against its source description; no code
  of it is used. The cube files carry 6 significant digits, which sets the
  tolerance;
* boundary: pyDelPhi's coulombic boundary is Σ q e^(−κr)/(ε_out r) with its
  Debye length 0.01990076478·√(ε_out T / I) Å; pbsolve's ``mdh`` uses
  e^(−κ(r−a))/(1+κa) and κ from CODATA constants;
* energies: pyDelPhi's grid energies against pbsolve's, and against pbsolve's
  solver run on pyDelPhi's maps, its ion term (Born accessibility from its node
  dielectric, its κ) and its boundary values, which isolates the solvers.

pyDelPhi initialises CUDA through numba even for ``-P cpu``; with numba 0.61
and a CUDA 13 driver that crashes, so it runs with numba's CUDA simulator
(``NUMBA_ENABLE_CUDASIM=1``), which leaves the CPU path unchanged.
"""
import argparse
import itertools
import os
import shutil
import subprocess
import sys
import tempfile
import warnings

import numpy as np

import pbsolve

BOHR = 0.529177210903  # Å
DEBYE_CONSTANT = 0.01990076478  # pyDelPhi's Debye length factor, Å (M/K)^½
SOLUTE_DENSITY = 0.02  # pyDelPhi: faces below this density get the outer ε (80, or 1 in the vacuum run)
SMOOTH_RATIO = 1.02  # pyDelPhi: six faces around a node that differ by less than this ratio are averaged
EPS_IN, EPS_OUT, CONTOUR, TEMPERATURE = 1.0, 80.0, 20.0, 298.15
# CODATA 2018 (as pbsolve's units.rs): e²/(4πε0 k) in Å K; pyDelPhi's value comes from CODATA 2006
EPK = 1.602176634e-19**2 / (4 * np.pi * 8.8541878128e-12 * 1.380649e-23) * 1e10
EPK_PYDELPHI = 167100.9162872952

SYNTHETIC = (
    np.array([[0.31, -0.42, 0.17], [1.83, 0.66, -0.29], [-1.12, 1.05, 0.88], [0.47, -1.71, -1.36]]),
    np.array([0.6, -0.8, 0.45, -0.25]),
    np.array([1.7, 1.5, 1.2, 1.9]),
)


def read_cube(path):
    """A pyDelPhi cube file: (data (nx, ny, nz), origin Å, spacing Å)."""
    with open(path) as fh:
        lines = fh.readlines()
    natom = int(lines[2].split()[0])
    origin = np.array([float(x) for x in lines[2].split()[1:4]]) * BOHR
    dims = [int(lines[3 + a].split()[0]) for a in range(3)]
    data = np.array(" ".join(lines[6 + natom:]).split(), dtype=float)
    return data.reshape(dims), origin, float(lines[3].split()[1]) * BOHR


def rounding(x):
    """Half a unit in the last place of the cubes' 6 significant digits."""
    ax = np.abs(x)
    return np.where(ax > 0, 0.5e-5 * 10.0 ** np.floor(np.log10(np.where(ax > 0, ax, 1.0))), 1e-300)


def density(coords, radii, sigma, origin, h, shape, dcut=np.inf, also=None):
    """ρ = 1 − Π(1 − g_i) on a grid, each g_i = exp(−d²/σ²R²) taken where d ≤ dcut and g_i > 1e-12.

    With ``also`` (a vector), atom i must also lie within dcut of the point shifted by it
    (pyDelPhi takes an atom into a face's density only if it is within reach of the node below).
    """
    logs = np.zeros(shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # log1p(−1) at a node on a nucleus
        for c, rad in zip(coords, radii):
            if rad <= 0:
                continue
            s = sigma * rad
            reach = min(dcut, s * np.sqrt(np.log(1e12)))
            lo = np.maximum(np.ceil((c - reach - origin) / h).astype(int), 0)
            hi = np.minimum(np.floor((c + reach - origin) / h).astype(int), np.array(shape) - 1)
            if np.any(hi < lo):
                continue
            ax = [origin[k] + h * np.arange(lo[k], hi[k] + 1) - c[k] for k in range(3)]
            d2 = ax[0][:, None, None] ** 2 + ax[1][None, :, None] ** 2 + ax[2][None, None, :] ** 2
            g = np.exp(-d2 / s**2)
            inside = d2 <= dcut**2
            if also is not None:
                inside &= ((ax[0] + also[0])[:, None, None] ** 2 + (ax[1] + also[1])[None, :, None] ** 2
                           + (ax[2] + also[2])[None, None, :] ** 2) <= dcut**2
            box = tuple(slice(lo[k], hi[k] + 1) for k in range(3))
            logs[box] += np.where(inside & (g > 1e-12), np.log1p(-g), 0.0)
    return -np.expm1(logs)


def smooth(faces, rho, node):
    """pyDelPhi's smoothing of the face dielectrics, in place.

    Where one of the six faces around an interior node has a density above
    SOLUTE_DENSITY and the six values differ by less than SMOOTH_RATIO
    (largest / smallest), the six faces and the node take their mean; first
    the nodes of even, then of odd i+j+k (with odd node counts no two nodes of
    one set share a face, so the order within a set does not matter).
    """
    inner = (slice(1, -1),) * 3
    views = lambda a: [a[0][inner], a[1][inner], a[2][inner], a[0][:-2, 1:-1, 1:-1],  # noqa: E731
                       a[1][1:-1, :-2, 1:-1], a[2][1:-1, 1:-1, :-2]]
    parity = np.indices(node.shape)[(slice(None),) + inner].sum(0) % 2
    dense = (np.stack(views(rho)) > SOLUTE_DENSITY).any(0)
    for p in (0, 1):
        f = np.stack(views(faces))
        lo, hi = f.min(0), f.max(0)
        sel = (parity == p) & dense & (lo > 1e-6) & (hi < SMOOTH_RATIO * lo)
        mean = f.sum(0) / 6.0
        for v in views(faces):
            v[sel] = mean[sel]
        node[inner][sel] = mean[sel]


def grid_energy(phi, coords, q, origin, h):
    """½ Σ q φ(r), φ interpolated trilinearly (DelPhi's and pbsolve's spl0 energy)."""
    g = (coords - origin) / h
    i0 = np.floor(g).astype(int)
    f = g - i0
    e = 0.0
    for corner in itertools.product((0, 1), repeat=3):
        w = np.prod(np.where(np.array(corner, bool), f, 1 - f), axis=1)
        e += np.sum(q * w * phi[i0[:, 0] + corner[0], i0[:, 1] + corner[1], i0[:, 2] + corner[2]])
    return 0.5 * e


def run_pydelphi(exe, workdir, pqr, center, n, h, salt, sigma, rion, threads):
    with open(os.path.join(workdir, "mol.pqr"), "w") as fh:
        fh.write(pqr)
    rho_c = (EPS_OUT - CONTOUR) / (EPS_OUT - EPS_IN)
    c = [float(x) for x in center]
    prm = f"""biomodel=pbe
surfacemethod=gaussiancutoff
dielectricmodel=gaussian
surfaceoffset=0.0
density_cutoff={rho_c!r}
surface_cutoff={CONTOUR!r}
gaussianmultiplier=1
sigma={float(sigma)!r}
grid_size={n}
scale={1.0 / h!r}
gridbox_type=cubic
acent(x={c[0]!r},y={c[1]!r},z={c[2]!r})
in(modpdb4,file="mol.pqr", format="pqr")
indi={EPS_IN!r}
gapdi={EPS_OUT!r}
exdi={EPS_OUT!r}
boundary_condition=coulombic
salt={float(salt)!r}
ionrad={float(rion)!r}
temper={TEMPERATURE!r}
linit=100000
max_rmsd=1.0e-12
max_delta_phi=1.0e-10
zeta_potential=OFF
out(eps, file="eps.cube", media="both", point="both", format="cube", precision="double")
out(phi, file="phi.cube", media="both", format="cube", precision="double")
"""
    with open(os.path.join(workdir, "params.prm"), "w") as fh:
        fh.write(prm)
    env = dict(os.environ, NUMBA_ENABLE_CUDASIM="1")
    p = subprocess.run([exe, "-f", "params.prm", "-P", "cpu", "-p", "double", "-t", str(threads), "-l", "mol",
                        "-o", "out.csv", "-O"], cwd=workdir, env=env, capture_output=True, text=True)
    with open(os.path.join(workdir, "run.log"), "w") as fh:
        fh.write(p.stdout + p.stderr)
    if p.returncode:
        raise RuntimeError(f"pydelphi-static failed ({p.returncode}); see {workdir}/run.log")
    head, vals = open(os.path.join(workdir, "out.csv")).read().strip().splitlines()[-2:]
    return {k: float(v) for k, v in zip(head.split("\t"), vals.split("\t")) if k != "LABEL"}


def pqr_text(coords, q, radii):
    """PQR for pyDelPhi, whose reader takes no chain column; pdb2pqr's precision (3 and 4 decimals) is kept."""
    return "".join("ATOM  %5d  C   MOL     1    %8.3f%8.3f%8.3f%8.4f%7.4f\n" % (i + 1, *x, c, r)
                   for i, (x, c, r) in enumerate(zip(coords, q, radii)))


def grid_center(coords):
    """Centre of the atoms' extent, shifted so that the atoms do not sit on grid lines."""
    return np.round((coords.max(0) + coords.min(0)) / 2, 3) + np.array([0.1, 0.2, -0.05])


def compare(coords, q, radii, pqr, center, n, h, salt, exe, workdir, sigma=0.93, rion=2.0, threads=4):
    """Run both programs on one shared grid; return the numbers of the report as a dict."""
    out = {"salt": salt}
    e_pyd = run_pydelphi(exe, workdir, pqr, center, n, h, salt, sigma, rion, threads)
    cube = lambda name: read_cube(os.path.join(workdir, name))[0]  # noqa: E731

    solver = pbsolve.Solver(platform="CPU", pdie=EPS_IN, sdie=EPS_OUT, temperature=TEMPERATURE, surface="gauss",
                            gauss_sigma=sigma, gauss_contour=CONTOUR, bcfl="mdh", ionic_strength=salt,
                            ion_radius=rion, tol=1e-13, preconditioner="multigrid")
    spec = pbsolve.GridSpec(center, [n] * 3, [(n - 1) * h] * 3)
    origin = np.array(center) - (n - 1) / 2 * h
    rmax = radii.max()
    dcut = max(3 * sigma * rmax, rmax + rion) + np.sqrt(3) / 2 * h
    out["dcut"] = dcut

    # maps: face densities with and without pyDelPhi's cutoff ((n, n, n) arrays, last face unused)
    offsets = [origin + h / 2 * np.eye(3)[d] for d in range(3)]
    rho_full = [density(coords, radii, sigma, o, h, (n,) * 3) for o in offsets]
    rho_cut = [density(coords, radii, sigma, o, h, (n,) * 3, dcut, also=-h / 2 * np.eye(3)[d])
               for d, o in enumerate(offsets)]
    rho_node = density(coords, radii, sigma, origin, h, (n,) * 3, dcut)
    rho_contour = (EPS_OUT - CONTOUR) / (EPS_OUT - EPS_IN)
    models = {}
    for media, ref in (("water", False), ("vacuum", True)):
        m = solver.maps(coords, q, radii, grid=spec, reference=ref)
        outer = 1.0 if ref else EPS_OUT
        faces_m = [np.where(r < SOLUTE_DENSITY, outer, r * EPS_IN + (1 - r) * EPS_OUT) for r in rho_cut]
        if ref:
            faces_m = [np.where(r < rho_contour, 1.0, f) for r, f in zip(rho_cut, faces_m)]
        node_m = EPS_IN + (EPS_OUT - EPS_IN) * (1 - rho_node)
        smooth(faces_m, rho_cut, node_m)
        models[media] = (faces_m, node_m)
        worst_raw = worst_model = 0.0
        off_raw = off_model = faces = 0
        max_formula = 0.0
        for d, key in enumerate(("eps_x", "eps_y", "eps_z")):
            sl = tuple(slice(0, n - 1) if k == d else slice(None) for k in range(3))
            pyd = cube(f"eps_{media}_mids_4th_dim{d}.cube")[sl]
            ours = m[key][sl]
            eps_full = rho_full[d][sl] * EPS_IN + (1 - rho_full[d][sl]) * EPS_OUT
            if ref:
                eps_full = np.where(eps_full <= CONTOUR, eps_full, EPS_IN)
            max_formula = max(max_formula, np.abs(eps_full - ours).max())
            model = faces_m[d][sl]
            tol = rounding(pyd)
            faces += pyd.size
            off_raw += int((np.abs(ours - pyd) > tol).sum())
            off_model += int((np.abs(model - pyd) > tol).sum())
            worst_raw = max(worst_raw, np.abs(ours - pyd).max())
            worst_model = max(worst_model, (np.abs(model - pyd) / tol).max())
        out[media] = dict(faces=faces, formula=max_formula, raw_max=worst_raw, raw_off=off_raw,
                          model_off=off_model, model_worst=worst_model)

    # boundary
    lb0 = EPK / TEMPERATURE
    kappa_d = np.sqrt(salt) / (DEBYE_CONSTANT * np.sqrt(EPS_OUT * TEMPERATURE)) if salt else 0.0
    bnd = np.zeros((n,) * 3, bool)
    bnd[[0, -1], :, :] = bnd[:, [0, -1], :] = bnd[:, :, [0, -1]] = True
    pts = origin + np.argwhere(bnd) * h
    phi_b = np.zeros(len(pts))
    for c, qq in zip(coords, q):
        dist = np.sqrt(((pts - c) ** 2).sum(1))
        phi_b += qq * np.exp(-kappa_d * dist) / dist
    phi_b0 = np.zeros(len(pts))
    for c, qq in zip(coords, q):
        dist = np.sqrt(((pts - c) ** 2).sum(1))
        phi_b0 += qq / dist
    phi_b *= lb0 / EPS_OUT
    phi_b0 *= lb0 / EPS_IN  # the vacuum run: ε = 1 outside the contour, no ions
    phi_w = cube("phi_water.cube")
    phi_v = cube("phi_vacuum.cube")
    out["boundary_rel"] = float((np.abs(phi_b - phi_w[bnd]) / np.abs(phi_w[bnd]).max()).max())
    phi_ours = solver.potential(coords, q, radii, grid=spec)[0]
    out["boundary_mdh_rel"] = float((np.abs(phi_ours[bnd] - phi_w[bnd]) / np.abs(phi_w[bnd]).max()).max())

    # energies
    res = solver.polar_solvation(coords, q, radii, grid=spec)
    m = solver.maps(coords, q, radii, grid=spec)
    eps_node = cube("eps_water.cube")
    kb2 = EPS_OUT * kappa_d**2 * np.exp(-lb0 / (2 * rion) * (1 / eps_node - 1 / EPS_OUT)) if salt else np.zeros_like(eps_node)
    inj = {}
    for media, phi_d, kap in (("water", phi_w, kb2), ("vacuum", phi_v, np.zeros_like(kb2))):
        eps = [cube(f"eps_{media}_mids_4th_dim{d}.cube") for d in range(3)]
        phi, _, _ = solver.solve_maps(*eps, kap, m["charge"], phi_d, m["origin"], m["spacing"])
        inj[media] = grid_energy(phi, coords, q, origin, h)
        out[f"phi_rms_{media}"] = float(np.sqrt(((phi - phi_d)[~bnd] ** 2).mean()))
    # pbsolve's own maps with pyDelPhi's definitions (cutoff, density floor, smoothing, Debye constant, boundary)
    kb2_def = EPS_OUT * kappa_d**2 * np.exp(-lb0 / (2 * rion) * (1 / models["water"][1] - 1 / EPS_OUT))
    out["node_eps_max"] = float(np.abs(models["water"][1] - eps_node).max())
    defs = {}
    for media, vals, kap in (("water", phi_b, kb2_def), ("vacuum", phi_b0, np.zeros_like(kb2_def))):
        b = np.zeros((n,) * 3)
        b[bnd] = vals
        phi, _, _ = solver.solve_maps(*models[media][0], kap, m["charge"], b, m["origin"], m["spacing"])
        defs[media] = grid_energy(phi, coords, q, origin, h)
    out["energies_kt"] = {
        "pydelphi": (e_pyd["E_grid_w"], e_pyd["E_grid_v"], e_pyd["E_rxn_corr_tot"]),
        "pbsolve on pydelphi's maps": (inj["water"], inj["vacuum"], inj["water"] - inj["vacuum"]),
        "pbsolve, pydelphi's definitions": (defs["water"], defs["vacuum"], defs["water"] - defs["vacuum"]),
        # energies in kT scale with e²/(4πε0 kT): the same with pyDelPhi's constant
        "  ... and pydelphi's constant": tuple(x * EPK_PYDELPHI / EPK for x in (
            defs["water"], defs["vacuum"], defs["water"] - defs["vacuum"])),
        "pbsolve": (res.solvated_kt, res.reference_kt, res.energy_kt),
    }
    return out


def report(name, r):
    print(f"\n== {name}, salt {r['salt']} M (Gaussian cutoff {r['dcut']:.4f} Å)")
    for media in ("water", "vacuum"):
        s = r[media]
        print(f"  {media:6s} faces {s['faces']}: pbsolve = formula to {s['formula']:.1e}; pbsolve vs pyDelPhi max |Δε| "
              f"{s['raw_max']:.3g}, {s['raw_off']} faces beyond the cube rounding; with pyDelPhi's definitions "
              f"{s['model_off']} faces (worst {s['model_worst']:.2f} × rounding)")
    print(f"  boundary: Σq e^(-κr)/(εr) with pyDelPhi's κ vs its values {r['boundary_rel']:.1e} (relative to the "
          f"largest); pbsolve mdh {r['boundary_mdh_rel']:.1e}")
    print(f"  potential rms, pbsolve on pyDelPhi's maps vs pyDelPhi: water {r['phi_rms_water']:.1e}, "
          f"vacuum {r['phi_rms_vacuum']:.1e} kT/e")
    print("  energies (kT; pyDelPhi writes 4 decimals)  solvated     reference      ΔG_polar")
    for k, (w, v, d) in r["energies_kt"].items():
        print(f"    {k:31s} {w:14.6f} {v:14.6f} {d:14.6f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pqr", help="PQR file (default: a synthetic 4-atom molecule)")
    ap.add_argument("--dims", type=int, default=65, help="nodes per axis, odd and 8k+1 (default 65)")
    ap.add_argument("--spacing", type=float, default=0.5)
    ap.add_argument("--salt", type=float, nargs="+", default=[0.0, 0.15])
    ap.add_argument("--sigma", type=float, default=0.93)
    ap.add_argument("--pydelphi", default=os.environ.get("PBSOLVE_PYDELPHI") or shutil.which("pydelphi-static"))
    ap.add_argument("--workdir", help="keep pyDelPhi's files here (default: a temporary directory)")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    if not a.pydelphi:
        sys.exit("pydelphi-static not found: pass --pydelphi or set PBSOLVE_PYDELPHI")
    if a.pqr:
        coords, q, radii = pbsolve.read_pqr(a.pqr)
        name = os.path.basename(a.pqr)
    else:
        coords, q, radii = SYNTHETIC
        name = "synthetic 4 atoms"
    for salt in a.salt:
        with tempfile.TemporaryDirectory() as tmp:
            wd = os.path.join(a.workdir, f"salt{salt}") if a.workdir else tmp
            os.makedirs(wd, exist_ok=True)
            report(name, compare(coords, q, radii, pqr_text(coords, q, radii), grid_center(coords), a.dims, a.spacing,
                                 salt, a.pydelphi, wd, sigma=a.sigma, threads=a.threads))


if __name__ == "__main__":
    main()
