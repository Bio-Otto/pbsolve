#!/usr/bin/env bash
# Builds real simulation files for scripts/validate_io.py with the actual MD
# packages (tleap, OpenMM, cpptraj, ParmEd, GROMACS) and dumps reference data
# with MDAnalysis/ParmEd. Needs an environment with:
#   conda create -n pbsolve-mdtools -c conda-forge python=3.12 gromacs ambertools openmm parmed mdanalysis netcdf4
# Usage: scripts/make_io_fixtures.sh OUTDIR        (then: python scripts/validate_io.py OUTDIR)
set -euo pipefail
OUT=$(realpath "${1:?usage: make_io_fixtures.sh OUTDIR}")
HERE=$(dirname "$(realpath "$0")")
mkdir -p "$OUT/amber" "$OUT/gmx"
cd "$OUT"
curl -sSfO https://files.rcsb.org/download/1YCR.pdb
curl -sSfO https://files.rcsb.org/download/1YCR.cif
# p53 peptide (chain B of 1YCR)
grep -E "^ATOM" 1YCR.pdb | awk 'substr($0,22,1)=="B"' > chainB.pdb; echo END >> chainB.pdb

cd "$OUT/amber"
cat > leap.in <<'LEAP'
source leaprc.protein.ff14SB
source leaprc.water.tip3p
set default PBRadii mbondi2
m = loadpdb ../chainB.pdb
solvatebox m TIP3PBOX 8.0
addionsrand m Na+ 4 Cl- 4
saveamberparm m pep.prmtop pep.inpcrd
quit
LEAP
tleap -f leap.in > leap.log
python - <<'PY'
import openmm as mm, openmm.app as app, openmm.unit as u
prm = app.AmberPrmtopFile("pep.prmtop"); crd = app.AmberInpcrdFile("pep.inpcrd")
s = prm.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=0.9 * u.nanometer, constraints=app.HBonds)
sim = app.Simulation(prm.topology, s, mm.LangevinMiddleIntegrator(300 * u.kelvin, 1 / u.picosecond, 0.002 * u.picoseconds),
                     mm.Platform.getPlatformByName("CPU"))
sim.context.setPositions(crd.positions); sim.context.setPeriodicBoxVectors(*crd.boxVectors)
sim.minimizeEnergy(maxIterations=200)
sim.reporters.append(app.DCDReporter("pep.dcd", 10)); sim.step(100)
PY
cat > cpptraj.in <<'CPP'
parm pep.prmtop
trajin pep.dcd
trajout pep.nc netcdf
trajout pep.mdcrd
trajout pep_last.ncrst restartnc onlyframes 10
trajout pep_last.rst7 restart onlyframes 10
run
CPP
cpptraj -i cpptraj.in > cpptraj.log
python - <<'PY'
import warnings, parmed as pmd
warnings.filterwarnings("ignore")
s = pmd.load_file("pep.prmtop", "pep.inpcrd"); s.save("pep.psf", overwrite=True)
sol = s["!:WAT,Na+,Cl-"]; sol.save("pep_solute.mol2", overwrite=True); sol.save("pep_solute.pdb", overwrite=True)
PY

cd "$OUT/gmx"
export GMX_MAXBACKUP=-1
printf "integrator=steep\nnsteps=200\ncutoff-scheme=Verlet\ncoulombtype=PME\nrcoulomb=0.9\nrvdw=0.9\n" > em.mdp
printf "integrator=md\nnsteps=100\ndt=0.001\nnstxout=10\nnstxout-compressed=10\ncutoff-scheme=Verlet\ncoulombtype=PME\nrcoulomb=0.9\nrvdw=0.9\nconstraints=h-bonds\ntcoupl=v-rescale\ntc-grps=System\ntau-t=0.1\nref-t=300\ngen-vel=yes\ngen-temp=300\n" > md.mdp
{
gmx pdb2gmx -f ../chainB.pdb -o pep.gro -p topol.top -ff amber99sb-ildn -water tip3p -ignh
gmx editconf -f pep.gro -o box.gro -d 0.8 -bt cubic
gmx solvate -cp box.gro -cs spc216.gro -o solv.gro -p topol.top
gmx grompp -f em.mdp -c solv.gro -p topol.top -o ions.tpr -maxwarn 2
echo SOL | gmx genion -s ions.tpr -o ions.gro -p topol.top -pname NA -nname CL -neutral -conc 0.1
gmx grompp -f em.mdp -c ions.gro -p topol.top -o em.tpr -maxwarn 2 && gmx mdrun -deffnm em -nt 4
gmx grompp -f md.mdp -c em.gro -p topol.top -o md.tpr -pp processed.top -maxwarn 2 && gmx mdrun -deffnm md -nt 4
printf "0\n" | gmx trjconv -f md.xtc -s md.tpr -o md_frames.pdb
} > gmx.log 2>&1

python "$HERE/validate_io.py" --dump-references "$OUT"
echo "fixtures and references in $OUT; now run: python scripts/validate_io.py $OUT"
