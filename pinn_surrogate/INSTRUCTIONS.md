# pinn_surrogate — how to run

All commands from this directory (`pinn_surrogate/`). Stages 1–2 need only numpy; Stages 3–5
need torch and matplotlib (no scipy anywhere). Every result goes into `LOG.md`; the write-up
is `REPORT.md`.

**Quick path to the report's results** (after installing, §0):

```bash
PYTHONPATH=src python3 -m unittest discover -s tests          # 63 tests, ~10 s
bash ablation.sh && SEED=1 bash ablation.sh                   # 12 runs -> runs/grid/
PYTHONPATH=src python3 -m pinn_surrogate.compare --runs runs/grid    # paired stats
PYTHONPATH=src python3 -m pinn_surrogate.figures --runs runs/grid --out figures
```

## 0. Install

**On your Mac (native, simplest):**

```bash
cd pinn_surrogate
python3 -m venv .venv && source .venv/bin/activate
pip install torch numpy matplotlib pytest
```

**In the Cowork Linux VM (no internet there).** The VM is aarch64 Linux with python 3.10. PyPI's
Linux torch wheel is the CUDA build and will not import without NVIDIA libraries, so download the
CPU build *on the Mac* into the repo, then install offline in the VM:

```bash
# on the Mac, from the repo root
python3 -m pip download torch --index-url https://download.pytorch.org/whl/cpu \
  --platform manylinux_2_28_aarch64 --python-version 3.10 --only-binary=:all: --no-deps \
  -d pinn_surrogate/wheels_cpu
python3 -m pip download pytest matplotlib --platform manylinux_2_28_aarch64 \
  --platform manylinux_2_17_aarch64 --platform manylinux2014_aarch64 --python-version 3.10 \
  --only-binary=:all: -d pinn_surrogate/wheels
# in the VM, from pinn_surrogate/
python3 -m pip install --user --no-index --find-links wheels pytest matplotlib
python3 -m pip install --user --no-index --no-deps wheels_cpu/torch-*.whl \
  wheels/filelock-* wheels/typing_extensions-* wheels/sympy-* wheels/mpmath-* \
  wheels/networkx-* wheels/jinja2-* wheels/markupsafe-* wheels/fsspec-*
python3 -c "import torch, matplotlib; print(torch.__version__, torch.get_num_threads())"
```

Both `wheels/` and `wheels_cpu/` are gitignored.

## 1. Tests (run these first)

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Without torch the torch tests are reported as `skipped`; with torch none should skip. The
torch files are `test_physics_torch.py`, `test_geometry_torch.py`, `test_model_torch.py`
(prints the parameter count and the worst inlet `1 - m`) and `test_train_torch.py`
(tiny end-to-end train / crash / `--resume` / single-case run, ~1 min). To run one file:

```bash
PYTHONPATH=src python3 -m unittest tests.test_physics_torch -v
```

## 2. Design numbers and parameter count (no training)

```bash
PYTHONPATH=src python3 -m pinn_surrogate.geometry_np   # inlet margin + S_u/S_v/S_p stats (numpy, ~30 s)
PYTHONPATH=src python3 -m pinn_surrogate.model         # parameter count by component (torch)
```

## 3. Stage 3 — textbook PINN on one case (physics only, no data)

```bash
# Timing probe first (~1 min): read "s/step" in the log lines.
PYTHONPATH=src python3 -m pinn_surrogate.single_case --steps 50 --report-every 10 \
    --threads 2 --out runs/single_case_probe

# Full run: wall tau / p errors vs Dan's arc_*_256 every 250 steps.
PYTHONPATH=src python3 -m pinn_surrogate.single_case --steps 3000 --threads 2 --out runs/single_case
```

Output: `runs/single_case/{single_case.log, metrics.jsonl, summary.json, model.pt}`.

## 4. Stage 4 — one training run

```bash
# Timing probe first (~1-2 min incl. data loading and a 2-solve evaluation):
PYTHONPATH=src python3 -m pinn_surrogate.train --n-train-geoms 20 --physics on --steps 30 \
    --log-every 10 --max-eval-solves 2 --threads 2 --out runs/probe_on

# One real run (default 3000 steps, cosine LR 1e-3 -> 1e-5, adaptive loss weights):
PYTHONPATH=src python3 -m pinn_surrogate.train --n-train-geoms 80 --physics on --steps 3000 \
    --threads 2 --out runs/n80_physon_s0

# If the VM restarted or the laptop slept: same command plus --resume
PYTHONPATH=src python3 -m pinn_surrogate.train --n-train-geoms 80 --physics on --steps 3000 \
    --threads 2 --out runs/n80_physon_s0 --resume

# Re-evaluate a finished run (writes eval.json and the figures again)
PYTHONPATH=src python3 -m pinn_surrogate.evaluate --run runs/n80_physon_s0
```

A run directory holds `config.json`, `train.log`, `metrics.jsonl` (step, wall/train seconds,
s/step, each loss, each lambda), `checkpoint.pt` (every 200 steps, for `--resume`),
`model.pt`, `train_summary.json`, `eval.json`, `wall_tau_test_B.png`, `error_vs_re.png`.
Without `--resume`, train.py refuses to reuse a directory that has a checkpoint.

If the probe shows much more than ~0.3 s/step for physics on, the 3000-step run will exceed
the 15-minute target; the main cost knob is `--pde-nodes-per-solve` (default 256, i.e.
2048 second-derivative points per step). Use the same value in every ablation arm.

## 5. Stage 5 — the ablation (6 runs per seed, 2 in parallel x 2 threads)

The report's grid: 3000 steps, `--physics-share 0.1` (the default), seeds 0 and 1, output in
`runs/grid/`.

```bash
nohup bash ablation.sh > runs_ablation.log 2>&1 &          # seed 0, ~30 min on 4 cores
SEED=1 nohup bash ablation.sh > runs_ablation_s1.log 2>&1 &   # seed 1 (after seed 0)
tail -f runs_ablation.log            # per-run logs in runs/grid/<name>.stdout.log
```

Resume-safe: after a restart just run it again — finished runs are skipped, interrupted ones
continue from their last checkpoint (bit-exact). It ends by calling summarize and compare.

**In the Cowork VM** every process started by a shell call is killed when the call returns
(even with nohup/setsid), and calls are capped at ~1.5–3 min. There, advance the grid in
chunks and call this repeatedly until it prints "all runs finished":

```bash
SEED=0 CHUNK_S=72 PAR=2 THREADS=2 ./ablation_chunked.sh
./status.sh 0                         # one line per run: last step, or DONE + test_B result
```

## 6. Summarize, compare, plot

```bash
PYTHONPATH=src python3 -m pinn_surrogate.summarize --runs runs/grid   # per-run table
PYTHONPATH=src python3 -m pinn_surrogate.compare --runs runs/grid     # physics on vs off, paired
PYTHONPATH=src python3 -m pinn_surrogate.figures --runs runs/grid --out figures
cat runs/grid/ablation_summary.md runs/grid/comparison.md
```

`compare` pairs the two arms on the same held-out solves and runs a two-sided exact sign test
(numpy only). `figures` writes `fig_scaling.png` (error and residual vs training size, every
seed) and `fig_wall_curves.png` (wall shear along one held-out FSG cushion, FEM vs both arms).

## 7. v2 losses: wall shear and near-wall gradients (added 2026-10-21)

Two optional fixed-weight loss terms; both default to 0, which reproduces v1 bit-for-bit.
Findings and the full diagnostics are in `analysis/README.md`.

```bash
# wall-shear loss at the 256 wall points (the v2 default to use: weight 1)
PYTHONPATH=src python3 -m pinn_surrogate.train --n-train-geoms 160 --physics off \
  --tau-weight 1 --steps 3000 --threads 4 --out runs/my_tau_run

# near-wall gradient loss needs Dan's gradients packed once (~7 s, numpy only):
PYTHONPATH=src python3 -m pinn_surrogate.pack_grads --dataset ../steady_state_dataset/dataset_v1_1k
PYTHONPATH=src python3 -m pinn_surrogate.train --n-train-geoms 160 --physics off \
  --tau-weight 1 --grad-weight 1 --grad-nodes-per-solve 256 --out runs/my_tau_grad_run

# the sweeps behind the README tables (VM: call repeatedly until "all runs finished")
./analysis/tau_sweep.sh      # --tau-weight 1, 10 x seeds 0, 1  -> runs/tau/
./analysis/grad_sweep.sh     # tau/grad 0/1 and 1/1 x seeds 0, 1 -> runs/grad/
python3 analysis/a6_compare.py runs/grid/n160_physoff_s0 runs/tau/w1_s0 runs/grad/t1g1_s0
```

Speed at 2 threads on the VM: data only ~0.03 s/step, + wall shear ~0.085, + both ~0.07–0.12.

## Stage 1–2 tools (numpy only)

```bash
PYTHONPATH=src python3 -m pinn_surrogate.reference   # reference physics vs Dan's saved arrays
PYTHONPATH=src python3 -m pinn_surrogate.stats       # -> data/nondim_stats.json

# Lint / format / compile (pack.py is excluded from formatting on purpose; its one
# PLC0206 finding is known and left alone because pack.py produced data/)
ruff check src tests
ruff format src tests
python3 -m py_compile src/pinn_surrogate/*.py tests/*.py
```

Using the data from Python:

```python
from pinn_surrogate.data import load_packed, trust_weights
pk = load_packed()                 # data/packed_v1.npz, SI units as Dan saved them
nd = pk.nondim()                   # x*, u*, p*, wall p*/tau*, Re
idx = pk.train_subset(80)          # solves of the first 80 train geometries (family-balanced)
w = trust_weights(pk.arrays["r_rel"])  # use on interior nodes only (see docstring)
```
