# Signature-Based Stochastic Control — Empirical Study

Companion code for the empirical study in Chapters 5 and 7:

- **Phase 1** (Chapter 5): Reproduction of Bank et al. (2024) Table 1 — optimal tracking of fractional Brownian motion using signature-based policies.
- **Phase 2** (Chapter 7): Signature-based optimal execution for fractional Ornstein-Uhlenbeck pairs trading under state-dependent dynamics.

---

## HPC Quick Start (bwUniCluster 3.0)

### Step 1: Log in

```bash
ssh <username>@uc3.scc.kit.edu
```

### Step 2: Create a workspace (60 days, extendable to 240)

```bash
ws_allocate thesis 60
```

Note down the path, or retrieve it later with:

```bash
ws_find thesis
```

To extend later (e.g. by 30 more days):

```bash
ws_extend thesis 30
```

### Step 3: Upload the code

From your **local machine**:

```bash
scp -r empiricalstudy_clean/ <username>@uc3.scc.kit.edu:$(ssh <username>@uc3.scc.kit.edu 'ws_find thesis')/
```

Or on the cluster, if the code is in a git repo:

```bash
cd $(ws_find thesis)
git clone <your-repo-url> empiricalstudy_clean
```

### Step 4: Set up the Python environment

```bash
cd $(ws_find thesis)/empiricalstudy_clean
bash setup_hpc.sh
```

This loads `devel/python/3.12`, creates a venv, installs all dependencies, and verifies `iisignature` + `logsigjoin`.

### Step 5: Preflight check (on a compute node)

```bash
cd $(ws_find thesis)/empiricalstudy_clean
sbatch scripts/slurm/preflight_phase2_cpu.sbatch
```

Check the output:

```bash
cat results/slurm/preflight_*.out
```

You should see `ALL CHECKS PASSED`.

### Step 6: Submit experiments

Navigate to the project directory first:

```bash
cd $(ws_find thesis)/empiricalstudy_clean
```

**Option A — Everything at once:**

```bash
bash scripts/slurm/submit_all_production.sh
```

This submits Phase 1 + Phase 2 (precompute + train). The `RUN_ID` is auto-generated from the current timestamp. To include robustness:

```bash
RUN_ROBUSTNESS=1 bash scripts/slurm/submit_all_production.sh
```

**Option B — Submit individually:**

```bash
# Phase 1 (independent, ~2h, 1 node)
sbatch scripts/slurm/phase1_production.sbatch

# Phase 2 precompute (~2h, 1 node)
PRE=$(sbatch --parsable --export=ALL,RUN_ID=$(date +%Y%m%d_%H%M%S) \
  scripts/slurm/phase2_precompute_production.sbatch)

# Phase 2 train (~12h, 1 node — starts after precompute finishes)
sbatch --dependency=afterok:$PRE --export=ALL,RUN_ID=$(date +%Y%m%d_%H%M%S) \
  scripts/slurm/phase2_train_production.sbatch

# Phase 2 robustness (optional, independent, ~2h)
sbatch --export=ALL,RUN_ID=$(date +%Y%m%d_%H%M%S) \
  scripts/slurm/phase2_robustness_production.sbatch
```

### Step 7: Monitor jobs

```bash
squeue -u $USER               # list your jobs
squeue --start                 # estimated start times
cat results/slurm/phase1_*.out # Phase 1 output
cat results/slurm/train_*.out  # Phase 2 training output
```

### Step 8: Post-run analysis

```bash
cd $(ws_find thesis)/empiricalstudy_clean
module load devel/python/3.12
source venv/bin/activate

# Phase 1: aggregate + plot
python scripts/aggregate_phase1.py
python scripts/plot_phase1_bank_insights.py --summary results/phase1/summary.csv

# Phase 2: review + plot
python scripts/review_phase2_smoke.py
python scripts/plot_phase2_h025_dashboard.py
```

---

## Resource summary

| Job | Node type | Cores | RAM | Walltime | Tasks |
|-----|-----------|-------|-----|----------|-------|
| Phase 1 | cpu (EPYC 9454) | 96 | 384 GiB | 3h | 30 |
| Phase 2 precompute | cpu | 96 | 384 GiB | 2h | 75 |
| Phase 2 train | cpu | 96 | 384 GiB | 12h | 600 |
| Phase 2 robustness | cpu | 96 | 384 GiB | 4h | 36 |

All jobs use a single node. Phase 1 and Phase 2 precompute can run in parallel on separate nodes.

---

## Local setup (for development/testing)

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

## Tests

```bash
python -m pytest tests/ -v
```

## Note on iisignature

Version 0.24 is required for `logsigjoin` (incremental log-signature computation). If `pip install` fails, ensure GCC and NumPy headers are available:

```bash
module load devel/python/3.12   # provides numpy headers
module load compiler/gnu        # provides GCC (if not already loaded)
```

## Configuration

- `config/default.yaml` — main production configuration (both Phase 1 and Phase 2)
- `config/robustness_low_kappa.yaml` — robustness check with kappa_0 = 5
- `config/robustness_high_sigma.yaml` — robustness check with sigma_min = 0.20

## Project structure

```
src/                  Core library
  fbm.py              Fractional Brownian motion simulation (Davies-Harte)
  signatures.py        Signature and log-signature computation
  policy.py            Linear and DNN policy architectures
  riccati.py           Riccati ODE solver for HJB baseline
  phase2_core.py       Spread generation, log-signature features, simulation
  variance_matching.py Variance-matching parameterization (L3)
  hjb_pairs.py         HJB Riccati baseline (H=0.5, L1)
  repro_manifest.py    Reproducibility manifest logging

scripts/              CLI entry points and analysis
  phase1_tracking.py   Bank et al. Table 1 reproduction
  aggregate_phase1.py  Merge parallel Phase 1 results
  plot_phase1_bank_insights.py  Phase 1 comparison plots
  phase2_precompute.py Precompute fBm + logsig features
  phase2_train.py      Train signature policies
  review_phase2_smoke.py  Phase 2 analysis & plotting
  plot_phase2_h025_dashboard.py  H=0.25 diagnostic plots

scripts/slurm/        SLURM batch scripts for bwUniCluster 3.0
  submit_all_production.sh  Master script (Phase 1 + Phase 2)
  submit_phase2_production.sh  Phase 2 only
  phase1_production.sbatch  Phase 1 training (30 jobs, ~2h)
  phase2_precompute_production.sbatch  (~2h)
  phase2_train_production.sbatch  (~12h)
  phase2_robustness_production.sbatch  (~4h)
  preflight_phase2_cpu.sbatch  Environment sanity check

tests/                Unit tests
config/               YAML configuration files
```
