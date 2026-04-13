# Task-Loss Reweighted MSE for Contextual Stochastic Optimization

This repository contains an implementation for re-weighting prediction loss using task error for ORIE6751's final project.


## Dependencies

-- `scikit-learn: 0.22.1`
-- `gurobi: 9.0.2`
-- `networkx: 2.5.1` 

The current experiment runner has been exercised in a newer environment as
well. The packages it needs are:

-- `numpy`
-- `pandas`
-- `scikit-learn`
-- `joblib`
-- `networkx`
-- `scipy`
-- `gurobipy`

## Experiments

The shortest-path experiments now have a script entry point at
`experiments/context_reweighting_experiment.py`.

Implementation details and review notes live in:

-- `docs/context_reweighting_implementation.md`
-- `docs/decision_weight_regression_method.tex`

The new context-only method uses:

-- a pilot least-squares cost model,
-- a scalar sampled-loss proxy based on realized regret,
-- cross-fitted context-only sample-weight estimation, and
-- a weighted least-squares refit.

By default the weight regression uses `RidgeCV`, so the weight model is tuned
inside the cross-fitting step with a small sklearn-native regularization sweep.

Useful presets:

-- `--preset deprecated-paper` reproduces the older shortest-path grid saved in
   `deprecated/res/linear_regression_-res.p`
-- `--preset may19` reproduces the newer notebook sweep behind the `may19`
   result CSVs
-- `--preset paper-figure5` reconstructs the paper's `SPO` comparison setting
   described around Figure 5

## Cloud Runbook

Minimal requirements:

-- Python `3.10+`
-- a working `gurobipy` install and valid Gurobi license
-- enough CPU/RAM for a long `SPO` run

Python packages:

-- `numpy`
-- `pandas`
-- `scikit-learn`
-- `joblib`
-- `networkx`
-- `scipy`
-- `gurobipy`

One-line install:

```bash
python -m pip install numpy pandas scikit-learn joblib networkx scipy gurobipy
```

Recommended cluster backend:

-- use `--parallel-backend processes`

Default cluster-style full Figure 5 run:

```bash
mkdir -p results && nohup python -u experiments/context_reweighting_experiment.py --preset paper-figure5 --parallel-backend processes --n-jobs 8 --batch-size 8 --verbose 0 --output results/paper_figure5_context.csv --run-log-path results/paper_figure5_context.log > results/paper_figure5_context.log 2>&1 &
```

Faster no-`SPO` run:

```bash
mkdir -p results && nohup python -u experiments/context_reweighting_experiment.py --preset paper-figure5 --skip-spo --parallel-backend processes --n-jobs 20 --batch-size 8 --verbose 0 --output results/paper_figure5_nospo.csv --run-log-path results/paper_figure5_nospo.log > results/paper_figure5_nospo.log 2>&1 &
```

Faster no-`SPO`, single-`mu=0.7925` run:

```bash
mkdir -p results && nohup python -u experiments/context_reweighting_experiment.py --preset paper-figure5 --skip-spo --mixture-weights 0.7925 --context-mixture-weights 0.7925 --parallel-backend processes --n-jobs 20 --batch-size 8 --verbose 0 --output results/paper_figure5_nospo_mu07925.csv --run-log-path results/paper_figure5_nospo_mu07925.log > results/paper_figure5_nospo_mu07925.log 2>&1 &
```

Mu-selection diagnostics run:

```bash
mkdir -p results && nohup python -u experiments/context_reweighting_experiment.py --preset paper-figure5 --skip-spo --compute-mu-diagnostics --diagnostic-cv-folds 5 --parallel-backend processes --n-jobs 8 --batch-size 8 --verbose 0 --output results/paper_figure5_mu_diagnostics.csv --run-log-path results/paper_figure5_mu_diagnostics.log > results/paper_figure5_mu_diagnostics.log 2>&1 &
```

Resume after interruption:

```bash
nohup python -u experiments/context_reweighting_experiment.py --preset paper-figure5 --parallel-backend processes --n-jobs 8 --batch-size 8 --verbose 0 --output results/paper_figure5_context.csv --run-log-path results/paper_figure5_context.log >> results/paper_figure5_context.log 2>&1 &
```

Progress:

```bash
tail -f results/paper_figure5_context.log
```

Status heartbeat:

```bash
cat results/paper_figure5_context.status.json
```

Run registry:

```bash
tail -n 20 results/run_history.md
```

Each invocation appends one row with the timestamp, preset, output CSV, log
file, and reconstructed Python command. You do not need to pass
`--launch-command` for normal use.

Watcher:

```bash
python scripts/watch_experiment_status.py --status-path results/paper_figure5_context.status.json
```

Background watcher:

```bash
nohup python scripts/watch_experiment_status.py --status-path results/paper_figure5_context.status.json > results/paper_figure5_context_watcher.log 2>&1 &
```

Watcher with Slack webhook:

```bash
SLACK_WEBHOOK_URL='https://hooks.slack.com/services/...' python scripts/watch_experiment_status.py --status-path results/paper_figure5_context.status.json
```

Background watcher with Slack webhook:

```bash
SLACK_WEBHOOK_URL='https://hooks.slack.com/services/...' nohup python scripts/watch_experiment_status.py --status-path results/paper_figure5_context.status.json > results/paper_figure5_context_watcher.log 2>&1 &
```

Background watcher with Slack webhook and terminal-alert mention:

```bash
SLACK_WEBHOOK_URL='https://hooks.slack.com/services/...' nohup python scripts/watch_experiment_status.py --status-path results/paper_figure5_context.status.json --slack-mention '<@U04FDFY1BTP>' > results/paper_figure5_context_watcher.log 2>&1 &
```

The watcher can also fall back to a PID or log path if needed:

```bash
python scripts/watch_experiment_status.py --pid 12345 --log-path results/paper_figure5_context.log
```

In `--log-path` mode, the watcher now parses the experiment's progress lines
directly, so it can still report `running` or `completed` even when the status
heartbeat is stale or unavailable.

The `--slack-mention` argument only prefixes terminal alerts such as
completion or failure. Progress updates remain untagged.

### Summarizing Figure 5 style results

After the run finishes:

```bash
python experiments/summarize_figure5_results.py \
  results/paper_figure5_context.csv \
  --output results/paper_figure5_context_summary.csv
```

This produces the best-over-mixture summary for:

-- `reweight_LS`
-- `context_reweight_LS`

alongside the mean `LS` and `SPO` results.

### 9. Practical notes

-- The run is long because `SPO` dominates runtime.
-- On the cluster, `joblib` with `--parallel-backend processes` behaved much
   better than `threads`.
-- Use the no-`SPO` and single-`mu=0.7925` commands above when you want a
   faster turnaround run.
-- For the full `SPO` run, use smaller batches and moderate worker counts first.
-- For the `mu`-diagnostics sweep, `joblib` worker `SIGKILL(-9)` errors should
   be interpreted as likely OS-level memory kills, not Python garbage
   collection issues.
-- If that happens on the diagnostics sweep, first reduce `--n-jobs` to `8`
   or `12`, or increase memory per worker before trying a larger process pool
   again.
