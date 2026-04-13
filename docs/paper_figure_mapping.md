# Paper Figure to Repo Mapping

This note maps the experiments described in `2211.05116v1.pdf` onto the code
and saved outputs present in this repository.

## Summary

The paper's shortest-path experiments are not represented by one single script
snapshot in the repo. Instead, the paper appears to correspond to a hybrid of:

- the older experiment flow in `deprecated/basis_localization.ipynb`
- the newer experiment flow in `Shortest Path Experiments.ipynb`

The main distinction is:

- the older flow matches the learning-curve style figures with the larger
  training-size grid and 5 mixture weights
- the newer flow matches the explicit `SPO` comparison machinery and the
  `LS` / `reweight_LS` / `SPO` tabular outputs

## Evidence from the PDF

From `2211.05116v1.pdf`:

- Figure 3:
  - shortest-path setup on a `5x5` grid
  - linear regression predictor
  - one round of reweighting
  - test set size `N = 10000`
  - `50` replications
  - `nu` ranged from `0` to `0.8`
- Figure 4:
  - repeated reweighting with `K in {1, 2, 3}`
- Figure 5:
  - `SPO+` comparison
  - linear regression predictor
  - `5` candidate mixture weights for reweighted LS
  - best out-of-sample mixture weight reported
- Figure 6:
  - same shortest-path setup as Figure 3
  - random forest predictor

The extracted figure labels in the PDF also show:

- linear-regression learning curves over training sizes that visually match
  `[100, 250, 500, 1000, 1500, 2000]`
- `mu` values consistent with the older 5-point grid
- Figure 5 panels indexed by polynomial degree and training sizes
  `[100, 1000, 2000]`

## Figure-by-Figure Mapping

### Figure 3: Linear regression learning curves

Paper description:

- shortest-path on `5x5`
- linear regression
- one reweighting round
- `50` reps
- test size `10000`
- multiple polynomial misspecification degrees

Closest repo source:

- `deprecated/basis_localization.ipynb`
- saved outputs:
  - `deprecated/res/linear_regression_-res.p`
  - `deprecated/res/linear_regression_misspec_*.p`
- plotting notebook:
  - `deprecated/plot_results.ipynb`

Matching details:

- `N_s = [100, 250, 500, 1000, 1500, 2000]`
- `polykernel_degree_vec = [1, 2, 4, 6, 8]`
- `n_test = 10000`
- `n_holdout = 1000`
- `LinearRegression`
- one-step weighted least squares
- 5 mixture weights from `np.linspace(1.0/n_wghts, 0.99, n_wghts)`

Mismatch:

- saved repo outputs use `12` reps, not `50`

Conclusion:

- Figure 3 is best matched by the older deprecated experiment flow.

### Figure 4: Repeated reweighting

Paper description:

- impact of `K in {1, 2, 3}` repeated reweightings
- linear regression
- shortest-path setup above

Closest repo source:

- the newer helper implementation in `reweighted_mse_helpers.py`
- the newer notebook `Shortest Path Experiments.ipynb`

Matching details:

- the newer helper directly supports `num_reweights`
- the paper's `K` study is conceptually much closer to this code path than to
  the older deprecated notebook

Conclusion:

- Figure 4 is closest to the newer experiment flow.

### Figure 5: Comparison to SPO+

Paper description:

- shortest-path setup outlined above
- linear regression
- compare predict-then-optimize, reweighted LS, and SPO+
- `5` candidate mixture weights for reweighted LS
- best out-of-sample mixture weight reported

Closest repo source:

- `Shortest Path Experiments.ipynb`
- `deprecated/may19_exp.csv`
- helper path now exposed through `run_replication_over_weights(...)`

Matching details:

- training-size grid `[100, 1000, 2000]`
- polynomial degree grid `[1, 2, 4, 6, 8]`
- includes `LS`, `reweight_LS`, and `SPO`

Mismatches:

- the newer notebook uses `4` mixture weights `[0.2, 0.4, 0.6, 0.8]`
- the newer notebook uses `8` reps
- the newer notebook uses `num_reweights = 3`
- the paper text says Figure 5 uses `5` candidate mixture weights and reads as
  a one-step reweighted comparison

Conclusion:

- Figure 5 is best matched by the newer `may19` code path structurally,
  but the exact paper configuration should be treated as a modified variant:
  - `N_s = [100, 1000, 2000]`
  - `degrees = [1, 2, 4, 6, 8]`
  - `n_reps = 50`
  - `5` candidate mixture weights
  - one reweighting step

### Figure 6: Random forest learning curves

Paper description:

- same shortest-path setup as Figure 3
- random forest predictor

Closest repo source:

- `deprecated/basis_localization.ipynb`

Matching details:

- explicit `RandomForestRegressor` exploration appears in the deprecated
  notebook
- Figure 6 uses the same larger training-size learning-curve style

Conclusion:

- Figure 6 is best matched by the older deprecated experiment flow.

## Practical Rerun Mapping

For reproducing old learning-curve style experiments with the new
context-conditional weights:

- use `--preset deprecated-paper`

For reproducing the `SPO` comparison family:

- use `--preset paper-figure5`

The `paper-figure5` preset is a reconstruction guided by the PDF:

- `n_train_values = [100, 1000, 2000]`
- `degrees = [1, 2, 4, 6, 8]`
- `mixture_weights = [0.2, 0.3975, 0.595, 0.7925, 0.99]`
- `n_test = 10000`
- `n_reps = 50`
- one reweighting step
- `SPO` included

## Caveat

The repository does not appear to contain the exact final paper run script.
What it does contain is enough to identify the nearest code path for each figure
and reconstruct a faithful rerun configuration.
