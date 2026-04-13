# Context-Only Reweighting Implementation

This note documents the implementation added for the cross-fitted context-only
reweighting method and how it fits into the existing code path.

## Goal

The existing repository already supported two relevant training styles:

- ordinary least squares cost prediction via `get_weighted_predictors(...)`
- a task-aware weighted least squares refit via `feasible_least_squares(...)`

The new implementation keeps that structure and inserts one extra step between
the pilot fit and the weighted refit:

1. fit a pilot cost model with ordinary least squares
2. compute one scalar regret-based raw weight per training example
3. regress those raw weights on context using cross-fitting
4. use the fitted context-only weights as sklearn `sample_weight`

This gives a minimal implementation of the note-style method without adding a
new model abstraction or a second training stack.

## Files Added Or Changed

- `reweighted_mse_helpers.py`
- `experiments/context_reweighting_experiment.py`
- `README.md`
- `docs/context_reweighting_implementation.md`
- `docs/decision_weight_regression_method.tex`
- `.gitignore`

## Implementation Structure

### 1. Self-contained shortest-path experiment helpers

`ShortestPathOracle` was moved into `reweighted_mse_helpers.py` so the
experiment code no longer depends on a notebook-local class definition.

This matters because `run_replication_over_weights(...)` already assumed the
existence of `ShortestPathOracle(graph_params)`, but before this change that
class only existed inside `Shortest Path Experiments.ipynb`.

### 2. Reuse of the existing sklearn fitting path

The cost model still uses `get_weighted_predictors(...)`, which fits one sklearn
regressor per cost dimension. The only change is that this helper now accepts:

- matrix-valued weights for the older edge-level reweighting path
- vector-valued weights for the new scalar context-only weighting path
- sklearn estimator instances as well as estimator classes

That lets the new method stay inside the same training paradigm as the
repository's existing code.

### 3. Raw decision weights from realized regret

The new path computes pilot regrets with the existing regret-evaluation logic:

- fit pilot predictors on `(X_train, c_train)`
- call `get_regret(...)`
- extract the per-example training regrets

Those regrets are converted into scalar raw weights with:

`w_i^raw = (1 - mu) + mu * phi(Reg_i)`

where the implementation uses:

- `Reg_i = |instance['opt_val'] - c_i^T x*(c_hat_i)|`
- `phi(r) = clip(r / q, 0, 1)`
- `q = empirical 0.9 quantile of the pilot regrets`

This makes the weights bounded, nonnegative, and numerically stable. The lower
bound is always `1 - mu`, and the upper bound is
`1 - mu + mu * max_weight` with default `max_weight = 1.0`.

The relevant helpers are:

- `get_raw_context_weights(...)`
- `_normalize_regret_weights(...)`

### 4. Cross-fitted alpha-weight regression

The core new step is `estimate_context_weights_cross_fitted(...)`.

For each fold:

1. hold out one fold of contexts
2. fit a weight regressor on the remaining folds using
   `(X_train[:, train_index].T, raw_weights[train_index])`
3. predict out-of-fold weights for the held-out fold
4. clip predictions back into `[1 - mu, 1 - mu + mu * max_weight]`

The default weight regressor is `RidgeCV`, which keeps the implementation small
and gives built-in hyperparameter tuning over a regularization grid:

`alphas = np.logspace(-4, 4, 9)`

Alternative weight regressors exposed by the experiment script:

- `LinearRegression`
- `RandomForestRegressor` with `GridSearchCV`

The cross-fitting logic uses sklearn-native tools:

- `KFold`
- `RidgeCV`
- `GridSearchCV`

### 5. Weighted refit

After the out-of-fold alpha weights are estimated, the final cost predictors are
fit with:

`get_weighted_predictors(regressor, c_train, X_train, fitted_weights)`

Because `fitted_weights` is one vector of length `n_train`, the same alpha
weight is applied to every cost coordinate of sample `i`. This is the intended
behavior for a context-only sample weighting scheme.

The full one-step method lives in:

- `feasible_context_least_squares(...)`

### 6. Experiment integration

`run_replication_over_weights(...)` now evaluates three families of methods in a
single replication:

- `LS`
- `reweight_LS`
- `context_reweight_LS`

The new result rows include:

- `algo='context_reweight_LS'`
- `mixture_weight`
- `avg_raw_weight`
- `avg_fitted_weight`

The script entry point is:

- `experiments/context_reweighting_experiment.py`

That script reproduces the notebook-style shortest-path setup but makes it
callable from the command line and configurable by flags.

It now also exposes named presets for the historical experiment grids already
present in the repository:

- `--preset deprecated-paper`
- `--preset may19`

## Why This Is Minimal

The implementation deliberately avoids:

- new model classes for the cost predictor
- custom CV or boosting infrastructure
- changes to the existing regret evaluator
- changes to the old edge-level reweighting baseline

Instead it reuses the repository's main abstractions:

- generated instance dictionaries
- shortest-path oracle evaluation
- per-dimension sklearn regression
- result collection through `run_replication_over_weights(...)`

## Default Hyperparameter Behavior

For the decision-weight regression:

- default model: `RidgeCV`
- default folds for cross-fitting: `5`
- default inner CV for tuning: `3`
- default clipping cap: `max_weight=1.0`
- default regret normalization scale: `90th` percentile

This is intentionally conservative. The weight model is tuned, but the tuning
surface stays small enough that the experiment remains easy to run.

## Example Command

```bash
python experiments/context_reweighting_experiment.py \
  --degrees 2 \
  --n-train-values 100,1000 \
  --n-reps 4 \
  --weight-model ridgecv \
  --output results/context_reweighting.csv
```

For a small non-linear weight regressor with explicit tuning:

```bash
python experiments/context_reweighting_experiment.py \
  --weight-model rf \
  --context-folds 5 \
  --context-weight-cv 3 \
  --output results/context_reweighting_rf.csv
```

## Known Simplifications

- The new method currently implements the one-step pilot-plus-refit variant, not
  the fully iterative or boosting variants from the note.
- The raw decision weight uses realized regret rather than a second sampled
  outcome, because the repository only observes one cost realization per context.
- The default transform clips regret to a bounded interval rather than using a
  more elaborate calibration map.

Those were deliberate choices to keep the implementation aligned with the
existing repository style and keep the code delta small enough for review.
