# Task-Loss Reweighted MSE for Contextual Stochastic Optimization

This repository contains an implementation for re-weighting prediction loss using task error for ORIE6751's final project.


## Dependencies

-- `scikit-learn: 0.22.1`
-- `gurobi: 9.0.2`
-- `networkx: 2.5.1` 

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
