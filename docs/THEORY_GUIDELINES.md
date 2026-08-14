# Projected-regret theory guidelines

Last updated: 2026-08-14 10:50 PDT

Notation authority: `docs/orthogonal_projected_regret_reweighting.tex`.

## Standing constraint

Do not claim uniform strict improvement of fixed-class one-step RWERM+ over
PTO, SPO+, or another DFL method. Such a result is impossible without further
structure. Oracle projected-regret reweighting can increase population
decision regret even when costs are deterministic given the context.

This constraint applies to proposed gradient, Newton, interpolation, and
decision-boundary arguments. Validation over a candidate set that contains
`alpha=0` gives a safety guarantee. It does not prove that projected regret
creates an improving candidate.

## Acceptable positive results

A positive theorem must do at least one of the following:

- prove a property that follows from the RWERM+ objective itself;
- derive strict improvement from operationally meaningful primitives in a
  stated problem class;
- give a finite-sample selection or testing guarantee; or
- state a condition that can be estimated on independent data and explain
  what mechanism the condition measures.

Do not use a condition that only renames the desired conclusion. Avoid
pairwise decision-face enumeration, artificial paths toward the unknown
conditional mean, and claims that a candidate reaches a boundary "sooner"
when the step size is freely tuned.

## Current conditional-theorem program

Use two separate results.

1. Prove the automatic prediction mechanism. Relative to PTO, RWERM+ shifts
   squared-error fit toward contexts according to `q^0-1`. In the linear model
   with an intercept in `Z`, the strength of the initial shift is an explicit
   residual moment.
2. Check decision conversion on an independent audit sample. Partition
   contexts using projected regret fixed before the audit. Compare the regret
   removed in the high-projected-regret group with the regret added elsewhere.
   This gives a multi-action conditional improvement theorem and a direct
   empirical diagnostic without decision-boundary assumptions.

The first result explains what projected regret guarantees. The second states
what the operational problem must supply. Keep these claims separate.
