# Projected-regret theory guidelines

Last updated: 2026-08-14 12:41 PDT

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

## Preferred explanation for the product block

Use the functional-gradient explanation for adding `q_t^0 H_t`. Freeze the
projected-regret function from the current predictor. The negative functional
gradient of its weighted squared loss is

```text
q_{alpha,t}^0 (mu^0-f_t)
= (1-alpha)(mu^0-f_t) + alpha q_t^0(mu^0-f_t).
```

If `h in H_t` approximates the ordinary residual, then

```text
(1-alpha)h + alpha q_t^0 h
```

belongs to `H_t + q_t^0 H_t` and approximates the weighted functional
gradient. A second copy of `H_t` can enlarge a norm-bounded class by increasing
its coefficient radius. It does not add new directions because
`span(H_t + H_t) = span(H_t)`. Multiplication by `q_t^0` usually adds
functions outside this span.

This is the primary algorithmic justification for the product block. It is
more precise than saying that projected regret identifies useful
interactions. The latter phrase can appear only as informal intuition.

Do not turn this explanation into a decision-regret claim. The product block
approximates a functional gradient of frozen weighted prediction loss. A
nonzero projection gives descent for that surrogate, not automatic descent
in decision regret.

## Supporting results, not the missing theorem

The finite-alpha error-transfer identity is comparative statics of the
weighted objective. It shows how the refit redistributes prediction fit. It
is not calibration and does not justify decision improvement.

The high-`q^0` versus low-`q^0` regret decomposition is an empirical audit. It
can explain an observed improvement after fitting. It does not prove that
projected regret creates an improving candidate.

## Unresolved main theorem

The paper still needs a simple, nontrivial existence result of the form

```text
inf_alpha Reg(c_{theta_alpha^0}) < Reg(c_{theta_0^0}),
```

or its product-block analogue. The assumptions must imply that projected
regret creates an improving candidate. They must not assume that the candidate
already has lower regret.

The preferred theorem should satisfy these requirements:

- cover multiple actions without enumerating decision faces;
- use a standard margin condition if a prediction-to-decision conversion is
  needed;
- state the misspecification structure in operational or statistical
  primitives;
- explain why projected regret is useful relative to ordinary residual
  fitting, not only why class expansion is useful;
- admit a meaningful empirical diagnostic; and
- avoid artificial interpolation paths, free-step boundary comparisons, and
  gradient-alignment assumptions that restate the desired conclusion.

Until this theorem is established, present population identification,
orthogonal estimation, validation safety, and the functional-gradient product
block rationale as separate valid contributions. Do not present their
combination as proof of improvement over PTO.
