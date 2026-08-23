# v2.0.0 — Consolidated Edition

**A Kohlrausch or Weibull Exponent Below One Admits a Classical Mixture**
*Pollard's Criterion as a Licence Condition, Two Predictions Refutable Against Existing Literature, and a Self-Audit of the Framework That Produced Them*

This is a **major** version. It merges the three papers of the programme into one
document, reordered by logical dependency rather than chronology, and it retracts
or relabels several claims of v1.0.0 and v1.1.0. Anyone citing an earlier version
should read the errata below before relying on it.

63 pages · 26 sections in 7 parts · 8 appendices.

---

## Why the title changed

The previous title sold the container — a framework with its own name and its
errata — and buried the one result usable outside quantum information. A panel of
three independent judges, scoring on discoverability, referee credibility and
honesty against the actual content, converged on stating the theorem instead of
listing the genres.

The word **Weibull** is in the title deliberately: the reliability function
`R(t) = exp(−(t/τ)^β)` is the same expression, and the criterion settles a live
interpretive question in that field, which does not currently know the result.

---

## What this version retracts

**The strongest claim of the previous drafts was too strong, and it was ours.**
`thm:gate` establishes *representability of the curve* — it exhibits a positive
measure whose Laplace transform reproduces the data. The map from mechanism to
curve is many-to-one, so exhibiting a classical mechanism that reproduces the
curve does not establish that the mechanism *is* classical. Every instance of
"certifies classicality" has been replaced by the weaker and sufficient
statement: **the data cannot be adduced as evidence of coherence, because a
classical account of them exists.**

A clause was added stating that the gate applies to the **confidence interval**,
not the point estimate. A fit returning β̂ = 0.95 ± 0.10 establishes nothing.

**Attribution corrected.** Pollard proved the representability direction; the
equivalence with complete monotonicity is Hausdorff–Bernstein–Widder; that β > 1
fails is elementary. None of the theorem is ours. The contribution is the
decision to use it as a gate on interpretation.

---

## Errata to v1.0.0 and v1.1.0

| | published | corrected |
|---|---|---|
| Critical qubit number | n\* = 2.6 | **n\* = 1.848** |
| LIGO ringdown significance | 13.8σ | **12.5σ** (internal error scaled by √(χ²/ν) = 1.669) |
| 18 kHz emergence mode | reported as detected | **below the Rayleigh limit of its own data** |
| Comoving density of remnants | Ω = 0.12 | corresponds to **Ω = 0.063**; the published expression also carries a spurious c² |
| Axiom 3 (β = 1 − α/2) | stated as theorem | **empirical postulate** — no derivation survives |

On the LIGO number we record something that does us no favours: our **first**
correction of it, 20.9σ, was itself wrong for neglecting the χ²/ν scaling, and the
originally published 13.8σ was closer to the defensible value than our correction
of it. An audit is not automatically right because it is an audit.

---

## Two entries added to the register after the document was otherwise finished

**D27 — an internal contradiction of this document.** Appendix C derives
β = α/2 and then reaches β = 1 − α/2 by asserting the convention α + 2β = 2. The
two coincide only at α = 1, and the step assumes precisely what it is invoked to
justify. It is also refuted by the no-go theorem in the same document, which
obtains t^(1+α) from the same bath. The passage is retained and marked SUPERSEDED
rather than deleted, because the corrected chain is only legible against what it
corrects.

**D6 — reopened.** It had been marked resolved by the generalized-Langevin
bridge. Reopening followed from noticing that the register's own text called that
bridge "a physical hypothesis", and that the test of the same bridge is
designated the decisive experiment of the programme. One does not test what is
resolved. Three mutually exclusive routes exist for the same pair of exponents;
adopting one discards the others by preference, not by proof.

---

## The decisive test, which this version does not perform

On a single sample, the anomalous-diffusion exponent and the stretched-relaxation
exponent must satisfy

```
β = (1 + α_diff)/2 ,   with  β ≥ 1/2  as a corollary
```

Both quantities are routinely tabulated, by groups with no stake in this
framework, for overlapping sample sets in the NMR relaxometry and
diffusion-weighted imaging literature. **No new experiment is required to attempt
this refutation.**

It is not performed here because an audit that also supplies the confirmation of
its own subject is not an audit.

---

## Endpoint, labelled CONJ throughout

Planck-scale black-hole remnants as dark matter, at
**M_min = m_P/(2√π) = 6.140 µg** — an exact identity, not the approximation
0.28 m_P under which it was published.

Evaluating it shows the usual dismissal to be right for the wrong reason. The
remnant carries **148.6 J** at galactic velocity and one transit is expected per
**0.86 km²·yr**, so rarity is not the obstacle. But the mean free path in rock is
**10²³ light years** and the probability of one interaction while crossing the
entire Earth is **10⁻³²**. The candidate is **cross-section-limited, not
exposure-limited**, and the two demand opposite experiments. Four falsifiers are
given, two of which need no new instrument.

Our own first classification of it reached the opposite verdict, by comparing
available energy against a detector threshold without asking whether that energy
could couple. Energy that cannot be deposited is not a signal.

---

## Reproducibility

```
354 regression tests · 37 published-number checks · 0 failures
138 equations registered with source, evidence level and declared domain
27 documented inconsistencies · 11 remain open
```

Where a published number could not be reproduced it is recorded as an erratum
rather than adjusted. Where a correction of ours proved wrong, that is recorded
too.

---

## Metadata correction that affects citation

`10.5281/zenodo.18186310` was cited in the corpus as a separate concept DOI for a
second paper. **It is not.** Verified against the Zenodo API: it is the version
DOI of v1.0.0 of *this* chain, whose concept DOI is `10.5281/zenodo.18186309`.
There is no independent deposit for a second paper.

The keywords of v1.1.0 described the work as "structure-dependent quantum
decoherence". The document no longer sustains that reading and those keywords are
withdrawn.
