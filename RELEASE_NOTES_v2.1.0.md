# v2.1.0 — The criterion, measured

**One Qubit, Two Verdicts — Pure Dephasing With No Classical Rate Density, Energy Relaxation With One**
*Complete Monotonicity Measured Without Fitting on Superconducting Hardware; Two Predictions Refutable Against Existing Literature; and a Self-Audit of the Framework That Produced Them*

This is a **minor** version by numbering and a substantive one by content. It adds
the experiment the programme had been missing and, on the strength of it,
withdraws a way of reading every fitted exponent in the corpus — including our
own. Anyone citing v2.0.0 for a statement about a fitted β should read
§1.4 and §7b before relying on it.

72 pages · 27 sections in 7 parts · 8 appendices.

---

## Why the title changed, again

The v2.0.0 title stated the theorem: β ≤ 1 admits a classical mixture. The
theorem is unchanged and remains correct. What the new title states is what the
theorem yields when it is finally applied to a device instead of to a fit of
one — and the answer is not uniform. **The same qubit gives opposite verdicts in
its two channels**: energy relaxation is representable as a classical mixture of
rates, pure dephasing is not.

An earlier draft of this title led with the correction — *a fitted exponent is
not the test*. That is true, it is the load-bearing methodological point, and it
is in the abstract. It is not the title, because a title that reads as an
erratum buries what a reader outside this framework can use.

---

## What is new

**§7b · The criterion applied directly.** 20 non-adjacent qubits of
`ibm_marrakesh`, one session, one fixed layout, 39 circuits, 958,452 shots.
Three arms — Ramsey, Hahn echo, T₁ reference — designed so that the gate can be
applied to the curve itself rather than to a fit of it.

| Channel | NNLS χ²/dof (200 free rates) | Verdict |
|---|---|---|
| T₁ — energy exchange | **0.99** | admits a positive rate density |
| echo — pure dephasing T_φ | **12.43** | **does not** |

The cause needs no model: the echo curve **rises**, in 10 of 220 consecutive
segments above 3σ and six above 5σ, peaking at z = +11.01, across 7 of 20
qubits. The T₁ control gives **0 of 220**, with backflow exactly zero in all
twenty; 0.30 false positives were expected. Confirmed independently three ways —
the Breuer–Laine–Piilo witness on the same counts (12/20 versus 0/20), the
residual against a one-rate Lindblad twin (+10.6σ at the last point versus
+1.4σ), and f_rev = 0.674 failing to correlate with that witness (ρ = −0.107),
which shows the surviving memory is not the quasi-static component the echo
already refocused.

**§24b · Method.** The working rules of the programme, each illustrated with the
case in this document where following it forced something to be given up. It
includes the passage of an earlier draft that was written, committed and then
removed for disposing of a measured result by appeal to the existence of prior
literature rather than to data.

**§24.6–24.9 · What the measurement leaves indicated but not decided.** The
mechanism (fluctuators: consistent, not identified), whether the split is a
property of the channel or of this device, the estimator that was withdrawn, and
— the uncomfortable one — whether any exponent in this corpus survives its own
goodness of fit.

---

## What this version withdraws

1. **Reading a fitted β without reporting its goodness of fit.** At σ/C ≈ 0.011
   the two-parameter Kohlrausch form is rejected in both dephasing arms
   (χ²/dof = 11.49 and 1598). A non-rejection at low statistical power is not
   evidence of adequacy, and Part II of the paper had been treating it as such.
   Recorded as **D43**, open, and it conditions how every exponent in Part II
   may be read.

2. **Every β-dependent quantity of the new run.** The pre-registered positive
   control fired: the T₁ arm returned β = 0.8664 ± 0.0285 where a single rate
   must give 1. Its premise proved false — T₁ is not a single rate on real
   hardware, shown two independent ways — but a premise falsified after the fact
   does not annul the clause. The distance to the pole β = 1 and the
   Laplace/Fourier branch assignment are withdrawn for that run. **We report no
   branch for any arm of it.**

3. **The universal claim that no platform violates β < 1** (**D29**), which
   survived into §26 of v2.0.0 after being corrected elsewhere. It is false
   against this paper's own §8 table. Now withdrawn in both places.

---

## What this version falsifies in the previous one

**D42.** §20 of v2.0.0 states that revival phenomena are ``slowdowns along the
flow, not reversals'', and that the decoherence arrow is one ``no local
operation can revert''. Both are falsified by measurement: the echo curve rises
with the control at zero, and a single X pulse — a local unitary — reverts 67 %
of the free-induction dephasing. §3's `thm:backflow_bound`, which *admits*
recoherence and bounds it, is the section the data support. The theorem itself
survives and receives its first experimental test: 2 of 20 qubits exceed the
bound at >3σ, of which **one** lies inside its stated hypotheses (β < 1) —
suggestive after multiplicity, not decisive.

**D44.** PRED-30.4 of §25 predicted that the Breuer–Laine–Piilo measure is
positive for β < 0.8, and §12 the stronger version: that it can be computed from
a fitted β without independent spectroscopy. Tested by the very protocol they
named, both fail. β < 0.8 is neither sufficient (2 of 6 such qubits give N = 0)
nor necessary (8 qubits above 0.8 give N > 0), and the within-arm correlation is
−0.299. The decisive comparison is between arms: median β of 0.9011 and 0.9070
— the same exponent to six thousandths — returning N of 0.00827 and exactly
0.00000. **A fitted β does not determine N.** This is a prediction of this
framework, refuted by an experiment this framework designed.

---

## Register

Forty-four entries: 22 resolved, 20 open, 1 closed, 1 invalidated. Of the open
ones, 13 are high impact. **D40 is closed** by §7b — its stated closing
condition was a design separating relaxation from dephasing under a single
definition, which is exactly this one. **D38 and D39 are not closed by it**, and
the paper says so explicitly: their closing conditions name the earlier
experiment specifically.

---

## Reproducibility

Every number in §7b is regenerated from the raw counts by a single extractor
that writes them alongside the SHA-256 of the counts file, with a verification
mode that fails if any of them has moved. The six figures are produced by a
script that **refuses to write** if its own numbers disagree with that file.
Raw counts are versioned compressed; losing them is what invalidated the
previous experimental corpus (D38), and it does not happen twice.

---

## What has not changed

The exact identities, the no-go theorem, the transferable criteria and the
dark-matter conjecture are as in v2.0.0, with the errata of that version intact.
The decisive test the programme identifies — β = (1 + α_diff)/2 on a single
sample, refutable against existing NMR and diffusion literature without new
experiments — has still not been performed here, for the reason given in v2.0.0:
an audit that supplies the confirmation of its own subject is not an audit.

---

## Scope, stated once more because it is easy to overstate

One device. One session. Twenty qubits. A single-qubit observable, blind to
entanglement by construction. The measurement establishes that the criterion is
applicable to hardware and that it separates two channels of the same qubit; it
establishes nothing about superconducting qubits in general, and the corrected
pre-registration for a second device is published in §24.7 rather than
attempted here.
