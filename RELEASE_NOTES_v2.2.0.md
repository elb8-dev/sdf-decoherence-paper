# v2.2.0 — The Impedance of Information

**DOI:** [10.5281/zenodo.22240350](https://doi.org/10.5281/zenodo.22240350) · concept DOI `10.5281/zenodo.18186309`

**The Impedance of Information — One Möbius Coordinate for Decoherence, Entropy and Extractable Work**
*Measured from a Superconducting Qubit to a Black-Hole Ringdown; Two Predictions Refutable Against Existing Literature; and a Self-Audit of the Framework That Produced Them*

81 pages · 33 sections in 7 parts · 8 appendices.

This version adds six sections and **corrects three errors of fact in the deposited
text**, one of which inverted the reading of a 38-event table. Anyone citing v2.1.0
for the LIGO exponent, for the remnant flux, or for any ΔAIC value should read
"What this version corrects" below.

---

## Why the title changed

The two previous titles named what the work *withdrew*: an exponent that admits a
classical mixture, and a qubit that gives two verdicts. This one names what it
*proves*, and the proof is an identity — the framework's reduced coordinate and
the Smith chart of transmission-line engineering are **the same Möbius
transformation**.

That is aspirational and literal at once. Nothing in the title is unproven.

---

## What is proved

With `Γ = 2x − 1` and `z = (1+Γ)/(1−Γ)`, six identities, symbolically:

```
z              = x/(1−x)          the normalised impedance IS the odds
ln z           = logit x          the coordinate IS the log of an impedance
z(1−x)·z(x)    = 1                x ↔ 1−x IS Z ↔ 1/Z (impedance ↔ admittance)
1 − Γ²         = 4x(1−x)          transmitted power IS 4·D·C̄/L²
e^{I_Q}        = 1 + z            accessible information IS ln(1 + impedance)
dH₂/dx         = −ln z            the coordinate IS the derivative of the entropy
```

and `d²H₂/dx² = −4/(1−Γ²)`: **the curvature of the entropy diverges exactly where
transmitted power vanishes.** The one-bit threshold `I_Q = ln 2` sits at `Γ = 0`,
which is **impedance matching**.

**Attribution, stated plainly:** the chart is Smith's and the line physics
Heaviside's. What is contributed is that the two coordinates are one — a change of
variables with no new data, and with one deficit stated rather than hidden: the
chart is a disc, this framework runs only its real diameter, and it has no
coordinate for reactance.

---

## What is measured

**The arrow separating the two channels is thermal.** On a qubit, the mirror
`x ↔ 1−x` is conjugation by `X`. Pure dephasing commutes with it exactly
(`0.000e+00` over 400 random states); amplitude damping does not (`9.000e−01`). And
the asymmetry is not a free parameter — it equals

```
tanh(ħω / 2k_B T)
```

verified symbolically: `1.000000` at the 15 mK of the cryostat, `0.379895` at
300 mK, `0.029986` at 4 K. **The dichotomy of v2.1.0 therefore holds at the
temperature at which it was measured, and is sized by a knob outside the fridge.**

`PRED` — a temperature sweep on the same apparatus must attenuate it to 0.38 at
300 mK and 0.12 at 1 K. A `T₁` arm still returning 0 of 220 rises when heated
**refutes this**.

---

## What this version corrects

Three errors of fact in the deposited text, all found by an adversarial audit of
the source against its own data files.

1. **The LIGO exponent in the body contradicted the paper's own erratum.**
   §22 quoted `β = 0.931 ± 0.003` at `13.8σ`; §15 had already corrected it to
   `0.93113 ± 0.00551` at `12.5σ`, with the internal error scaled by the Birge
   ratio `1.669` the catalogue requires. The body now carries the corrected value.

2. **The ΔAIC sign convention was stated inverted relative to the tabulated
   column.** The appendix declared `ΔAIC = AIC_GR − AIC_SDF` and that positive
   values favour SDF, while tabulating the column as stored — in which 34 of 38
   events are **negative** with a median `R² = 0.995`. **The stated reading was
   backwards for every entry.** Convention and reading are now both corrected.

3. **The remnant table used the cosmological mean density for a terrestrial
   rate.** The number density, mean separation and flux rows are now labelled as
   the cosmological mean, and are explicitly separated from the exposure row,
   which carries the local-halo figure. They must not be read as one calculation.

---

## What the register records

**Forty-six entries: 24 resolved, 19 open, 1 partially answered, 1 closed, 1
invalidated.**

- **D24 — resolved, and it carried no number until now.** The framework writes `σ`
  for the entropy and computes `H₂(D)`. Under pure dephasing the two coincide by a
  symmetry of `H₂`; under amplitude damping they separate by **0.6931 nats — the
  whole range of a qubit entropy** — and at full application the state is *pure*,
  `S_vN = 0`, where the framework reports its **maximum**.
- **D43 — partially answered.** The ringdown fits are admissible: `χ²/dof` median
  **1.023**, none above 2, with `R²` median 0.995 and SNR median 59, so the
  non-rejection carries weight. The hardware fits are not (11.49 and 1598). Open
  for hardware, answered for ringdown.
- **D45 — new, open, high impact.** The reduced coordinate is intensive and bounded
  by one bit; a black hole is extreme only in the extensive account. **The
  framework has nowhere to put a black hole**, and that is not fixed by choosing a
  better point.
- **D46 — new, resolved.** The binding ladder omitted helium, which is the
  electronic ceiling, and did not declare its conversion (Landauer at 300 K).

---

## What is not claimed

The passive-state construction is Pusz–Woronowicz and Lenard's, the cycle
efficiency Carnot's, and the `k_BT ln 2` per erased bit Landauer's. **What this
framework adds is the criterion for *which channel* satisfies the equation, never
the equation.** The two measured channels differ in one bit, four cells are
possible, the experiment occupied three, and the empty one — reversible energy
exchange, ordinary strong-coupling circuit electrodynamics — is where the criterion
can next be refuted.

---

## Unchanged

The exact identities, the no-go theorem, the transferable criteria and the
dark-matter conjecture stand as in v2.1.0, with that version's errata intact. The
decisive test the programme identifies — `β = (1 + α_diff)/2` on a single sample,
refutable against existing NMR and diffusion literature without new experiments —
is still not performed here.

---

## Scope

One device, one session, twenty qubits, and a single-qubit observable blind to
entanglement by construction. The identities of the first section are mathematics
and carry no such limit; **everything measured does**.
