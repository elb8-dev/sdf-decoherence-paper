# One Qubit, Two Verdicts

### Pure Dephasing With No Classical Rate Density, Energy Relaxation With One

> **v2.1.0 — the criterion, measured.** Adds the experiment the programme lacked: the
> criterion applied *directly* to a real decay curve, without fitting. On one device it
> answers differently in two channels of the same qubit, and it withdraws a way of reading
> every fitted exponent in the corpus — including our own. See `RELEASE_NOTES_v2.1.0.md`.
>
> **v2.0.0 — consolidated edition.** Merges the three papers of the programme into one
> document, reordered by logical dependency. See `RELEASE_NOTES_v2.0.0.md` for what this
> version **retracts**: the overclaim that a fitted exponent certifies classicality (it
> certifies *compatibility* with it), the LIGO significance (12.5σ, and our own first
> correction was worse than the number it corrected), and Axiom 3, relabelled from theorem
> to empirical postulate.

*Previously: SDF: Universal Stretched-Exponential Decoherence*

[![Concept DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18186309.svg)](https://doi.org/10.5281/zenodo.18186309)
[![v2.1.0](https://img.shields.io/badge/v2.1.0-10.5281%2Fzenodo.22238971-blue)](https://doi.org/10.5281/zenodo.22238971)
[![v2.0.0](https://img.shields.io/badge/v2.0.0-10.5281%2Fzenodo.22068343-blue)](https://doi.org/10.5281/zenodo.22068343)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Paper

**Title:** Multi-State Validation and Anti-Quantum Extension of Universal Stretched-Exponential Decoherence: From IBM Quantum Hardware to Gravitational Scales

**Author:** Eloy René Becerra Daly (Independent Researcher, Spain)

**Contact:** daly@icam.es

**Paper:** 40 pages, 8 figures, 30 tables, 6 appendices, 46 references

---

## Abstract

I establish that the degradation functional $D(\gamma) = L[1 - \exp(-(\gamma/\tau)^\beta)]$ is the *unique* solution to a physically motivated first-order ODE, providing a universal stretched-exponential description of decoherence across all tested physical scales. The key parameter $\beta$ characterizes deviations from pure exponential decay ($\beta = 1$), with the theoretical relation $\beta = 1 - \alpha/2$ derived rigorously from Lindblad master equations with Lévy-stable noise.

### Key Results

| Platform | β | σ(β) | Significance |
|----------|---|------|--------------|
| **IBM Quantum** | 0.852 | ±0.028 | 5.2σ from β=1 |
| **LIGO GWTC-1/2/3** | 0.931 | ±0.003 | 13.8σ from β=1 |
| IonQ | 0.94 | ±0.02 | — |
| Quantinuum | 0.97 | ±0.01 | — |
| Simulation | 1.00 | ±0.001 | Theoretical limit |

### Theoretical Contributions

- **Axiomatic foundation:** D(γ) derived from Picard–Lindelöf existence and uniqueness (19 theorems, 4 conjectures)
- **Information geometry:** Decoherence embedded in Bures–Uhlmann manifold; geodesic curvature κ_g ∝ |1−β|
- **Entropic phase space:** Metric signature (2,d) with decoherence and entropy as timelike dimensions
- **Biological extension:** Diffusion–decoherence correspondence β_SDF = α_diff/2; Λ-hierarchy produces biological scales
- **Information conservation:** Exact D + C = 1 verified on hardware
- **Emergent gravity:** G_eff(r) = G·D(r/ℓ_P) regularises singularities; dark matter candidate M_min ≈ 6 ng
- **Entropic duality:** Every coalescence decomposes into anti-entropic product (β_f → 1) and entropic medium (β_M < 1)
- **Anti-quantum irreversibility:** D × D̄ > 0 formalized with spectral predictions

### Multi-Scale Hierarchy

τ(Λ) = τ_P/Λ validated across nuclear (3,558 nuclei), atomic (118 elements), and molecular (38 bond types) scales, spanning 10⁻⁴⁴–10¹⁷ s.

---

## Repository Contents

```
sdf-decoherence-paper/
├── paper/
│   └── SDF_Universal_Decoherence_Becerra_Daly.pdf   # Full manuscript (40 pp)
├── data/
│   ├── ibm_quantum_fits.csv           # IBM Quantum hardware (42 configurations)
│   ├── ligo_ringdown_38events.csv     # LIGO GWTC-1/2/3 ringdown (38 BBH events)
│   ├── ligo_catalog_full.csv          # Full GWTC catalog with physical parameters
│   ├── cross_platform_validation.csv  # β measurements across 5 platforms
│   └── lindblad_derivation_results.csv # Theoretical β-α values
├── scripts/
│   ├── lindblad_beta_derivation.py    # β = 1 - α/2 derivation
│   ├── ligo_ringdown_analysis.py      # LIGO envelope fitting
│   ├── alpha_measurement.py           # Independent α measurement
│   ├── validate_all.py                # Full validation suite
│   └── utils/
│       └── fitting_functions.py       # Common fitting utilities
└── README.md                          # This file
```

---

## Data Description

### IBM Quantum Hardware (`data/ibm_quantum_fits.csv`)

- **Source:** IBM Quantum Experience, backend `ibm_fez`
- **Date:** December 2025
- **Contents:** 42 configurations across 8 quantum state types (GHZ, W, Dicke, NOON, etc.)
- **Columns:** `state_type`, `n_qubits`, `replica`, `D_n`, `metric`, `backend`, `n_shots`, `execution_time`, `job_id`, `transpiled_depth`

### LIGO Gravitational Waves (`data/ligo_ringdown_38events.csv`)

- **Source:** GWTC-1, GWTC-2, GWTC-2.1, GWTC-3 catalogs via GWOSC
- **Contents:** 38 binary black hole merger events with ringdown analysis
- **Columns:** `event_name`, `beta`, `beta_err`, `tau`, `tau_err`, `r_squared`, `chi2_dof`, `delta_aic`, `snr`, `mass_1_msun`, `mass_2_msun`, `mass_final_msun`, `spin_final`, `distance_mpc`, `catalog`

### Cross-Platform Validation (`data/cross_platform_validation.csv`)

- **Contents:** Summary of β measurements across all 5 experimental platforms
- **Purpose:** Demonstrates universality of the β = 1 - α/2 relationship

---

## Requirements

```bash
pip install numpy scipy pandas matplotlib
```

---

## Usage

### Reproduce Lindblad β-α Derivation

```python
python scripts/lindblad_beta_derivation.py
```

### Analyze LIGO Ringdown Data

```python
python scripts/ligo_ringdown_analysis.py
```

### Run Full Validation Suite

```python
python scripts/validate_all.py
```

---

## Changelog

### v1.1.0 (2026-02-10)

**Revised and expanded edition.** 148 issues resolved (7 critical, 7 mathematical, 29 cross-references, 12 bibliographic). Major additions:

- **§2b Information Geometry:** Bures–Uhlmann manifold embedding, Fisher metric, geodesic curvature characterisation
- **§2c Biological Extension:** Λ-hierarchy at biological scales, diffusion–decoherence correspondence, protein folding predictions
- **§11 Gravitational Connections:** Significantly expanded — entropic duality, spatial decoherence operator, triple signature transition
- **§12 Anti-Quantum Irreversibility:** Formalised D × D̄ > 0 with spectral predictions
- **Appendix D:** Negative results documentation
- **Appendix E:** Complete LIGO GWTC-1/2/3 catalog (38 events)
- **Appendix F:** Formal information conservation proofs
- **Symbol ξ** introduced for β(n) scaling exponent (resolving α overloading)
- All cross-references converted to LaTeX `\ref{}` (29 edits)
- Full bibliography audit: 46 entries, all cited, DOIs added
- Convention footnote clarifying β_SDF = β_Lindblad/2

### v1.0.0 (2026-01-08)

- Initial release: 32 pages, 9 figures, 3 appendices

---

## Citation

If you use this data or code, please cite:

```bibtex
@software{becerra2026sdf,
  author = {Becerra Daly, Eloy Ren\'{e}},
  title = {Multi-State Validation and Anti-Quantum Extension of Universal
           Stretched-Exponential Decoherence: From {IBM} Quantum Hardware
           to Gravitational Scales},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.22238971},
  url = {https://doi.org/10.5281/zenodo.22238971},
  version = {v2.1.0}
}
```

---

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

- **Data:** CC BY 4.0
- **Code:** MIT License
- **Paper:** Author retains copyright; CC BY 4.0 for preprint

---

## Acknowledgements

- IBM Quantum for hardware access
- LIGO Scientific Collaboration and Virgo Collaboration for public gravitational wave data (GWOSC)
- The open-source scientific Python community

---

## Contact

For questions or collaborations:

**Eloy René Becerra Daly**  
Email: daly@icam.es
