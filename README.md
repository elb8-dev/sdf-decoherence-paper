# SDF: Universal Stretched-Exponential Decoherence

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18186310.svg)](https://doi.org/10.5281/zenodo.18186310)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Paper

**Title:** SDF: Universal Stretched-Exponential Decoherence from Quantum to Gravitational Scales

**Author:** Eloy R. Becerra Daly (Independent Researcher, Spain)

**Contact:** daly@icam.es

---

## Abstract

We present a universal stretched-exponential decoherence law $D(\gamma) = L[1 - \exp(-(\gamma/\tau)^\beta)]$ validated across five experimental platforms spanning quantum to gravitational scales. The key parameter $\beta$ characterizes deviations from pure exponential decay ($\beta = 1$), with the theoretical relation $\beta = 1 - \alpha/2$ derived rigorously from Lindblad master equations.

### Key Results

| Platform | β | σ(β) | Significance |
|----------|---|------|--------------|
| **IBM Quantum** | 0.852 | ±0.028 | 5.2σ from β=1 |
| **LIGO GWTC-3** | 0.931 | ±0.005 | 13.8σ from β=1 |
| Ion traps | 0.94 | ±0.02 | — |
| Quantinuum | 0.97 | ±0.01 | — |
| Simulation | 1.00 | ±0.001 | Theoretical limit |

The consistent observation of $\beta < 1$ across independent platforms suggests a universal mechanism connecting quantum decoherence to emergent gravitational phenomena.

---

## Repository Contents

```
sdf-decoherence-paper/
├── paper/
│   └── SDF_Universal_Decoherence_Becerra_Daly.pdf   # Full manuscript
├── data/
│   ├── ibm_quantum_fits.csv           # IBM Quantum hardware (42 configurations)
│   ├── ligo_ringdown_38events.csv     # LIGO GWTC-3 ringdown (38 BBH events)
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

## Citation

If you use this data or code, please cite:

```bibtex
@software{becerra2026sdf,
  author = {Becerra Daly, Eloy René},
  title = {SDF: Universal Stretched-Exponential Decoherence from Quantum to Gravitational Scales},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.18186310},
  url = {https://doi.org/10.5281/zenodo.18186310},
  version = {v1.0.0}
}
```

---

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

- **Data:** CC BY 4.0
- **Code:** MIT License
- **Paper:** Author retains copyright; CC BY 4.0 for preprint

---

## Acknowledgments

- IBM Quantum for hardware access
- LIGO Scientific Collaboration and Virgo Collaboration for public gravitational wave data (GWOSC)
- The open-source scientific Python community

---

## Contact

For questions or collaborations:

**Eloy R. Becerra Daly**  
Email: daly@icam.es
