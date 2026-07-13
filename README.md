# Riemannian Procrustes Analysis (RPA) Reimplementation

This repository implements the Riemannian Procrustes Analysis (RPA) transfer learning framework for Brain-Computer Interfaces (BCIs). RPA aligns covariance matrices from different subjects or sessions to reduce variability and improve classification performance.

## Features Implemented So Far

- **SPD Matrices Operations** (`rpa.core.spd_matrices`):
  - Verification of symmetric positive-definiteness (`is_spd`).
  - Matrix operations: logarithm (`matrix_log`), exponential (`matrix_exp`), square root (`matrix_sqrt`), and inverse square root (`matrix_inv_sqrt`).
- **Riemannian Distance** (`rpa.core.riemann_base`):
  - Affine-Invariant Riemannian Distance (AIRM) calculation (`riemann_distance`).
- **Test Suite**:
  - Full test coverage for SPD operations and Riemannian distance calculations using `pytest`.

## Project Structure

```text
├── rpa/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── riemann_base.py      # Riemannian distance base
│   │   └── spd_matrices.py      # SPD matrix helper utilities
│   ├── transfer_learning/
│   │   ├── __init__.py
│   │   └── rpa.py               # RPA core alignment methods (placeholder)
│   └── utils/
│       └── __init__.py
├── tests/
│   ├── test_riemann.py          # Test suite for Riemannian calculations
│   └── test_spd.py              # Test suite for SPD matrix functions
├── requirements.txt             # Project dependencies
├── pytest.ini                   # Pytest path configurations
└── README.md                    # Project documentation
```

## Getting Started

### 1. Installation

First, install the required dependencies (such as `numpy` and `scipy`):
```bash
pip install -r requirements.txt
```

### 2. Running Tests

To run the unit tests, simply execute:
```bash
pytest
```
All tests will be discovered automatically and run.
