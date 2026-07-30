# Riemannian Procrustes Analysis for EEG Transfer Learning

This repository is an in-progress Python implementation of **Riemannian Procrustes Analysis (RPA)** for transfer learning with EEG covariance matrices.

The project follows the paper:

> Pedro L. C. Rodrigues, Christian Jutten, and Marco Congedo,  
> **"Riemannian Procrustes Analysis: Transfer Learning for Brain-Computer Interfaces"**,  
> IEEE Transactions on Biomedical Engineering, 2018.

RPA is a geometry-aware transfer learning method designed to reduce the mismatch between EEG recordings from different subjects or sessions. It works with **symmetric positive definite (SPD) matrices**, usually covariance matrices extracted from EEG trials.

---

## Project Goal

EEG signals vary significantly across subjects and sessions. This makes it difficult to reuse a classifier trained on one subject for another subject.

This project aims to implement the RPA pipeline step by step:

1. Represent EEG trials as SPD covariance matrices.
2. Use Riemannian geometry to operate on these matrices correctly.
3. Recenter source and target covariance distributions to the identity matrix.
4. Stretch the target distribution to match source dispersion.
5. Rotate the target distribution using class-wise landmarks.
6. Classify transformed target trials using an MDM classifier.

---

## Current Status

This repository is currently under active development.

### Implemented

#### Core SPD Matrix Operations

- SPD validation
- Matrix square root
- Matrix inverse square root
- Matrix logarithm
- Matrix exponential
- SPD matrix power

#### Riemannian Geometry

- Affine-Invariant Riemannian Distance
- Logarithmic map
- Exponential map
- Riemannian / Karcher mean

#### Transfer Learning Helpers

- Center covariance matrices around the identity
- Recolor centered covariance matrices using a reference mean
- Mean alignment helper
- Dispersion calculation
- Stretching covariance matrices along geodesics from identity
- Class-wise Riemannian means
- Orthogonal rotation of covariance matrices

#### Testing

- Unit tests for SPD matrix operations
- Unit tests for Riemannian geometry functions
- Unit tests for transfer-learning alignment helpers

#### Classification

- Minimum Distance to Mean classifier

---

## Not Yet Implemented

The full RPA pipeline is not complete yet.

The following components are still in progress:

- RPA class following the full paper algorithm
- Estimation of the orthogonal rotation matrix $U$
- Integration with real EEG datasets
- Benchmarking against baseline methods
- Example notebooks and visualizations

---

## Repository Structure

```text
rpa-project/
│
├── rpa/
│   ├── __init__.py
│   │
│   ├── classification/
│   │   ├── __init__.py
│   │   └── mdm.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── spd_matrices.py
│   │   └── riemann_base.py
│   │
│   ├── transfer_learning/
│   │   ├── __init__.py
│   │   ├── alignment.py
│   │   └── rpa.py
│   │
│   └── utils/
│       └── __init__.py
│
├── tests/
│   ├── test_spd.py
│   ├── test_riemann.py
│   └── test_alignment.py
│
├── examples/
│
├── requirements.txt
└── README.md
```

---

## Main Modules

### `rpa/core/spd_matrices.py`

This file contains basic operations for SPD matrices.

Implemented functions include:

```python
is_symmetric()
is_positive_definite()
is_spd()
matrix_sqrt()
matrix_inv_sqrt()
matrix_log()
matrix_exp()
matrix_power()
```

These functions are the low-level mathematical building blocks used throughout the project.

---

### `rpa/core/riemann_base.py`

This file contains core Riemannian geometry operations on the SPD manifold.

Implemented functions include:

```python
riemannian_distance()
log_map()
exp_map()
riemannian_mean()
```

These functions allow covariance matrices to be compared, averaged, and mapped between the SPD manifold and tangent spaces.

---

### `rpa/transfer_learning/alignment.py`

This file contains transfer-learning helper transformations.

Implemented functions include:

```python
center_covariances()
recolor_covariances()
align_mean_to_reference()
dispersion()
stretch_covariances()
class_means()
rotate_covariances()
```

Important note:

`align_mean_to_reference()` is a generic mean-alignment helper. It is **not** the full RPA algorithm from the paper.

The paper-style RPA pipeline mainly uses:

```python
center_covariances()
dispersion()
stretch_covariances()
class_means()
rotate_covariances()
```

---

## Mathematical Background

### SPD Matrices

Covariance matrices are usually **symmetric positive definite**, or SPD.

A matrix $C$ is symmetric if:

$$
C = C^T
$$

It is positive definite if:

$$
x^T C x > 0
$$

for every non-zero vector $x$.

SPD matrices do not form an ordinary Euclidean vector space. They lie on a curved space called the **SPD manifold**. Because of this, operations such as distance, averaging, and interpolation should respect the geometry of the manifold.

Implemented in:

```text
rpa/core/spd_matrices.py
```

---

### Affine-Invariant Riemannian Distance

The distance between two SPD matrices $A$ and $B$ is computed as:

$$
d(A,B) = \left\| \log\left(A^{-1/2} B A^{-1/2}\right) \right\|_F
$$

This is used to measure distances between covariance matrices on the SPD manifold.

Implemented as:

```python
riemannian_distance()
```

in:

```text
rpa/core/riemann_base.py
```

---

### Logarithmic and Exponential Maps

The logarithmic map projects an SPD matrix from the curved manifold to a tangent space:

$$
\mathrm{Log}_P(X) = P^{1/2} \log\left(P^{-1/2} X P^{-1/2}\right) P^{1/2}
$$

The exponential map performs the inverse operation:

$$
\mathrm{Exp}_P(V) = P^{1/2} \exp\left(P^{-1/2} V P^{-1/2}\right) P^{1/2}
$$

These maps allow us to move between the SPD manifold and a flat tangent space.

Implemented as:

```python
log_map()
exp_map()
```

in:

```text
rpa/core/riemann_base.py
```

---

### Riemannian Mean

The Riemannian mean of covariance matrices $C_1, C_2, \dots, C_N$ is the matrix $G$ that minimizes:

$$
G = \arg\min_{P \succ 0} \sum_{i=1}^{N} d^2(P, C_i)
$$

This is the geometric center of a set of SPD matrices.

Implemented as:

```python
riemannian_mean()
```

The Riemannian mean is used for:

- centering covariance distributions
- computing class-wise landmarks
- estimating dispersion
- MDM classification

---

## Paper-Faithful RPA Pipeline

The RPA paper does **not** simply transform the source data into the target mean space.

Instead, the method follows this sequence:

```text
Source covariance matrices
        ↓
Recenter to identity

Target covariance matrices
        ↓
Recenter to identity
        ↓
Stretch to match source dispersion
        ↓
Rotate to match source class landmarks
```

Then classification is performed using an MDM classifier.

The intended final training and testing procedure is:

```text
Training data:
    recentered source data
    +
    rotated labeled target data

Testing data:
    rotated unlabeled target data
```

---

## Difference Between Mean Alignment and RPA

This repository includes a helper function:

```python
align_mean_to_reference()
```

This function aligns the Riemannian mean of one covariance distribution to another reference distribution.

However, this is **not full RPA**.

Full RPA includes:

1. recentering
2. stretching
3. rotation
4. MDM classification

So `align_mean_to_reference()` is kept as a useful helper, but the main RPA implementation will be built separately in:

```text
rpa/transfer_learning/rpa.py
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running Tests

Run the full test suite:

```bash
pytest -v
```

Run a specific test file:

```bash
pytest tests/test_alignment.py -v
```

---

## Example Usage

### Center Covariance Matrices

```python
import numpy as np

from rpa.transfer_learning.alignment import center_covariances

covariances = np.array([
    [
        [2.0, 0.2],
        [0.2, 3.0],
    ],
    [
        [4.0, 0.5],
        [0.5, 5.0],
    ],
])

centered, mean = center_covariances(covariances)

print(centered.shape)
print(mean)
```

---

### Compute Riemannian Mean

```python
import numpy as np

from rpa.core.riemann_base import riemannian_mean

covariances = np.array([
    [
        [2.0, 0.2],
        [0.2, 3.0],
    ],
    [
        [4.0, 0.5],
        [0.5, 5.0],
    ],
])

mean = riemannian_mean(covariances)

print(mean)
```

---

### Compute Dispersion

```python
import numpy as np

from rpa.transfer_learning.alignment import dispersion

covariances = np.array([
    [
        [2.0, 0.2],
        [0.2, 3.0],
    ],
    [
        [4.0, 0.5],
        [0.5, 5.0],
    ],
])

d = dispersion(covariances)

print(d)
```

---

### Stretch Covariance Matrices

```python
import numpy as np

from rpa.transfer_learning.alignment import stretch_covariances

covariances = np.array([
    [
        [2.0, 0.2],
        [0.2, 3.0],
    ],
    [
        [4.0, 0.5],
        [0.5, 5.0],
    ],
])

stretched = stretch_covariances(covariances, scale=0.5)

print(stretched)
```

---

### Compute Class-Wise Means

```python
import numpy as np

from rpa.transfer_learning.alignment import class_means

covariances = np.array([
    [
        [2.0, 0.2],
        [0.2, 3.0],
    ],
    [
        [3.0, 0.1],
        [0.1, 4.0],
    ],
    [
        [6.0, 0.3],
        [0.3, 7.0],
    ],
    [
        [7.0, 0.2],
        [0.2, 8.0],
    ],
])

labels = np.array([0, 0, 1, 1])

means = class_means(covariances, labels)

print(means.keys())
```

---

## Development Roadmap

### Phase 1 — Mathematical Foundation

- [x] SPD validation
- [x] Matrix square root
- [x] Matrix inverse square root
- [x] Matrix logarithm
- [x] Matrix exponential
- [x] Matrix power
- [x] Riemannian distance
- [x] Log map
- [x] Exp map
- [x] Riemannian mean

### Phase 2 — RPA Transformation Helpers

- [x] Centering
- [x] Recoloring
- [x] Mean alignment helper
- [x] Dispersion
- [x] Stretching
- [x] Class-wise means
- [x] Rotation helper

### Phase 3 — Classification

- [x] MDM classifier
- [x] MDM tests

### Phase 4 — Full RPA Pipeline

- [ ] RPA class
- [ ] Target labeled/unlabeled split handling
- [ ] Paper-style training and prediction pipeline
- [ ] Rotation matrix $U$ estimation

### Phase 5 — Experiments

- [ ] Synthetic SPD dataset
- [ ] EEG covariance extraction
- [ ] MOABB dataset support
- [ ] Accuracy benchmarking
- [ ] Visualization of recentering, stretching, and rotation

---

## Reference

### Paper

Pedro L. C. Rodrigues, Christian Jutten, and Marco Congedo,  
**"Riemannian Procrustes Analysis: Transfer Learning for Brain-Computer Interfaces"**,  
IEEE Transactions on Biomedical Engineering, 2018.

### Official Author Repository

```text
https://github.com/plcrodrigues/RPA
```

This repository is used as a conceptual reference while this project aims to build a cleaner, modular, and test-driven implementation.

---

## Disclaimer

This project is currently a learning-focused and research-oriented reimplementation.

It is not yet a complete reproduction of the paper results.

The current implementation provides the mathematical and geometric foundation for RPA, but the full algorithm and experimental benchmarks are still under development.
