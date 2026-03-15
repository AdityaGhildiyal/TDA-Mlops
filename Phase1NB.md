# Phase 1 — Appendix A: Mathematical Foundations of TDA

**File:** `notebooks/phase1_appendix_A.ipynb`  
**Thesis role:** Appendix A — complete worked derivation of persistent homology from first principles  
**Status:** Days 1–5 complete (21 cells)

---

## What This Notebook Does

This notebook proves, step by step, that topological features of data can be computed
algorithmically. It starts from the definition of a simplicial complex, builds boundary
matrices by hand, computes Betti numbers, constructs a full VR filtration, tracks
persistence pairs manually, applies Takens embedding to time series, and verifies
every result against Ripser — the same library used in the production pipeline.

By the end of Appendix A, any examiner can follow the full chain:

```
Point cloud → Simplicial complex → Boundary matrices → Betti numbers
→ Persistence diagram → Takens embedding → Anomaly detection → Stability theorem
```

---

## Cell-by-Cell Explanation

---

### Cell 1 — Imports and version check

```python
import numpy as np
import matplotlib.pyplot as plt
import ripser
from persim import plot_diagrams

print("ripser:", ripser.__version__)
```

**What it does:** Loads all required libraries and prints the Ripser version.  
**Why it matters:** Establishes the computational environment. Version pinning ensures
the notebook is reproducible — an examiner cloning the repo gets identical results.

**Libraries used:**

- `numpy` — matrix operations for boundary matrices
- `matplotlib` — persistence diagram and filtration plots
- `ripser` — fast Vietoris-Rips persistent homology (what the production library uses)
- `persim` — persistence image and diagram utilities

---

### Cell 2 — Boundary matrices ∂₁ and ∂₂

```python
d1 = np.array([[-1, 0, +1], [+1, -1, 0], [0, +1, -1]], dtype=float)
d2 = np.array([[+1], [+1], [+1]], dtype=float)
print("∂1 @ ∂2 =\n", d1 @ d2)
```

**What it does:** Constructs the boundary matrices for a filled triangle {A, B, C}
and verifies the fundamental lemma of homology.

**The simplicial complex:**

```
    C
   / \
  /   \
 A-----B
```

**∂₁ (edges → vertices):** Each column is an edge. Entry is +1 at the head, -1 at the tail.

```
     AB   BC   CA
A  [ -1    0   +1 ]
B  [ +1   -1    0 ]
C  [  0   +1   -1 ]
```

**∂₂ (triangle → edges):** The filled triangle ABC has all three edges as its boundary.

```
     ABC
AB [  +1 ]
BC [  +1 ]
CA [  +1 ]
```

**The fundamental lemma:** `∂₁ ∘ ∂₂ = 0`

This says "the boundary of a boundary is always empty." It is the cornerstone of
homology theory. Without this identity, homology groups could not be defined.

**Expected output:**

```
∂1 @ ∂2 =
 [[0.]
  [0.]
  [0.]]
```

---

### Cell 3 — Computing Betti numbers

```python
def betti(d_k, d_k1): ...

b0 = 3 - matrix_rank(d1)       # β₀ = 1
b1 = betti(d1, d2)             # β₁ = 0  (filled triangle)
b1_empty = betti(d1, d2_empty) # β₁ = 1  (empty triangle)
```

**What it does:** Computes β₀ and β₁ using the rank-nullity theorem, and demonstrates
the critical difference between a filled and an empty triangle.

**The formula:**

```
βₖ = dim(Ker ∂ₖ) − dim(Im ∂ₖ₊₁)
   = (n_cols − rank(∂ₖ)) − rank(∂ₖ₊₁)
```

**Betti number reference table:**

| Symbol | Counts               | Filled triangle | Empty triangle |
| ------ | -------------------- | --------------- | -------------- |
| β₀     | Connected components | 1               | 1              |
| β₁     | Independent loops    | 0               | 1              |
| β₂     | Enclosed voids       | 0               | 0              |

**Expected output:**

```
β₀ = 1
β₁ = 0

Empty triangle: β₁ = 1
```

---

### Cell 4 — Ripser on a noisy circle

```python
theta = np.linspace(0, 2*np.pi, 80)
X = np.column_stack([np.cos(theta), np.sin(theta)])
X += np.random.normal(0, 0.05, X.shape)
result = ripser.ripser(X, maxdim=1)
plot_diagrams(result['dgms'], show=True)
```

**What it does:** Generates 80 points on a noisy circle and runs Ripser. Plots the
persistence diagram. One point in H₁ far from the diagonal confirms β₁ = 1.

**Connection to anomaly detection:** In the production pipeline, anomalous data
produces persistence diagram points far from the diagonal.

---

### Cell 5 — BoundaryMatrix class

```python
class BoundaryMatrix:
    def __init__(self, matrix):
        self.M = np.array(matrix, dtype=int)   # store raw, NO % 2 here

    def reduce(self):
        M = (self.M % 2)   # mod 2 ONLY during reduction
        ...

    def betti(self, higher_boundary=None):
        ...
```

**What it does:** Encapsulates boundary matrix operations into a reusable class.
This is the same code that lives in `tda_detect/utils.py`.

**Critical design decision — Z/2Z arithmetic:**
The `reduce()` method works over Z/2Z (integers mod 2). The `__init__` stores
the raw matrix (with -1 entries intact) so that `matrix_rank` in `betti()` works
correctly. The `% 2` conversion happens only inside `reduce()`.

**⚠️ Known bug to avoid:** Never apply `% 2` in `__init__`. In Python/NumPy,
`(-1) % 2 == 1`, which would corrupt -1 entries to +1 and produce wrong ranks.

**The column reduction algorithm (Edelsbrunner & Harer Ch. 4):**

```
For each column j (left to right):
    Find its lowest 1 (= largest row index with a 1)
    If another column already owns that pivot:
        XOR column j with that column
        Repeat
    Else:
        Claim this pivot for column j
```

---

### Cell 6 — Verify BoundaryMatrix on the triangle

**Expected output:**

```
Filled triangle: b0=1, b1=0
Empty triangle:  b0=1, b1=1
```

---

### Cell 7 — Persistence pairs from reduced matrices

**Pairing rule:**

- A non-zero column j in reduced ∂ₖ₊₁ with pivot at row i → persistence pair (birth=i, death=j)
- A zero column in reduced ∂ₖ that is not killed → essential class (death = ∞)

**Expected output (empty triangle):**

```
Paired (birth simplex -> death simplex): {}
Essential (never-dying) classes: columns [2]
```

---

### Cell 8 — Two disconnected loops

**Expected topology:**

- β₀ = 2: two disconnected components (triangle ABC and triangle DEF)
- β₁ = 2: two independent loops (one per triangle)

**Expected output:**

```
Two empty triangles: b0=2, b1=2
```

---

### Cell 9 — VR filtration: distance matrix and edge birth times

**What it does:** Places 4 points as a unit square, computes all pairwise distances,
and labels every edge with the ε value at which it appears.

**Expected output:**

```
Edge birth times:
  edge (0, 1): ε = 1.000
  edge (0, 3): ε = 1.000
  edge (1, 2): ε = 1.000
  edge (2, 3): ε = 1.000
  edge (0, 2): ε = 1.414
  edge (1, 3): ε = 1.414
```

---

### Cell 10 — VR filtration: three snapshots

**Reading the three panels:**

| ε value | What appears                        | Topology                     |
| ------- | ----------------------------------- | ---------------------------- |
| 0.60    | 4 isolated vertices                 | β₀ = 4, β₁ = 0               |
| 1.05    | All 4 side edges                    | β₀ = 1, β₁ = 1 (square loop) |
| 1.50    | Both diagonals + 2 filled triangles | β₀ = 1, β₁ = 0 (loop filled) |

Loop persistence = 1.414 − 1.000 = 0.414.

---

### Cell 11 — Hand-computed H0 persistence pairs using Union-Find

**Expected output:**

```
Edge 0-1 at ε=1.000 → merges components → H0 pair: (birth=0.000, death=1.000)
Edge 0-3 at ε=1.000 → merges components → H0 pair: (birth=0.000, death=1.000)
Edge 1-2 at ε=1.000 → merges components → H0 pair: (birth=0.000, death=1.000)
Edge 2-3 at ε=1.000 → creates loop (H1 event)
...
1 essential H0 class: (birth=0.000, death=∞)
```

**Why only 3 H0 pairs for 4 vertices:** Exactly 3 merges are needed to go from 4
components to 1. The 4th component is the essential class.

---

### Cell 12 — Hand-drawn persistence diagram

Plots the persistence diagram from Cell 11 with no Ripser involvement. Points far
from the diagonal are long-lived features; points near the diagonal are noise.

---

### Cell 13 — Verify against Ripser

**Expected Ripser H0 output:**

```
birth=0.000, death=1.000  ← matches hand-computed pair (×3)
birth=0.000, death=∞      ← matches essential class
```

**Expected Ripser H1 output:**

```
birth=1.000, death=1.414  ← the square loop
```

Ripser output exactly matches the hand computation. ✓

---

## Part IV — Takens Embedding: Time Series → Topology

_Cells 14–21 cover the Day 4 + Day 5 content._

Takens' embedding theorem (1981) states that the topology of the attractor of a
dynamical system can be reconstructed from a single scalar time series:

```
x(t) = ( s(t), s(t+τ), s(t+2τ), ..., s(t+(d-1)τ) )
```

A periodic signal (normal heartbeat) traces a closed loop → β₁ = 1.
An anomalous signal breaks the loop → β₁ changes. This is the basis of
`tda_detect/drift.py`.

---

### Cell 14 — Takens embedding function

```python
def takens_embed(signal, dim, tau):
    N = len(signal)
    n_points = N - (dim - 1) * tau
    X = np.zeros((n_points, dim))
    for i in range(dim):
        X[:, i] = signal[i * tau : i * tau + n_points]
    return X
```

**Output shape:** `(N − (dim−1)·τ, dim)`. Each row is a sliding delay vector.

**Sanity check:** For signal `[1,2,3,4,5,6,7,8]`, dim=3, tau=1:

```
Row 0 = [1, 2, 3]
Row 1 = [2, 3, 4]
...
```

---

### Cell 15 — Normal signal: clean sine wave → clean loop

**Setup:**

- Signal: sin(t), 4 full cycles, 500 samples
- τ = 31 (≈ quarter period for best phase-space reconstruction)
- Embedding dimension: 3

**Expected result:** 3D plot shows a clean closed loop → β₁ = 1.

**Why τ = quarter period:** For a sine wave, quarter-period delay gives coordinates
(sin, cos, −sin) which trace a circle. This is the theoretically optimal delay.

---

### Cell 16 — Compute H₁ on normal signal

**Expected output:**

- One dominant H₁ point far from diagonal (the loop)
- Many near-diagonal H₁ points (sampling noise from finite points on the loop)

```
Normal signal — Ripser H1 diagram:
  birth=..., death=..., persistence=~1.2  ← the dominant loop feature
```

The persistence value of this dominant feature is the _topological fingerprint_
of normal operation. Stored and compared against in drift detection.

---

### Cell 17 — Anomalous signal: inject spikes

**Anomaly type:** Sharp up-down spike pairs (amplitude 3.5×) at t = 100, 200, 310.

**Simulates:**

- Cardiac arrhythmia in ECG
- Transducer fault in manufacturing sensor
- Traffic burst in network monitoring

Two-panel plot shows normal (blue) vs anomalous (crimson) signals.

---

### Cell 18 — Phase space comparison: normal vs anomalous

Side-by-side 3D Takens embedding plots.

- **Left (normal):** Clean closed loop — β₁ = 1 visible to the naked eye
- **Right (anomalous):** Loop broken by spike arms — topology disturbed

This is the most visually striking result in Appendix A and the plot most likely
to be remembered by examiners.

---

### Cell 19 — Compare persistence diagrams

Side-by-side persistence diagrams confirm the visual observation:

- **Normal:** One dominant H₁ feature (the loop), all others near diagonal
- **Anomalous:** Dominant H₁ feature reduced or split; new features at unusual positions

```
Normal — max H1 persistence  : ~1.2
Anomaly — max H1 persistence : smaller (loop broken)
```

This quantitative change is what the model in `tda_detect/models.py` learns to score.

---

### Cell 20 — Wasserstein distance

```python
from persim import wasserstein
d_wasserstein = wasserstein(result_normal['dgms'][1], result_anomaly['dgms'][1])
```

**Wasserstein distance** = optimal transport cost between two persistence diagrams.

In `tda_detect/drift.py`:

- Compute W between a reference window and each new sliding window
- If W > threshold → drift detected → trigger model retraining
- This is the **Topological Wasserstein Drift Detector** — the novel MLOps contribution

**Expected output:**

```
Wasserstein distance between normal and anomalous H1 diagrams:
  W = [non-trivial positive value confirming diagrams differ]
```

---

### Cell 21 — Stability theorem demo (mathematical capstone)

**The stability theorem (Cohen-Steiner et al., 2007):**

```
W∞(D(f), D(g)) ≤ ‖f − g‖∞
```

Small perturbations to the input produce small changes in the persistence diagram.
This guarantees the anomaly detector does not fire on random noise.

**Demo:** We perturb the normal signal with increasing noise levels σ = 0.00, 0.02,
0.05, 0.10, 0.20, 0.40 and measure the Wasserstein distance from the clean reference
diagram.

**Expected output:**

```
noise σ = 0.00  |  L∞ ≈ 0.000  |  W(dgm) = 0.0000
noise σ = 0.02  |  L∞ ≈ 0.060  |  W(dgm) = small
noise σ = 0.05  |  L∞ ≈ 0.150  |  W(dgm) = moderate
noise σ = 0.40  |  L∞ ≈ 1.200  |  W(dgm) = large
```

The plot shows W(D₁, D₂) staying below the stability bound line, confirming the
theorem numerically.

**Final print:**

```
=== Appendix A Complete ===
Full chain demonstrated:
  Simplicial complex → Boundary matrices → Betti numbers
  → VR filtration → Persistence pairs → Takens embedding
  → Anomaly detection → Wasserstein drift detection → Stability theorem
```

---

## Running the Notebook

```bash
# Activate environment
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

# Launch Jupyter
jupyter notebook notebooks/phase1_appendix_A.ipynb
```

Run Cell 1 first (imports), then all subsequent cells in order.

---

## Connection to the Production Pipeline

| Notebook concept                  | Production code                                     | Phase   |
| --------------------------------- | --------------------------------------------------- | ------- |
| `BoundaryMatrix` class            | `tda_detect/utils.py`                               | Phase 1 |
| VR filtration at increasing ε     | `tda_detect/features.py` — `TDAFeatureExtractor`    | Phase 2 |
| Ripser on point cloud             | `tda_detect/features.py` — `TDAFeatureExtractor`    | Phase 2 |
| Persistence diagram               | Input to `persim.PersistenceImager`                 | Phase 2 |
| Anomaly = point far from diagonal | Anomaly score in `tda_detect/models.py`             | Phase 3 |
| Takens embedding                  | `tda_detect/features.py` — `takens_embed()`         | Phase 2 |
| Wasserstein distance              | `tda_detect/drift.py` — Drift Detector              | Phase 4 |
| Stability theorem                 | Mathematical guarantee for Phase 4 threshold tuning | Phase 4 |

---

## Tests

The `BoundaryMatrix` class from Cell 5 is tested in `tests/test_features.py`:

```bash
python -m pytest tests/test_features.py -v
# 3 passed: filled triangle, empty triangle, two components
```

---

## Commit History

```
feat: Day 1 - folder structure, environment, cells 1-4
feat: Day 2 - BoundaryMatrix class, cells 5-8, 3 tests passing
feat: Day 3 - VR filtration, union-find, hand-computed persistence diagram
feat: Day 4 - Takens embedding, normal vs anomalous, Wasserstein distance
docs: Day 5 - markdown polish, stability theorem cell, updated PHASE1_NOTEBOOK.md
```
