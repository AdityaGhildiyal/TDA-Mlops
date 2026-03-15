# Phase 1 — Appendix A: Mathematical Foundations of TDA

**File:** `notebooks/phase1_appendix_A.ipynb`  
**Thesis role:** Appendix A — complete worked derivation of persistent homology from first principles  
**Status:** Days 1–3 complete

---

## What This Notebook Does

This notebook proves, step by step, that topological features of data can be computed
algorithmically. It starts from the definition of a simplicial complex, builds boundary
matrices by hand, computes Betti numbers, constructs a full VR filtration, tracks
persistence pairs manually, and verifies every result against Ripser — the same library
used in the production pipeline.

By the end of Appendix A, any examiner should be able to follow the full chain:

```
Point cloud → Simplicial complex → Boundary matrices → Betti numbers → Persistence diagram
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
homology theory. The output `[[0], [0], [0]]` confirms it holds for our triangle.
Without this identity, homology groups could not be defined.

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

**Key insight — the most common examiner question:**  
A filled triangle has β₁ = 0 because the loop AB → BC → CA is the boundary of the
filled face ABC. In homology terms, the cycle is also a boundary — it is "trivial."  
An empty triangle has β₁ = 1 because the loop exists but has no filler to make it
trivial. This is a genuine topological hole.

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

**What it does:** Generates 80 points on a noisy circle and runs Ripser to compute
persistent homology up to dimension 1. Plots the persistence diagram.

**Why a noisy circle:** A perfect circle is topologically a loop — it should have
β₁ = 1. Adding Gaussian noise (σ = 0.05) tests the stability theorem: small
perturbations should not destroy the topological signal.

**Reading the persistence diagram:**

- Each point (b, d) represents a topological feature born at radius b and dying at radius d
- Points near the diagonal: short-lived features = noise
- Points far from the diagonal: long-lived features = genuine topology
- One point in H₁ far from the diagonal = the loop of the circle

**Connection to anomaly detection:** In the production pipeline, anomalous data
produces persistence diagram points far from the diagonal. This cell gives the
visual intuition for why.

---

### Cell 5 — BoundaryMatrix class

```python
class BoundaryMatrix:
    def __init__(self, matrix):
        self.M = np.array(matrix, dtype=int)   # store raw

    def reduce(self):
        M = (self.M % 2)   # convert to Z/2Z for reduction
        ...

    def betti(self, higher_boundary=None):
        ...
```

**What it does:** Encapsulates boundary matrix operations into a reusable class.
This is the same code that lives in `tda_detect/utils.py` for use by the rest
of the pipeline.

**Critical design decision — Z/2Z arithmetic:**  
The `reduce()` method works over Z/2Z (integers mod 2). This means:

- All entries become 0 or 1
- Addition becomes XOR
- Signs disappear

Why? Working over Z/2Z eliminates the sign bookkeeping from boundary operators,
making the algorithm simpler and exactly matching how Ripser computes internally.
The `__init__` stores the raw matrix (with -1 entries intact) so that `matrix_rank`
in `betti()` works correctly over the integers. The `% 2` conversion happens only
inside `reduce()`.

**The column reduction algorithm (Edelsbrunner & Harer Ch. 4):**

```
For each column j (left to right):
    Find its lowest 1 (= largest row index with a 1)
    If another column already owns that pivot:
        XOR column j with that column (eliminates the shared pivot)
        Repeat
    Else:
        Claim this pivot for column j
```

This is the standard persistence algorithm. After reduction, paired columns
(those that got reduced to zero) correspond to topological features that die,
and unpaired non-zero columns correspond to features that are born and never die
(essential classes).

---

### Cell 6 — Verify BoundaryMatrix on the triangle

```python
d1 = BoundaryMatrix([[-1, 0, +1], [+1, -1, 0], [0, +1, -1]])
d2_filled = BoundaryMatrix([[+1], [+1], [+1]])
d2_empty  = BoundaryMatrix(np.zeros((3, 0), dtype=int))
```

**What it does:** Rebuilds the triangle example using the new class and verifies
it produces identical results to Cell 3. Also reduces ∂₁ and prints the pivot
structure.

**Reading the reduced matrix output:**

```
Reduced ∂1:
 [[1 0 0]
  [1 1 0]
  [0 1 0]]
Pivots: {1: 0, 2: 1}
```

Column 0 has pivot at row 1 (edge AB kills vertex B's component).  
Column 1 has pivot at row 2 (edge BC kills vertex C's component).  
Column 2 reduces to zero — it is in the kernel (the loop CA is a cycle).  
This zero column in ∂₁ with no corresponding pivot in ∂₂ is the β₁ = 1 loop
of the empty triangle.

**Expected output:**

```
Filled triangle: b0=1, b1=0
Empty triangle:  b0=1, b1=1
```

---

### Cell 7 — Persistence pairs from reduced matrices

```python
def persistence_pairs(d_k_reduced, d_k1_reduced):
    pivots = d_k1_reduced.pivot_col
    ...
    # Essential = zero columns in reduced d_k not killed by d_k1
```

**What it does:** Extracts (birth, death) pairs from the reduced boundary matrices.
This is the direct output of the persistence algorithm before it gets packaged
into a diagram.

**Pairing rule:**

- A non-zero column j in reduced ∂ₖ₊₁ with pivot at row i means: simplex i is born,
  simplex j kills it → persistence pair (birth=i, death=j)
- A zero column in reduced ∂ₖ that is not killed by anything → essential class (death = ∞)

**For the empty triangle:**

- ∂₂ is empty (no 2-simplices), so no pairs
- Column 2 of reduced ∂₁ is zero and unkilled → essential H₁ class
- This is the loop, which lives forever

**Expected output:**

```
=== Empty triangle persistence ===
Paired (birth simplex -> death simplex): {}
Essential (never-dying) classes: columns [2]
```

---

### Cell 8 — Two disconnected loops

```python
d1_two = BoundaryMatrix([
    [-1, 0, +1, 0, 0, 0],  # A
    [+1,-1,  0, 0, 0, 0],  # B
    [ 0,+1, -1, 0, 0, 0],  # C
    [ 0, 0,  0,-1, 0,+1],  # D
    [ 0, 0,  0,+1,-1, 0],  # E
    [ 0, 0,  0, 0,+1,-1],  # F
])
```

**What it does:** Tests the multi-component, multi-loop case — two separate empty
triangles with no shared vertices.

**Expected topology:**

- β₀ = 2: two disconnected components (triangle ABC and triangle DEF)
- β₁ = 2: two independent loops (one per triangle)

**Why this matters:** This example tests that the boundary matrix framework handles
disconnected complexes correctly. The block-diagonal structure of the matrix
(top-left = triangle ABC, bottom-right = triangle DEF) directly reflects the
topological disconnection.

**Expected output:**

```
Two empty triangles: b0=2, b1=2
```

---

### Cell 9 — VR filtration: distance matrix and edge birth times

```python
points = np.array([[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]])
D = cdist(points, points)
edge_births = {(i,j): D[i,j] for i,j in edges}
```

**What it does:** Places 4 points as a unit square, computes all pairwise distances,
and labels every possible edge with the ε value at which it appears in the
Vietoris-Rips filtration.

**Key concept — the VR filtration:** As ε increases from 0 to ∞, the VR complex
grows by adding edges (and eventually triangles) between points that fall within
distance ε. This cell makes that process completely explicit:

- Side edges (length 1.000) appear first
- Diagonal edges (length 1.414) appear later

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

**What it does:** Draws the simplicial complex at ε = 0.60, ε = 1.05, and ε = 1.50,
showing how the complex grows as ε increases.

**Reading the three panels:**

| ε value | What appears                        | Topology                     |
| ------- | ----------------------------------- | ---------------------------- |
| 0.60    | 4 isolated vertices                 | β₀ = 4, β₁ = 0               |
| 1.05    | All 4 side edges                    | β₀ = 1, β₁ = 1 (square loop) |
| 1.50    | Both diagonals + 2 filled triangles | β₀ = 1, β₁ = 0 (loop filled) |

**The key observation:** At ε = 1.05, a loop appears (β₁ = 1). At ε = 1.414, the
diagonals fill the loop with two triangles (β₁ drops back to 0). The persistence
of this loop is its lifetime: 1.414 − 1.000 = 0.414.

---

### Cell 11 — Hand-computed H0 persistence pairs using Union-Find

```python
parent = list(range(4))

def find(x): ...
def union(x, y): ...

for (i, j), birth in sorted_edges:
    merged = union(i, j)
    if merged:
        h0_pairs.append((0.0, birth))
```

**What it does:** Manually tracks how connected components merge as edges appear,
using a Union-Find (disjoint-set) data structure. Each merge event produces one
H0 persistence pair.

**The Union-Find algorithm:**

- Each vertex starts as its own component
- When an edge appears, if its two endpoints are in different components, merge them
  → one component dies: H0 pair (birth=0, death=ε)
- If the endpoints are already connected, the edge creates a loop (H1 event)
- The last surviving component never dies → essential H0 class (death = ∞)

**Expected output:**

```
Edge 0-1 at ε=1.000 → merges components → H0 pair: (birth=0.000, death=1.000)
Edge 0-3 at ε=1.000 → merges components → H0 pair: (birth=0.000, death=1.000)
Edge 1-2 at ε=1.000 → merges components → H0 pair: (birth=0.000, death=1.000)
Edge 2-3 at ε=1.000 → creates loop (H1 event)
Edge 0-2 at ε=1.414 → creates loop (H1 event)
Edge 1-3 at ε=1.414 → creates loop (H1 event)

1 essential H0 class: (birth=0.000, death=∞)
```

**Why only 3 H0 pairs for 4 vertices:** With 4 vertices (4 components), exactly 3
merges are needed to reach 1 component. The 4th component is the essential class.

---

### Cell 12 — Hand-drawn persistence diagram

**What it does:** Plots the persistence diagram built entirely from the Union-Find
computation in Cell 11 — no Ripser involved.

**Reading the diagram:**

- Blue dots at (0.000, 1.000): three components that were born at ε=0 and died at ε=1
- Blue triangle + arrow: the essential H0 class that was born at ε=0 and never dies
- The diagonal (dashed line): features on the diagonal have zero lifetime = noise

**The key insight printed at the bottom:**

```
Points far from diagonal = long-lived features = topologically significant
Points near diagonal     = short-lived features = noise
```

This is the mathematical foundation for anomaly detection: anomalous data produces
persistence points far from the diagonal.

---

### Cell 13 — Verify against Ripser

```python
result = ripser.ripser(points, maxdim=1)
plot_diagrams(result['dgms'], show=True)
```

**What it does:** Runs the same 4-point square through Ripser and compares the
output against the hand-computed results from Cells 11–12.

**Expected Ripser H0 output:**

```
birth=0.000, death=1.000   ← matches hand-computed pair
birth=0.000, death=1.000   ← matches hand-computed pair
birth=0.000, death=1.000   ← matches hand-computed pair
birth=0.000, death=∞       ← matches essential class
```

**Expected Ripser H1 output:**

```
birth=1.000, death=1.414   ← the square loop (born when 4 side edges appear, dies when diagonal fills it)
```

**Why this matters:** The Ripser output exactly matches the hand computation. This
confirms that our manual Union-Find approach is not just an approximation — it is
the same algorithm Ripser implements at high speed. The H1 pair (1.000, 1.414) was
not computed by hand (that requires H1 reduction, covered in Cell 7 for the triangle
case), but Ripser finds it automatically.

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

Every concept demonstrated here maps directly to production code:

| Notebook concept                      | Production code                                  | Phase   |
| ------------------------------------- | ------------------------------------------------ | ------- |
| `BoundaryMatrix` class                | `tda_detect/utils.py`                            | Phase 1 |
| VR filtration at increasing ε         | `tda_detect/features.py` — `TDAFeatureExtractor` | Phase 2 |
| Ripser on point cloud                 | `tda_detect/features.py` — `TDAFeatureExtractor` | Phase 2 |
| Persistence diagram                   | Input to `persim.PersistenceImager`              | Phase 2 |
| Anomaly = point far from diagonal     | Anomaly score in `tda_detect/models.py`          | Phase 3 |
| Wasserstein distance between diagrams | `tda_detect/drift.py`                            | Phase 4 |

---

## Tests

The `BoundaryMatrix` class from Cell 5 is tested in `tests/test_features.py`:

```bash
python -m pytest tests/test_features.py -v
# 3 passed: filled triangle, empty triangle, two components
```
