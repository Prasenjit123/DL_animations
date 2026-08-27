# Interactive Polynomial Curve Fitting

<p align="center">
  <strong>A visual introduction to polynomial model complexity, underfitting, and potential overfitting.</strong>
</p>

<p align="center">
  Explore polynomial regression interactively by changing the model degree while keeping the noisy observations fixed.
</p>

---

## Overview

Polynomial regression is one of the simplest ways to introduce an important machine-learning idea:

> **Increasing model complexity gives a model more flexibility, but greater flexibility does not guarantee better generalization.**

This project turns that idea into an interactive visual experiment.

A known function is sampled over a fixed domain, Gaussian noise is added to the observations, and polynomial models of increasing degree are fitted to the **same dataset**. The learner can then move through the polynomial degrees and observe how the fitted curve changes.

The visualization combines:

- the underlying function,
- noisy observations,
- the fitted polynomial,
- training Mean Squared Error (MSE),
- and a simple model-complexity interpretation.

The result is designed primarily for **Machine Learning and Data Science teaching**, classroom demonstrations, lectures, and self-learning.

---

## Demo

The interactive visualization is provided as:

```text
polynomial-curve-fitting.html
```

Generate the HTML with:

```bash
python polynomial-curve-fitting.py
```

Then open the generated HTML file in a modern web browser.

The application runs in the browser after the HTML file has been generated. No additional assets, GIFs, screenshots, or supporting files are required for the application itself.

---

## What the visualization demonstrates

The experiment follows a simple progression:

```text
Simple polynomial
       ↓
Limited flexibility
       ↓
Underfitting
       ↓
More flexible polynomial
       ↓
Better representation of the observations
       ↓
Increasing complexity
       ↓
Highly flexible polynomial
       ↓
Potential fitting of noise
```

The important distinction is between **training fit** and **generalization**.

A polynomial can become increasingly good at fitting the observations used during training without necessarily becoming better at predicting unseen observations.

---

## Core experiment

For a selected underlying function `f(x)`, the application generates observations according to:

```text
yᵢ = f(xᵢ) + εᵢ
```

where `εᵢ` is Gaussian observation noise.

The model is a polynomial of degree `d`:

```text
ŷ(x) = a₀ + a₁x + a₂x² + ... + a_d xᵈ
```

The degree can be changed interactively.

The key experimental control is:

> **The noisy observations stay fixed while the polynomial degree changes.**

This makes the effect of model complexity much easier to see.

---

## Available underlying functions

The current application provides eight functions:

| Function | Expression |
|---|---|
| Sine | `sin(x)` |
| Cosine | `cos(x)` |
| Double-frequency sine | `sin(2x)` |
| Double-frequency cosine | `cos(2x)` |
| Mixed periodic function | `sin(x) + 0.3cos(3x)` |
| Quadratic | `x²` |
| Cubic | `x³` |
| Gaussian-shaped function | `exp(-x²)` |

The function selector is part of the interactive control panel, allowing the same model-complexity experiment to be repeated with different underlying relationships.

---

## Default settings

| Parameter | Default |
|---|---:|
| Underlying function | `sin(x)` |
| Number of observations | `50` |
| Noise level | `0.50` |
| Maximum polynomial degree | `30` |
| Random seed | `42` |
| Smooth plotting samples | `800` |
| Domain | `0` to `2π` |

These settings provide a moderately noisy dataset and enough polynomial degrees to make the change in model flexibility visually apparent.

---

## Interactive controls

### True Function

Select the function used to generate the synthetic observations.

Available choices include:

```text
sin(x)
cos(x)
sin(2x)
cos(2x)
sin(x) + 0.3cos(3x)
x²
x³
exp(-x²)
```

### Number of Data Points

Controls the number of observations in the generated dataset.

**Allowed range:** `5–500`

**Default:** `50`

More observations provide a denser sample of the underlying relationship.

### Noise Level

Controls the magnitude of the Gaussian observation noise.

**Allowed range:** `0–5`

**Step:** `0.05`

**Default:** `0.50`

A larger noise level produces more dispersed observations.

### Maximum Polynomial Degree

Controls the highest polynomial degree available to the slider and animation.

The application also limits the maximum degree so that it remains below the number of observations. This avoids requesting a polynomial with as many or more coefficients than available observations.

**Default:** `30`

### Random Seed

Controls deterministic dataset generation.

Using the same seed with the same settings reproduces the same noisy sample.

### Apply Settings

Reads the current control values, regenerates the dataset, resets the polynomial degree to `1`, and updates the visualization.

### New Sample

Regenerates the dataset using the currently displayed settings and seed. Because the random-number generator is deterministic, using the same seed reproduces the same sample; to obtain a different sample, use **Random Sample** or change the seed.

### Random Sample

Creates a new random seed and generates a new noisy sample.

### Play / Pause

Automatically progresses through the available polynomial degrees.

### Speed

Cycles through four animation speeds:

```text
1× → 2× → 3× → 4× → 1×
```

### Polynomial Degree

The slider directly controls the degree of the polynomial displayed in the graph.

---

## What is shown on the graph?

The graph contains three primary visual elements.

### 1. True function

The function that generated the observations.

For the default experiment:

```text
y = sin(x)
```

It acts as the ground-truth reference.

### 2. Noisy observations

The synthetic training observations are displayed as markers.

### 3. Polynomial fit

The selected polynomial is displayed as a smooth curve.

This gives the learner an immediate visual comparison:

```text
Underlying relationship
        vs.
Observed noisy data
        vs.
Learned polynomial model
```

---

## Training MSE

The application calculates the training Mean Squared Error for the selected polynomial:

```text
MSE = (1/N) × Σ (yᵢ − ŷᵢ)²
```

where:

- `yᵢ` is the observed training value,
- `ŷᵢ` is the polynomial prediction,
- `N` is the number of observations.

The training MSE is updated whenever the polynomial degree changes.

It is displayed in the information panel and in the plot title.

### Why show MSE?

The curve provides a qualitative view of the fit, while MSE provides a numerical measure of training error.

Together they help illustrate:

```text
Increasing degree
        ↓
Increasing flexibility
        ↓
Potentially lower training error
```

However:

> **Training MSE alone cannot determine whether a model generalizes well.**

A model can have a very small training error and still perform poorly on unseen data.

---

## Model-complexity interpretation

The visualization uses the following simple pedagogical labels:

| Polynomial degree | Interpretation |
|---:|---|
| `1–2` | **UNDERFITTING** |
| `3–5` | **REASONABLE FIT** |
| `6–9` | **INCREASING COMPLEXITY** |
| `10–30` | **POTENTIAL OVERFITTING** |

These labels are intended to support teaching rather than define universal statistical thresholds.

In particular:

> A high polynomial degree does not automatically mean that a model is overfit.

Formal overfitting is fundamentally associated with poor performance on unseen data. The current application reports training MSE only, so the term **potential overfitting** is deliberately used.

---

## Reproducibility

Reproducibility is built into the application.

The browser-side implementation uses a deterministic seeded random-number generator. When the dataset is regenerated, the generator is reset to the selected seed.

Therefore:

```text
Same function
+ Same number of observations
+ Same noise level
+ Same random seed
        ↓
Same generated dataset
```

This makes the visualization suitable for repeatable classroom demonstrations and instructional material.

Changing the seed creates a different noisy realization.

---

## How the fitting works

The polynomial coefficients are estimated using least squares.

For a degree-`d` polynomial:

```text
ŷ(x) = a₀ + a₁x + a₂x² + ... + a_d xᵈ
```

the browser constructs the corresponding least-squares system and solves for the coefficient vector.

Conceptually:

```text
Training observations
        ↓
Construct least-squares system
        ↓
Solve for coefficients
        ↓
Evaluate polynomial
        ↓
Calculate training MSE
        ↓
Render fitted curve
```

The HTML contains its own JavaScript implementation for the matrix operations and linear-system solution. The fitting therefore happens directly in the browser rather than through NumPy or scikit-learn.

---

## Why the implementation is browser-based

The generated HTML contains the complete interactive application.

The browser performs:

- synthetic data generation,
- function evaluation,
- polynomial fitting,
- prediction,
- MSE calculation,
- graph rendering,
- degree changes,
- animation.

This makes the project lightweight from the Python side.

The Python program acts as an HTML generator; the generated HTML is the actual interactive application.

---

## Project architecture

```text
polynomial-curve-fitting/
│
├── polynomial-curve-fitting.py
├── polynomial-curve-fitting.html
├── README.md
└── 
    
    
    
```

### `polynomial-curve-fitting.py`

Generates the HTML application.

The generator itself uses Python's standard library.

### `polynomial-curve-fitting.html`

Contains the complete browser-side visualization, including the HTML interface, CSS styling, JavaScript logic, numerical fitting routines, and Plotly visualization.

### `README.md`

Project documentation.

---

## Requirements

### Python

Python 3 is sufficient to generate the HTML.

No Python scientific-computing package is required by the generator.

You do **not** need:

- NumPy
- SciPy
- scikit-learn
- Matplotlib

### Web browser

A modern browser with JavaScript enabled is required.

The HTML application loads Plotly.js from a CDN, so internet access is normally required when opening the generated page.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Prasenjit123/DL_animations.git
```

Move into the project directory:

```bash
cd DL_animations/polynomial-curve-fitting
```

No additional Python package installation is required for the HTML generator.

---

## Usage

Run:

```bash
python polynomial-curve-fitting.py
```

The script generates:

```text
polynomial-curve-fitting.html
```

Open the generated HTML file in a modern browser.

Then:

1. Select an underlying function.
2. Choose the number of observations.
3. Set the noise level.
4. Choose the maximum polynomial degree.
5. Set the random seed.
6. Click **Apply Settings**.
7. Move the degree slider.
8. Observe the fitted curve and training MSE.
9. Use **Play** to animate the degree progression.
10. Change the speed if required.

---

## Recommended classroom demonstration

A useful teaching sequence is:

### Step 1 — Show the observations

Start with:

```text
Function: sin(x)
Points:   50
Noise:    0.50
Seed:     42
```

Ask students to identify the underlying pattern.

### Step 2 — Start with degree 1

Explain that a linear model has limited flexibility.

The model cannot reproduce the nonlinear shape particularly well.

### Step 3 — Increase the degree

Move gradually through degrees `2`, `3`, `4`, and `5`.

Discuss how additional polynomial terms increase model flexibility.

### Step 4 — Observe training MSE

Connect the visual fit with the numerical training error.

### Step 5 — Continue to higher degrees

Allow the model to become increasingly flexible.

Observe the resulting changes in the fitted curve.

### Step 6 — Discuss noise

Explain that a highly flexible model can respond not only to the underlying relationship but also to random fluctuations in the training observations.

### Step 7 — Introduce generalization

Ask:

> What happens if we give the model new observations that it has never seen?

This provides a natural transition to:

- train/test splits,
- validation data,
- test error,
- model selection,
- bias-variance trade-off,
- regularization.

---

## The most important conceptual distinction

This visualization should not be interpreted as:

```text
Low degree  = good
High degree = bad
```

The actual lesson is more general:

```text
Model complexity
        ↓
Model flexibility
        ↓
Ability to fit training data
        ↓
Potential effect on generalization
```

The goal is to help learners understand why machine-learning models must balance **fitting observed data** with **performing well on unseen data**.

---

## Numerical considerations

High-degree polynomial models can become numerically sensitive.

A polynomial basis contains terms such as:

```text
1, x, x², x³, ..., xᵈ
```

As `d` increases, these terms can have very different magnitudes. The resulting least-squares system can therefore become poorly conditioned.

This is an important numerical issue in polynomial regression.

The current implementation prioritizes transparency because the project is educational. For production numerical work, more stable approaches may be preferable, including:

- input scaling,
- orthogonal polynomial bases,
- QR-based least squares,
- SVD-based least squares,
- regularized regression.

---

## Limitations

The current project is intentionally focused on visualizing model complexity.

It does not currently provide:

- train/test splitting,
- validation-set evaluation,
- test MSE,
- cross-validation,
- automatic model selection,
- Ridge regression,
- Lasso regression,
- confidence intervals,
- statistical significance testing,
- formal bias-variance decomposition.

The model-complexity labels are also fixed pedagogical categories rather than data-driven statistical conclusions.

---

## Possible extensions

The current visualization provides a natural foundation for a broader interactive machine-learning teaching framework.

### Generalization

Add an independent test dataset and display training error alongside test error.

### Error curves

Plot:

```text
Polynomial degree
        vs.
Training MSE
        and
Test MSE
```

This would provide a direct visual demonstration of generalization.

### Bias-variance trade-off

Generate many independent datasets and visualize the relationship between:

```text
Bias
Variance
Irreducible noise
Expected prediction error
```

### Regularization

Add Ridge and Lasso regression to demonstrate how regularization constrains model complexity.

### Residual analysis

Allow learners to inspect:

```text
Residual = observed value − predicted value
```

### Coefficient visualization

Display polynomial coefficients as the degree increases.

### Numerical-stability comparison

Compare raw polynomial bases with scaled or orthogonal representations.

---

## Technical summary

| Component | Implementation |
|---|---|
| Data | Synthetic |
| Default function | `sin(x)` |
| Domain | `0` to `2π` |
| Noise | Gaussian |
| Default observations | `50` |
| Default maximum degree | `30` |
| Fitting | Least-squares polynomial |
| Solver | JavaScript matrix solver |
| Error metric | Training MSE |
| Visualization | Plotly.js |
| Interface | HTML + CSS + JavaScript |
| Randomness | Deterministic seeded generator |
| Python dependencies | Standard library |
| Runtime | Modern web browser |

---

## Repository philosophy

This project is part of a broader effort to create **visual, interactive explanations of Machine Learning and Deep Learning concepts**.

The emphasis is on turning mathematical or algorithmic ideas into demonstrations that learners can manipulate rather than merely observe.

For polynomial curve fitting, the essential learning progression is:

```text
Equation
   ↓
Data
   ↓
Model
   ↓
Increasing complexity
   ↓
Training fit
   ↓
Potential overfitting
   ↓
Generalization
```

The visualization is intended to be a teaching instrument first and a numerical-regression library second.

---

## License

No project-specific license is currently defined by the supplied project materials.

If this repository is intended for public redistribution, add an explicit `LICENSE` file and update this section accordingly.

---

## Author

**Dr. Prasenjit Dey**

Part of the **DL Animations** collection of interactive visualizations for Machine Learning and Deep Learning education.

---

## Acknowledgement of scope

This repository is deliberately small: the objective is not to provide a complete polynomial-regression framework, but to make one important machine-learning concept visually intuitive.

The central takeaway is:

```text
A model can become better at fitting the data it has seen
without necessarily becoming better at predicting data it has not seen.
```

That distinction is the foundation for understanding model selection, generalization, bias-variance trade-offs, and regularization.
