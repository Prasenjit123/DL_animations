# Interactive Polynomial Curve Fitting

An interactive, browser-based visualization for understanding **polynomial curve fitting, model complexity, underfitting, and potential overfitting**.

The project provides a controlled visual experiment in which the same noisy observations are fitted with polynomial models of increasing degree. A degree slider and animation make it easy to see how additional model flexibility changes the fitted curve and the training error.

> **Educational note:** The interpretation labels in this project are pedagogical. In particular, "potential overfitting" does not constitute a formal statistical test of overfitting. Formal overfitting assessment requires evaluation on unseen data.

---

## Overview

Polynomial regression is easy to define mathematically, but the effect of increasing model complexity is often easier to understand visually than from equations alone.

This project generates synthetic observations from a known function:

```text
y = f(x) + ε
```

where `ε` represents observation noise.

A polynomial model of degree `d` is then fitted to the observations:

```text
ŷ(x) = a₀ + a₁x + a₂x² + ... + a_d x^d
```

The degree is increased from a simple model to increasingly flexible models while keeping the observations fixed.

The resulting visualization lets learners directly compare:

```text
True function
      +
Noisy observations
      +
Polynomial approximation
      +
Training MSE
      +
Model-complexity interpretation
```

---

## Key idea

The central teaching message is:

```text
Increasing polynomial degree
            ↓
Increasing model flexibility
            ↓
Greater ability to fit training observations
            ↓
Increasingly complex fitted curves
            ↓
Possible fitting of noise
            ↓
Potential overfitting
```

The project is therefore particularly useful as a visual introduction to:

- Polynomial regression
- Model capacity
- Underfitting
- Overfitting
- Training error
- Generalization
- Bias-variance trade-off
- Regularization

---

## Features

- Interactive Plotly visualization
- Eight selectable underlying functions
- Configurable number of observations
- Configurable noise level
- Configurable maximum polynomial degree
- Reproducible seeded dataset generation
- Random sample generation
- Manual polynomial-degree slider
- Automatic degree animation
- Four animation speeds
- Training MSE calculation
- True-function reference curve
- Noisy training observations
- Browser-side polynomial fitting
- Self-contained JavaScript numerical solver
- Responsive Plotly graph
- No Python scientific-computing dependencies required

---

## Available functions

The current interface provides eight choices for the underlying function:

| Name | Function |
|---|---|
| Sine | `sin(x)` |
| Cosine | `cos(x)` |
| Double-frequency sine | `sin(2x)` |
| Double-frequency cosine | `cos(2x)` |
| Mixed periodic function | `sin(x) + 0.3cos(3x)` |
| Quadratic | `x²` |
| Cubic | `x³` |
| Gaussian-shaped function | `exp(-x²)` |

This allows the same model-complexity experiment to be demonstrated with functions having different shapes.

---

## Default configuration

The default interface settings are:

| Parameter | Default |
|---|---:|
| True function | `sin(x)` |
| Number of data points | `50` |
| Noise level | `0.50` |
| Maximum polynomial degree | `30` |
| Random seed | `42` |
| Smooth plotting samples | `800` |
| X range | `0` to `2π` |

The HTML implementation defines these defaults directly in the interface and JavaScript application state. fileciteturn0file0L261-L370

---

## How the data are generated

For each selected function, the application generates evenly spaced `x` values from `0` to `2π`.

For every observation:

```text
yᵢ = f(xᵢ) + εᵢ
```

The noise is Gaussian, generated from a standard normal random variable and multiplied by the selected noise level.

The application uses a deterministic seeded random-number generator. The generator is reset when the dataset is regenerated, so the same seed and settings reproduce the same sequence of noise values. fileciteturn0file0L602-L681

This gives the experiment an important property for teaching:

```text
Same settings + same seed
            ↓
Same dataset
            ↓
Same polynomial-fitting experiment
```

---

## Why a fixed dataset matters

When demonstrating model complexity, changing the dataset at the same time as changing the model makes the experiment difficult to interpret.

This project keeps the observations fixed while the polynomial degree changes.

Therefore, when moving from degree 1 to degree 10, for example, the primary experimental change is the **model's flexibility**, not the data.

That makes the visualization suitable for explaining the effect of model capacity in a controlled setting.

---

## Polynomial fitting

For a selected degree `d`, the browser constructs the least-squares polynomial system using the observations.

The coefficient vector is represented as:

```text
[a₀, a₁, a₂, ..., a_d]
```

The implementation constructs the required matrix and right-hand-side vector from powers of the observed `x` values, then solves the resulting linear system.

Conceptually:

```text
Training observations
        ↓
Construct least-squares system
        ↓
Solve for polynomial coefficients
        ↓
Evaluate polynomial
        ↓
Render fitted curve
```

The actual HTML contains an explicit matrix solver with pivot selection and row operations, followed by the polynomial-fitting and prediction routines. fileciteturn0file0L778-L1055

This browser-side implementation means the generated HTML can perform the fitting without requiring NumPy or another Python numerical package.

---

## Training Mean Squared Error

For the selected polynomial, the application calculates training Mean Squared Error:

```text
MSE = (1/N) Σ (yᵢ - ŷᵢ)²
```

where:

- `yᵢ` is the observed training value,
- `ŷᵢ` is the polynomial prediction,
- `N` is the number of training observations.

The MSE is recalculated whenever the polynomial degree changes and is displayed in the information panel and plot title. fileciteturn0file0L1462-L1606

### Why training MSE is useful here

Training MSE provides a numerical measure of how closely the current polynomial fits the observations used for training.

It complements the visual evidence:

```text
Visual observation:
"How closely does the curve follow the points?"

Numerical observation:
"How large is the training MSE?"
```

However:

> **Training MSE is not a measure of generalization.**

A model can achieve a very small training error and still perform poorly on unseen data.

---

## Underfitting and potential overfitting

The visualization uses a simple pedagogical interpretation based on polynomial degree:

| Degree | Interpretation |
|---:|---|
| 1–2 | **UNDERFITTING** |
| 3–5 | **REASONABLE FIT** |
| 6–9 | **INCREASING COMPLEXITY** |
| 10–30 | **POTENTIAL OVERFITTING** |

These categories are implemented directly in the HTML application. fileciteturn0file0L1497-L1534

### Important qualification

These thresholds should **not** be interpreted as universal statistical rules.

For example, a degree-10 polynomial is not automatically overfit. Whether a model is overfit depends on the data, noise, sample size, model class, and most importantly its performance on unseen observations.

The project intentionally uses the term:

```text
POTENTIAL OVERFITTING
```

rather than claiming that high degree alone proves overfitting.

---

## Interactive controls

### True Function

Select the function used to generate the synthetic observations.

### Number of Data Points

Controls how many observations are generated.

The current interface accepts values from 5 to 500. fileciteturn0file0L300-L316

### Noise Level

Controls the magnitude of the Gaussian observation noise.

The default is `0.50`, with the interface allowing values from `0` to `5` in increments of `0.05`. fileciteturn0file0L319-L336

### Maximum Polynomial Degree

Controls the upper limit of the degree slider.

The interface accepts degrees from 1 upward, and the application prevents the maximum degree from reaching the number of observations. This preserves the intended relationship between the number of observations and polynomial coefficients. fileciteturn0file0L1135-L1187

### Random Seed

Controls reproducible dataset generation.

### Apply Settings

Reads the current settings, regenerates the dataset, resets the degree to 1, and updates the visualization.

### New Sample

Regenerates the observations using the currently displayed settings and seed.

### Random Sample

Creates a new random seed and generates a new noisy dataset.

### Play / Pause

Automatically progresses through the available polynomial degrees.

### Speed

Cycles through four animation speeds:

```text
1× → 2× → 3× → 4× → 1×
```

The underlying timing values are defined in the HTML application. fileciteturn0file0L585-L600

### Polynomial Degree

The slider provides direct control over the currently displayed model degree.

---

## Visualization layout

The interface is organized into four major areas:

```text
┌──────────────────────────────────────────────────────────┐
│              Interactive Polynomial Curve Fitting        │
├──────────────────────────────────────────────────────────┤
│ Function | Points | Noise | Max Degree | Seed | Controls │
├──────────────────────────────────────────────────────────┤
│                    Polynomial Degree                     │
│             ─────────────●──────────────                 │
├──────────────────────────────────────────────────────────┤
│ Function | Points | Noise | Seed | Degree | MSE | Status │
├──────────────────────────────────────────────────────────┤
│                                                          │
│                     Plotly Visualization                 │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

The graph contains:

1. The true underlying function
2. Noisy observations
3. The currently selected polynomial fit

The application uses a dense set of plotting points to render smooth curves while retaining the original observations for fitting and MSE calculation. fileciteturn0file0L1286-L1347

---

## Animation

The Play control automatically increases the polynomial degree:

```text
Degree 1
   ↓
Degree 2
   ↓
Degree 3
   ↓
...
   ↓
Maximum degree
   ↓
Degree 1
   ↺
```

The application uses JavaScript timers to advance between degrees. When the maximum degree is reached, the animation returns to degree 1. fileciteturn0file0L1856-L1941

This is useful for lectures because the instructor can let the model-complexity progression play without manually moving the slider.

---

## Reproducibility

Reproducibility is built into the data-generation process.

The application maintains a seed state and uses a deterministic random-number generator. When `generateData()` is called, the generator is reset to the selected seed before the noisy observations are created. fileciteturn0file0L1211-L1279

Therefore, for a fixed configuration:

```text
Function
+ Number of points
+ Noise level
+ Random seed
        ↓
Deterministic noisy dataset
```

This is particularly useful when:

- preparing lecture demonstrations,
- recording instructional material,
- comparing two configurations,
- reproducing a classroom example,
- debugging the visualization.

---

## Numerical considerations

High-degree polynomial fitting can become numerically sensitive.

A polynomial basis contains powers such as:

```text
1, x, x², x³, ..., xᵈ
```

As the degree increases, these terms can differ substantially in magnitude and the associated least-squares system can become poorly conditioned.

The current application intentionally implements the fitting procedure explicitly in JavaScript for transparency and education.

This makes the project useful for explaining the mathematics, but it is not intended to replace numerically robust regression libraries in production applications.

For production numerical work, one would typically consider techniques such as:

- input scaling,
- orthogonal polynomial bases,
- QR-based least squares,
- SVD-based least squares,
- regularization,
- numerically stable polynomial representations.

---

## A subtle but important modeling lesson

One of the most important lessons in this visualization is that **fit quality and model quality are not the same thing**.

Suppose two models produce:

```text
Model A:
Higher training MSE
Simpler curve

Model B:
Lower training MSE
Highly complex curve
```

It is tempting to declare Model B better because its training error is lower.

That conclusion is not justified without evaluating how the models perform on data they did not see during training.

The correct conceptual distinction is:

```text
Training performance
        ≠
Generalization performance
```

This is the natural point at which to introduce:

- train/test splits,
- validation sets,
- cross-validation,
- bias-variance trade-off,
- regularization,
- model selection.

---

## Recommended teaching sequence

A practical classroom demonstration can follow this sequence.

### 1. Show the noisy observations

Start with the default `sin(x)` configuration.

Ask students to identify the underlying pattern.

### 2. Show degree 1

Explain that a linear model has limited representational capacity.

### 3. Increase the degree gradually

Move through degrees 2–5.

Discuss the increasing flexibility of the polynomial.

### 4. Introduce training MSE

Use the displayed MSE to connect visual fit with a quantitative training measure.

### 5. Continue toward high degrees

Observe how the fitted curve can become increasingly complex.

### 6. Discuss noise fitting

Explain that a sufficiently flexible model can begin responding to individual noisy observations.

### 7. Make the overfitting distinction

Emphasize that visual complexity and low training error alone do not formally prove overfitting.

### 8. Introduce unseen data

Use this visualization as the transition to test error and generalization.

---

## Project architecture

The project is intentionally lightweight:

```text
polynomial-curve-fitting/
│
├── polynomial-curve-fitting.py
├── polynomial-curve-fitting.html
└── README.md
```

### Python generator

`polynomial-curve-fitting.py` is the source generator.

Its job is to write the complete HTML application to disk.

The Python generator does not perform the browser-side regression itself.

### HTML application

`polynomial-curve-fitting.html` is the actual interactive application.

It contains:

- HTML interface
- CSS styling
- JavaScript state
- seeded random generation
- function evaluation
- polynomial fitting
- polynomial prediction
- training MSE calculation
- Plotly graph generation
- animation controls

The application initializes itself by generating the default dataset when the page loads. fileciteturn0file0L2022-L2029

---

## Why the application runs in the browser

The numerical and visualization logic is embedded in the generated HTML.

This has several advantages:

- No Python server is required.
- No backend is required.
- The interaction is immediate.
- The HTML can be copied and shared as a single application file.
- Students can inspect the JavaScript implementation.
- The polynomial fitting logic is visible rather than hidden behind a machine-learning library.

The main external dependency is Plotly.js, loaded by the HTML document. fileciteturn0file0L8-L14

---

## Requirements

### Python

Python 3 is sufficient for generating the HTML.

The generator uses Python's standard library and does not require NumPy, SciPy, Matplotlib, or scikit-learn.

### Browser

A modern JavaScript-enabled browser is required to run the generated visualization.

Because Plotly.js is loaded from a CDN, internet access is normally required when opening the generated HTML.

---

## Installation

Clone or download the repository and enter the project directory:

```bash
cd polynomial-curve-fitting
```

No Python package installation is required for the HTML generator.

---

## Usage

Run the generator:

```bash
python polynomial-curve-fitting.py
```

Then open:

```text
polynomial-curve-fitting.html
```

in a modern browser.

Once the page is open:

1. Select the underlying function.
2. Set the number of observations.
3. Set the noise level.
4. Set the maximum degree.
5. Choose a random seed.
6. Click **Apply Settings**.
7. Move the polynomial-degree slider.
8. Use **Play** to animate the progression.
9. Observe both the fitted curve and training MSE.

---

## Reproducing a specific experiment

For a reproducible classroom example, record the following five settings:

```text
Function:       sin(x)
Points:         50
Noise:          0.50
Seed:           42
Maximum degree: 30
```

Another person using the same settings will obtain the same deterministic noisy sample.

This makes the project suitable for:

- lecture notes,
- demonstrations,
- screenshots,
- instructional videos,
- practical exercises,
- student assignments.

---

## Suggested repository structure

For a source-oriented GitHub repository:

```text
polynomial-curve-fitting/
│
├── polynomial-curve-fitting.py    # HTML generator
├── polynomial-curve-fitting.html  # Generated interactive application
├── README.md                       # Documentation
└── LICENSE                         # Optional, but recommended
```

Generated artifacts can either be committed for convenient distribution or regenerated from the Python source.

---

## Limitations

This project is intentionally focused on a single educational concept.

It currently does not provide:

- Train/test split
- Validation-set evaluation
- Test MSE
- Cross-validation
- Ridge regression
- Lasso regression
- Automatic model selection
- Confidence intervals
- Statistical hypothesis testing
- Formal bias-variance decomposition

The fixed interpretation thresholds are also pedagogical rather than statistically universal.

These limitations are appropriate for the project's current purpose: **making polynomial model complexity visually intuitive**.

---

## Future extensions

The current architecture provides a natural foundation for a more comprehensive regression-learning environment.

### 1. Train/test split

Generate separate training and test observations and display both.

### 2. Training vs. test error

Plot:

```text
Polynomial degree
        ↓
Training MSE
        +
Test MSE
```

This would provide a direct visual demonstration of generalization and overfitting.

### 3. Bias-variance visualization

Repeat the experiment across many independently generated datasets and estimate:

```text
Bias
Variance
Irreducible noise
Expected prediction error
```

### 4. Regularization

Add Ridge and Lasso regression to demonstrate how regularization controls model complexity.

### 5. Residual analysis

Display residuals:

```text
eᵢ = yᵢ - ŷᵢ
```

and allow students to inspect systematic patterns.

### 6. Coefficient visualization

Plot polynomial coefficients as the degree increases.

This would provide a second perspective on model complexity.

### 7. Numerical-stability comparison

Compare the current explicit polynomial basis with scaled or orthogonal polynomial representations.

### 8. Model-selection experiment

Allow the learner to choose a model based on validation performance rather than training MSE alone.

---

## Educational value

This project is especially suitable for:

- Machine Learning courses
- Data Science courses
- Regression lectures
- Introductory statistical learning
- Model-complexity demonstrations
- Underfitting/overfitting discussions
- Bias-variance lectures
- Regularization introductions
- Interactive classroom teaching
- Self-learning

It is most effective when used as a visual bridge between the intuitive idea of "a more flexible model" and the formal statistical concept of generalization.

---

## Technical summary

| Component | Implementation |
|---|---|
| Data source | Synthetic |
| Default function | `sin(x)` |
| Domain | `0 ≤ x ≤ 2π` |
| Noise | Gaussian |
| Default observations | 50 |
| Polynomial degree | 1–30 by default |
| Fitting | Least-squares polynomial fit |
| Solver | Browser-side JavaScript matrix solver |
| Error metric | Training MSE |
| Visualization | Plotly.js |
| Interaction | HTML + JavaScript |
| Randomness | Deterministic seeded generator |
| Runtime | Modern web browser |
| Python dependencies | Standard library only |

---

## Design philosophy

The project intentionally favors **transparency over abstraction**.

Instead of hiding the numerical process behind a machine-learning framework, the HTML exposes the essential steps:

```text
Generate observations
        ↓
Evaluate the underlying function
        ↓
Add Gaussian noise
        ↓
Construct polynomial least-squares system
        ↓
Solve for coefficients
        ↓
Predict over a dense grid
        ↓
Calculate training MSE
        ↓
Render the result
```

This makes the project suitable not only for demonstration, but also for discussion of what polynomial regression is actually doing computationally.

---

## Frequently asked questions

### Does the Python program perform the polynomial fitting?

No. The Python program generates the HTML application. The fitting is performed by JavaScript in the browser.

### Do I need NumPy?

No. The generator does not require NumPy.

### Do I need scikit-learn?

No. The project does not depend on scikit-learn.

### Does the project use an external dataset?

No. The dataset is generated synthetically.

### Is the dataset reproducible?

Yes, when the same function, number of points, noise level, and seed are used.

### Does a high-degree polynomial always overfit?

No. High degree means greater flexibility, not automatic overfitting.

### Why does the application say "potential overfitting"?

Because the interface is designed for education. The label highlights a region of high model complexity without claiming that unseen-data performance has been measured.

### Is training MSE enough to select the best degree?

No. Training MSE alone cannot establish generalization performance.

### Can the project demonstrate test error?

Not in its current form. A train/test extension would be required.

---

## License

No project-specific license is currently established by the supplied project materials.

If this repository is intended for public distribution, add an explicit `LICENSE` file and replace this section with the corresponding license terms.

---

## Author

**Dr. Prasenjit Dey**

Part of an educational collection of interactive visualizations for Machine Learning and Deep Learning concepts.

---

## Final perspective

The purpose of this project is not to teach that "low degree is good" and "high degree is bad."

The deeper lesson is:

```text
Model complexity determines flexibility.
Flexibility affects training fit.
Training fit alone does not determine generalization.
```

In compact form:

```text
Good training fit
        ≠
Good generalization
```

The interactive visualization provides an intuitive starting point for understanding why machine-learning models must balance **fitting the observed data** with **learning patterns that remain useful on unseen data**.

That principle is the foundation for the subsequent study of validation, model selection, bias-variance trade-off, and regularization.
