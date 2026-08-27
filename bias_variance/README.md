# Bias–Variance Visualization with a Sine Function

<p align="center">
  <strong>An interactive visualization of model bias, variance, underfitting, and overfitting.</strong>
</p>

<p align="center">
  Generate repeated noisy samples from a sine function and compare a low-degree polynomial with a higher-degree polynomial across those samples.
</p>

---

## Overview

The bias–variance trade-off is easier to understand when the effect of changing the training sample can be seen directly.

This project creates repeated noisy datasets from a known sine function and fits two polynomial models to each experiment:

```text
Low-degree polynomial
        ↓
High-bias / underfitting behavior

High-degree polynomial
        ↓
High-variance / overfitting behavior
```

The visualization keeps the fitted models from previous experiments on the graphs while replacing the displayed sample points with the current dataset.

After several experiments, the learner can compare:

- how similar the low-degree fitted models remain across datasets,
- how much the high-degree fitted models change across datasets,
- how model complexity affects sensitivity to the training sample.

The project is designed primarily as an **educational visualization**, not as a statistical estimator of bias and variance.

---

## Important clarification

The labels **High Bias / Underfitting** and **High Variance / Overfitting** describe the intended pedagogical interpretation of the two model settings.

The application does **not** explicitly calculate:

- expected prediction error,
- statistical bias,
- statistical variance,
- irreducible noise,
- test error,
- cross-validation error,
- or a formal bias–variance decomposition.

Instead, it demonstrates the behavior associated with these concepts by fitting models to multiple independently generated training samples.

This distinction is important:

> **The visualization demonstrates the intuition behind bias and variance; it does not numerically estimate the full bias–variance decomposition.**

---

## Core experiment

The true function is fixed as:

```text
f(x) = sin(x)
```

For each experiment, the application generates `n` input values uniformly over:

```text
0 ≤ x ≤ 6
```

and creates noisy observations according to:

```text
yᵢ = sin(xᵢ) + εᵢ
```

where the noise is generated from a zero-mean Gaussian distribution:

```text
εᵢ ~ N(0, σ²)
```

The value of `σ` is controlled by the **Noise Level** setting.

The generated sample is then used to fit:

1. a lower-degree polynomial for the bias/underfitting demonstration;
2. a higher-degree polynomial for the variance/overfitting demonstration.

---

## Why repeated datasets matter

A single training dataset cannot visually demonstrate variance very well.

Variance is about how much a learned model changes when the training sample changes.

This project therefore repeats the experiment:

```text
Dataset 1 → Model 1
Dataset 2 → Model 2
Dataset 3 → Model 3
...
Dataset N → Model N
```

The fitted models from previous experiments remain visible.

This creates two visual collections:

```text
Low-degree models:
  ─ similar shapes across samples

High-degree models:
  ─ more noticeable changes across samples
```

That difference is the central visual idea of the project.

---

## Bias / Underfitting view

The left graph is labeled:

```text
High Bias / Underfitting
```

The default bias-model degree is:

```text
1
```

so the model is a straight line:

```text
ŷ = w₁x + b
```

A straight line has limited flexibility and cannot reproduce the nonlinear sine function well over the full interval.

When different noisy samples are generated, the fitted low-degree models tend to remain relatively simple and similar in overall form.

This is used to illustrate the intuition of:

```text
High bias
Low variance
Underfitting
```

The exact amount of bias or variance is not calculated by the application.

---

## Variance / Overfitting view

The right graph is labeled:

```text
High Variance / Overfitting
```

The default variance-model degree is:

```text
10
```

The model is therefore substantially more flexible than the default degree-1 model.

A higher-degree polynomial can respond more strongly to variations in the individual training samples.

When the dataset changes, the fitted polynomial can therefore change substantially.

This is used to illustrate the intuition of:

```text
Lower bias
Higher variance
Potential overfitting
```

Again, the application does not calculate a formal bias–variance decomposition.

---

## Model fitting

The application uses NumPy's polynomial least-squares fitting:

```python
np.polyfit(...)
```

For each experiment, the bias and variance models are fitted independently using the selected polynomial degrees.

The fitted coefficient vectors are stored so that the models from previous experiments can remain visible.

The corresponding fitted curves are evaluated with:

```python
np.polyval(...)
```

---

## Same-sample mode

The interface provides:

```text
Use same sample for both graphs
```

This option is enabled by default.

When enabled, the same noisy dataset is used for both polynomial fits:

```text
                 Same dataset
                 /          \
                /            \
               ↓              ↓
        Low-degree        High-degree
          model              model
```

This is useful because it isolates the effect of **model complexity**.

When the option is disabled, the two models are fitted using different noisy samples generated from related but different random-number streams.

That mode can be useful for experimentation, but it is less controlled for a direct model-complexity comparison.

---

## Reproducible experiment sequence

The application uses a deterministic seed scheme.

For experiment number `r`, the seed is constructed as:

```text
experiment_seed = seed + r × 7919
```

The resulting seed is used to initialize NumPy's random-number generator.

Therefore, with the same:

- random seed,
- number of points,
- noise level,
- polynomial degrees,
- and experiment sequence,

the generated experiment sequence is reproducible.

This makes the visualization suitable for lectures and repeatable demonstrations.

---

## Current sample points

The interface includes:

```text
Show current sample points
```

When enabled, only the **most recent dataset** is displayed as points.

The previously fitted models remain visible.

Therefore, after several experiments the graph conceptually contains:

```text
True sine function
+
Previously fitted models
+
Current sample points
```

When the option is disabled, the fitted models remain while the sample points are hidden.

Changing this option only redraws the graphs; it does not generate a new dataset or remove previously fitted models.

---

## Experiments

The **Experiments** setting determines how many experiments the Play control will generate.

Current interface limits:

```text
Minimum: 1
Maximum: 100
Default: 10
```

For example, with:

```text
Experiments = 10
```

the Play control generates experiments until the current experiment number reaches 10.

Previously fitted models remain visible throughout the sequence.

---

## Controls

### True Function

The current implementation provides one function:

```text
sin(x)
```

The selector is intentionally fixed to this single option.

### Data Points

Controls the requested number of observations per experiment.

```text
Minimum: 5
Default: 25
Maximum in the web interface: 500
```

The application also ensures that there are enough observations for the selected polynomial degrees.

Internally, the effective number of points is increased when necessary so that it is at least:

```text
bias degree + 2
variance degree + 3
```

This means the actual number used for fitting can be larger than the value entered by the user when the selected polynomial degrees require additional observations.

### Noise Level

Controls the standard deviation of the Gaussian noise.

Web interface:

```text
Minimum: 0
Maximum: 3
Step: 0.05
Default: 0.30
```

Higher noise produces more variation around the underlying sine function.

### Experiments

Controls how many samples the Play control will generate.

```text
Minimum: 1
Maximum: 100
Default: 10
```

### Bias Degree

Controls the polynomial degree used for the left-hand model.

Web interface:

```text
Minimum: 1
Maximum: 5
Default: 1
```

### Variance Degree

Controls the polynomial degree used for the right-hand model.

Web interface:

```text
Minimum: 2
Maximum: 18
Default: 10
```

Higher values allow a more flexible polynomial model.

### Random Seed

Controls the deterministic experiment sequence.

Default:

```text
42
```

### Speed

Controls the delay between automatically generated experiments.

Available settings:

```text
0.5×
1×
2×
5×
```

The corresponding delays in the web application are:

| Speed | Delay |
|---:|---:|
| `0.5×` | `1300 ms` |
| `1×` | `700 ms` |
| `2×` | `350 ms` |
| `5×` | `120 ms` |

### View

The **View** control cycles through:

```text
View: Normal
View: Zoom In
View: Zoom Out
```

Changing the view only redraws the existing graphs. It does not generate new observations or remove fitted models.

---

## Buttons

### Apply Settings

Applies the current settings and resets the experiment state.

It removes previously stored fitted models and returns the experiment counter to zero.

### Next Sample

Generates one new experiment.

The process is:

```text
Generate noisy sample
        ↓
Fit low-degree polynomial
        ↓
Fit high-degree polynomial
        ↓
Store both models
        ↓
Replace displayed sample points
        ↓
Redraw both graphs
```

### Play

Automatically generates experiments until the requested experiment count is reached.

The Play button changes to Pause while the sequence is running.

### Reset

The current web implementation maps Reset to the same reset operation as Apply Settings.

It therefore clears the stored fitted models and returns the visualization to its initial state.

### Show Current Sample Points

Toggles visibility of the current sample points without generating new data.

### Use Same Sample for Both Graphs

Controls whether the bias and variance models use exactly the same noisy dataset within each experiment.

---

## Visualization

The web application displays two graph panels.

### Left: High Bias / Underfitting

The left panel contains:

- the true sine function,
- all previously fitted low-degree models,
- optionally the current sample points.

### Right: High Variance / Overfitting

The right panel contains:

- the true sine function,
- all previously fitted high-degree models,
- optionally the current sample points.

The previous fitted models remain visible after each new experiment.

This creates a visual history of model behavior across different datasets.

---

## Plot ranges

The application supports three view modes.

### Normal view

Bias graph:

```text
x: 0 to 6
y: -1.5 to 1.5
```

Variance graph:

```text
x: 0 to 6
y: -2.0 to 1.5
```

### Zoom In

Bias graph:

```text
x: 0.5 to 5.5
y: -1.25 to 1.25
```

Variance graph:

```text
x: 0.5 to 5.5
y: -1.5 to 1.5
```

### Zoom Out

Bias graph:

```text
x: -0.5 to 6.5
y: -2.0 to 2.0
```

Variance graph:

```text
x: -0.5 to 6.5
y: -2.5 to 2.5
```

The variance curves are additionally clipped to:

```text
-3 ≤ y ≤ 3
```

before plotting. This is a visualization safeguard for extreme high-degree polynomial values and does not modify the fitted polynomial coefficients.

---

## Curve resolution

The displayed curves are evaluated on a grid of:

```text
700 x-values
```

covering:

```text
0 ≤ x ≤ 6
```

This grid is used for plotting the true sine function and fitted polynomial curves.

---

## Web architecture

The current `index.html` is a self-contained browser application that combines:

```text
HTML
CSS
JavaScript
Pyodide
NumPy
Matplotlib
```

The browser loads Pyodide and then loads NumPy and Matplotlib into the Python runtime.

The JavaScript layer:

- reads the interface controls,
- passes settings to Python,
- requests new experiments,
- requests redraws,
- controls Play/Pause,
- controls the experiment speed,
- controls the view mode,
- updates the displayed graph images and status.

The Python code running inside Pyodide:

- generates the samples,
- fits the polynomials,
- stores fitted models,
- creates the Matplotlib figures,
- encodes the figures as images,
- returns the image data to JavaScript.

---

## How the browser version works

The web application loads Pyodide from its CDN and requests:

```text
NumPy
Matplotlib
```

The Python runtime is initialized directly in the browser.

The generated Matplotlib figures are converted to PNG images in memory and returned to the JavaScript interface.

Therefore, the web version does not require a Python server.

A normal static web host can serve the HTML file.

---

## Running the web version

The current repository contains:

```text
bias_variance/
├── index.html
├── bias_variance_sine_gui.py
└── README_bias_variance.md
```

To use the web version, open:

```text
index.html
```

in a modern browser.

The page requires internet access because Pyodide and its NumPy/Matplotlib packages are loaded from external resources.

The repository's `index.html` is therefore the primary browser-based implementation described by this README.

---

## GitHub Pages

Because the web application executes Python in the browser through Pyodide, it can be served as a static page.

The repository already contains:

```text
index.html
```

which is the appropriate entry point for a GitHub Pages deployment.

No Python web server is required.

The main requirement is that the browser can access the external Pyodide resources.

---

## Desktop Python implementation

The repository also contains:

```text
bias_variance_sine_gui.py
```

This is a separate desktop implementation using Tkinter, NumPy, and Matplotlib.

It provides a similar conceptual experiment but should not be confused with the browser implementation.

The desktop program:

- creates a native Tkinter interface,
- generates the noisy datasets locally in Python,
- fits the two polynomial models with NumPy,
- displays the graphs using Matplotlib,
- supports the same general experiment workflow,
- includes its own speed and view controls.

The web implementation and desktop implementation share the same core educational idea but are separate programs.

---

## Desktop requirements

For the Python GUI version, the source imports:

```text
tkinter
numpy
matplotlib
```

`tkinter` is normally distributed with standard Python installations, although availability can depend on the operating system and Python distribution.

The external Python packages required by the desktop implementation are:

```bash
pip install numpy matplotlib
```

Run the desktop application with:

```bash
python bias_variance_sine_gui.py
```

The browser version does not require this desktop Python installation.

---

## Recommended settings

For a clear introductory demonstration, the current defaults are:

```text
True Function:    sin(x)
Data Points:      25
Noise Level:      0.30
Experiments:      10
Bias Degree:      1
Variance Degree:  10
Random Seed:      42
Speed:            1×
```

Keep:

```text
Use same sample for both graphs = ON
Show current sample points = ON
```

Then:

1. Click **Apply Settings**.
2. Click **Next Sample** several times, or click **Play**.
3. Watch the fitted models accumulate.
4. Compare the stability of the low-degree models with the variation of the high-degree models.

For a cleaner final visualization, turn **Show current sample points** off after generating the desired number of experiments.

---

## Educational interpretation

The visualization is intended to support the following intuition:

```text
Simpler model
    ↓
Less flexibility
    ↓
May systematically miss the true relationship
    ↓
Higher bias
```

and:

```text
More flexible model
    ↓
Greater sensitivity to the training sample
    ↓
Can fit sample-specific fluctuations
    ↓
Higher variance
```

The repeated-model view is especially useful because it makes variance visible as **variation across fitted models produced from different training samples**.

---

## What this project does not calculate

The application does not explicitly compute:

```text
Bias = ...
Variance = ...
Expected test error = ...
```

It also does not produce a numerical bias–variance curve.

Instead, it provides a visual demonstration of model behavior across repeated samples.

For a formal bias–variance decomposition, a separate experiment would need to generate many training datasets, evaluate models at selected input locations, and calculate the relevant expectations and variances.

---

## Important caution about the labels

The interface calls the two models:

```text
High Bias / Underfitting
High Variance / Overfitting
```

These labels are useful for the intended teaching scenario, but they should not be interpreted as universal properties of every polynomial of those degrees.

For example:

```text
Degree 10 ≠ automatically overfitting
Degree 1 ≠ automatically high bias for every dataset
```

The behavior depends on:

- the underlying function,
- noise level,
- sample size,
- input distribution,
- polynomial degree,
- and evaluation procedure.

The current sine-function experiment is deliberately configured to make the intended contrast visible.

---

## Numerical stability

High-degree polynomial fitting can be numerically sensitive.

The web implementation uses:

```python
np.polyfit(...)
```

and NumPy's polynomial evaluation through:

```python
np.polyval(...)
```

The visualization additionally clips variance-model predictions to:

```text
[-3, 3]
```

for plotting.

This clipping is only a display safeguard. It does not change the fitted coefficients or the underlying model.

The application also ensures that the effective number of observations is sufficiently large relative to the selected polynomial degrees before fitting.

---

## Suggested teaching sequence

### 1. Begin with one dataset

Use:

```text
Bias degree = 1
Variance degree = 10
```

Generate one sample.

Ask:

> Which model appears more flexible?

### 2. Generate several samples

Click **Next Sample** repeatedly.

Observe that the previous fitted models remain.

### 3. Compare the left graph

Look at the collection of low-degree lines.

They tend to retain a relatively simple overall shape.

### 4. Compare the right graph

Look at the collection of high-degree curves.

Their shapes can change much more substantially from one sample to another.

### 5. Connect the observation to variance

Explain that variance concerns the sensitivity of a learned model to changes in the training sample.

### 6. Connect the observation to bias

Explain that a model with insufficient flexibility can systematically fail to represent the underlying nonlinear relationship.

### 7. Introduce the trade-off

Use the visualization as a transition to:

- model complexity,
- generalization,
- training vs. test error,
- bias–variance trade-off,
- regularization,
- model selection.

---

## Limitations

The current visualization is intentionally focused on intuition.

It does not currently include:

- a test dataset,
- validation data,
- test error,
- cross-validation,
- a numerical bias estimate,
- a numerical variance estimate,
- expected prediction error,
- irreducible-noise estimation,
- automatic degree selection,
- regularization,
- confidence intervals.

The project should therefore be understood as a **visual bias–variance demonstration**, not as a complete statistical analysis tool.

---

## Possible extensions

### Numerical bias and variance

For fixed test inputs, generate many independent training datasets and calculate:

```text
Average prediction
Bias
Variance
Mean Squared Error
```

### Bias–variance decomposition

Add a numerical panel showing:

```text
Expected error
= Bias²
+ Variance
+ Irreducible noise
```

### Degree sweep

Allow the learner to compare several polynomial degrees simultaneously.

### Error curves

Plot average training and test error against polynomial degree.

### Regularization

Add Ridge regression and compare its behavior with unregularized high-degree polynomial fitting.

### Interactive sample size

Allow learners to see how increasing the number of observations affects the variability of high-degree models.

### Interactive noise

Increase the noise level and observe how the fitted models respond.

---

## Technical summary

| Component | Current web implementation |
|---|---|
| Underlying function | `sin(x)` |
| Input range | `[0, 6]` |
| Default observations | `25` |
| Default noise | `0.30` |
| Default experiments | `10` |
| Default bias degree | `1` |
| Default variance degree | `10` |
| Bias degree range | `1–5` |
| Variance degree range | `2–18` |
| Data-point input range | `5–500` |
| Noise input range | `0–3` |
| Experiment range | `1–100` |
| Default random seed | `42` |
| Polynomial fitting | `numpy.polyfit` |
| Polynomial evaluation | `numpy.polyval` |
| Plotting | Matplotlib |
| Browser Python runtime | Pyodide |
| Browser control layer | JavaScript |
| Curve samples | `700` |
| Variance display clipping | `[-3, 3]` |
| Web output | PNG images embedded in the page |
| Entry point | `index.html` |

---

## Key takeaway

The project makes one central idea visible:

```text
Change the training sample
        ↓
Refit the same type of model
        ↓
Observe how much the fitted model changes
```

A simple model may remain relatively stable but fail to capture the true nonlinear relationship.

A highly flexible model may capture the training samples much more closely but become substantially more sensitive to the particular sample used for training.

That visual contrast provides an intuitive entry point into the **bias–variance trade-off**.

---

## Author

**Dr. Prasenjit Dey**

Part of the **DL Animations** collection of interactive visualizations for Machine Learning and Deep Learning education.

---

## License

No explicit open-source license file is currently present in the `bias_variance` directory.

If this project is intended for public redistribution, add a `LICENSE` file and document the applicable terms here.
