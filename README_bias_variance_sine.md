# Bias vs Variance Visualization using a Sine Curve

This project is an interactive visualization of the **bias–variance concept in machine learning** using the true function

\[
y = \sin(x)
\]

The application shows two synchronized graphs:

- **Left graph:** High Bias / Underfitting
- **Right graph:** High Variance / Overfitting

The project runs directly in the browser using **Python through Pyodide**, together with **NumPy** and **Matplotlib**.

---

## Project Objective

The main goal of this project is to visually demonstrate the difference between **bias** and **variance**.

For every experiment:

1. A new noisy dataset is generated around the true sine function.
2. The same dataset is used for both the bias and variance models.
3. A low-degree polynomial is fitted for the bias graph.
4. A high-degree polynomial is fitted for the variance graph.
5. The old data points are removed.
6. The previously fitted models remain visible.

After several experiments, the difference between bias and variance becomes easy to observe.

---

## Bias Graph

The graph on the **left side** represents high bias or underfitting.

By default, the model degree is:

```text
Bias Degree = 1
```

Therefore, the model is approximately

\[
\hat{y} = wx + b
\]

A straight line cannot properly represent the nonlinear sine function.

As a result, different datasets usually produce similar simple fitted lines.

This represents:

- High Bias
- Low Variance
- Underfitting

---

## Variance Graph

The graph on the **right side** represents high variance or overfitting.

By default, the polynomial degree is:

```text
Variance Degree = 10
```

A high-degree polynomial is flexible enough to follow small variations and noise in the training data.

Therefore, small changes in the generated dataset can produce significantly different fitted curves.

This represents:

- Low Bias
- High Variance
- Overfitting

---

## Synchronized Experiments

Both graphs are synchronized.

For example:

```text
Experiment 1

Bias Graph     -> 1 fitted line
Variance Graph -> 1 fitted curve
```

After pressing **Next Sample** again:

```text
Experiment 2

Bias Graph     -> 2 fitted lines
Variance Graph -> 2 fitted curves
```

After 10 experiments:

```text
Bias Graph     -> 10 fitted models
Variance Graph -> 10 fitted models
```

The old sample points are removed after every experiment, while the fitted models remain on the graph.

---

## Data Generation

For every experiment, input values are sampled from:

\[
x_i \sim U(0,6)
\]

The corresponding noisy target values are generated using:

\[
y_i = \sin(x_i) + \epsilon_i
\]

where

\[
\epsilon_i \sim \mathcal{N}(0,\sigma^2)
\]

and \(\sigma\) is controlled by the **Noise Level** setting.

---

## Model Fitting

### Bias Model

For the default degree:

```text
Bias Degree = 1
```

the model becomes:

\[
\hat{y} = w_1x + b
\]

This model is too simple for the true sine function.

---

### Variance Model

For the default degree:

```text
Variance Degree = 10
```

the model becomes a high-degree polynomial:

\[
\hat{y}
=
w_{10}x^{10}
+
w_9x^9
+
\cdots
+
w_1x
+
b
\]

This model is highly flexible and can fit noise in the training samples.

---

## Controls

The application contains the following controls.

### True Function

```text
sin(x)
```

The current project uses the sine function as the true underlying function.

---

### Data Points

Controls the number of randomly generated data points.

Recommended value:

```text
25
```

---

### Noise Level

Controls the amount of random noise added to the sine function.

Recommended value:

```text
0.30
```

Increasing this value makes the data more scattered.

---

### Experiments

Controls how many experiments are generated automatically when using **Play**.

Recommended value:

```text
10
```

---

### Bias Degree

Controls the polynomial degree of the bias model.

Recommended value:

```text
1
```

A smaller value produces a simpler model and stronger underfitting.

---

### Variance Degree

Controls the polynomial degree of the variance model.

Recommended value:

```text
10
```

A larger value produces a more flexible model and stronger variance.

---

### Random Seed

Controls random number generation.

Recommended value:

```text
42
```

Using the same seed makes the sequence of experiments reproducible.

---

### Speed

Controls the delay between automatically generated experiments when using the **Play** button.

Available options include:

```text
0.5x
1x
2x
5x
```

---

## Buttons

### Apply Settings

Applies the current settings and restarts the experiment.

---

### Next Sample

Generates one new noisy dataset.

For each click:

```text
Generate new data
        |
        v
Fit low-degree model
        |
        +----> Bias Graph
        |
        v
Fit high-degree model
        |
        +----> Variance Graph
```

The same generated sample is used for both models when synchronization is enabled.

---

### Play

Automatically generates experiments one after another until the specified number of experiments is reached.

---

### Reset

Removes all fitted lines and curves and returns the visualization to the initial state.

---

## Show Current Sample Points

When enabled, the most recently generated sample points are displayed.

When a new experiment is created:

- previous sample points disappear
- new sample points appear
- previous fitted lines remain

Disable this option if you want a clean final graph containing only the true function and fitted models.

---

## Use Same Sample for Both Graphs

This option keeps the bias and variance demonstrations synchronized.

When enabled:

```text
Dataset 1
   |----------------|
   |                |
Bias Model     Variance Model
```

Therefore, the difference between the graphs comes from **model complexity**, not from using different datasets.

---

## Recommended Settings

For a clear bias–variance demonstration, use:

```text
True Function:      sin(x)
Data Points:        25
Noise Level:        0.30
Experiments:        10
Bias Degree:        1
Variance Degree:    10
Random Seed:        42
```

Then click:

```text
Apply Settings
```

and either repeatedly click:

```text
Next Sample
```

or click:

```text
Play
```

---

## Technologies Used

This project uses:

- HTML
- CSS
- JavaScript
- Python
- Pyodide
- NumPy
- Matplotlib

---

## How Python Runs Inside the HTML File

The project uses **Pyodide**, which allows Python code to run directly inside a web browser using WebAssembly.

The HTML file loads Pyodide from a CDN and then loads:

```python
numpy
matplotlib
```

The Python code performs:

- random data generation
- sine-function calculation
- polynomial regression
- model storage
- graph generation

JavaScript controls the buttons and communicates with the Python code.

---

## How to Run

No Python installation is required for this version.

Download:

```text
bias_variance_sine_demo.html
```

Then open the file in a modern browser such as:

- Google Chrome
- Microsoft Edge
- Firefox

An internet connection is required initially because Pyodide, NumPy and Matplotlib are loaded from the CDN.

---

## GitHub Repository Structure

A simple repository can contain:

```text
bias-variance-sine/
│
├── bias_variance_sine_demo.html
└── README.md
```

---

## GitHub Pages

Because Python is executed in the browser using Pyodide, this version can be hosted as a static webpage.

For GitHub Pages, it is convenient to rename:

```text
bias_variance_sine_demo.html
```

to:

```text
index.html
```

Your repository can then look like:

```text
bias-variance-sine/
│
├── index.html
└── README.md
```

After enabling GitHub Pages, the interactive bias–variance visualization can run directly in the browser.

---

## Bias–Variance Interpretation

The visualization illustrates the classic bias–variance trade-off.

| Model | Bias | Variance | Behavior |
|---|---|---|---|
| Low-degree polynomial | High | Low | Underfitting |
| High-degree polynomial | Low | High | Overfitting |

The left graph therefore demonstrates a model that is too simple, while the right graph demonstrates a model that is too sensitive to the training sample.

---

## Educational Purpose

This project can be used to teach:

- Bias
- Variance
- Bias–variance trade-off
- Underfitting
- Overfitting
- Polynomial regression
- Noise in datasets
- Random sampling
- Model complexity
- Repeated experiments
- Generalization in machine learning

---

## Expected Result

After several experiments, the left side should contain several relatively similar simple fitted lines.

The right side should contain several more flexible and noticeably different fitted curves.

This visually shows why:

```text
Simple Model
    -> High Bias
    -> Low Variance
```

while:

```text
Complex Model
    -> Lower Bias
    -> High Variance
```

---

## License

This project can be used and modified for educational and research demonstrations.
