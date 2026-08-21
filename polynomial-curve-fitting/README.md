# Interactive Polynomial Curve Fitting

An interactive and presentation-ready visualization of **polynomial curve fitting**, designed to demonstrate how increasing model degree changes the flexibility of a fitted polynomial and can lead from **underfitting to increasing complexity and potential overfitting**.

The project generates two outputs from the same Python program:

1. A **standalone interactive HTML visualization** with a degree slider.
2. A **high-quality 1920×1080 MP4 animation** showing the polynomial fit progressing from degree 1 through degree 30.

The implementation uses a fixed noisy sample of a sine function and fits polynomial models of increasing degree to the same observations.

---

## 1. Project Overview

The visualization starts with a set of noisy observations sampled from:

\[
y=\sin(x)
\]

The observed data are generated as:

\[
y_i=\sin(x_i)+\epsilon_i
\]

where the noise term is sampled from a normal distribution:

\[
\epsilon_i\sim\mathcal{N}(0,\sigma^2)
\]

The program then fits polynomial models of increasing degree:

\[
y=a_0+a_1x+a_2x^2+\cdots+a_dx^d
\]

for:

\[
d=1,2,3,\ldots,30.
\]

The resulting fitted curves can be examined interactively using the HTML output or sequentially through the generated MP4 animation.

---

# 2. Educational Objective

The primary purpose of this project is to provide a visual explanation of **model complexity in polynomial curve fitting**.

A low-degree polynomial may be too simple to represent the structure of the data.

As the polynomial degree increases, the model becomes more flexible.

At sufficiently high degrees, the model can begin fitting noise and exhibit highly complex behavior.

The visualization therefore provides an intuitive progression:

```text
Low model complexity
        ↓
Underfitting
        ↓
More flexible model
        ↓
Reasonable / increasing complexity
        ↓
High model complexity
        ↓
Potential overfitting
```

The classification shown in the visualization is:

| Polynomial degree | Interpretation |
|---:|---|
| 1–2 | UNDERFITTING |
| 3–5 | REASONABLE FIT |
| 6–9 | INCREASING COMPLEXITY |
| 10–30 | POTENTIAL OVERFITTING |

These labels are implemented directly in the Python program.

> **Important:** These labels are pedagogical interpretations used by the visualization. They are not a statistical test that formally proves overfitting for every dataset.

---

# 3. Dataset Generation

The program does not load an external dataset.

Instead, it generates a synthetic dataset so that the experiment is completely reproducible.

The random seed is fixed:

```python
np.random.seed(42)
```

This means that running the program with the same configuration produces the same noisy observations.

---

## 3.1 Number of Observations

The current configuration uses:

```python
NUMBER_OF_POINTS = 50
```

Therefore, 50 observations are generated.

The x-values are uniformly spaced between:

\[
0
\]

and:

\[
2\pi.
\]

The implementation uses:

```python
x_data = np.linspace(
    0,
    2 * np.pi,
    NUMBER_OF_POINTS
)
```

---

## 3.2 True Function

The underlying function is:

\[
y=\sin(x)
\]

implemented as:

```python
y_true_data = np.sin(x_data)
```

The true function is also plotted as a dashed reference curve.

---

## 3.3 Noise

Gaussian noise is added to the true observations.

The current noise level is:

```python
NOISE_LEVEL = 0.50
```

The noise is generated using:

```python
noise = np.random.normal(
    loc=0,
    scale=NOISE_LEVEL,
    size=NUMBER_OF_POINTS
)
```

Therefore:

\[
\epsilon\sim\mathcal{N}(0,0.5^2).
\]

The observed data are then:

```python
y_data = y_true_data + noise
```

This creates a controlled environment in which the effect of polynomial
model complexity can be visualized.

---

# 4. Polynomial Fitting

For every degree from 1 through 30, the program fits a polynomial model to the same 50 observations.

The fitting operation uses NumPy's polynomial representation:

```python
np.polynomial.Polynomial.fit(
    x_data,
    y_data,
    degree
)
```

The fitted model is then evaluated at two locations:

1. The smooth plotting grid, to draw the fitted curve.
2. The original training points, to calculate training MSE.

The implementation is encapsulated in:

```python
def fit_polynomial(degree):
```

---

# 5. Training Mean Squared Error

For every polynomial degree, the program calculates the training Mean Squared Error (MSE).

The MSE is:

\[
MSE=
\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat{y}_i)^2
\]

where:

- \(y_i\) is the observed value,
- \(\hat{y}_i\) is the model prediction,
- \(N\) is the number of observations.

The implementation calculates:

```python
mse = np.mean(
    (y_data - y_prediction) ** 2
)
```

The resulting MSE is displayed dynamically in the HTML visualization and in every frame of the MP4 animation.

---

# 6. Why Training MSE Is Shown

Training MSE provides a useful numerical companion to the visual demonstration.

As model degree increases, the polynomial has more flexibility to adapt to the training observations.

This makes it possible to observe the relationship between:

```text
Model complexity
       ↓
Training fit
       ↓
Training MSE
```

However, training MSE alone cannot establish generalization performance.

A model may have a very low training MSE while performing poorly on unseen data.

Therefore, this visualization should be used to introduce the concept of complexity and overfitting, rather than as a complete model-selection experiment.

---

# 7. Polynomial Degrees

The maximum polynomial degree is controlled by:

```python
MAX_DEGREE = 30
```

The program pre-computes every model:

```text
Degree 1
Degree 2
Degree 3
...
Degree 30
```

The fitted curves and MSE values are stored in:

```python
curves = {}
mse_values = {}
```

This means the interactive HTML slider can switch between already-computed models without repeatedly fitting the polynomial during interaction.

---

# 8. Smooth Curve Rendering

The polynomial models are evaluated on a dense grid of 1000 x-values:

```python
x_plot = np.linspace(
    0,
    2 * np.pi,
    1000
)
```

The dense grid is used only for smooth visualization of the fitted polynomial curves.

The original 50 observations remain the training data used for fitting and MSE calculation.

---

# 9. Interactive HTML Visualization

The HTML output provides an interactive Plotly visualization.

The graph contains three main visual elements:

### 1. True Function

\[
y=\sin(x)
\]

shown as a dashed reference curve.

### 2. Noisy Observations

The 50 generated observations are shown as markers.

### 3. Polynomial Fit

The currently selected polynomial degree is shown as a solid fitted curve.

The HTML visualization therefore allows the user to compare:

```text
True function
       +
Noisy observations
       +
Polynomial approximation
```

for every degree from 1 through 30.

---

# 10. Degree Slider

The HTML interface provides a slider containing:

```text
1  2  3  4  5  ...  28  29  30
```

The slider is controlled through Plotly animation frames.

For every degree, the visualization updates:

- Polynomial curve
- Polynomial degree
- Training MSE
- Interpretation label

For example:

```text
Polynomial Curve Fitting — Degree 5
Training MSE = ...
REASONABLE FIT
```

The slider therefore provides direct visual control over model complexity.

---

# 11. Interactive Animation Speed

The HTML animation is configured using:

```python
HTML_FRAME_DURATION = 700
HTML_TRANSITION_DURATION = 300
```

These values control the duration of each animation frame and its transition.

The HTML is therefore not simply a static graph with a slider; it contains Plotly animation frames for the polynomial degrees.

---

# 12. Model Complexity Interpretation

The visualization uses the following educational categorization.

## Degree 1–2: Underfitting

A very low-degree polynomial has limited flexibility.

It may fail to capture the nonlinear structure present in the data.

---

## Degree 3–5: Reasonable Fit

Moderate-degree models generally have more flexibility and can provide a visually reasonable approximation of the noisy observations.

The visualization labels this region:

```text
REASONABLE FIT
```

---

## Degree 6–9: Increasing Complexity

As the degree continues to increase, the model becomes increasingly flexible.

The visualization labels this region:

```text
INCREASING COMPLEXITY
```

---

## Degree 10–30: Potential Overfitting

High-degree polynomials can become extremely flexible.

They may begin to follow noise and produce increasingly complex shapes.

The visualization therefore labels this region:

```text
POTENTIAL OVERFITTING
```

The word **potential** is intentional. Formal overfitting assessment requires evaluation on unseen data, not only inspection of the training curve.

---

# 13. Coordinate System

The interactive HTML visualization uses:

### X-axis

\[
0\leq x\leq2\pi
\]

### Y-axis

\[
-2.5\leq y\leq2.5
\]

The same coordinate ranges are used for the video visualization.

The axes are configured with clear titles, tick labels, and a white Plotly background.

---

# 14. Teaching Annotation

The visualization includes an explanatory annotation:

```text
Model Complexity

Low degree → Underfitting
Moderate degree → Good fit
High degree → Overfitting
```

This annotation is intended to help students connect the mathematical visualization with the conceptual progression of model complexity.

The annotation appears in both the interactive HTML visualization and the generated video.

---

# 15. High-Quality MP4 Animation

In addition to the interactive HTML file, the program generates an MP4 video.

The output file is:

```text
polynomial_curve_fitting_animation.mp4
```

The video is generated using:

```python
imageio
```

and encoded using:

```text
libx264
```

The video resolution is configured as:

```python
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
```

Therefore, the resulting animation is Full HD:

\[
1920\times1080.
\]

This resolution is suitable for:

- Classroom presentations
- Lecture slides
- Project demonstrations
- Online teaching
- Screen recording
- Educational videos

---

# 16. Video Frame Rate

The MP4 is generated at:

```python
fps = 30
```

Therefore, the video contains 30 frames per second.

The program uses a frame-based approach to control how long each polynomial degree remains visible.

---

# 17. Video Timing

The current configuration uses:

```python
VIDEO_HOLD_TIME = 0.175
VIDEO_TRANSITION_TIME = 0.075
```

At 30 frames per second:

### Hold time

\[
0.175\times30\approx5
\]

frames.

### Transition time

\[
0.075\times30\approx2
\]

frames.

Therefore, each degree is displayed for approximately:

\[
0.25\text{ seconds}
\]

under the current settings.

With 30 degrees, the resulting animation is approximately:

\[
30\times0.25=7.5\text{ seconds}
\]

of degree progression, excluding minor encoding and rounding effects.

The Python script itself reports an approximate duration in its final console output, but the exact duration depends on the integer frame counts generated from the configured timing values.

---

# 18. Video Generation Process

For each degree from 1 through 30, the program:

1. Creates a dedicated high-resolution Plotly figure.
2. Adds the true sine function.
3. Adds the 50 noisy observations.
4. Adds the polynomial fitted at the current degree.
5. Adds the training MSE.
6. Adds the model-complexity interpretation.
7. Converts the Plotly figure to PNG.
8. Converts the PNG to an image array.
9. Appends the required number of frames to the MP4 writer.

This produces a presentation-oriented video in which the polynomial fit changes degree by degree.

---

# 19. Video and HTML Use Different Presentation Configurations

The HTML and video outputs are intentionally configured separately.

The HTML visualization uses:

```python
width = 1200
height = 750
```

while the video uses:

```python
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
```

The video also uses larger fonts and margins to improve readability at Full HD resolution.

This prevents the video from simply being a low-resolution capture of the browser visualization.

---

# 20. Output Files

Running the program creates two primary outputs:

```text
interactive_polynomial_curve_fitting.html
```

and:

```text
polynomial_curve_fitting_animation.mp4
```

Both files are saved in:

```python
OUTPUT_FOLDER = Path.cwd()
```

Therefore, they are written to the current working directory.

The HTML file is automatically opened in the default browser.

The MP4 remains available as a normal video file that can be opened with a standard media player.

---

# 21. Output Summary

| Output | Format | Purpose |
|---|---|---|
| Interactive visualization | HTML | Interactive exploration |
| Polynomial animation | MP4 | Presentation and teaching |
| HTML resolution | 1200 × 750 | Browser visualization |
| Video resolution | 1920 × 1080 | Full HD presentation |
| Polynomial degrees | 1–30 | Model complexity progression |
| Data points | 50 | Synthetic noisy observations |
| Noise standard deviation | 0.50 | Controlled noise level |
| True function | \(\sin(x)\) | Ground-truth reference |
| MSE | Training MSE | Numerical fit measure |

---

# 22. Configuration Reference

All major settings are grouped near the beginning of the Python file.

```python
NUMBER_OF_POINTS = 50
NOISE_LEVEL = 0.50
MAX_DEGREE = 30

HTML_FRAME_DURATION = 700
HTML_TRANSITION_DURATION = 300

VIDEO_HOLD_TIME = 0.175
VIDEO_TRANSITION_TIME = 0.075

VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080

OUTPUT_FOLDER = Path.cwd()
```

This design makes the experiment easy to modify without changing the main implementation logic.

---

# 23. Changing the Number of Data Points

Change:

```python
NUMBER_OF_POINTS = 50
```

For example:

```python
NUMBER_OF_POINTS = 30
```

will generate 30 observations.

Similarly:

```python
NUMBER_OF_POINTS = 100
```

will generate 100 observations.

The same value automatically updates the generated dataset and the observation label shown in the plots.

---

# 24. Changing the Noise Level

Change:

```python
NOISE_LEVEL = 0.50
```

For example:

```python
NOISE_LEVEL = 0.25
```

creates less noisy observations.

```python
NOISE_LEVEL = 0.75
```

creates more noisy observations.

The value is used as the standard deviation of the Gaussian noise distribution.

Higher noise makes the fitting problem visually more challenging and can make the difference between fitting the underlying function and fitting individual observations easier to discuss.

---

# 25. Changing the Maximum Polynomial Degree

Change:

```python
MAX_DEGREE = 30
```

For example:

```python
MAX_DEGREE = 15
```

generates degrees 1–15.

```python
MAX_DEGREE = 40
```

generates degrees 1–40.

The same maximum degree controls:

- Polynomial fitting
- HTML animation frames
- HTML slider
- Video generation
- Console reporting

---

# 26. Changing HTML Animation Speed

The HTML animation uses:

```python
HTML_FRAME_DURATION = 700
HTML_TRANSITION_DURATION = 300
```

Reducing these values makes the interactive animation faster.

Increasing them makes it slower.

For example:

```python
HTML_FRAME_DURATION = 350
HTML_TRANSITION_DURATION = 150
```

will make the HTML animation approximately twice as fast in terms of frame timing.

---

# 27. Changing Video Speed

The video speed is controlled independently using:

```python
VIDEO_HOLD_TIME = 0.175
VIDEO_TRANSITION_TIME = 0.075
```

To make the video faster, reduce these values.

For example:

```python
VIDEO_HOLD_TIME = 0.0875
VIDEO_TRANSITION_TIME = 0.0375
```

approximately halves the time allocated to each degree.

To make the video slower, increase them.

The video frame rate remains:

```python
fps = 30
```

---

# 28. Reproducibility

The program uses:

```python
np.random.seed(42)
```

This fixes the random number generator before generating the noise.

Therefore, with the same:

- Number of points
- Noise level
- Random seed
- Function
- Polynomial configuration

the generated dataset is reproducible.

Changing the seed will produce a different noisy sample.

---

# 29. Important Distinction: Training Error vs Generalization

The visualization displays **training MSE**.

Training MSE measures how closely the polynomial fits the observations used to train it.

It does not measure how well the model will perform on unseen observations.

For a rigorous study of overfitting, a separate validation or test dataset should be introduced and the following should be compared:

\[
Training\ Error
\]

versus:

\[
Validation/Test\ Error.
\]

This project intentionally keeps the experiment simple and focuses on visualizing the effect of increasing polynomial degree.

---

# 30. Numerical Considerations

High-degree polynomial fitting can become numerically challenging.

The program uses:

```python
np.polynomial.Polynomial.fit(...)
```

which uses NumPy's polynomial fitting representation rather than directly constructing an unscaled Vandermonde system in the most naive form.

Nevertheless, very high polynomial degrees can still exhibit numerical sensitivity and highly oscillatory behavior.

The current experiment is therefore intentionally limited to:

```python
MAX_DEGREE = 30
```

---

# 31. Educational Interpretation of Overfitting

A high-degree polynomial may pass very closely through noisy observations.

This can make the training error small while producing a visually complex function.

However, visual complexity alone is not sufficient evidence of statistical overfitting.

A rigorous definition requires poor generalization to unseen data.

Therefore, this project uses the label:

```text
POTENTIAL OVERFITTING
```

rather than claiming that every degree from 10 to 30 is statistically overfit.

This distinction is important when using the animation for academic teaching.

---

# 32. Suggested Teaching Sequence

A useful classroom sequence is:

### Step 1 — Show the data

Explain that the observations are noisy samples from:

\[
y=\sin(x).
\]

### Step 2 — Start with degree 1

Show that a linear model is too restrictive to capture the nonlinear structure.

### Step 3 — Increase the degree

Move through degrees 2, 3, 4 and 5.

Discuss increasing flexibility.

### Step 4 — Observe training MSE

Explain how the model becomes increasingly capable of fitting the training observations.

### Step 5 — Continue to higher degrees

Observe the increasing complexity of the fitted curve.

### Step 6 — Discuss overfitting

Explain why a highly flexible model can begin to model noise rather than the underlying relationship.

### Step 7 — Introduce test error

Use the visualization as a bridge to the more rigorous concept of generalization and the bias-variance trade-off.

---

# 33. Project Structure

Recommended GitHub organization:

```text
polynomial-curve-fitting/
│
├── polynomial_curve_fitting.py
└── README.md
```

After running the program locally, the directory may contain:

```text
polynomial-curve-fitting/
│
├── polynomial_curve_fitting.py
├── README.md
├── interactive_polynomial_curve_fitting.html
└── polynomial_curve_fitting_animation.mp4
```

The HTML and MP4 files are generated artifacts.

They do not need to be committed to the source repository unless the generated educational outputs are intentionally being distributed with the project.

---

# 34. Technologies

| Technology | Purpose |
|---|---|
| Python | Main implementation |
| NumPy | Data generation and polynomial fitting |
| Plotly | Interactive visualization |
| ImageIO | MP4 frame writing |
| FFmpeg / libx264 | H.264 video encoding |
| HTML | Standalone interactive output |
| JavaScript | Plotly animation controls |

---

# 35. Key Features

- Synthetic noisy sine-wave dataset
- Reproducible random sampling
- Configurable number of observations
- Configurable noise level
- Polynomial fitting from degree 1 to 30
- Pre-computed polynomial curves
- Training MSE calculation
- Interactive Plotly HTML visualization
- Degree slider
- Animated degree transitions
- Model-complexity interpretation
- True-function reference curve
- Noisy observation markers
- Full HD 1920×1080 MP4 output
- H.264 video encoding
- Configurable video timing
- Automatic browser launch for HTML
- Local output generation

---

# 36. Intended Use

This project is intended for educational and demonstration purposes, particularly for:

- Machine Learning lectures
- Data Science courses
- Polynomial regression demonstrations
- Model-complexity discussions
- Underfitting and overfitting demonstrations
- Bias-variance discussions
- Interactive classroom teaching
- Lecture presentations
- Educational video creation
- Self-learning

---

# 37. Limitations

The current implementation uses a single synthetic dataset generated from:

\[
y=\sin(x)+\epsilon.
\]

It does not currently provide:

- Train/test split
- Validation-set evaluation
- Test MSE
- Cross-validation
- Regularization
- Ridge regression
- Lasso regression
- Automatic model selection
- Confidence intervals
- Statistical significance testing

These could be added as extensions if the project is later expanded into a more complete polynomial regression teaching framework.

---

# 38. Future Extensions

Potential extensions include:

1. Train/test data visualization.
2. Training MSE versus test MSE plots.
3. Bias-variance visualization.
4. Ridge and Lasso regularization.
5. Adjustable polynomial degree from the interface.
6. Adjustable noise level from the interface.
7. Different underlying functions.
8. Multiple random seeds.
9. Dataset regeneration.
10. Residual visualization.
11. Error curves versus polynomial degree.
12. Comparison of several polynomial models.
13. Interactive coefficient controls.
14. Additional export formats.

---

# 39. Summary

This project provides a compact but comprehensive visual demonstration of polynomial model complexity.

Starting with 50 noisy observations generated from:

\[
y=\sin(x),
\]

the program fits polynomial models from degree 1 through degree 30.

For every degree, it provides:

- The fitted polynomial curve
- The original noisy observations
- The true sine function
- Training MSE
- A pedagogical model-complexity interpretation

The project produces both an interactive HTML visualization and a Full HD MP4 animation, making it suitable for both interactive exploration and classroom presentation.

The central learning progression is:

\[
\boxed{
\text{Low Complexity}
\rightarrow
\text{Better Fit}
\rightarrow
\text{High Flexibility}
\rightarrow
\text{Potential Overfitting}
}
\]

---

# 40. Author

**Dr. Prasenjit Dey**

Part of the **DL Animations** collection of interactive visualizations for Machine Learning and Deep Learning education.

---

# 41. License

Please refer to the license of the parent repository for the applicable terms of use, modification, and distribution.
