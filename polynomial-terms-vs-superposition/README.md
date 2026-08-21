# Polynomial Terms vs Superposition

An interactive mathematical visualization for understanding the relationship between **individual polynomial terms** and their **cumulative superposition**.

The visualization uses two synchronized side-by-side plots:

- **Left panel:** Individual polynomial terms
- **Right panel:** Cumulative superposition of polynomial terms

The visualization progressively increases the polynomial order from **1 to 20**, allowing the viewer to observe how individual powers of \(x\) and their cumulative combinations evolve.

This project is designed as an educational visualization for teaching polynomial functions, basis functions, coefficients, linear combinations, and polynomial model construction.

---

## 1. Overview

A polynomial can be represented as a combination of powers of the input variable:

\[
y = m_1x + m_2x^2 + m_3x^3 + \cdots + m_nx^n
\]

where:

- \(x\) is the input variable,
- \(m_1,m_2,\ldots,m_n\) are coefficients,
- \(n\) is the maximum polynomial order.

This visualization separates the concept into two complementary views.

### Individual Polynomial Terms

The left panel displays the terms individually:

\[
y=x
\]

\[
y=x^2
\]

\[
y=x^3
\]

\[
\vdots
\]

\[
y=x^{20}
\]

### Superposition

The right panel displays the cumulative addition of those terms:

\[
y=x
\]

\[
y=x+x^2
\]

\[
y=x+x^2+x^3
\]

\[
\vdots
\]

\[
y=x+x^2+x^3+\cdots+x^{20}
\]

This side-by-side design makes the distinction between an individual polynomial term and a combination of polynomial terms visually explicit.

---

## 2. What This Project Demonstrates

The main educational idea is:

> Individual polynomial terms can be viewed as building blocks that can be combined to construct a higher-order polynomial expression.

For example, consider order 4.

The **Individual Terms** panel contains:

\[
x,\quad x^2,\quad x^3,\quad x^4
\]

as separate functions.

The **Superposition** panel contains the cumulative functions:

\[
x
\]

\[
x+x^2
\]

\[
x+x^2+x^3
\]

\[
x+x^2+x^3+x^4
\]

The two panels therefore provide two different perspectives on the same set of polynomial components.

---

## 3. Individual Terms

The left panel is labelled:

**Individual Terms**

At each order, the corresponding polynomial term is revealed.

| Order | Individual term |
|---:|---|
| 1 | \(y=x\) |
| 2 | \(y=x^2\) |
| 3 | \(y=x^3\) |
| 4 | \(y=x^4\) |
| ... | ... |
| 20 | \(y=x^{20}\) |

The code calculates each term independently according to:

\[
y=m_kx^k
\]

where \(m_k\) is the coefficient associated with the \(k\)-th term.

In the current implementation all coefficients are set to 1:

\[
m_1=m_2=\cdots=m_{20}=1.
\]

Therefore, the displayed individual terms are simply:

\[
x,x^2,x^3,\ldots,x^{20}.
\]

---

## 4. Superposition

The right panel is labelled:

**Superposition**

Unlike the left panel, the right panel displays cumulative combinations of the polynomial terms.

The progression is:

### Order 1

\[
y=x
\]

### Order 2

\[
y=x+x^2
\]

### Order 3

\[
y=x+x^2+x^3
\]

### Order 4

\[
y=x+x^2+x^3+x^4
\]

and so on.

At order 20:

\[
y=x+x^2+x^3+\cdots+x^{20}.
\]

The terms are added cumulatively, so each higher-order superposition contains all preceding terms.

---

## 5. Individual Terms vs Superposition

This is the central distinction demonstrated by the project.

### Individual Terms

The left side answers:

> What does each polynomial power look like by itself?

For order 3:

\[
x,\quad x^2,\quad x^3
\]

are separate curves.

### Superposition

The right side answers:

> What happens when the polynomial terms are progressively added together?

For order 3:

\[
x
\]

\[
x+x^2
\]

\[
x+x^2+x^3
\]

are cumulative curves.

Therefore:

```text
INDIVIDUAL TERMS

x
x²
x³
...
x²⁰


SUPERPOSITION

x
x + x²
x + x² + x³
...
x + x² + x³ + ... + x²⁰
```

---

## 6. Mathematical Formulation

The individual basis term of order \(k\) can be written as:

\[
\phi_k(x)=m_kx^k
\]

The cumulative superposition through order \(n\) is:

\[
f_n(x)=\sum_{k=1}^{n}m_kx^k
\]

In the current implementation:

\[
m_k=1
\]

for every \(k\).

Therefore:

\[
f_n(x)=\sum_{k=1}^{n}x^k.
\]

For example:

\[
f_1(x)=x
\]

\[
f_2(x)=x+x^2
\]

\[
f_3(x)=x+x^2+x^3
\]

and:

\[
f_{20}(x)=x+x^2+x^3+\cdots+x^{20}.
\]

---

## 7. Maximum Polynomial Order

The current implementation supports polynomial terms through order 20.

The main configuration is:

```python
MAX_ORDER = 20
```

Therefore, the visualization generates:

\[
x^1,x^2,x^3,\ldots,x^{20}.
\]

The same maximum order controls both the individual-term and superposition calculations.

The value can be changed directly in the Python source.

For example:

```python
MAX_ORDER = 10
```

would restrict the visualization to order 10.

Similarly:

```python
MAX_ORDER = 30
```

would extend the visualization to order 30.

---

## 8. Coefficients

The current experiment intentionally uses:

\[
m_1=m_2=\cdots=m_{20}=1.
\]

This is implemented as:

```python
coefficients = [1.0] * MAX_ORDER
```

The coefficients are explicitly incorporated into the calculation. This makes the implementation suitable for future experiments with different coefficient values.

For example, changing the coefficient of a particular term changes its contribution to both the individual term and the cumulative superposition.

---

## 9. Coordinate System

Both panels use the same visible Cartesian coordinate system.

### X-axis

\[
-2\leq x\leq2
\]

### Y-axis

\[
-5\leq y\leq5
\]

The axis configuration includes:

- Integer tick spacing
- Grid lines
- Zero lines
- Visible axis borders
- x-axis label
- y-axis label

Using the same coordinate ranges allows the two panels to be compared directly.

---

## 10. Curve Evaluation Range

There is an intentional distinction between the visible x-axis range and the range over which the curves are numerically evaluated.

The visible x-axis is:

\[
[-2,2]
\]

but the curves are evaluated over:

\[
[-1,1].
\]

The implementation uses:

```python
X_DRAW_MIN = -1
X_DRAW_MAX = 1
```

while the visible axis uses:

```python
X_AXIS_MIN = -2
X_AXIS_MAX = 2
```

This design is important for high-order polynomial terms.

For example:

\[
2^{10}=1024
\]

and:

\[
2^{20}=1,048,576.
\]

Therefore, high-order powers of \(x\) grow extremely rapidly as \(|x|\) moves beyond 1.

Restricting curve evaluation to \([-1,1]\) keeps the high-order polynomial terms within a visually useful region while retaining additional coordinate context around the curves.

---

## 11. Curve Resolution

The curves are sampled using:

```python
N_POINTS = 1200
```

The x-values are generated with:

```python
x = np.linspace(
    X_DRAW_MIN,
    X_DRAW_MAX,
    N_POINTS
)
```

Each curve is therefore represented using 1200 sample points before being rendered by Plotly.

---

## 12. Interactive Visualization

The generated visualization is a standalone HTML application based on Plotly.

It contains:

```text
                    Polynomial Terms vs Superposition
                                   |
                  +----------------+----------------+
                  |                                 |
                  v                                 v
          Individual Terms                    Superposition
              Plot                                Plot
                  |                                 |
                  +----------------+----------------+
                                   |
                             Shared Order
                              and Controls
```

Both plots are synchronized to the same current polynomial order.

The HTML page contains the two Plotly graphs together with custom HTML, CSS, and JavaScript controls.

---

## 13. Progressive Curve Visibility

All polynomial curves are calculated when the Python program runs.

The interactive interface then controls which curves are visible.

For a current order \(n\), the first \(n\) curves are made visible.

For example:

### Order 1

Left:

\[
x
\]

Right:

\[
x
\]

### Order 3

Left:

\[
x,\quad x^2,\quad x^3
\]

Right:

\[
x,\quad x+x^2,\quad x+x^2+x^3
\]

### Order 20

Left:

\[
x,x^2,\ldots,x^{20}
\]

Right:

\[
x,\;x+x^2,\;\ldots,\;x+x^2+\cdots+x^{20}.
\]

The two panels are therefore synchronized throughout the animation.

---

## 14. Interactive Controls

The HTML interface provides six controls.

### Previous

Moves the visualization back by one order.

Example:

```text
Order 8 → Order 7
```

The button is disabled at order 1.

### Next Order

Moves forward by one order.

Example:

```text
Order 7 → Order 8
```

The button is disabled at order 20.

### Play

Starts automatic progression through the polynomial orders.

The animation proceeds one order at a time.

If the visualization is already at the final order, Play starts the sequence again from order 1.

### Pause

Stops automatic progression while keeping the current order visible.

### Reset

Stops the animation and returns the visualization to:

```text
Order 1 / 20
```

### Speed

The animation speed can be cycled through:

```text
0.5×
1×
2×
4×
```

The corresponding delays are:

| Speed | Delay |
|---:|---:|
| 0.5× | 2000 ms |
| 1× | 1000 ms |
| 2× | 500 ms |
| 4× | 250 ms |

If the animation is currently running, changing the speed updates the active timer.

---

## 15. Dynamic Equation Display

The interface displays the current order and the corresponding equations above the two plots.

For example, at order 3:

```text
Order 3 / 20

Left: y = x^3

Right: y = x + x^2 + x^3
```

The left equation represents the current individual polynomial term.

The right equation represents the cumulative superposition through the current order.

The equations are generated dynamically from the current order.

---

## 16. Synchronization Between the Two Panels

The two panels always represent the same current order.

For example, when the interface is at:

```text
Order 5 / 20
```

the left panel displays the individual terms through \(x^5\), while the right panel displays cumulative superpositions through the fifth order.

### Left

\[
x,\;x^2,\;x^3,\;x^4,\;x^5
\]

### Right

\[
x
\]

\[
x+x^2
\]

\[
x+x^2+x^3
\]

\[
x+x^2+x^3+x^4
\]

\[
x+x^2+x^3+x^4+x^5
\]

A single `currentOrder` state controls both plots.

---

## 17. Responsive Layout

The two plots are arranged side by side using a two-column CSS grid:

```text
+----------------------+----------------------+
|                      |                      |
|   Individual Terms   |    Superposition     |
|                      |                      |
|                      |                      |
+----------------------+----------------------+
```

For narrower screens, the layout automatically changes to a single-column arrangement.

This allows the visualization to remain usable on smaller displays.

---

## 18. Plot Configuration

Each Plotly figure is configured with a nominal size of:

```python
width = 900
height = 760
```

The overall HTML page uses a maximum content width of approximately 1900 pixels.

The figures use a clean white Plotly theme with:

- Grid lines
- Zero lines
- Mirrored axis borders
- Integer tick spacing
- Clear x and y labels
- No legend, since the equation is shown dynamically above the plots

---

## 19. Output

The Python program generates the following HTML file:

```text
polynomial_terms_vs_superposition_1_to_20.html
```

The output path is created using:

```python
Path.cwd()
```

Therefore, the file is saved in the current working directory from which the Python program is executed.

The program then automatically opens the generated HTML file in the system's default web browser.

---

## 20. Standalone HTML

The generated visualization is contained in a single HTML document.

The Plotly library is included once for the first graph and reused by the second graph.

The resulting HTML contains:

- Plotly visualizations
- HTML interface
- CSS styling
- JavaScript animation logic
- Mathematical equation display
- Interactive controls

After the file has been generated, it can be opened directly in a modern web browser.

---

## 21. Requirements

The project requires:

- Python 3.x
- NumPy
- Plotly

Install the required packages with:

```bash
pip install numpy plotly
```

---

## 22. How to Run

Navigate to the project directory:

```bash
cd polynomial-terms-vs-superposition
```

Run the Python program:

```bash
python polynomial_terms_vs_superposition.py
```

The program will:

1. Create the x-values.
2. Generate the individual polynomial terms.
3. Generate the cumulative superposition curves.
4. Generate the corresponding equation labels.
5. Create the left Plotly figure.
6. Create the right Plotly figure.
7. Build the complete HTML interface.
8. Add the interactive JavaScript controls.
9. Save the HTML file.
10. Open the HTML file automatically in the default browser.

---

## 23. Configuration

The main experimental parameters are located near the beginning of the Python file.

### Maximum Order

```python
MAX_ORDER = 20
```

Change this value to control the highest polynomial order.

For example:

```python
MAX_ORDER = 30
```

will extend the visualization to order 30.

---

### Coefficients

The current implementation uses:

```python
coefficients = [1.0] * MAX_ORDER
```

Therefore:

\[
m_1=m_2=\cdots=m_n=1.
\]

Different coefficient values can be introduced for experiments involving weighted polynomial terms.

---

### Visible X-axis

```python
X_AXIS_MIN = -2
X_AXIS_MAX = 2
```

### Visible Y-axis

```python
Y_AXIS_MIN = -5
Y_AXIS_MAX = 5
```

### Curve Evaluation Range

```python
X_DRAW_MIN = -1
X_DRAW_MAX = 1
```

### Curve Resolution

```python
N_POINTS = 1200
```

---

## 24. Example: Extending to Order 30

To extend the visualization from order 20 to order 30, change:

```python
MAX_ORDER = 20
```

to:

```python
MAX_ORDER = 30
```

The coefficient array is automatically resized because it is generated using:

```python
coefficients = [1.0] * MAX_ORDER
```

The visualization will then generate individual terms through:

\[
x^{30}
\]

and cumulative superpositions through:

\[
x+x^2+x^3+\cdots+x^{30}.
\]

No manual modification of the mathematical loops is required.

---

## 25. Educational Interpretation

This visualization can be used to introduce polynomial models from a basis-function perspective.

A learner can first observe the individual functions:

\[
x,\quad x^2,\quad x^3,\quad\ldots
\]

and then observe how these components can be progressively combined:

\[
x+x^2,
\]

\[
x+x^2+x^3,
\]

and so on.

This provides an intuitive transition toward the general polynomial expression:

\[
f(x)=\sum_{k=1}^{n}m_kx^k.
\]

The visualization can therefore serve as a bridge between elementary polynomial functions and topics such as:

- Polynomial regression
- Polynomial feature expansion
- Basis-function representations
- Linear combinations
- Parameterized mathematical models
- Model complexity

---

## 26. Important Mathematical Scope

This project is a **mathematical visualization**, not a polynomial regression training implementation.

The coefficients are fixed rather than learned from data:

\[
m_k=1.
\]

The program does not perform coefficient estimation, optimization, or loss minimization.

Therefore, the visualization demonstrates the **structure and behavior of polynomial terms and their cumulative combination**, rather than the training of a polynomial regression model.

---

## 27. What This Project Does Not Implement

The current implementation does not perform:

- Gradient descent
- Least-squares coefficient estimation
- Polynomial regression training
- Training/test data splitting
- Loss minimization
- Regularization
- Cross-validation
- Bias-variance analysis
- Automated model selection

Its purpose is specifically to visualize the relationship between individual polynomial terms and their cumulative superposition.

---

## 28. Project Structure

Recommended GitHub organization:

```text
polynomial-terms-vs-superposition/
│
├── polynomial_terms_vs_superposition.py
└── README.md
```

After running the Python program locally:

```text
polynomial-terms-vs-superposition/
│
├── polynomial_terms_vs_superposition.py
├── polynomial_terms_vs_superposition_1_to_20.html
└── README.md
```

The HTML file is a generated artifact.

It does not need to be committed to the repository unless the interactive visualization itself is intentionally being published as part of the repository.

---

## 29. Technologies

| Technology | Purpose |
|---|---|
| Python | Main implementation |
| NumPy | Numerical computation |
| Plotly | Interactive graphing |
| HTML | Standalone visualization |
| CSS | Page layout and styling |
| JavaScript | Animation and interactive controls |

---

## 30. Key Features

- Two synchronized polynomial visualizations
- Individual polynomial terms
- Cumulative polynomial superposition
- Polynomial orders 1–20
- Configurable maximum order
- All coefficients initially set to 1
- Cartesian coordinate system
- Visible x-axis from -2 to +2
- Visible y-axis from -5 to +5
- Curve evaluation from -1 to +1
- 1200 points per curve
- Dynamic equation display
- Previous-order navigation
- Next-order navigation
- Play control
- Pause control
- Reset control
- 0.5×, 1×, 2× and 4× animation speeds
- Synchronized left and right panels
- Responsive layout
- Standalone HTML output
- Automatic browser launch

---

## 31. Intended Use

This project is intended for educational and demonstration purposes, particularly for:

- University lectures
- Machine Learning courses
- Data Science courses
- Mathematical demonstrations
- Polynomial regression introductions
- Basis-function explanations
- Interactive classroom teaching
- Self-learning

The visualization is particularly useful when a static polynomial equation is not sufficient to communicate how individual terms contribute to a multi-term polynomial expression.

---

## 32. Author

**Dr. Prasenjit Dey**

Part of the **DL Animations** collection of interactive visualizations for Machine Learning and Deep Learning education.

---

## 33. License

Please refer to the license of the parent repository for the applicable terms of use, modification, and distribution.
