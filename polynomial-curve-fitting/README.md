# Polynomial Curve Fitting

An interactive visualization for demonstrating **polynomial curve fitting
and the effect of increasing model complexity on noisy observations**.

The program generates noisy observations from a known sine function,
fits polynomial models of increasing degree, and visualizes how the fitted
curve changes from a simple model to increasingly complex models.

Two outputs are generated:

- An interactive HTML visualization with a polynomial-degree slider.
- A high-quality MP4 video showing the progression from degree 1 to
  degree 30.

---

## Overview

The underlying relationship used in this demonstration is:

\[
y = \sin(x)
\]

A finite set of observations is sampled from this function and Gaussian
noise is added to simulate observed data.

Polynomial models are then fitted to the noisy observations:

\[
\hat{y}
=
w_0 + w_1x + w_2x^2 + \cdots + w_dx^d
\]

where \(d\) is the polynomial degree.

The program fits models from:

\[
d=1
\]

through:

\[
d=30
\]

and allows the effect of increasing polynomial degree to be observed
visually.

---

## What the Visualization Shows

The visualization contains three main elements:

1. **True function**  
   The original \(y=\sin(x)\) function.

2. **Noisy observations**  
   The sampled observations after Gaussian noise has been added.

3. **Polynomial fit**  
   The polynomial fitted to the noisy observations for the currently
   selected degree.

As the polynomial degree increases, the model becomes increasingly
flexible.

This provides a visual demonstration of the progression:

```text
Low-degree model
       ↓
Limited flexibility
       ↓
Underfitting

Increasing degree
       ↓
Greater flexibility
       ↓
Improved fit to observations

High-degree model
       ↓
Very high flexibility
       ↓
May begin fitting noise
       ↓
Potential overfitting
