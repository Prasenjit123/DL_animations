# Interactive Polynomial Curve Fitting

An interactive visualization of polynomial curve fitting designed to
demonstrate how increasing model complexity affects the fitting of noisy
observations.

The project provides **two outputs**:

1. A standalone interactive HTML animation with a polynomial-degree slider.
2. A high-quality MP4 video showing the progression from degree 1 to
   degree 30.

The visualization is intended as an educational tool for explaining
polynomial regression, model complexity, underfitting, overfitting, and
training error.

---

## 1. Concept

Suppose we observe data generated from an unknown relationship.

In this visualization, the underlying true function is:

\[
y = \sin(x)
\]

However, the observations contain random noise.

The objective is to fit polynomial models of increasing degree to these
noisy observations.

A polynomial model of degree \(d\) can be written as:

\[
\hat{y}
=
w_0 + w_1x + w_2x^2 + \cdots + w_dx^d
\]

The project demonstrates what happens as the degree of the polynomial
increases.

---

## 2. Data Generation

The visualization generates **50 observations** between \(0\) and
\(2\pi\).

The true function is:

\[
y = \sin(x)
\]

Random Gaussian noise is then added to the true function:

\[
y_{\text{observed}}
=
\sin(x) + \epsilon
\]

where:

\[
\epsilon \sim \mathcal{N}(0,\sigma^2)
\]

The current noise level is:

```text
Noise standard deviation = 0.50
