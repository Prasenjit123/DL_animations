# Momentum Gradient Descent: Epoch-by-Epoch Sigmoid Animation

An educational Python visualization of **Gradient Descent with Momentum** applied to a sigmoid function.

This project is designed to make the behavior of momentum-based optimization visually intuitive by showing, **epoch by epoch**, how the sigmoid curve changes, how the parameters oscillate because of momentum, and how the optimization eventually converges.

---

## 🎯 Project Objective

The main objective is to visualize the following optimization behavior:

\[
\text{Initial State}
\rightarrow
\text{Momentum}
\rightarrow
\text{Overshooting}
\rightarrow
\text{Oscillation}
\rightarrow
\text{Damped Oscillation}
\rightarrow
\text{Convergence}
\]

Instead of showing only the final result, the animation allows us to observe the **complete optimization trajectory**.

---

## 📊 Dataset

The demonstration uses two data points:

| Input \(x\) | Target \(y\) |
|---:|---:|
| 0.35 | 0.5 |
| 3.50 | 0.5 |

The model is a single sigmoid unit:

\[
\hat{y} = \sigma(wx+b)
\]

where

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

Since both target values are \(0.5\), the desired final predictions are:

\[
\hat{y}_1 \rightarrow 0.5
\]

and

\[
\hat{y}_2 \rightarrow 0.5
\]

For both points to simultaneously produce \(0.5\), the solution is:

\[
\boxed{w\rightarrow0,\quad b\rightarrow0}
\]

---

## 🧠 Optimization Method

The project uses **Gradient Descent with Momentum**.

The velocity update is:

\[
v_t=\beta v_{t-1}+\nabla J(\theta_t)
\]

and the parameter update is:

\[
\theta_{t+1}
=
\theta_t-\eta v_t
\]

where:

- \(\eta\) = learning rate
- \(\beta\) = momentum coefficient
- \(v_t\) = accumulated velocity
- \(\nabla J(\theta_t)\) = current gradient

### Parameters used

```text
Learning rate = 1.0
Momentum      = 0.90
Initial w     = 4.0
Initial b     = -6.0
Epochs        = 80
```

These settings are chosen to make the optimization dynamics visually noticeable.

---

## 🎬 What the Animation Shows

The visualization contains three synchronized views.

### 1. Sigmoid Curve

The left panel shows the sigmoid function:

\[
\sigma(wx+b)
\]

changing after every optimization step.

The two target points remain fixed at:

\[
y=0.5
\]

The animation allows you to observe the sigmoid becoming flatter as \(w\) approaches zero.

A short trail of previous sigmoid curves is also displayed so that the movement of the curve is easy to see.

---

### 2. Parameter Oscillation

The upper-right panel displays the trajectories of:

\[
w
\]

and

\[
b
\]

over the epochs.

Because momentum retains information from previous updates, the parameters can overshoot the optimum.

The trajectory therefore illustrates:

```text
Current gradient
      ↓
Accumulated momentum
      ↓
Overshooting
      ↓
Oscillation
      ↓
Damped oscillation
      ↓
Convergence
```

---

### 3. Loss Convergence

The lower-right panel displays the Binary Cross-Entropy loss over the optimization process.

As the model predictions approach the target value \(0.5\), the loss decreases toward its minimum.

---

## ⏱️ Epoch-by-Epoch Animation

The animation intentionally runs at:

\[
\boxed{1\text{ second per epoch}}
\]

Therefore, the visualization proceeds as:

```text
Epoch 0
   ↓
1 second
   ↓
Epoch 1
   ↓
1 second
   ↓
Epoch 2
   ↓
1 second
   ↓
Epoch 3
   ↓
...
```

This slow progression is intentional. It allows the viewer to clearly observe the change in the sigmoid curve after each optimization step.

---

## 📁 Project Structure

A simple project structure is sufficient:

```text
Momentum-Gradient-Descent/
│
├── momentum_sigmoid_animation.py
├── momentum_sigmoid_animation.gif
└── README.md
```

### Files

**`momentum_sigmoid_animation.py`**

Main Python program that performs the optimization and generates the animation.

**`momentum_sigmoid_animation.gif`**

Generated epoch-by-epoch visualization.

**`README.md`**

Project documentation.

---

## ⚙️ Requirements

Python 3.8 or later is recommended.

Required libraries:

```text
numpy
matplotlib
pillow
```

Install them using:

```bash
pip install numpy matplotlib pillow
```

---

## ▶️ How to Run

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

Move into the project directory:

```bash
cd YOUR_REPOSITORY
```

Install the dependencies:

```bash
pip install numpy matplotlib pillow
```

Run the Python program:

```bash
python momentum_sigmoid_animation.py
```

The program will:

1. Initialize the sigmoid model.
2. Run Gradient Descent with Momentum.
3. Store the parameters after every epoch.
4. Create the epoch-by-epoch animation.
5. Save the animation as a GIF.
6. Display the live animation.

---

## 💾 Output

The generated animation is saved as:

```text
momentum_sigmoid_animation.gif
```

The program automatically determines the output location and prints the complete file path in the terminal.

Example:

```text
Animation saved successfully.

Full path:
C:\...\Momentum-Gradient-Descent\momentum_sigmoid_animation.gif

File exists: True
```

---

## 🔬 Mathematical Details

For the sigmoid model:

\[
z=wx+b
\]

\[
\hat{y}=\sigma(z)
\]

The Binary Cross-Entropy loss is:

\[
J=
-\frac{1}{n}
\sum_{i=1}^{n}
\left[
y_i\log(\hat y_i)
+
(1-y_i)\log(1-\hat y_i)
\right]
\]

For sigmoid activation combined with Binary Cross-Entropy:

\[
\frac{\partial J}{\partial w}
=
\frac{1}{n}
\sum_i
(\hat y_i-y_i)x_i
\]

and

\[
\frac{\partial J}{\partial b}
=
\frac{1}{n}
\sum_i
(\hat y_i-y_i)
\]

Momentum then modifies the optimization trajectory.

---

## 💡 Why Momentum Is Interesting

Ordinary Gradient Descent uses only the current gradient:

\[
\theta_{t+1}
=
\theta_t-\eta\nabla J(\theta_t)
\]

Momentum introduces a velocity term:

\[
v_t
=
\beta v_{t-1}
+
\nabla J(\theta_t)
\]

Therefore, the optimizer has a form of **memory**.

This can cause the optimizer to continue moving in a direction even after the current gradient has changed.

Consequently, the parameters may:

- move rapidly toward the optimum,
- overshoot it,
- move back,
- overshoot again,
- gradually reduce their oscillations,
- and eventually converge.

This is the primary phenomenon that this visualization is designed to demonstrate.

---

## 🎓 Educational Use

This project is particularly useful for explaining:

- Gradient Descent
- Gradient Descent with Momentum
- Learning rate
- Momentum coefficient
- Parameter updates
- Overshooting
- Oscillations
- Convergence
- Sigmoid activation
- Binary Cross-Entropy
- Optimization dynamics

The animation is intended as a **teaching visualization**, rather than as a benchmark for optimization performance.

---

## 📌 Key Takeaway

The most important concept demonstrated by this project is:

> **Momentum changes the path taken by Gradient Descent, not the objective being optimized.**

The optimizer can overshoot the optimum because it carries information from previous updates. With appropriate learning-rate and momentum settings, these oscillations can gradually decay and the parameters converge.

In this example:

\[
\boxed{
w\rightarrow0,\qquad
b\rightarrow0
}
\]

and therefore:

\[
\boxed{
\sigma(wx+b)\rightarrow0.5
}
\]

for both training points.

---

## 📜 License

This project is intended for educational and academic use.

You may modify, extend, and reuse the code for teaching, learning, demonstrations, and research with appropriate attribution.
