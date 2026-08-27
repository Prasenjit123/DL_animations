# Momentum Gradient Descent: Epoch-by-Epoch Sigmoid Animation

<p align="center">
  <strong>A visual demonstration of Gradient Descent with Momentum on a single sigmoid unit.</strong>
</p>

<p align="center">
  Watch the sigmoid curve, model parameters, and Binary Cross-Entropy loss evolve together, one optimization step at a time.
</p>

---

## Overview

Gradient Descent is usually introduced through an update equation and a final converged solution. That is mathematically correct, but it can hide an important part of optimization:

> **The optimizer does not jump directly to the solution. It follows a trajectory through parameter space.**

This project makes that trajectory visible.

A single sigmoid unit is trained on two observations whose target values are both `0.5`. Gradient Descent with Momentum starts from a deliberately non-optimal parameter setting and updates the weight `w` and bias `b` for 80 optimization steps.

The generated animation synchronizes three views:

```text
Sigmoid curve
      +
Parameter trajectories
      +
Binary Cross-Entropy loss
```

This makes it possible to see how the parameter updates change the model output and how the loss responds to those changes.

---

## What the experiment demonstrates

The main educational focus is the behavior of **Gradient Descent with Momentum**.

The visualization makes several effects visible:

- movement toward a lower-loss region,
- momentum carried from previous updates,
- overshooting,
- oscillation around the low-loss region,
- gradual reduction in the size of the oscillations,
- movement toward the parameter solution that produces predictions close to `0.5`.

The exact trajectory depends on the learning rate, momentum coefficient, starting parameters, and dataset.

This project uses a deliberately chosen configuration that makes the dynamics visually noticeable.

---

## Model

The model contains a single sigmoid unit:

```text
z = wx + b

ŷ = sigmoid(z)

sigmoid(z) = 1 / (1 + exp(-z))
```

where:

- `x` is the input,
- `w` is the weight,
- `b` is the bias,
- `z` is the pre-activation value,
- `ŷ` is the predicted output.

The implementation clips the sigmoid input to the range `[-50, 50]` before evaluating the exponential. This prevents extreme exponent values from causing numerical overflow during the visualization.

---

## Dataset

The experiment uses exactly two observations:

| Input `x` | Target `y` |
|---:|---:|
| `0.35` | `0.50` |
| `3.50` | `0.50` |

Both target values are `0.5`.

For a sigmoid function, an output of exactly `0.5` occurs when its input is zero:

```text
sigmoid(0) = 0.5
```

Therefore, for both observations to produce `0.5` simultaneously:

```text
0.35w + b = 0
3.50w + b = 0
```

which gives:

```text
w = 0
b = 0
```

Thus `(w, b) = (0, 0)` is the parameter solution corresponding to perfect predictions of `0.5` for both training points.

The animation starts away from this solution so that the optimization trajectory can be observed.

---

## Loss function

The project uses **Binary Cross-Entropy (BCE)**:

```text
J = -(1/N) × Σ [ yᵢ log(ŷᵢ) + (1 − yᵢ) log(1 − ŷᵢ) ]
```

For the two observations in this experiment, the minimum possible BCE occurs when both predictions are `0.5`.

At that point:

```text
BCE = -log(0.5) ≈ 0.693147
```

The implementation adds a small numerical constant `1e-12` inside the logarithms to avoid evaluating `log(0)`.

---

## Gradient Descent with Momentum

The implementation uses the following momentum formulation.

First, the gradients are calculated:

```text
dw = mean((ŷ − y) × x)

db = mean(ŷ − y)
```

The velocity variables are then updated:

```text
velocity_w = β × velocity_w + dw

velocity_b = β × velocity_b + db
```

Finally, the model parameters are updated:

```text
w = w − η × velocity_w

b = b − η × velocity_b
```

where:

- `η` is the learning rate,
- `β` is the momentum coefficient,
- `velocity_w` is the accumulated update for `w`,
- `velocity_b` is the accumulated update for `b`.

The important difference from ordinary Gradient Descent is the velocity term.

Instead of responding only to the current gradient, the update also retains part of the previous velocity.

Conceptually:

```text
Current gradient
       +
Previous update direction
       ↓
Momentum
       ↓
Parameter update
```

---

## Configuration

The current source code uses:

| Parameter | Value |
|---|---:|
| Learning rate | `1.0` |
| Momentum coefficient | `0.90` |
| Number of optimization steps | `80` |
| Initial weight `w` | `4.0` |
| Initial bias `b` | `-6.0` |
| Initial velocity `w` | `0.0` |
| Initial velocity `b` | `0.0` |

These values are explicitly defined in the Python program.

The relatively large learning rate and momentum coefficient are part of the demonstration design. They produce visible movement and oscillation rather than an almost imperceptible trajectory.

---

## What happens during optimization?

The parameter trajectory is not monotonic.

The optimizer can move past the low-loss region and then return toward it because momentum carries information from earlier updates.

A simplified conceptual picture is:

```text
Start
  ↓
Large update
  ↓
Rapid movement
  ↓
Overshoot
  ↓
Reverse direction
  ↓
Oscillation
  ↓
Smaller oscillations
  ↓
Movement toward the low-loss region
```

The actual numerical trajectory is generated directly by the Python implementation.

It is therefore more accurate to describe the animation as **illustrating oscillatory momentum dynamics and movement toward the optimum**, rather than claiming that every step monotonically reduces the loss or that the 80-step run reaches exact convergence.

---

## What the animation contains

The figure is divided into three synchronized views.

```text
┌──────────────────────────────────┬──────────────────────┐
│                                  │ Parameter trajectories│
│          Sigmoid curve            ├──────────────────────┤
│                                  │ Loss convergence      │
│                                  │                      │
└──────────────────────────────────┴──────────────────────┘
```

### 1. Sigmoid curve

The large left panel displays the current sigmoid function over:

```text
x ∈ [-6, 6]
```

with output limits:

```text
y ∈ [-0.05, 1.05]
```

The two training observations remain fixed at:

```text
(0.35, 0.5)
(3.50, 0.5)
```

The current model predictions at those inputs are shown as points.

A horizontal reference line at `y = 0.5` is also displayed.

---

### 2. Parameter trajectories

The upper-right panel tracks:

```text
w
b
```

against optimization step.

The plot shows both complete trajectories and the current parameter values.

This allows the viewer to connect the mathematical update rule with the actual movement of the parameters.

---

### 3. Loss

The lower-right panel displays Binary Cross-Entropy against optimization step.

The current loss is marked as the animation progresses.

The loss does not have to decrease at every individual step when momentum and the selected learning rate produce an oscillatory trajectory.

That behavior is part of what this visualization is designed to make visible.

---

## Sigmoid history trail

The main sigmoid panel retains a short history of previous curves.

The implementation displays up to eight previous sigmoid curves as a visual trail.

This provides a direct view of how the model function moves from one optimization state to the next.

The trail is not an additional optimization method; it is purely a visualization device.

---

## Epoch and frame interpretation

The Python program stores the initial state before performing any updates.

Then it performs:

```text
80 optimization updates
```

As a result, the stored histories contain:

```text
1 initial state + 80 updated states = 81 states
```

The animation therefore contains 81 frames:

```text
Frame 0  → initial parameters
Frame 1  → after update 1
Frame 2  → after update 2
...
Frame 80 → after update 80
```

The animation interval is:

```text
1000 ms per frame
```

and the GIF is written with:

```text
1 frame per second
```

Therefore, the generated GIF is approximately 81 seconds long, subject to the exact behavior of the GIF encoder and playback software.

---

## Animation output

The Python program creates:

```text
momentum_sigmoid_animation.gif
```

The output path is determined automatically.

When the script is executed as a normal Python file, the GIF is saved in the same directory as the Python script.

When the code is executed in an interactive environment where `__file__` is unavailable, the program falls back to the current working directory.

The program prints the output location and verifies whether the generated file exists.

---

## Project structure

The repository currently contains:

```text
gradient-descent/
│
├── momentum_sigmoid_animation.py
├── momentum_sigmoid_animation.gif
└── README.md
```

### `momentum_sigmoid_animation.py`

The complete Python implementation.

It:

1. defines the dataset,
2. defines the sigmoid function,
3. defines Binary Cross-Entropy,
4. initializes the model,
5. performs Gradient Descent with Momentum,
6. stores parameter, prediction, and loss histories,
7. constructs the three-panel visualization,
8. creates the animation,
9. saves the GIF,
10. displays the animation window.

### `momentum_sigmoid_animation.gif`

The generated epoch-by-epoch animation.

### `README.md`

Project documentation.

---

## Requirements

The Python program imports:

```python
numpy
matplotlib
pillow
```

`pathlib` is part of the Python standard library.

Install the required external packages with:

```bash
pip install numpy matplotlib pillow
```

Python 3 is required.

---

## How to run

Clone the repository:

```bash
git clone https://github.com/Prasenjit123/DL_animations.git
```

Move into the project directory:

```bash
cd DL_animations/gradient-descent
```

Install the dependencies:

```bash
pip install numpy matplotlib pillow
```

Run the program:

```bash
python momentum_sigmoid_animation.py
```

The program will:

1. initialize the dataset,
2. initialize `w = 4.0` and `b = -6.0`,
3. run 80 momentum-based optimization updates,
4. store the optimization history,
5. create the animated figure,
6. save the GIF as `momentum_sigmoid_animation.gif`,
7. display the animation using Matplotlib.

---

## Output

The generated file is:

```text
momentum_sigmoid_animation.gif
```

The script saves it in the directory determined by:

```text
Directory containing the Python file
```

when executed normally.

The program also prints the complete output path and whether the file exists after saving.

---

## Numerical behavior of the current configuration

The target solution is:

```text
w = 0
b = 0
```

However, the script runs for a fixed 80 optimization updates rather than iterating until a convergence tolerance is reached.

Therefore, the correct interpretation of the final frame is:

> The optimizer has moved toward the low-loss solution and the oscillations have become much smaller, but the final state after 80 updates should not be interpreted as mathematically exact convergence.

This distinction is important for teaching optimization.

A finite number of optimization steps is not the same thing as a formal convergence criterion.

---

## Why the loss can temporarily increase

A common misconception is:

> "Gradient Descent must reduce the loss at every iteration."

That is not generally true for the momentum configuration used here.

The current update contains a velocity term:

```text
velocityₜ = β × velocityₜ₋₁ + gradientₜ
```

so the optimizer can retain movement from previous iterations.

With a sufficiently aggressive learning rate and momentum coefficient, the parameter update can carry the model past a lower-loss region.

The result can be:

```text
Loss decreases
      ↓
Momentum carries the parameters further
      ↓
Loss increases temporarily
      ↓
Direction changes
      ↓
Loss decreases again
```

The visualization makes this behavior directly observable.

---

## Why both `w` and `b` are important

The sigmoid input is:

```text
z = wx + b
```

so both parameters affect the location and shape of the sigmoid transition.

In this particular experiment, the target is `0.5` at both input locations.

The desired condition is therefore:

```text
wx + b = 0
```

at both:

```text
x = 0.35
x = 3.50
```

which forces:

```text
w = 0
b = 0
```

The parameter plot therefore provides an unusually simple setting for visualizing how two parameters jointly move toward a known solution.

---

## Educational use

This visualization is particularly useful for introducing:

- Gradient Descent
- Gradient Descent with Momentum
- Learning rate
- Momentum coefficient
- Velocity
- Parameter updates
- Sigmoid activation
- Binary Cross-Entropy
- Optimization trajectories
- Overshooting
- Oscillation
- Damped oscillation
- Finite-step optimization
- Convergence concepts

It is best used as a teaching visualization rather than as a benchmark for optimizer performance.

---

## Suggested teaching sequence

### Step 1 — Start with the model

Introduce:

```text
ŷ = sigmoid(wx + b)
```

Explain the roles of `w` and `b`.

### Step 2 — Examine the data

Show that both target values are `0.5`.

Ask:

> What sigmoid input produces an output of 0.5?

This leads to:

```text
z = 0
```

### Step 3 — Derive the target parameter solution

For both observations:

```text
0.35w + b = 0
3.50w + b = 0
```

Therefore:

```text
w = 0
b = 0
```

### Step 4 — Introduce the initial state

Start from:

```text
w = 4
b = -6
```

The model is intentionally far from the target solution.

### Step 5 — Introduce the gradient

Explain how the gradients determine the local descent direction.

### Step 6 — Add momentum

Explain that momentum accumulates the previous update direction.

### Step 7 — Play the animation

Ask students to watch all three panels simultaneously:

```text
Sigmoid curve
     ↕
Parameters
     ↕
Loss
```

### Step 8 — Discuss oscillation

Point out that the parameter path and loss need not move monotonically.

### Step 9 — Discuss convergence

Explain the difference between:

```text
moving toward an optimum
```

and:

```text
meeting a formal convergence criterion
```

---

## Important implementation detail

The project uses the standard sigmoid-plus-BCE gradient simplification:

```text
dw = mean((ŷ − y) × x)

db = mean(ŷ − y)
```

This follows from combining the derivative of Binary Cross-Entropy with the derivative of the sigmoid activation.

The implementation therefore does not numerically approximate the gradients using finite differences.

---

## Numerical safeguards

Two small numerical safeguards are used.

### Sigmoid clipping

Before evaluating the sigmoid, the input is clipped:

```text
z ∈ [-50, 50]
```

This prevents extreme exponential values from causing numerical overflow.

### Logarithm safeguard

The BCE calculation uses:

```text
ε = 1e-12
```

inside the logarithms:

```text
log(ŷ + ε)

log(1 − ŷ + ε)
```

This prevents taking the logarithm of exactly zero.

These safeguards are numerical implementation details and do not change the conceptual optimization procedure.

---

## Limitations

The current demonstration is intentionally small.

It uses:

- only two training observations,
- one sigmoid unit,
- one weight,
- one bias,
- one fixed learning rate,
- one fixed momentum coefficient,
- a fixed 80-step run,
- Binary Cross-Entropy as the loss,
- no train/test split,
- no validation data,
- no optimizer comparison.

The purpose is to isolate the behavior of momentum rather than provide a general-purpose optimization framework.

---

## Possible extensions

The current visualization could be extended in several directions.

### Compare ordinary Gradient Descent and Momentum

Run both methods from the same initial parameters and compare their trajectories.

### Interactive learning rate

Allow the user to change `η` and observe how aggressive updates affect the trajectory.

### Interactive momentum

Allow the user to change `β` and observe the effect of accumulated velocity.

### Convergence tolerance

Replace the fixed number of updates with a convergence criterion such as:

```text
|Jₜ − Jₜ₋₁| < tolerance
```

or a gradient-norm criterion.

### Parameter-space visualization

Plot the trajectory directly in `(w, b)` space.

### Loss surface

Visualize the Binary Cross-Entropy as a function of `w` and `b` and overlay the optimization path.

### Optimizer comparison

Compare:

```text
Gradient Descent
Momentum
Nesterov Momentum
AdaGrad
RMSProp
Adam
```

using the same objective and initialization.

### Multiple datasets

Use more observations to demonstrate that the same optimization principles extend beyond this two-point example.

---

## Technical summary

| Component | Current implementation |
|---|---|
| Model | Single sigmoid unit |
| Parameters | `w`, `b` |
| Training samples | 2 |
| Targets | Both `0.5` |
| Activation | Sigmoid |
| Loss | Binary Cross-Entropy |
| Optimizer | Gradient Descent with Momentum |
| Learning rate | `1.0` |
| Momentum | `0.90` |
| Initial `w` | `4.0` |
| Initial `b` | `-6.0` |
| Optimization updates | `80` |
| Stored animation states | `81` |
| Sigmoid plot range | `[-6, 6]` |
| Sigmoid output range | `[-0.05, 1.05]` |
| Previous-curve trail | Up to 8 curves |
| Animation interval | `1000 ms` |
| GIF frame rate | `1 fps` |
| Output format | GIF |
| Visualization | Matplotlib |
| Numerical computation | NumPy |
| GIF writer | Pillow |

---

## Key takeaway

The most important lesson is not simply that momentum can make optimization faster.

The deeper idea is:

```text
Gradient
   ↓
Current descent direction

Momentum
   ↓
Retains information from previous updates

Gradient + Momentum
   ↓
A different optimization trajectory
```

That trajectory can overshoot, oscillate, and then settle toward a low-loss region.

In this particular example, the ideal parameter solution is:

```text
w = 0
b = 0
```

which produces:

```text
sigmoid(wx + b) = 0.5
```

for both training inputs.

The animation makes the path toward that solution visible rather than showing only the final parameter values.

---

## Author

**Dr. Prasenjit Dey**

Part of the **DL Animations** collection of interactive visualizations for Machine Learning and Deep Learning education.

---

## License

No explicit open-source license file is currently present in the `gradient-descent` project directory.

If this repository is intended for public redistribution, add a `LICENSE` file and state the applicable terms here.

