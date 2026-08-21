import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path


# ============================================================
# 1. FIND THE CORRECT OUTPUT FOLDER
# ============================================================

# If running a normal .py file:
# save the animation in the same folder as the Python file.

try:
    OUTPUT_FOLDER = Path(__file__).resolve().parent

except NameError:
    # If running in Jupyter / interactive Python:
    # save in the current working folder.
    OUTPUT_FOLDER = Path.cwd()


OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_FOLDER
    /
    "momentum_sigmoid_animation.gif"
)


print()
print("=" * 65)
print("ANIMATION OUTPUT LOCATION")
print("=" * 65)
print(f"Folder : {OUTPUT_FOLDER}")
print(f"File   : {OUTPUT_FILE}")
print("=" * 65)
print()


# ============================================================
# 2. DATA
# ============================================================

X = np.array([
    0.35,
    3.50
])

Y = np.array([
    0.50,
    0.50
])


# ============================================================
# 3. SIGMOID
# ============================================================

def sigmoid(z):

    z = np.clip(
        z,
        -50,
        50
    )

    return 1.0 / (
        1.0 + np.exp(-z)
    )


# ============================================================
# 4. BINARY CROSS-ENTROPY
# ============================================================

def binary_cross_entropy(w, b):

    z = w * X + b

    predictions = sigmoid(z)

    eps = 1e-12

    loss = -np.mean(
        Y * np.log(predictions + eps)
        +
        (1 - Y)
        *
        np.log(1 - predictions + eps)
    )

    return loss


# ============================================================
# 5. MOMENTUM PARAMETERS
# ============================================================

learning_rate = 1.0

momentum = 0.90

number_of_epochs = 80


# ============================================================
# 6. INITIAL PARAMETERS
# ============================================================

w = 4.0

b = -6.0

velocity_w = 0.0

velocity_b = 0.0


# ============================================================
# 7. STORE INITIAL STATE
# ============================================================

w_history = [
    w
]

b_history = [
    b
]

loss_history = [
    binary_cross_entropy(
        w,
        b
    )
]

prediction_history = [
    sigmoid(
        w * X + b
    )
]


# ============================================================
# 8. MOMENTUM GRADIENT DESCENT
# ============================================================

for epoch in range(
    number_of_epochs
):

    # --------------------------------------------------------
    # Forward propagation
    # --------------------------------------------------------

    z = (
        w * X
        +
        b
    )

    predictions = sigmoid(z)


    # --------------------------------------------------------
    # Gradients
    # --------------------------------------------------------

    dw = np.mean(
        (predictions - Y)
        *
        X
    )

    db = np.mean(
        predictions - Y
    )


    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    velocity_w = (
        momentum
        *
        velocity_w
        +
        dw
    )

    velocity_b = (
        momentum
        *
        velocity_b
        +
        db
    )


    # --------------------------------------------------------
    # Parameter update
    # --------------------------------------------------------

    w = (
        w
        -
        learning_rate
        *
        velocity_w
    )

    b = (
        b
        -
        learning_rate
        *
        velocity_b
    )


    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    w_history.append(w)

    b_history.append(b)

    loss_history.append(
        binary_cross_entropy(
            w,
            b
        )
    )

    prediction_history.append(
        sigmoid(
            w * X + b
        )
    )


# Convert to NumPy arrays

w_history = np.array(
    w_history
)

b_history = np.array(
    b_history
)

loss_history = np.array(
    loss_history
)

prediction_history = np.array(
    prediction_history
)


# ============================================================
# 9. PRINT RESULTS
# ============================================================

print()
print("=" * 65)
print("TRAINING RESULT")
print("=" * 65)

print(
    f"Initial w = "
    f"{w_history[0]:.6f}"
)

print(
    f"Initial b = "
    f"{b_history[0]:.6f}"
)

print()

print(
    f"Final w = "
    f"{w_history[-1]:.6f}"
)

print(
    f"Final b = "
    f"{b_history[-1]:.6f}"
)

print()

print(
    f"Final prediction at x=0.35 = "
    f"{prediction_history[-1, 0]:.6f}"
)

print(
    f"Final prediction at x=3.50 = "
    f"{prediction_history[-1, 1]:.6f}"
)

print()

print(
    f"Final loss = "
    f"{loss_history[-1]:.8f}"
)

print("=" * 65)
print()


# ============================================================
# 10. CREATE FIGURE
# ============================================================

fig = plt.figure(
    figsize=(15, 9)
)

grid = fig.add_gridspec(
    2,
    2,
    width_ratios=[1.55, 1],
    height_ratios=[1, 1]
)


ax_sigmoid = fig.add_subplot(
    grid[:, 0]
)

ax_parameter = fig.add_subplot(
    grid[0, 1]
)

ax_loss = fig.add_subplot(
    grid[1, 1]
)


# ============================================================
# 11. SIGMOID PLOT
# ============================================================

x_plot = np.linspace(
    -6,
    6,
    600
)


ax_sigmoid.set_xlim(
    -6,
    6
)

ax_sigmoid.set_ylim(
    -0.05,
    1.05
)

ax_sigmoid.set_xlabel(
    "Input x",
    fontsize=12
)

ax_sigmoid.set_ylabel(
    "Sigmoid output",
    fontsize=12
)

ax_sigmoid.set_title(
    "Sigmoid Curve: Epoch-by-Epoch",
    fontsize=14,
    fontweight="bold"
)

ax_sigmoid.grid(
    True,
    alpha=0.25
)


# Target points

ax_sigmoid.scatter(
    X,
    Y,
    s=120,
    zorder=10,
    label="Target = 0.5"
)


# Target line

ax_sigmoid.axhline(
    0.5,
    linestyle="--",
    linewidth=1.5,
    label="Target y = 0.5"
)


# ------------------------------------------------------------
# Previous sigmoid curves
# ------------------------------------------------------------

number_of_trails = 8

trail_lines = []

for i in range(
    number_of_trails
):

    line, = ax_sigmoid.plot(
        [],
        [],
        linewidth=1
    )

    trail_lines.append(
        line
    )


# Current sigmoid

current_sigmoid, = ax_sigmoid.plot(
    [],
    [],
    linewidth=3,
    label="Current sigmoid"
)


# Prediction points

prediction_points = ax_sigmoid.scatter(
    X,
    prediction_history[0],
    s=110,
    zorder=12
)


# Information box

info_text = ax_sigmoid.text(
    -5.65,
    0.94,
    "",
    fontsize=11,
    verticalalignment="top",
    bbox=dict(
        boxstyle="round,pad=0.5",
        alpha=0.9
    )
)


ax_sigmoid.legend(
    loc="lower right"
)


# ============================================================
# 12. PARAMETER PLOT
# ============================================================

ax_parameter.set_xlim(
    0,
    number_of_epochs
)


parameter_min = min(
    w_history.min(),
    b_history.min()
)

parameter_max = max(
    w_history.max(),
    b_history.max()
)

parameter_range = (
    parameter_max
    -
    parameter_min
)


if parameter_range == 0:

    parameter_range = 1


margin = (
    0.15
    *
    parameter_range
)


ax_parameter.set_ylim(
    parameter_min - margin,
    parameter_max + margin
)


ax_parameter.set_xlabel(
    "Epoch"
)

ax_parameter.set_ylabel(
    "Parameter value"
)

ax_parameter.set_title(
    "Momentum: Oscillation and Convergence",
    fontsize=13,
    fontweight="bold"
)

ax_parameter.grid(
    True,
    alpha=0.25
)


# Zero line

ax_parameter.axhline(
    0,
    linestyle="--",
    linewidth=1.5
)


# w

w_line, = ax_parameter.plot(
    [],
    [],
    linewidth=2,
    label="w"
)


# b

b_line, = ax_parameter.plot(
    [],
    [],
    linewidth=2,
    label="b"
)


# Current w

w_point, = ax_parameter.plot(
    [],
    [],
    marker="o",
    markersize=8
)


# Current b

b_point, = ax_parameter.plot(
    [],
    [],
    marker="o",
    markersize=8
)


ax_parameter.legend()


# ============================================================
# 13. LOSS PLOT
# ============================================================

ax_loss.set_xlim(
    0,
    number_of_epochs
)

ax_loss.set_ylim(
    0,
    max(loss_history) * 1.10
)

ax_loss.set_xlabel(
    "Epoch"
)

ax_loss.set_ylabel(
    "Binary Cross-Entropy"
)

ax_loss.set_title(
    "Loss Convergence",
    fontsize=13,
    fontweight="bold"
)

ax_loss.grid(
    True,
    alpha=0.25
)


loss_line, = ax_loss.plot(
    [],
    [],
    linewidth=2
)


loss_point, = ax_loss.plot(
    [],
    [],
    marker="o",
    markersize=8
)


# ============================================================
# 14. MAIN TITLE
# ============================================================

main_title = fig.suptitle(
    "",
    fontsize=17,
    fontweight="bold"
)


# ============================================================
# 15. ANIMATION FUNCTION
# ============================================================

def update(frame):

    current_w = w_history[frame]

    current_b = b_history[frame]

    current_predictions = (
        prediction_history[frame]
    )

    current_loss = (
        loss_history[frame]
    )


    # --------------------------------------------------------
    # Update sigmoid
    # --------------------------------------------------------

    current_curve = sigmoid(
        current_w * x_plot
        +
        current_b
    )

    current_sigmoid.set_data(
        x_plot,
        current_curve
    )


    # --------------------------------------------------------
    # Update sigmoid trail
    # --------------------------------------------------------

    for j, trail_line in enumerate(
        trail_lines
    ):

        history_index = (
            frame
            -
            (
                number_of_trails
                -
                j
            )
        )

        if history_index >= 0:

            old_curve = sigmoid(
                w_history[
                    history_index
                ]
                *
                x_plot
                +
                b_history[
                    history_index
                ]
            )

            trail_line.set_data(
                x_plot,
                old_curve
            )

            trail_line.set_linewidth(
                0.5 + 0.15 * j
            )

        else:

            trail_line.set_data(
                [],
                []
            )


    # --------------------------------------------------------
    # Update predictions
    # --------------------------------------------------------

    prediction_points.set_offsets(
        np.column_stack(
            (
                X,
                current_predictions
            )
        )
    )


    # --------------------------------------------------------
    # Update information
    # --------------------------------------------------------

    info_text.set_text(
        f"Epoch = {frame}\n\n"
        f"w = {current_w:.5f}\n"
        f"b = {current_b:.5f}\n\n"
        f"ŷ(0.35) = "
        f"{current_predictions[0]:.5f}\n"
        f"ŷ(3.50) = "
        f"{current_predictions[1]:.5f}\n\n"
        f"Loss = "
        f"{current_loss:.6f}"
    )


    # --------------------------------------------------------
    # Parameter trajectories
    # --------------------------------------------------------

    iterations = np.arange(
        frame + 1
    )


    w_line.set_data(
        iterations,
        w_history[:frame + 1]
    )

    w_point.set_data(
        [frame],
        [w_history[frame]]
    )


    b_line.set_data(
        iterations,
        b_history[:frame + 1]
    )

    b_point.set_data(
        [frame],
        [b_history[frame]]
    )


    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    loss_line.set_data(
        iterations,
        loss_history[:frame + 1]
    )

    loss_point.set_data(
        [frame],
        [loss_history[frame]]
    )


    # --------------------------------------------------------
    # Main title
    # --------------------------------------------------------

    if frame == 0:

        main_title.set_text(
            "Momentum Gradient Descent\n"
            "INITIAL STATE — Epoch 0"
        )

    else:

        main_title.set_text(
            "Momentum Gradient Descent\n"
            f"Epoch {frame} / "
            f"{number_of_epochs}"
            f"    |    "
            f"w = {current_w:.4f}"
            f"    |    "
            f"b = {current_b:.4f}"
        )


    return (
        current_sigmoid,
        prediction_points,
        info_text,
        w_line,
        w_point,
        b_line,
        b_point,
        loss_line,
        loss_point,
        main_title
    )


# ============================================================
# 16. CREATE ANIMATION
# ============================================================

animation = FuncAnimation(
    fig,
    update,
    frames=len(w_history),
    interval=1000,          # 1 SECOND PER EPOCH
    repeat=False,
    blit=False,
    cache_frame_data=False
)


# ============================================================
# 17. SAVE GIF
# ============================================================

print()
print("Saving animation...")
print("Please wait...")
print()


try:

    writer = PillowWriter(
        fps=1
    )

    animation.save(
        str(OUTPUT_FILE),
        writer=writer,
        dpi=120
    )

    print()
    print("=" * 65)
    print("SUCCESS!")
    print("=" * 65)
    print(
        "Animation saved successfully."
    )
    print()
    print(
        f"Full path:\n{OUTPUT_FILE}"
    )
    print()
    print(
        f"File exists: "
        f"{OUTPUT_FILE.exists()}"
    )
    print("=" * 65)
    print()


except Exception as error:

    print()
    print("=" * 65)
    print("ERROR WHILE SAVING ANIMATION")
    print("=" * 65)
    print(
        f"{type(error).__name__}: {error}"
    )
    print()
    print(
        "Try installing Pillow with:"
    )
    print()
    print(
        "pip install pillow"
    )
    print("=" * 65)
    print()


# ============================================================
# 18. SHOW LIVE ANIMATION
# ============================================================

plt.tight_layout(
    rect=[
        0,
        0,
        1,
        0.94
    ]
)

plt.show()
