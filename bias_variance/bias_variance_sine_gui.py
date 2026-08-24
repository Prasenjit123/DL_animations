import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class BiasVarianceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bias vs Variance - Sine Curve Demo")
        self.root.geometry("1500x850")

        # -----------------------------
        # Experiment state
        # -----------------------------
        self.run = 0
        self.bias_models = []
        self.variance_models = []

        self.current_x = None
        self.current_y = None

        self.is_playing = False

        # Zoom modes:
        # 0 = normal
        # 1 = zoom in
        # 2 = zoom out
        self.zoom_mode = 0

        self.zoom_labels = [
            "View: Normal",
            "View: Zoom In",
            "View: Zoom Out",
        ]

        # -----------------------------
        # Build UI
        # -----------------------------
        self.build_controls()
        self.build_graphs()

        # Initial plot
        self.reset_experiment()

    # =========================================================
    # UI
    # =========================================================

    def build_controls(self):
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(fill="x")

        # True Function
        ttk.Label(control_frame, text="True Function").grid(
            row=0, column=0, sticky="w", padx=5
        )
        self.function_box = ttk.Combobox(
            control_frame,
            values=["sin(x)"],
            width=12,
            state="readonly",
        )
        self.function_box.set("sin(x)")
        self.function_box.grid(row=1, column=0, padx=5, pady=5)

        # Data points
        ttk.Label(control_frame, text="Data Points").grid(
            row=0, column=1, sticky="w", padx=5
        )
        self.num_points_var = tk.IntVar(value=25)
        ttk.Entry(
            control_frame,
            textvariable=self.num_points_var,
            width=10,
        ).grid(row=1, column=1, padx=5, pady=5)

        # Noise
        ttk.Label(control_frame, text="Noise Level").grid(
            row=0, column=2, sticky="w", padx=5
        )
        self.noise_var = tk.DoubleVar(value=0.30)
        ttk.Entry(
            control_frame,
            textvariable=self.noise_var,
            width=10,
        ).grid(row=1, column=2, padx=5, pady=5)

        # Number of experiments
        ttk.Label(control_frame, text="Experiments").grid(
            row=0, column=3, sticky="w", padx=5
        )
        self.experiments_var = tk.IntVar(value=10)
        ttk.Entry(
            control_frame,
            textvariable=self.experiments_var,
            width=10,
        ).grid(row=1, column=3, padx=5, pady=5)

        # Bias degree
        ttk.Label(control_frame, text="Bias Degree").grid(
            row=0, column=4, sticky="w", padx=5
        )
        self.bias_degree_var = tk.IntVar(value=1)
        ttk.Entry(
            control_frame,
            textvariable=self.bias_degree_var,
            width=10,
        ).grid(row=1, column=4, padx=5, pady=5)

        # Variance degree
        ttk.Label(control_frame, text="Variance Degree").grid(
            row=0, column=5, sticky="w", padx=5
        )
        self.variance_degree_var = tk.IntVar(value=10)
        ttk.Entry(
            control_frame,
            textvariable=self.variance_degree_var,
            width=10,
        ).grid(row=1, column=5, padx=5, pady=5)

        # Seed
        ttk.Label(control_frame, text="Random Seed").grid(
            row=0, column=6, sticky="w", padx=5
        )
        self.seed_var = tk.IntVar(value=42)
        ttk.Entry(
            control_frame,
            textvariable=self.seed_var,
            width=10,
        ).grid(row=1, column=6, padx=5, pady=5)

        # Buttons
        self.apply_button = ttk.Button(
            control_frame,
            text="Apply Settings",
            command=self.reset_experiment,
        )
        self.apply_button.grid(row=1, column=7, padx=5)

        self.next_button = ttk.Button(
            control_frame,
            text="Next Sample",
            command=self.next_experiment,
        )
        self.next_button.grid(row=1, column=8, padx=5)

        self.play_button = ttk.Button(
            control_frame,
            text="Play",
            command=self.toggle_play,
        )
        self.play_button.grid(row=1, column=9, padx=5)

        self.reset_button = ttk.Button(
            control_frame,
            text="Reset",
            command=self.reset_experiment,
        )
        self.reset_button.grid(row=1, column=10, padx=5)

        self.zoom_button = ttk.Button(
            control_frame,
            text=self.zoom_labels[self.zoom_mode],
            command=self.cycle_zoom,
        )
        self.zoom_button.grid(row=1, column=11, padx=5)

        # Second row
        second_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        second_frame.pack(fill="x")

        ttk.Label(second_frame, text="Speed").pack(side="left", padx=(5, 5))

        self.speed_var = tk.StringVar(value="1x")
        self.speed_box = ttk.Combobox(
            second_frame,
            textvariable=self.speed_var,
            values=["0.5x", "1x", "2x", "5x"],
            width=8,
            state="readonly",
        )
        self.speed_box.pack(side="left", padx=(0, 15))

        self.show_points_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            second_frame,
            text="Show current sample points",
            variable=self.show_points_var,
            command=self.redraw,
        ).pack(side="left", padx=10)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(
            second_frame,
            textvariable=self.status_var,
        ).pack(side="right", padx=10)

    def build_graphs(self):
        graph_frame = ttk.Frame(self.root, padding=10)
        graph_frame.pack(fill="both", expand=True)

        self.figure, (self.ax_bias, self.ax_variance) = plt.subplots(
            1, 2, figsize=(14, 6)
        )

        self.figure.subplots_adjust(
            left=0.06,
            right=0.98,
            bottom=0.10,
            top=0.90,
            wspace=0.20,
        )

        self.canvas = FigureCanvasTkAgg(self.figure, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # =========================================================
    # SETTINGS
    # =========================================================

    def get_settings(self):
        try:
            n = max(5, int(self.num_points_var.get()))
            noise = max(0.0, float(self.noise_var.get()))
            experiments = max(1, int(self.experiments_var.get()))
            bias_degree = max(1, int(self.bias_degree_var.get()))
            variance_degree = max(2, int(self.variance_degree_var.get()))
            seed = int(self.seed_var.get())

            # Need enough points for polynomial fitting
            n = max(n, bias_degree + 2, variance_degree + 3)

            return {
                "n": n,
                "noise": noise,
                "experiments": experiments,
                "bias_degree": bias_degree,
                "variance_degree": variance_degree,
                "seed": seed,
            }

        except ValueError:
            messagebox.showerror(
                "Invalid Input",
                "Please enter valid numeric values.",
            )
            return None

    # =========================================================
    # EXPERIMENT
    # =========================================================

    def reset_experiment(self):
        self.is_playing = False
        self.play_button.config(text="Play")

        self.run = 0
        self.bias_models.clear()
        self.variance_models.clear()

        self.current_x = None
        self.current_y = None

        self.status_var.set("Ready - click Next Sample or Play")

        self.redraw()

    def next_experiment(self):
        settings = self.get_settings()

        if settings is None:
            return

        n = settings["n"]
        noise = settings["noise"]
        bias_degree = settings["bias_degree"]
        variance_degree = settings["variance_degree"]
        seed = settings["seed"]

        # Synchronized seed for this experiment
        experiment_seed = seed + self.run * 7919

        rng = np.random.default_rng(experiment_seed)

        # Generate a NEW noisy dataset
        x = np.sort(rng.uniform(0.0, 6.0, n))
        y = np.sin(x) + rng.normal(0.0, noise, n)

        # Low complexity model -> high bias
        bias_coeff = np.polyfit(
            x,
            y,
            bias_degree,
        )

        # High complexity model -> high variance
        variance_coeff = np.polyfit(
            x,
            y,
            variance_degree,
        )

        # Keep all fitted models
        self.bias_models.append(bias_coeff.copy())
        self.variance_models.append(variance_coeff.copy())

        # Replace old points with current points
        self.current_x = x
        self.current_y = y

        self.run += 1

        self.status_var.set(
            f"Experiment {self.run} | "
            f"Bias degree: {bias_degree} | "
            f"Variance degree: {variance_degree}"
        )

        self.redraw()

    # =========================================================
    # PLAY / PAUSE
    # =========================================================

    def toggle_play(self):
        if self.is_playing:
            self.is_playing = False
            self.play_button.config(text="Play")
            return

        self.is_playing = True
        self.play_button.config(text="Pause")

        self.play_step()

    def play_step(self):
        if not self.is_playing:
            return

        settings = self.get_settings()

        if settings is None:
            self.is_playing = False
            self.play_button.config(text="Play")
            return

        if self.run >= settings["experiments"]:
            self.is_playing = False
            self.play_button.config(text="Play")
            self.status_var.set(
                f"Finished - {self.run} experiments"
            )
            return

        self.next_experiment()

        speed_delay = {
            "0.5x": 1300,
            "1x": 700,
            "2x": 350,
            "5x": 120,
        }

        delay = speed_delay.get(
            self.speed_var.get(),
            700,
        )

        self.root.after(delay, self.play_step)

    # =========================================================
    # ZOOM
    # =========================================================

    def cycle_zoom(self):
        # Normal -> Zoom In -> Zoom Out -> Normal
        self.zoom_mode = (self.zoom_mode + 1) % 3

        self.zoom_button.config(
            text=self.zoom_labels[self.zoom_mode]
        )

        # Only redraw:
        # no new data
        # no model deletion
        # no experiment change
        self.redraw()

    # =========================================================
    # DRAWING
    # =========================================================

    def configure_axis(self, ax, kind):
        if self.zoom_mode == 1:
            # Zoom In
            ax.set_xlim(0.5, 5.5)

            if kind == "bias":
                ax.set_ylim(-1.25, 1.25)
                ax.set_yticks(
                    np.arange(-1.0, 1.01, 0.5)
                )
            else:
                ax.set_ylim(-1.5, 1.5)
                ax.set_yticks(
                    np.arange(-1.5, 1.51, 0.5)
                )

            ax.set_xticks(np.arange(1, 6, 1))

        elif self.zoom_mode == 2:
            # Zoom Out
            ax.set_xlim(-0.5, 6.5)

            if kind == "bias":
                ax.set_ylim(-2.0, 2.0)
                ax.set_yticks(
                    np.arange(-2.0, 2.01, 0.5)
                )
            else:
                ax.set_ylim(-2.5, 2.5)
                ax.set_yticks(
                    np.arange(-2.5, 2.51, 0.5)
                )

            ax.set_xticks(np.arange(0, 7, 1))

        else:
            # Normal
            ax.set_xlim(0.0, 6.0)

            if kind == "bias":
                ax.set_ylim(-1.5, 1.5)
                ax.set_yticks(
                    np.arange(-1.5, 1.51, 0.5)
                )
            else:
                ax.set_ylim(-2.0, 1.5)
                ax.set_yticks(
                    np.arange(-2.0, 1.51, 0.5)
                )

            ax.set_xticks(np.arange(0, 7, 1))

    def draw_graph(self, ax, models, kind):
        ax.clear()

        x_grid = np.linspace(
            0.0,
            6.0,
            700,
        )

        # True sine function
        ax.plot(
            x_grid,
            np.sin(x_grid),
            color="black",
            linewidth=2.5,
            label="True function",
            zorder=5,
        )

        # Draw all previously fitted models
        for coeff in models:
            y_fit = np.polyval(
                coeff,
                x_grid,
            )

            if kind == "variance":
                # Prevent extreme high-degree values
                # from destroying the graph scale
                y_fit = np.clip(
                    y_fit,
                    -3.0,
                    3.0,
                )

            ax.plot(
                x_grid,
                y_fit,
                linewidth=1.25,
                alpha=0.80,
                zorder=2,
            )

        # Only the latest sample points are shown
        if (
            self.show_points_var.get()
            and self.current_x is not None
        ):
            ax.scatter(
                self.current_x,
                self.current_y,
                s=22,
                facecolors="white",
                edgecolors="black",
                linewidths=0.7,
                alpha=0.80,
                zorder=6,
            )

        self.configure_axis(
            ax,
            kind,
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")

        if kind == "bias":
            ax.set_title(
                f"High Bias / Underfitting\n"
                f"{self.run} fitted model"
                + ("s" if self.run != 1 else "")
            )
        else:
            ax.set_title(
                f"High Variance / Overfitting\n"
                f"{self.run} fitted model"
                + ("s" if self.run != 1 else "")
            )

        ax.tick_params(
            direction="in",
            top=True,
            right=True,
        )

        ax.grid(False)

    def redraw(self):
        self.draw_graph(
            self.ax_bias,
            self.bias_models,
            "bias",
        )

        self.draw_graph(
            self.ax_variance,
            self.variance_models,
            "variance",
        )

        self.canvas.draw_idle()


# =============================================================
# RUN APPLICATION
# =============================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = BiasVarianceApp(root)
    root.mainloop()
