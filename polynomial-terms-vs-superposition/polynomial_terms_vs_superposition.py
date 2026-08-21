import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import webbrowser


# ============================================================
# 1. MAIN SETTINGS
# ============================================================

MAX_ORDER = 20

# All coefficients are 1
# m1 = m2 = ... = m20 = 1

coefficients = [1.0] * MAX_ORDER


# ============================================================
# 2. VISIBLE AXIS RANGE
# ============================================================

X_AXIS_MIN = -2
X_AXIS_MAX = 2

Y_AXIS_MIN = -5
Y_AXIS_MAX = 5


# ============================================================
# 3. ACTUAL CURVE RANGE
# ============================================================

# We evaluate the curves only from -1 to +1.
#
# The visible x-axis remains -2 to +2.

X_DRAW_MIN = -1
X_DRAW_MAX = 1

N_POINTS = 1200


# ============================================================
# 4. CREATE X VALUES
# ============================================================

x = np.linspace(
    X_DRAW_MIN,
    X_DRAW_MAX,
    N_POINTS
)


# ============================================================
# 5. CREATE INDIVIDUAL CURVES
# ============================================================
#
# LEFT:
#
# y = x
# y = x²
# y = x³
# ...
# y = x²⁰
# ============================================================

individual_curves = []

for order in range(1, MAX_ORDER + 1):

    y = (
        coefficients[order - 1]
        * x ** order
    )

    individual_curves.append(y)


# ============================================================
# 6. CREATE SUPERPOSITION CURVES
# ============================================================
#
# RIGHT:
#
# Order 1:
# y = x
#
# Order 2:
# y = x + x²
#
# Order 3:
# y = x + x² + x³
#
# ...
#
# Order 20:
# y = x + x² + ... + x²⁰
# ============================================================

superposition_curves = []

for order in range(1, MAX_ORDER + 1):

    y = np.zeros_like(x)

    for k in range(1, order + 1):

        y += (
            coefficients[k - 1]
            * x ** k
        )

    superposition_curves.append(y)


# ============================================================
# 7. EQUATION LABELS
# ============================================================

individual_equations = []

for order in range(1, MAX_ORDER + 1):

    if order == 1:

        individual_equations.append(
            "y = x"
        )

    else:

        individual_equations.append(
            f"y = x^{order}"
        )


superposition_equations = []

for order in range(1, MAX_ORDER + 1):

    terms = []

    for k in range(1, order + 1):

        if k == 1:

            terms.append("x")

        else:

            terms.append(
                f"x^{k}"
            )

    superposition_equations.append(
        "y = " + " + ".join(terms)
    )


# ============================================================
# 8. CREATE LEFT FIGURE
# ============================================================

left_fig = go.Figure()


for order in range(1, MAX_ORDER + 1):

    left_fig.add_trace(

        go.Scatter(

            x=x,

            y=individual_curves[
                order - 1
            ],

            mode="lines",

            name=individual_equations[
                order - 1
            ],

            visible=(order == 1),

            line=dict(
                width=3
            )

        )

    )


# ============================================================
# 9. LEFT FIGURE LAYOUT
# ============================================================

left_fig.update_layout(

    width=900,

    height=760,

    title=dict(

        text=(
            "<b>Individual Terms</b>"
            "<br>"
            "<sup>Through Order 1</sup>"
        ),

        x=0.5,

        font=dict(
            size=22
        )

    ),

    xaxis=dict(

        title="<b>x</b>",

        range=[
            X_AXIS_MIN,
            X_AXIS_MAX
        ],

        dtick=1,

        zeroline=True,

        zerolinewidth=3,

        showgrid=True,

        gridwidth=1,

        showline=True,

        linewidth=2,

        mirror=True

    ),

    yaxis=dict(

        title="<b>y</b>",

        range=[
            Y_AXIS_MIN,
            Y_AXIS_MAX
        ],

        dtick=1,

        zeroline=True,

        zerolinewidth=3,

        showgrid=True,

        gridwidth=1,

        showline=True,

        linewidth=2,

        mirror=True

    ),

    template="plotly_white",

    showlegend=False,

    margin=dict(

        l=65,

        r=20,

        t=85,

        b=55

    )

)


# ============================================================
# 10. CREATE RIGHT FIGURE
# ============================================================

right_fig = go.Figure()


for order in range(1, MAX_ORDER + 1):

    right_fig.add_trace(

        go.Scatter(

            x=x,

            y=superposition_curves[
                order - 1
            ],

            mode="lines",

            name=superposition_equations[
                order - 1
            ],

            visible=(order == 1),

            line=dict(
                width=3
            )

        )

    )


# ============================================================
# 11. RIGHT FIGURE LAYOUT
# ============================================================

right_fig.update_layout(

    width=900,

    height=760,

    title=dict(

        text=(
            "<b>Superposition</b>"
            "<br>"
            "<sup>Through Order 1</sup>"
        ),

        x=0.5,

        font=dict(
            size=22
        )

    ),

    xaxis=dict(

        title="<b>x</b>",

        range=[
            X_AXIS_MIN,
            X_AXIS_MAX
        ],

        dtick=1,

        zeroline=True,

        zerolinewidth=3,

        showgrid=True,

        gridwidth=1,

        showline=True,

        linewidth=2,

        mirror=True

    ),

    yaxis=dict(

        title="<b>y</b>",

        range=[
            Y_AXIS_MIN,
            Y_AXIS_MAX
        ],

        dtick=1,

        zeroline=True,

        zerolinewidth=3,

        showgrid=True,

        gridwidth=1,

        showline=True,

        linewidth=2,

        mirror=True

    ),

    template="plotly_white",

    showlegend=False,

    margin=dict(

        l=65,

        r=20,

        t=85,

        b=55

    )

)


# ============================================================
# 12. CONVERT PLOTS TO HTML
# ============================================================

left_html = left_fig.to_html(

    full_html=False,

    include_plotlyjs=True

)


right_html = right_fig.to_html(

    full_html=False,

    include_plotlyjs=False

)


# ============================================================
# 13. COMPLETE HTML
# ============================================================

html = f"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>
Polynomial Terms vs Superposition
</title>


<style>

* {{
    box-sizing: border-box;
}}


html,
body {{

    margin: 0;

    padding: 0;

    width: 100%;

    min-height: 100%;

    background: #eeeeee;

    font-family:
        Arial,
        Helvetica,
        sans-serif;

}}


#container {{

    width: 100%;

    max-width: 1900px;

    margin: 0 auto;

    padding: 8px;

}}


#header {{

    background: white;

    border-radius: 8px;

    padding: 8px 15px;

    text-align: center;

    margin-bottom: 8px;

    box-shadow:
        0 1px 5px
        rgba(0,0,0,0.15);

}}


#mainTitle {{

    font-size: 26px;

    font-weight: bold;

}}


#currentEquation {{

    margin-top: 4px;

    font-size: 17px;

    font-weight: bold;

    min-height: 44px;

}}


#controls {{

    background: white;

    border-radius: 8px;

    padding: 8px;

    display: flex;

    align-items: center;

    justify-content: center;

    gap: 8px;

    flex-wrap: wrap;

    margin-bottom: 8px;

    box-shadow:
        0 1px 5px
        rgba(0,0,0,0.15);

}}


button {{

    padding:
        8px 14px;

    font-size: 14px;

    font-weight: bold;

    border:
        1px solid #555;

    border-radius: 6px;

    background: white;

    cursor: pointer;

}}


button:hover {{

    background: #eeeeee;

}}


button:disabled {{

    opacity: 0.4;

    cursor: not-allowed;

}}


#status {{

    font-weight: bold;

    font-size: 15px;

    padding:
        5px 10px;

}}


#plots {{

    display: grid;

    grid-template-columns:
        1fr 1fr;

    gap: 8px;

    width: 100%;

}}


.plotBox {{

    background: white;

    border-radius: 8px;

    overflow: hidden;

    min-width: 0;

    box-shadow:
        0 1px 5px
        rgba(0,0,0,0.15);

}}


.plotTitle {{

    text-align: center;

    font-size: 18px;

    font-weight: bold;

    padding:
        6px 0 0 0;

}}


@media (max-width: 1000px) {{

    #plots {{

        grid-template-columns:
            1fr;

    }}

}}

</style>

</head>


<body>


<div id="container">


<!-- ======================================================
     HEADER
     ====================================================== -->

<div id="header">

<div id="mainTitle">

Polynomial Terms vs Superposition

</div>


<div id="currentEquation">

Order 1 / 20

</div>

</div>


<!-- ======================================================
     CONTROLS
     ====================================================== -->

<div id="controls">


<button
    id="previousButton"
    type="button"
>

◀ Previous

</button>


<button
    id="nextButton"
    type="button"
>

Next Order ▶

</button>


<button
    id="playButton"
    type="button"
>

▶ Play

</button>


<button
    id="pauseButton"
    type="button"
>

⏸ Pause

</button>


<button
    id="resetButton"
    type="button"
>

↻ Reset

</button>


<button
    id="speedButton"
    type="button"
>

Speed: 1×

</button>


<span id="status">

Order 1 / 20

</span>


</div>


<!-- ======================================================
     TWO LARGE PLOTS
     ====================================================== -->

<div id="plots">


<div class="plotBox">

<div class="plotTitle">

Individual Terms

</div>

{left_html}

</div>


<div class="plotBox">

<div class="plotTitle">

Superposition

</div>

{right_html}

</div>


</div>


</div>


<script>


// ==========================================================
// 1. SETTINGS
// ==========================================================

const MAX_ORDER = 20;


// ==========================================================
// 2. STATE
// ==========================================================

let currentOrder = 1;

let isPlaying = false;

let timer = null;

let speedIndex = 1;


// ==========================================================
// 3. SPEED
// ==========================================================

const speeds = [

    2000,

    1000,

    500,

    250

];


const speedLabels = [

    "0.5×",

    "1×",

    "2×",

    "4×"

];


// ==========================================================
// 4. FIND THE TWO PLOTLY GRAPHS
// ==========================================================

const graphs =
    document.querySelectorAll(
        ".plotly-graph-div"
    );


const leftGraph =
    graphs[0];


const rightGraph =
    graphs[1];


// ==========================================================
// 5. CHECK THAT BOTH GRAPHS EXIST
// ==========================================================

if (!leftGraph || !rightGraph) {{

    console.error(
        "Could not find both Plotly graphs."
    );

}}


// ==========================================================
// 6. UPDATE BOTH GRAPHS
// ==========================================================

function updatePlots() {{

    if (!leftGraph || !rightGraph) {{

        return;

    }}


    // ------------------------------------------------------
    // Determine which curves are visible
    // ------------------------------------------------------

    const visibility = [];


    for (

        let i = 0;

        i < MAX_ORDER;

        i++

    ) {{

        visibility.push(
            i < currentOrder
        );

    }}


    // ------------------------------------------------------
    // Update LEFT graph
    // ------------------------------------------------------

    Plotly.restyle(

        leftGraph,

        {{

            visible:
                visibility

        }}

    );


    // ------------------------------------------------------
    // Update RIGHT graph
    // ------------------------------------------------------

    Plotly.restyle(

        rightGraph,

        {{

            visible:
                visibility

        }}

    );


    // ------------------------------------------------------
    // Current individual equation
    // ------------------------------------------------------

    let leftEquation;


    if (currentOrder === 1) {{

        leftEquation = "y = x";

    }}

    else {{

        leftEquation =
            "y = x^"
            +
            currentOrder;

    }}


    // ------------------------------------------------------
    // Current superposition equation
    // ------------------------------------------------------

    let rightEquation = "y = x";


    for (

        let k = 2;

        k <= currentOrder;

        k++

    ) {{

        rightEquation +=
            " + x^"
            +
            k;

    }}


    // ------------------------------------------------------
    // Update header
    // ------------------------------------------------------

    document
        .getElementById(
            "currentEquation"
        )
        .innerHTML =

            "<b>Order "
            +
            currentOrder
            +
            " / "
            +
            MAX_ORDER
            +
            "</b>"
            +
            "<br>"
            +
            "Left: "
            +
            leftEquation
            +
            " &nbsp;&nbsp;&nbsp; "
            +
            "|"
            +
            " &nbsp;&nbsp;&nbsp; "
            +
            "Right: "
            +
            rightEquation;


    // ------------------------------------------------------
    // Update status
    // ------------------------------------------------------

    document
        .getElementById(
            "status"
        )
        .innerHTML =

            "Order "
            +
            currentOrder
            +
            " / "
            +
            MAX_ORDER;


    // ------------------------------------------------------
    // Button states
    // ------------------------------------------------------

    document
        .getElementById(
            "previousButton"
        )
        .disabled =
            currentOrder === 1;


    document
        .getElementById(
            "nextButton"
        )
        .disabled =
            currentOrder === MAX_ORDER;

}}


// ==========================================================
// 7. NEXT ORDER
// ==========================================================

function nextOrder() {{

    if (
        currentOrder < MAX_ORDER
    ) {{

        currentOrder++;

        updatePlots();

    }}

}}


// ==========================================================
// 8. PREVIOUS ORDER
// ==========================================================

function previousOrder() {{

    if (
        currentOrder > 1
    ) {{

        currentOrder--;

        updatePlots();

    }}

}}


// ==========================================================
// 9. PAUSE
// ==========================================================

function pauseAnimation() {{

    isPlaying = false;


    if (timer !== null) {{

        clearTimeout(timer);

        timer = null;

    }}


    document
        .getElementById(
            "playButton"
        )
        .innerHTML =
            "▶ Play";

}}


// ==========================================================
// 10. PLAY
// ==========================================================

function playAnimation() {{

    if (isPlaying) {{

        return;

    }}


    // If already at final order,
    // start again from order 1.

    if (
        currentOrder >= MAX_ORDER
    ) {{

        currentOrder = 1;

        updatePlots();

    }}


    isPlaying = true;


    document
        .getElementById(
            "playButton"
        )
        .innerHTML =
            "▶ Playing";


    scheduleNextStep();

}}


// ==========================================================
// 11. NEXT ANIMATION STEP
// ==========================================================

function scheduleNextStep() {{

    if (!isPlaying) {{

        return;

    }}


    if (
        currentOrder >= MAX_ORDER
    ) {{

        pauseAnimation();

        return;

    }}


    timer = setTimeout(

        function() {{

            if (!isPlaying) {{

                return;

            }}


            currentOrder++;

            updatePlots();

            scheduleNextStep();

        }},

        speeds[
            speedIndex
        ]

    );

}}


// ==========================================================
// 12. RESET
// ==========================================================

function resetGraph() {{

    pauseAnimation();

    currentOrder = 1;

    updatePlots();

}}


// ==========================================================
// 13. PREVIOUS BUTTON
// ==========================================================

document
    .getElementById(
        "previousButton"
    )
    .addEventListener(

        "click",

        previousOrder

    );


// ==========================================================
// 14. NEXT BUTTON
// ==========================================================

document
    .getElementById(
        "nextButton"
    )
    .addEventListener(

        "click",

        nextOrder

    );


// ==========================================================
// 15. PLAY BUTTON
// ==========================================================

document
    .getElementById(
        "playButton"
    )
    .addEventListener(

        "click",

        playAnimation

    );


// ==========================================================
// 16. PAUSE BUTTON
// ==========================================================

document
    .getElementById(
        "pauseButton"
    )
    .addEventListener(

        "click",

        pauseAnimation

    );


// ==========================================================
// 17. RESET BUTTON
// ==========================================================

document
    .getElementById(
        "resetButton"
    )
    .addEventListener(

        "click",

        resetGraph

    );


// ==========================================================
// 18. SPEED BUTTON
// ==========================================================

document
    .getElementById(
        "speedButton"
    )
    .addEventListener(

        "click",

        function() {{

            speedIndex =

                (
                    speedIndex + 1
                )
                %
                speeds.length;


            this.innerHTML =

                "Speed: "
                +
                speedLabels[
                    speedIndex
                ];


            // Restart the timer if animation
            // is currently running.

            if (isPlaying) {{

                clearTimeout(timer);

                scheduleNextStep();

            }}

        }}

    );


// ==========================================================
// 19. INITIALIZE
// ==========================================================

updatePlots();


</script>


</body>

</html>
"""


# ============================================================
# 14. SAVE HTML FILE
# ============================================================

output_file = (

    Path.cwd()
    /
    "polynomial_terms_vs_superposition_1_to_20.html"

)


with open(

    output_file,

    "w",

    encoding="utf-8"

) as file:

    file.write(html)


# ============================================================
# 15. OPEN IN DEFAULT BROWSER
# ============================================================

webbrowser.open(
    output_file.resolve().as_uri()
)


# ============================================================
# 16. CONSOLE OUTPUT
# ============================================================

print()
print("=" * 70)
print("POLYNOMIAL TERMS VS SUPERPOSITION")
print("=" * 70)
print()
print("Maximum order :", MAX_ORDER)
print()
print("Individual terms:")
print("  y = x")
print("  y = x²")
print("  y = x³")
print("  ...")
print("  y = x²⁰")
print()
print("Superposition:")
print("  y = x")
print("  y = x + x²")
print("  y = x + x² + x³")
print("  ...")
print("  y = x + x² + ... + x²⁰")
print()
print("Curve range    : -1 to +1")
print("Visible X-axis : -2 to +2")
print("Visible Y-axis : -5 to +5")
print()
print("HTML file:")
print(output_file)
print()
print("Browser opened automatically.")
print("=" * 70)
