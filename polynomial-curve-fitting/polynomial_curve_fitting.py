from pathlib import Path

# ============================================================
# Interactive Polynomial Curve Fitting
# HTML generator
# ============================================================

HTML_FILE = Path(__file__).with_name("polynomial-curve-fitting.html")

HTML = r"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>
Interactive Polynomial Curve Fitting
</title>

<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>


<style>

body {

    margin: 0;
    padding: 0;

    font-family:
        Arial,
        Helvetica,
        sans-serif;

    background: #f4f4f4;
}


#main {

    width: 95%;
    max-width: 1450px;

    margin: 20px auto;
}


#title {

    text-align: center;

    font-size: 30px;

    font-weight: bold;

    margin-bottom: 15px;
}


#controls {

    background: white;

    border-radius: 10px;

    padding: 18px;

    box-shadow:
        0px 2px 8px
        rgba(0,0,0,0.15);

    display: flex;

    flex-wrap: wrap;

    gap: 18px;

    align-items: end;

    justify-content: center;

    margin-bottom: 15px;
}


.control {

    display: flex;

    flex-direction: column;

    gap: 5px;
}


.control label {

    font-weight: bold;

    font-size: 14px;
}


select,
input[type="number"] {

    font-size: 16px;

    padding: 7px;

    border-radius: 5px;

    border: 1px solid #999;

}


button {

    font-size: 15px;

    padding: 9px 16px;

    border-radius: 6px;

    border: none;

    cursor: pointer;

    background: #444;

    color: white;
}


#applyButton {

    background: #222;
}


#newSampleButton {

    background: #333;
}


#randomSampleButton {

    background: #555;
}


#playButton {

    background: #444;
}


#speedButton {

    background: #666;
}


#degreeArea {

    background: white;

    padding: 15px 25px;

    border-radius: 10px;

    margin-bottom: 15px;

    box-shadow:
        0px 2px 8px
        rgba(0,0,0,0.12);
}


#degreeSlider {

    width: 100%;
}


#degreeValue {

    text-align: center;

    font-size: 22px;

    font-weight: bold;

    margin-top: 5px;
}


#information {

    background: white;

    border-radius: 10px;

    padding: 12px;

    text-align: center;

    font-size: 17px;

    margin-bottom: 15px;

    box-shadow:
        0px 2px 8px
        rgba(0,0,0,0.12);
}


#plot {

    background: white;

    border-radius: 10px;

    padding: 5px;

    box-shadow:
        0px 2px 8px
        rgba(0,0,0,0.15);
}


</style>

</head>


<body>


<div id="main">


<div id="title">

Interactive Polynomial Curve Fitting

</div>


<!-- =====================================================
     CONTROLS
     ===================================================== -->

<div id="controls">


<!-- TRUE FUNCTION -->

<div class="control">

<label>
True Function
</label>

<select id="functionSelect">

<option value="sin(x)">
sin(x)
</option>

<option value="cos(x)">
cos(x)
</option>

<option value="sin(2x)">
sin(2x)
</option>

<option value="cos(2x)">
cos(2x)
</option>

<option value="sin(x) + 0.3cos(3x)">
sin(x) + 0.3cos(3x)
</option>

<option value="x²">
x²
</option>

<option value="x³">
x³
</option>

<option value="exp(-x²)">
exp(-x²)
</option>

</select>

</div>


<!-- NUMBER OF DATA POINTS -->

<div class="control">

<label>
Number of Data Points
</label>

<input
    id="pointsInput"
    type="number"
    min="5"
    max="500"
    value="50"
>

</div>


<!-- NOISE -->

<div class="control">

<label>
Noise Level
</label>

<input
    id="noiseInput"
    type="number"
    min="0"
    max="5"
    step="0.05"
    value="0.50"
>

</div>


<!-- MAXIMUM DEGREE -->

<div class="control">

<label>
Maximum Polynomial Degree
</label>

<input
    id="maxDegreeInput"
    type="number"
    min="1"
    max="50"
    value="30"
>

</div>


<!-- SEED -->

<div class="control">

<label>
Random Seed
</label>

<input
    id="seedInput"
    type="number"
    value="42"
>

</div>


<!-- APPLY -->

<div class="control">

<button
    id="applyButton"
    onclick="generateData()"
>
Apply Settings
</button>

</div>


<!-- NEW SAMPLE -->

<div class="control">

<button
    id="newSampleButton"
    onclick="generateNewSample()"
>
New Sample
</button>

</div>


<!-- RANDOM SAMPLE -->

<div class="control">

<button
    id="randomSampleButton"
    onclick="generateRandomSample()"
>
Random Sample
</button>

</div>


<!-- PLAY -->

<div class="control">

<button
    id="playButton"
    onclick="toggleAnimation()"
>
▶ Play
</button>

</div>


<!-- SPEED -->

<div class="control">

<button
    id="speedButton"
    onclick="changeSpeed()"
>
Speed: 1×
</button>

</div>


</div>


<!-- =====================================================
     DEGREE SLIDER
     ===================================================== -->

<div id="degreeArea">

<label>

<b>
Polynomial Degree
</b>

</label>


<input
    id="degreeSlider"
    type="range"
    min="1"
    max="30"
    value="1"
    step="1"
    oninput="changeDegree()"
>


<div id="degreeValue">

Degree = 1

</div>


</div>


<!-- =====================================================
     INFORMATION
     ===================================================== -->

<div id="information">

True function:
<b id="functionInfo">
sin(x)
</b>

&nbsp;&nbsp; | &nbsp;&nbsp;

Data points:
<b id="pointsInfo">
50
</b>

&nbsp;&nbsp; | &nbsp;&nbsp;

Noise:
<b id="noiseInfo">
0.50
</b>

&nbsp;&nbsp; | &nbsp;&nbsp;

Seed:
<b id="seedInfo">
42
</b>

&nbsp;&nbsp; | &nbsp;&nbsp;

Polynomial degree:
<b id="degreeInfo">
1
</b>

&nbsp;&nbsp; | &nbsp;&nbsp;

Training MSE:
<b id="mseInfo">
-
</b>

&nbsp;&nbsp; | &nbsp;&nbsp;

<b id="interpretationInfo">
UNDERFITTING
</b>

</div>


<!-- =====================================================
     PLOT
     ===================================================== -->

<div id="plot">

</div>


</div>


<script>


// ========================================================
// GLOBAL VARIABLES
// ========================================================

let xData = [];

let yData = [];

let xPlot = [];

let yTrue = [];

let currentFunction = "sin(x)";

let noiseLevel = 0.50;

let numberOfPoints = 50;

let maxDegree = 30;

let currentDegree = 1;

let randomSeed = 42;

let animationTimer = null;

let isPlaying = false;

let speedIndex = 0;


// ========================================================
// ANIMATION SPEEDS
// ========================================================

const speeds = [

    700,   // 1×

    350,   // 2×

    233,   // 3×

    175    // 4×

];


// ========================================================
// SEEDED RANDOM NUMBER GENERATOR
// ========================================================

let seedState = 42;


function seededRandom() {

    /*
       Mulberry32-style deterministic generator.
       Same seed = same random sequence.
    */

    seedState += 0x6D2B79F5;

    let t = seedState;

    t =
        Math.imul(
            t ^ (t >>> 15),
            t | 1
        );

    t ^=
        t
        +
        Math.imul(
            t ^ (t >>> 7),
            t | 61
        );

    return (
        (
            t ^
            (t >>> 14)
        )
        >>> 0
    )
    /
    4294967296;

}


// ========================================================
// NORMAL RANDOM NUMBER
// ========================================================

function randomNormal() {

    let u = 0;

    let v = 0;


    while (u === 0) {

        u = seededRandom();

    }


    while (v === 0) {

        v = seededRandom();

    }


    return Math.sqrt(
        -2 * Math.log(u)
    )
    *
    Math.cos(
        2 * Math.PI * v
    );

}


// ========================================================
// TRUE FUNCTION
// ========================================================

function evaluateFunction(
    functionName,
    x
) {


    if (
        functionName === "sin(x)"
    ) {

        return Math.sin(x);

    }


    if (
        functionName === "cos(x)"
    ) {

        return Math.cos(x);

    }


    if (
        functionName === "sin(2x)"
    ) {

        return Math.sin(2 * x);

    }


    if (
        functionName === "cos(2x)"
    ) {

        return Math.cos(2 * x);

    }


    if (
        functionName ===
        "sin(x) + 0.3cos(3x)"
    ) {

        return (
            Math.sin(x)
            +
            0.3 * Math.cos(3 * x)
        );

    }


    if (
        functionName === "x²"
    ) {

        return x * x;

    }


    if (
        functionName === "x³"
    ) {

        return x * x * x;

    }


    if (
        functionName === "exp(-x²)"
    ) {

        return Math.exp(
            -x * x
        );

    }


    return Math.sin(x);

}


// ========================================================
// MATRIX SOLVER
// ========================================================

function solveLinearSystem(
    A,
    b
) {


    let n = b.length;

    let M = [];


    for (
        let i = 0;
        i < n;
        i++
    ) {

        M[i] =
            A[i].slice();

        M[i].push(
            b[i]
        );

    }


    for (
        let i = 0;
        i < n;
        i++
    ) {


        let maxRow = i;


        for (
            let k = i + 1;
            k < n;
            k++
        ) {

            if (
                Math.abs(
                    M[k][i]
                )
                >
                Math.abs(
                    M[maxRow][i]
                )
            ) {

                maxRow = k;

            }

        }


        let temp = M[i];

        M[i] =
            M[maxRow];

        M[maxRow] =
            temp;


        if (
            Math.abs(
                M[i][i]
            )
            <
            1e-12
        ) {

            M[i][i] =
                1e-12;

        }


        let pivot =
            M[i][i];


        for (
            let j = i;
            j <= n;
            j++
        ) {

            M[i][j] /=
                pivot;

        }


        for (
            let k = 0;
            k < n;
            k++
        ) {

            if (
                k !== i
            ) {

                let factor =
                    M[k][i];


                for (
                    let j = i;
                    j <= n;
                    j++
                ) {

                    M[k][j]
                    -=
                    factor *
                    M[i][j];

                }

            }

        }

    }


    let solution = [];


    for (
        let i = 0;
        i < n;
        i++
    ) {

        solution.push(
            M[i][n]
        );

    }


    return solution;

}


// ========================================================
// POLYNOMIAL FITTING
// ========================================================

function polynomialFit(
    degree
) {


    let n =
        xData.length;


    let size =
        degree + 1;


    let A = [];

    let b = [];


    for (
        let i = 0;
        i < size;
        i++
    ) {

        A[i] = [];

        b[i] = 0;


        for (
            let j = 0;
            j < size;
            j++
        ) {

            let sum = 0;


            for (
                let k = 0;
                k < n;
                k++
            ) {

                sum +=
                    Math.pow(
                        xData[k],
                        i + j
                    );

            }


            A[i][j] =
                sum;

        }


        for (
            let k = 0;
            k < n;
            k++
        ) {

            b[i] +=
                yData[k]
                *
                Math.pow(
                    xData[k],
                    i
                );

        }

    }


    return solveLinearSystem(
        A,
        b
    );

}


// ========================================================
// POLYNOMIAL PREDICTION
// ========================================================

function polynomialPredict(
    coefficients,
    x
) {


    let y = 0;


    for (
        let i = 0;
        i < coefficients.length;
        i++
    ) {

        y +=
            coefficients[i]
            *
            Math.pow(
                x,
                i
            );

    }


    return y;

}


// ========================================================
// READ SETTINGS
// ========================================================

function readSettings() {


    currentFunction =
        document
        .getElementById(
            "functionSelect"
        )
        .value;


    numberOfPoints =
        parseInt(
            document
            .getElementById(
                "pointsInput"
            )
            .value
        );


    noiseLevel =
        parseFloat(
            document
            .getElementById(
                "noiseInput"
            )
            .value
        );


    maxDegree =
        parseInt(
            document
            .getElementById(
                "maxDegreeInput"
            )
            .value
        );


    randomSeed =
        parseInt(
            document
            .getElementById(
                "seedInput"
            )
            .value
        );


    if (
        !Number.isFinite(
            numberOfPoints
        )
        ||
        numberOfPoints < 5
    ) {

        numberOfPoints = 5;

        document
        .getElementById(
            "pointsInput"
        )
        .value = 5;

    }


    if (
        !Number.isFinite(
            noiseLevel
        )
        ||
        noiseLevel < 0
    ) {

        noiseLevel = 0.50;

        document
        .getElementById(
            "noiseInput"
        )
        .value = 0.50;

    }


    if (
        !Number.isFinite(
            maxDegree
        )
        ||
        maxDegree < 1
    ) {

        maxDegree = 1;

    }


    /*
       A polynomial of degree d has d+1
       coefficients.

       We therefore keep:
       
       maximum degree <= number of data points - 1
    */

    if (
        maxDegree >= numberOfPoints
    ) {

        maxDegree =
            numberOfPoints - 1;

        document
        .getElementById(
            "maxDegreeInput"
        )
        .value =
            maxDegree;

    }


    if (
        !Number.isFinite(
            randomSeed
        )
    ) {

        randomSeed = 42;

        document
        .getElementById(
            "seedInput"
        )
        .value = 42;

    }

}


// ========================================================
// GENERATE DATA
// ========================================================

function generateData() {


    readSettings();


    stopAnimation();


    /*
       Reset deterministic random generator.

       This is the important part:

       Same seed
       +
       same settings
       =
       same dataset.
    */

    seedState =
        randomSeed;


    xData = [];

    yData = [];


    for (
        let i = 0;
        i < numberOfPoints;
        i++
    ) {


        let x =
            (
                2
                *
                Math.PI
                *
                i
            )
            /
            (
                numberOfPoints - 1
            );


        let y =
            evaluateFunction(
                currentFunction,
                x
            );


        y +=
            noiseLevel
            *
            randomNormal();


        xData.push(x);

        yData.push(y);

    }


    // Smooth curve

    xPlot = [];

    yTrue = [];


    for (
        let i = 0;
        i < 800;
        i++
    ) {


        let x =
            (
                2
                *
                Math.PI
                *
                i
            )
            /
            799;


        xPlot.push(x);


        yTrue.push(

            evaluateFunction(
                currentFunction,
                x
            )

        );

    }


    // Reset degree

    currentDegree = 1;


    let slider =
        document
        .getElementById(
            "degreeSlider"
        );


    slider.max =
        maxDegree;


    slider.value =
        1;


    updatePlot();

}


// ========================================================
// NEW SAMPLE
// ========================================================

function generateNewSample() {


    /*
       Keep all current settings,
       but regenerate the noisy observations
       using the seed currently shown.
    */

    readSettings();

    generateData();

}


// ========================================================
// RANDOM SAMPLE
// ========================================================

function generateRandomSample() {


    /*
       Generate a new random seed.
    */

    randomSeed =
        Math.floor(
            Math.random()
            *
            1000000000
        );


    document
    .getElementById(
        "seedInput"
    )
    .value =
        randomSeed;


    generateData();

}


// ========================================================
// CHANGE DEGREE
// ========================================================

function changeDegree() {


    currentDegree =
        parseInt(

            document
            .getElementById(
                "degreeSlider"
            )
            .value

        );


    updatePlot();

}


// ========================================================
// UPDATE PLOT
// ========================================================

function updatePlot() {


    let coefficients =
        polynomialFit(
            currentDegree
        );


    let polynomialY = [];


    for (
        let i = 0;
        i < xPlot.length;
        i++
    ) {

        polynomialY.push(

            polynomialPredict(
                coefficients,
                xPlot[i]
            )

        );

    }


    // ====================================================
    // TRAINING MSE
    // ====================================================

    let mse = 0;


    for (
        let i = 0;
        i < xData.length;
        i++
    ) {

        let prediction =
            polynomialPredict(
                coefficients,
                xData[i]
            );


        mse +=
            Math.pow(
                yData[i]
                -
                prediction,
                2
            );

    }


    mse /=
        xData.length;


    // ====================================================
    // INTERPRETATION
    // ====================================================

    let interpretation = "";


    if (
        currentDegree <= 2
    ) {

        interpretation =
            "UNDERFITTING";

    }

    else if (
        currentDegree <= 5
    ) {

        interpretation =
            "REASONABLE FIT";

    }

    else if (
        currentDegree <= 9
    ) {

        interpretation =
            "INCREASING COMPLEXITY";

    }

    else {

        interpretation =
            "POTENTIAL OVERFITTING";

    }


    // ====================================================
    // UPDATE INFORMATION
    // ====================================================

    document
    .getElementById(
        "degreeValue"
    )
    .innerHTML =
        "Degree = "
        +
        currentDegree;


    document
    .getElementById(
        "functionInfo"
    )
    .innerHTML =
        currentFunction;


    document
    .getElementById(
        "pointsInfo"
    )
    .innerHTML =
        numberOfPoints;


    document
    .getElementById(
        "noiseInfo"
    )
    .innerHTML =
        noiseLevel.toFixed(2);


    document
    .getElementById(
        "seedInfo"
    )
    .innerHTML =
        randomSeed;


    document
    .getElementById(
        "degreeInfo"
    )
    .innerHTML =
        currentDegree;


    document
    .getElementById(
        "mseInfo"
    )
    .innerHTML =
        mse.toFixed(6);


    document
    .getElementById(
        "interpretationInfo"
    )
    .innerHTML =
        interpretation;


    // ====================================================
    // GRAPH DATA
    // ====================================================

    let graphData = [

        {

            x: xPlot,

            y: yTrue,

            mode: "lines",

            name:
                "True function: "
                +
                currentFunction,

            line: {

                dash: "dash",

                width: 4

            }

        },


        {

            x: xData,

            y: yData,

            mode: "markers",

            name:
                numberOfPoints
                +
                " noisy observations",

            marker: {

                size: 9

            }

        },


        {

            x: xPlot,

            y: polynomialY,

            mode: "lines",

            name:
                "Polynomial degree = "
                +
                currentDegree,

            line: {

                width: 5

            }

        }

    ];


    // ====================================================
    // Y AXIS RANGE
    // ====================================================

    let allY =
        yData
        .concat(yTrue)
        .concat(polynomialY);


    let ymin =
        Math.min(
            ...allY
        );


    let ymax =
        Math.max(
            ...allY
        );


    let margin =
        (
            ymax - ymin
        )
        *
        0.10;


    if (
        margin < 0.5
    ) {

        margin = 0.5;

    }


    // ====================================================
    // LAYOUT
    // ====================================================

    let layout = {

        title: {

            text:

                "<b>"
                +
                "Polynomial Curve Fitting — Degree "
                +
                currentDegree
                +
                "</b>"
                +
                "<br>"
                +
                "<sup>"
                +
                "True function: y = "
                +
                currentFunction
                +
                " &nbsp; | &nbsp; "
                +
                "Training MSE = "
                +
                mse.toFixed(6)
                +
                " &nbsp; | &nbsp; "
                +
                interpretation
                +
                "</sup>",

            x: 0.5,

            font: {

                size: 24

            }

        },


        xaxis: {

            title: "x",

            range: [

                0,

                2 * Math.PI

            ]

        },


        yaxis: {

            title: "y",

            range: [

                ymin - margin,

                ymax + margin

            ]

        },


        template:
            "plotly_white",


        height: 650,


        margin: {

            l: 70,

            r: 50,

            t: 110,

            b: 80

        },


        legend: {

            orientation:
                "h",

            y:
                -0.15

        }

    };


    Plotly.react(

        "plot",

        graphData,

        layout,

        {

            responsive:
                true

        }

    );

}


// ========================================================
// PLAY / PAUSE
// ========================================================

function toggleAnimation() {


    if (
        isPlaying
    ) {

        stopAnimation();

        return;

    }


    isPlaying =
        true;


    document
    .getElementById(
        "playButton"
    )
    .innerHTML =
        "⏸ Pause";


    playNextDegree();

}


// ========================================================
// NEXT DEGREE
// ========================================================

function playNextDegree() {


    if (
        !isPlaying
    ) {

        return;

    }


    currentDegree++;


    if (
        currentDegree >
        maxDegree
    ) {

        currentDegree = 1;

    }


    document
    .getElementById(
        "degreeSlider"
    )
    .value =
        currentDegree;


    updatePlot();


    animationTimer =
        setTimeout(

            playNextDegree,

            speeds[
                speedIndex
            ]

        );

}


// ========================================================
// STOP ANIMATION
// ========================================================

function stopAnimation() {


    isPlaying =
        false;


    if (
        animationTimer
    ) {

        clearTimeout(
            animationTimer
        );

        animationTimer =
            null;

    }


    document
    .getElementById(
        "playButton"
    )
    .innerHTML =
        "▶ Play";

}


// ========================================================
// CHANGE SPEED
// ========================================================

function changeSpeed() {


    speedIndex++;


    if (
        speedIndex >=
        speeds.length
    ) {

        speedIndex = 0;

    }


    let displayedSpeed =
        Math.pow(
            2,
            speedIndex
        );


    document
    .getElementById(
        "speedButton"
    )
    .innerHTML =
        "Speed: "
        +
        displayedSpeed
        +
        "×";


}


// ========================================================
// INITIALIZE
// ========================================================

generateData();


</script>


</body>

</html>"""

def generate_html(output_file=HTML_FILE):
    """Write the interactive polynomial curve-fitting HTML file."""
    output_file = Path(output_file)
    output_file.write_text(HTML, encoding="utf-8")
    print(f"HTML generated successfully: {output_file.resolve()}")


if __name__ == "__main__":
    generate_html()
