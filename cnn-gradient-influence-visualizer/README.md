# CNN Neuron Influence

Compact Streamlit visualization of the lecture concept

∂h_j / ∂x_i

A hidden convolutional neuron in pretrained AlexNet is selected and its activation is backpropagated to the input image.

## Default behavior

The app defaults to **Strongest activation**, which automatically selects the strongest hidden neuron in the chosen convolutional layer. This avoids an arbitrary neuron whose gradient may be uninformative.

Manual channel/row/column selection is also available.

## Difference from other projects

- Filter visualization: learned convolutional weights.
- Neuron visualization: inputs that maximally activate a neuron.
- Occlusion: modify an image region and measure prediction change.
- Neuron influence: compute ∂h_j/∂x_i for a fixed image and hidden neuron.

## Run

```bash
streamlit run app.py
```
