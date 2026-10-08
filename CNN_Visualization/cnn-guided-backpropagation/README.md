# CNN Guided Backpropagation

A compact Streamlit demonstration of guided backpropagation.

## What it demonstrates

For one selected convolutional feature-map neuron:

1. Run a forward pass.
2. Select one neuron in a convolutional feature map.
3. Treat that neuron as the backward target.
4. Compute ordinary input gradients.
5. Compute guided-backpropagation input gradients.
6. Compare the two visualizations.

## Guided backpropagation rule

During the backward pass through ReLU, the gradient is retained only when:

- the forward activation is positive, and
- the incoming gradient is positive.

This suppresses negative gradients flowing from the upper layer.

## Difference from input-pixel influence

The previous input-pixel influence experiment visualizes ordinary gradients.

This experiment adds the guided-backpropagation rule and compares:

- ordinary backpropagation
- guided backpropagation

for the same selected convolutional neuron.

## Run

```bash
streamlit run app.py
```
