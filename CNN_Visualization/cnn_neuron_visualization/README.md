# CNN Neuron Visualization — AlexNet-style CNN + MNIST

## Important
This is an AlexNet-style CNN trained on MNIST, not the ImageNet AlexNet checkpoint.

The interactive experiment answers:

> For one selected convolutional filter, which images produce the
> strongest response, and where in each image did that response occur?

## Controls

- Convolutional layer: conv1 ... conv5
- Filter/channel: select any channel valid for that layer
- Activation used for ranking:
  - POST-RELU: actual feature-map response after ReLU
  - PRE-RELU: raw convolution output before ReLU
- Images to scan: up to 10,000 MNIST test images
- Top-K: number of strongest examples

For each image, the score is:

    score(image) = max(feature_map_of_selected_filter)

The images are then sorted by this score.

Each result shows:
1. Original MNIST image
2. Strongest spatial location (yellow dot)
3. Theoretical receptive field (red box)
4. Exact receptive-field crop
5. Feature-map coordinate and activation value

## Run in Spyder

Open `cnn_neuron_visualization_alexnet_mnist.py` and press Run.

The first run downloads MNIST if necessary and trains the model.
The trained checkpoint is saved under:

    results/alexnet_mnist.pt

Later runs reuse the checkpoint.

## Why POST-RELU and PRE-RELU?

They answer slightly different teaching questions.

POST-RELU:
- Shows the signal that actually survives the ReLU.
- Negative convolution responses become zero.
- Recommended default for demonstrating "what activates a filter."

PRE-RELU:
- Shows the raw signed convolution response.
- Useful when explaining what ReLU changes.

## 4 GB GPU

The model is intentionally modest and uses inference mode during scanning.
The scanner uses small batches and does not build autograd graphs.
