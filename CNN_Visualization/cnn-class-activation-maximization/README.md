# Improved CNN Activation Maximization

A Streamlit teaching app that optimizes input pixels for a chosen ImageNet-1K class using pretrained AlexNet.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

The CNN weights stay fixed. The app optimizes a low-resolution image and upsamples it, with mild smoothness and pixel regularization to reduce noisy patterns. Generated images remain synthetic and are not expected to look like photographs.
