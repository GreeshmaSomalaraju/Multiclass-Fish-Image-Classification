# Multiclass Fish Image Classification using Deep Learning

# Overview
This project classifies fish species using CNN and transfer learning models such as VGG16, ResNet50, MobileNet, InceptionV3, and EfficientNetB0.

# Objective
- Compare pre trained models
- Identify best-performing model
- Deploy using Streamlit

---

# Models Used
- CNN (Custom)
- VGG16
- ResNet50 ✅ (Best)
- MobileNet
- InceptionV3
- EfficientNetB0

---

# Dataset Structure

data/
├── train/
├── val/
└── test/

# Approach

# Data Preprocessing & Augmentation

Rescale images to '[0,1]' range.
Apply augmentation techniques:
Rotation
Zoom
Flipping
Random cropping

# Model Training

Train a CNN model from scratch.
Experiment with five pre-trained models:
VGG16
ResNet50
MobileNet
InceptionV3
EfficientNetB0
Fine-tune models using the fish dataset.
Save the best model The models are saved in .h5 format for future use.

# Model Evaluation

Compare models based on:
Accuracy
Precision
Recall
F1-score
Confusion Matrix
Visualize training history (accuracy & loss curves).

# Deployment

A Streamlit web app with features:
Image Upload – Users upload fish images.
Prediction – Displays fish category.
Confidence Score – Shows model certainty.

