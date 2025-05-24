# Chest X-Ray Disease Detection using Deep Learning

This project implements a deep learning-based pipeline to detect thoracic diseases from frontal chest X-ray images. It uses a DenseNet-121 architecture trained on a subset of the NIH ChestX-ray14 dataset and incorporates Grad-CAM for interpreting model predictions.

This implementation is adapted from the open-source work by [Laurent Veyssier](https://github.com/LaurentVeyssier/Chest-X-Ray-Medical-Diagnosis-with-Deep-Learning) and further structured for educational and research purposes.

---

## Project Motivation

Accurate and scalable diagnosis of chest diseases from X-rays is critical in clinical settings, especially where radiologist access is limited. This project explores the application of convolutional neural networks (CNNs) for multi-label classification of common thoracic pathologies.

---

## Core Concepts

### 1. Multi-label Classification

Unlike traditional single-label classification, patients may present with multiple coexisting conditions (e.g., Effusion and Atelectasis). This requires:
- Using **sigmoid activation** per output class
- Calculating **binary cross-entropy loss** independently for each disease
- Using metrics like **F1-score** and **ROC-AUC** per label

### 2. DenseNet-121 Architecture

DenseNet is a convolutional neural network that connects each layer to every other layer in a feed-forward fashion. Benefits:
- Efficient parameter usage
- Strong gradient flow
- Great performance on medical image tasks

### 3. Grad-CAM (Gradient-weighted Class Activation Mapping)

Grad-CAM allows visualization of the image regions that influence the model’s predictions. This helps:
- Improve interpretability
- Assist clinical trust
- Identify model failure cases

---

## Dataset

- **Source**: NIH ChestX-ray14 dataset
- **Images**: 112,000+ frontal-view X-rays
- **Labels**: 14 thoracic diseases including Pneumonia, Mass, Edema, Cardiomegaly, Infiltration, etc.
- **Subset Used**: A manually sampled and resized subset (~1,000 images) for demonstration and model development

---

## Implementation Summary

- **Model**: DenseNet-121 pretrained on ImageNet
- **Loss Function**: Binary Cross-Entropy
- **Optimizer**: Adam
- **Input Processing**: Grayscale conversion, resizing to 224x224, normalization
- **Visualization**: Grad-CAM with overlayed heatmaps

---

## Directory Structure
├── ChestXRay_Medical_Diagnosis_Deep_Learning.ipynb # Main Jupyter notebook
├── densenet.7z # Pretrained model weights
├── images/ # Sample Grad-CAM outputs
├── utils/ # Helper functions for model and visualization
└── README.md


---

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/samhitasari05/chest-xray-disease-detection.git
cd chest-xray-disease-detection

2. Extract pretrained model weights:
# Unzip the .7z file or use weights provided in notebook

3. Install dependencies:
pip install -r requirements.txt

4. Run the Jupyter notebook:
jupyter notebook ChestXRay_Medical_Diagnosis_Deep_Learning.ipynb

Results
Achieved strong multi-label classification performance on test data

Demonstrated interpretable decision-making via Grad-CAM overlays

Visual feedback aligns with expected pathology regions in many cases

Future Work
Fine-tune on larger subset or full ChestX-ray14 dataset

Expand to include DICOM input and PACS integration

Compare Grad-CAM with other interpretability tools (e.g., LIME, SHAP)

Integrate with clinical decision dashboards using Streamlit

License and Attribution
This work is adapted from the original project by Laurent Veyssier and retains core structure and methodology under the MIT License.

All enhancements including documentation, restructuring, and downstream planning are made by Samhita Sarikonda for academic and portfolio use.

