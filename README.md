# GAN-Autoencoder Classification Pipeline  
**Author:** Vandana B.S  

---

## Overview

This project presents a deep learning pipeline combining:

- Generative Adversarial Networks (GAN)
- Convolutional Autoencoder
- Latent Feature Extraction
- Classical Machine Learning Classifiers

The system is designed for **histopathological image classification** and maps **5 raw tumor classes into 3 clinically meaningful categories**:

- Normal  
- Benign  
- Malignant  

The extracted latent representations are used to train multiple classifiers including SVM, Random Forest, KNN, and Logistic Regression.

---

## Dataset

This project uses the publicly available **Osteosarcoma-Tumor-Assessment dataset** from The Cancer Imaging Archive (TCIA).

- **DOI:** https://doi.org/10.7937/tcia.2019.bvhjhdas  
- **License:** CC BY 3.0  
- **Source:** The Cancer Imaging Archive (TCIA)

### Citation

> Leavey, P., Sengupta, A., Rakheja, D., Daescu, O., Arunachalam, H. B., & Mishra, R. (2019). Osteosarcoma data from UT Southwestern/UT Dallas for Viable and Necrotic Tumor Assessment (Osteosarcoma-Tumor-Assessment) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/tcia.2019.bvhjhdas

---

## Important Note

- The dataset is **not included in this repository** due to licensing restrictions.
- Only source code is provided.
- Users must manually download the dataset from TCIA.

---

## Class Mapping (5 → 3 Groups)

The original labels are mapped into three clinical categories:

| Original Class         | Mapped Class |
|----------------------|--------------|
| Non-Tumor           | Normal       |
| Viable              | Benign       |
| Tumor               | Malignant    |
| Non-Viable-Tumor    | Malignant    |
| viable non-viable   | Malignant    |

---

## Methodology

### 1. Data Loading
- CSV-based dataset loading using custom PyTorch Dataset
- Image paths constructed dynamically
- Labels extracted from CSV file

---

### 2. Preprocessing
- Image resizing to 64×64
- Tensor normalization
- Label mapping to 3 classes

---

### 3. Autoencoder Feature Learning
A convolutional autoencoder is trained to learn compact latent representations.

**Architecture:**
- Encoder: 3 convolution layers (stride=2)
- Decoder: 3 transposed convolution layers
- Loss function: Mean Squared Error (MSE)

---

### 4. GAN Module (Optional Extension)
A simple GAN architecture is included:

- Generator: latent vector → synthetic image  
- Discriminator: real vs fake image classification  

*(Used for future data augmentation research)*

---

### 5. Feature Extraction
- Latent vectors are extracted from encoder output
- Flattened feature vectors are used for classification

---

### 6. Classification Models

The following machine learning models are trained on latent features:

- Support Vector Machine (SVM)
- Random Forest (200 estimators)
- K-Nearest Neighbors (KNN)
- Logistic Regression

---

## Pipeline Steps

1. Load CSV dataset  
2. Map labels (5 → 3 classes)  
3. Load images using custom PyTorch Dataset  
4. Train autoencoder  
5. Extract latent features  
6. Train ML classifiers  
7. Evaluate performance  

---

## Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Classification report per class  

---

## Requirements

Install dependencies:

```bash
pip install torch torchvision scikit-learn pandas numpy pillow
