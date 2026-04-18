# 🧠 GAN + Autoencoder + Machine Learning Pipeline for Histopathological Image Classification

## 📌 Overview

This project presents a hybrid deep learning and machine learning framework for histopathological image classification. It integrates:

- Generative Adversarial Networks (GAN)
- Convolutional Autoencoder
- Latent Feature Extraction
- Classical Machine Learning Classifiers

The system is designed to classify histopathological images and map **5 raw tumor-related classes** into **3 clinically meaningful categories**:

- Normal  
- Benign  
- Malignant  

The learned latent representations from the autoencoder are used as input features for multiple machine learning classifiers.

---

## ⚙️ Methodology

The pipeline follows these main stages:

1. **Image Preprocessing**
   - Resize images
   - Normalize pixel values
   - Load dataset using structured folders

2. **Label Mapping**
   - Converts 5 raw classes into 3 clinical groups:
     - Normal
     - Benign
     - Malignant

3. **Feature Learning**
   - Convolutional Autoencoder is trained to reconstruct input images
   - Encoder learns compressed latent representations

4. **Latent Feature Extraction**
   - Encoder output is used as feature vectors

5. **Classification**
   - Multiple machine learning models are trained:
     - Support Vector Machine (SVM)
     - Random Forest
     - K-Nearest Neighbors (KNN)
     - Logistic Regression

6. **Evaluation**
   - Accuracy
   - Precision, Recall, F1-score
   - Classification report

---

## 🧪 Models Used

- **GAN (Conceptual/Support Module)**  
  Used for generative enhancement (optional extension in pipeline)

- **Convolutional Autoencoder**  
  Learns compact feature representation from images

- **Classical ML Classifiers**
  - SVM (RBF kernel)
  - Random Forest (200 estimators)
  - KNN (k=5)
  - Logistic Regression

---

## 📂 Dataset

This project uses the publicly available **Osteosarcoma-Tumor-Assessment dataset** from The Cancer Imaging Archive (TCIA).

- 🔗 DOI: https://doi.org/10.7937/tcia.2019.bvhjhdas  
- 📜 License: CC BY 3.0  
- 📦 Source: The Cancer Imaging Archive (TCIA)

### 📖 Citation

> Leavey, P., Sengupta, A., Rakheja, D., Daescu, O., Arunachalam, H. B., & Mishra, R. (2019).  
> Osteosarcoma data from UT Southwestern/UT Dallas for Viable and Necrotic Tumor Assessment (Osteosarcoma-Tumor-Assessment) [Data set].  
> The Cancer Imaging Archive. https://doi.org/10.7937/tcia.2019.bvhjhdas

---

## ⚠️ Important Note

- The dataset is **not included in this repository** due to licensing restrictions.
- Only the source code is provided.
- Users must manually download the dataset from TCIA before running the project.

---
## Dataset Setup

Users must:
1. Download the dataset from TCIA:
   https://doi.org/10.7937/tcia.2019.bvhjhdas

2. Organize it as:
   dataset/
     ├── train/
     └── test/

3. Update the dataset path in the code:
   train_path = "path/to/train"
   test_path  = "path/to/test"

## 🚀 Project Goal

To demonstrate how deep feature learning (Autoencoder) combined with classical machine learning can effectively classify medical images into clinically meaningful categories.

---

## 🧑‍💻 Author

**Vandana B.S**

---
