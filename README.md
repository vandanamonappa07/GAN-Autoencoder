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
- After download, structure the dataset as follows:
