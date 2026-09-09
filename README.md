# Pneumonia Detection from Chest X-Rays using Deep Learning

A deep learning project that classifies chest X-ray images into two categories: **Normal** and **Pneumonia**.

The project uses a Convolutional Neural Network (CNN) built with **TensorFlow/Keras**. The dataset was stored in **Amazon S3**, while **Amazon SageMaker** was used as the development and model training environment. After training, the model was evaluated on validation and unseen test data, and a prediction function was used to classify individual X-ray images.

> **Disclaimer:** This project was developed for academic and learning purposes. It is not clinically validated and should not be used for medical diagnosis or treatment decisions.

---

## Overview

Pneumonia is a respiratory infection that can be identified using medical imaging such as chest X-rays. This project explores the use of deep learning for automatically identifying pneumonia-related patterns in chest X-ray images.

The task is treated as a **binary image classification problem**:

- **NORMAL** - X-ray classified as normal
- **PNEUMONIA** - X-ray classified as showing pneumonia

The project covers the workflow from storing and preparing the dataset to training, evaluating, and testing the CNN model.

### Project Workflow

```text
Chest X-Ray Dataset
        │
        ▼
    Amazon S3
        │
        ▼
 Amazon SageMaker
        │
        ▼
Image Preprocessing
        │
        ▼
   CNN Model
        │
        ▼
     Training
        │
        ▼
Validation & Testing
        │
        ▼
Individual X-Ray Prediction
        │
        ▼
NORMAL / PNEUMONIA
