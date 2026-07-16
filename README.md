# Brain Tumor MRI Classification

This project focuses on classifying brain MRI images using deep learning.  
The goal was to build a custom CNN model from scratch, compare it with pretrained transfer learning models, and deploy the best model using Streamlit.

## Project Overview

The dataset contains brain MRI images classified into four categories:

- Glioma
- Meningioma
- No Tumor
- Pituitary Tumor

The project includes the full machine learning workflow:

1. Dataset loading and exploration
2. Image preprocessing
3. Data augmentation
4. Custom CNN model training
5. Transfer learning using pretrained models
6. Model evaluation and comparison
7. Deployment using Streamlit

## Models Used

Three models were trained and compared:

- Custom CNN built from scratch
- MobileNetV2
- EfficientNetB0

## Evaluation Metrics

The models were compared using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Training time
- Overfitting analysis
- Accuracy and loss curves

## Results

| Model | Test Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Custom CNN | 82% | 82% | 82% | 81% |
| MobileNetV2 | 85% | 86% | 85% | 85% |
| EfficientNetB0 | 84% | 85% | 84% | 84% |

MobileNetV2 achieved the best overall performance and was selected for deployment.

## Streamlit App

The final model was deployed using Streamlit.  
The app allows users to upload a brain MRI image and returns:

- Predicted class
- Confidence score
- Class probability chart

## How to Run the App

Install dependencies:

```bash
pip install -r requirements.txt
