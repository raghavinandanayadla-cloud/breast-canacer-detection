# Breast Cancer Classification Using Machine Learning
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)
![Status](https://img.shields.io/badge/Project-Completed-green.svg)
![Dataset](https://img.shields.io/badge/Dataset-Wisconsin%20Breast%20Cancer-red.svg)

---

## Project Overview

This project aims to classify breast tumors as **Benign (Non-Cancerous)** or **Malignant (Cancerous)** using various Machine Learning algorithms. The Breast Cancer Wisconsin (Diagnostic) Dataset was used to train and evaluate multiple models. The project also analyzes the impact of **Standard Scaling** and **SMOTE (Synthetic Minority Over-sampling Technique)** on model performance.

---

## Dataset Description

The Breast Cancer Wisconsin (Diagnostic) Dataset contains features computed from digitized images of breast mass cell nuclei. The objective is to predict whether a tumor is benign or malignant based on these features.

### Dataset Information

- **Total Samples:** 569
- **Features:** 30 Numerical Features
- **Target Variable:** Diagnosis
- **Missing Values:** None

### Target Classes

| Value | Class |
|---------|---------|
| 0 | Benign (B) |
| 1 | Malignant (M) |

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn
- Imbalanced-Learn (SMOTE)

---

## Machine Learning Models Implemented

### Logistic Regression
Predicts the probability of a sample belonging to a particular class using a logistic function.

### Decision Tree
A tree-based model that makes predictions by splitting data into decision nodes.

### Random Forest
An ensemble method that combines multiple decision trees to improve accuracy and reduce overfitting.

### Support Vector Machine (SVM)

#### Linear Kernel
Suitable for linearly separable datasets.

#### RBF Kernel
Handles complex non-linear relationships by mapping data into higher dimensions.

#### Polynomial Kernel
Creates polynomial decision boundaries for classification.

### K-Nearest Neighbors (KNN)
Classifies a sample based on the majority class of its nearest neighbors.

### Bagging Classifier
Builds multiple models on different subsets of the training data and combines their predictions.

### AdaBoost Classifier
Boosting technique that improves performance by focusing on previously misclassified samples.

### Voting Classifier
Combines predictions from multiple models using majority voting or probability averaging.

### Stacking Classifier
Uses multiple base models and a meta-model to make final predictions.

### Linear Discriminant Analysis (LDA)
Reduces feature dimensions while maximizing class separation.

---

## Data Preprocessing

### Standard Scaling

Standard Scaling transforms features to:

- Mean = 0
- Standard Deviation = 1

This helps distance-based algorithms such as SVM and KNN perform better.

### SMOTE

SMOTE (Synthetic Minority Over-sampling Technique) balances the dataset by generating synthetic samples of the minority class.

**Benefits:**
- Reduces class imbalance
- Improves minority class detection
- Enhances model generalization

---

## Model Performance Comparison

| Model | Accuracy Before | Accuracy After Scaling & SMOTE |
|---------|---------|---------|
| Logistic Regression | 93.8% | 98.0% |
| Decision Tree | 93.8% | 96.0% |
| Random Forest | 96.5% | 97.0% |
| SVM (Linear) | 93.8% | 96.0% |
| SVM (RBF) | 62.2% | 97.0% |
| SVM (Polynomial) | 61.4% | 90.0% |
| KNN | 75.4% | 96.0% |

---

## Results and Analysis

### Logistic Regression
- Accuracy improved from **93.8%** to **98.0%**
- ROC-AUC Score: **0.997**

### Decision Tree
- Accuracy improved from **93.8%** to **96.0%**
- Achieved **100% Recall** for malignant tumors

### Random Forest
- Accuracy improved from **96.5%** to **97.0%**
- ROC-AUC Score: **0.999**

### SVM (RBF Kernel)
- Largest performance improvement
- Accuracy increased from **62.2%** to **97.0%**

### SVM (Polynomial Kernel)
- Accuracy increased from **61.4%** to **90.0%**

### KNN
- Accuracy increased from **75.4%** to **96.0%**
- Benefited significantly from feature scaling

---

## ROC-AUC Evaluation

The ROC Curve measures a model's ability to distinguish between benign and malignant tumors.

### AUC Interpretation

- **1.0** → Perfect Classification
- **0.9 – 1.0** → Excellent Model
- **0.8 – 0.9** → Good Model
- **0.5** → Random Guessing

### Best ROC-AUC Scores

| Model | ROC-AUC |
|---------|---------|
| Random Forest | 0.999 |
| Logistic Regression | 0.997 |
| SVM (RBF) | 0.997 |

---

## Key Findings

- Standard Scaling significantly improved SVM and KNN performance.
- SMOTE improved the detection of malignant tumors.
- Random Forest achieved the highest ROC-AUC score.
- Logistic Regression delivered excellent performance despite being a simple model.
- SVM with RBF Kernel showed the greatest improvement after preprocessing.

---

## Conclusion

This project demonstrates the importance of data preprocessing in machine learning. Applying **Standard Scaling** and **SMOTE** significantly improved the performance of most classification models, particularly SVM and KNN.

Among all models tested, **Random Forest**, **Logistic Regression**, and **SVM (RBF Kernel)** achieved the best overall results for breast cancer classification.

Machine learning can serve as an effective tool for supporting early breast cancer detection and assisting healthcare professionals in diagnostic decision-making.

---

---

# Ensemble Learning Techniques and Performance Evaluation

## What is Ensemble Learning?

Ensemble Learning is a machine learning approach that combines multiple models to improve prediction accuracy, robustness, and generalization. Instead of relying on a single model, ensemble methods leverage the strengths of multiple learners to produce better results.

### Advantages of Ensemble Learning

- Reduces overfitting
- Improves prediction accuracy
- Enhances model stability
- Handles complex datasets effectively
- Provides better generalization on unseen data

---

## Ensemble Models Implemented

### Random Forest Classifier

Random Forest is a bagging-based ensemble algorithm that combines multiple Decision Trees trained on different subsets of data.

#### Key Features
- Reduces variance and overfitting
- Handles high-dimensional datasets
- Provides feature importance scores
- Robust against noisy data

#### Performance
| Metric | Score |
|----------|----------|
| Accuracy | 97.0% |
| ROC-AUC | 0.999 |

---

### Bagging Classifier

Bagging (Bootstrap Aggregating) trains multiple models on randomly sampled subsets of the training dataset and combines their predictions.

#### Working Principle
1. Create multiple bootstrap samples.
2. Train a base estimator on each sample.
3. Aggregate predictions using majority voting.

#### Benefits
- Reduces variance
- Improves model stability
- Prevents overfitting

---

### AdaBoost Classifier

AdaBoost (Adaptive Boosting) is a boosting algorithm that sequentially trains weak learners while assigning higher weights to previously misclassified samples.

#### Working Principle
1. Train a weak learner.
2. Increase weight of misclassified samples.
3. Train next learner focusing on difficult cases.
4. Combine weighted predictions.

#### Benefits
- Improves classification performance
- Focuses on difficult observations
- Reduces bias

---

### Voting Classifier

Voting Classifier combines multiple machine learning models and predicts the final class using voting strategies.

#### Types of Voting

##### Hard Voting
Predicts the class receiving the majority vote.

##### Soft Voting
Predicts the class based on averaged class probabilities.

#### Benefits
- Utilizes strengths of multiple algorithms
- Often achieves better performance than individual models
- Reduces prediction variance

---

### Stacking Classifier

Stacking combines multiple base models and trains a meta-model on their predictions.

#### Architecture

Level 0 Models:
- Logistic Regression
- Decision Tree
- Random Forest
- SVM

Level 1 Model:
- Meta Learner (Logistic Regression)

#### Benefits
- Captures strengths of diverse models
- Improves predictive performance
- Handles complex decision boundaries

---

## Evaluation Metrics Used

To comprehensively evaluate model performance, multiple classification metrics were used.

---

### Accuracy

## Evaluation Metrics Used

To comprehensively evaluate model performance, multiple classification metrics were used.

---

### Accuracy

Accuracy measures the percentage of correctly classified samples.

#### Formula

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

#### Where

- **TP** = True Positives
- **TN** = True Negatives
- **FP** = False Positives
- **FN** = False Negatives

#### Interpretation

Higher accuracy indicates better overall classification performance.

---

### Precision

Precision measures how many predicted positive cases are actually positive.

#### Formula

$$
Precision = \frac{TP}{TP + FP}
$$

#### Importance

In medical diagnosis, high precision reduces false alarms and unnecessary treatments.

---

### Recall (Sensitivity)

Recall measures the model's ability to correctly identify positive cases.

#### Formula

$$
Recall = \frac{TP}{TP + FN}
$$

#### Importance

For breast cancer detection, high recall is critical because missing malignant cases can have severe consequences.

---

### F1 Score

The F1 Score balances Precision and Recall.

#### Formula

$$
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$

#### Importance

Useful when classes are imbalanced because it considers both Precision and Recall.

---

### ROC Curve

The Receiver Operating Characteristic (ROC) Curve visualizes the trade-off between:

- True Positive Rate (Recall)
- False Positive Rate (FPR)

A curve closer to the top-left corner indicates better classification performance.

---

### Confusion Matrix Interpretation

| Actual / Predicted | Benign | Malignant |
|-------------------|---------|------------|
| Benign | True Negative (TN) | False Positive (FP) |
| Malignant | False Negative (FN) | True Positive (TP) |

#### Medical Perspective

- **False Positive (FP):** Patient is predicted to have cancer but actually does not.
- **False Negative (FN):** Patient has cancer but the model predicts otherwise.
- **False Negatives are generally more dangerous in cancer diagnosis because they may delay treatment.**

---

---

## Live Web Application

A user-friendly web interface has been developed using **Streamlit** to make breast cancer prediction accessible without requiring programming knowledge.

### Features

- Interactive and responsive web interface
- Real-time breast cancer prediction
- Input all diagnostic features through a simple form
- Instant classification as **Benign** or **Malignant**
- Machine Learning model integration for accurate predictions
- Easy deployment using Streamlit Cloud

### Live Demo

🚀 **Try the Application Here:**

https://breast-cancer-detection9.streamlit.app/

### Application Workflow

1. Enter the diagnostic feature values.
2. Click the **Predict** button.
3. The trained machine learning model processes the inputs.
4. The application displays the predicted diagnosis:
   - **Benign (Non-Cancerous)**
   - **Malignant (Cancerous)**

### Technologies Used for Deployment

- Streamlit
- Python
- Scikit-Learn
- Pandas
- NumPy

### Deployment Architecture

```text
User Input
     ↓
Streamlit Interface
     ↓
Trained ML Model
     ↓
Prediction Engine
     ↓
Diagnosis Result
```

---

## Screenshots

| Home Page | EDA |
|------------|------------------|
| ![Home](assets/home.png) | ![Prediction](assets/eda.png) |

| model | metrics |
|------------|------------------|
| ![Home](assets/model.png) | ![Prediction](assets/metrics.png) |

| c |
|------------|
| ![Home](assets/c.png) |

| ensemble | 
|------------|
| ![Home](assets/ensenble.png) | 

| new diagnosis | 
|------------|
| ![Home](assets/new.png) | 


---


## Future Improvements

- Hyperparameter Tuning using Grid Search
- Cross-Validation for more robust evaluation
- Deep Learning Models (ANN/CNN)
- Explainable AI using SHAP and LIME

---



**Ragaa**

Machine Learning Project on Breast Cancer Classification using Python and Scikit-Learn.
