# Applied Machine Learning for Industrial Data Analysis

## Project Overview

This project applies a complete machine learning pipeline to the **SECOM Semiconductor Manufacturing Dataset** to detect manufacturing failures using high-dimensional sensor data.

The aim of the project is to analyse industrial sensor readings, reduce dimensionality, identify hidden process patterns, and build classification models that can support early fault detection in semiconductor manufacturing.

Semiconductor manufacturing produces large volumes of sensor data from equipment and process monitoring systems. Even small deviations in sensor readings can lead to defective wafers. This project explores how machine learning can help identify these rare failures before they cause quality or production issues.

## Research Aim

The main aim of this project is to build an end-to-end machine learning workflow for semiconductor fault detection using the SECOM dataset.

The project focuses on:

- Cleaning high-dimensional industrial sensor data
- Handling missing values and noisy features
- Exploring class imbalance between Pass and Fail outcomes
- Applying Principal Component Analysis (PCA)
- Performing clustering to identify hidden process groups
- Training and evaluating classification models
- Critically analysing model performance for rare failure detection

## Dataset

**Dataset:** SECOM Semiconductor Manufacturing Dataset  
**Source:** UCI Machine Learning Repository  
**Dataset ID:** 179  

The dataset contains semiconductor manufacturing sensor measurements and a binary quality outcome.

### Dataset Summary

| Property | Value |
|---|---:|
| Total samples | 1,567 |
| Original sensor features | 590 |
| Pass samples | 1,463 |
| Fail samples | 104 |
| Failure rate | 6.6% |
| Final cleaned features | 446 |

The original labels were encoded as:

- `-1` = Pass
- `1` = Fail

For this project, the labels were converted to:

- `0` = Pass
- `1` = Fail

## Problem Type

This is a **binary classification problem** with severe class imbalance.

The main challenge is that only **6.6%** of the samples represent failures. This means accuracy alone is not a reliable metric, because a model could predict every sample as “Pass” and still achieve high accuracy while detecting no failures.

For this reason, the project focuses more on:

- Recall for the Fail class
- Precision for the Fail class
- F1-score
- Macro F1-score
- AUC-PR

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- SciPy

## Machine Learning Techniques

- Missing value analysis
- Median imputation
- Standardisation using StandardScaler
- Outlier analysis using Z-score and IQR
- Principal Component Analysis (PCA)
- K-Means Clustering
- Hierarchical Clustering
- SMOTE oversampling
- k-Nearest Neighbours
- Decision Tree
- Support Vector Machine
- Hyperparameter tuning using GridSearchCV and RandomizedSearchCV

## Project Workflow

```text
Raw SECOM Dataset
        |
        v
Data Loading
        |
        v
Missing Value Analysis
        |
        v
Feature Cleaning
- Remove constant features
- Remove high-missing features
- Median imputation
- Remove zero-variance features
        |
        v
Feature Scaling
        |
        v
Exploratory Data Analysis
        |
        v
Outlier Analysis
        |
        v
PCA Dimensionality Reduction
        |
        v
Clustering Analysis
- K-Means
- Hierarchical Clustering
        |
        v
Train/Test Split
        |
        v
SMOTE on Training Data
        |
        v
Model Training
- k-NN
- Decision Tree
- SVM
        |
        v
Model Evaluation
        |
        v
Critical Analysis
