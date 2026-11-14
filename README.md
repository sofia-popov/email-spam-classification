# Email Spam Classification Using Ensemble Learning

A machine learning project developed for **CSDS 340: Introduction to Machine Learning** at Case Western Reserve University, in collaboration with Phillip Kornberg.

## Project Overview

This project focuses on developing a personalized email spam classifier that distinguishes spam from legitimate emails while minimizing false positives.

We explored multiple classification algorithms, preprocessing techniques, and ensemble configurations to identify an effective approach for handling missing data and improving spam detection.

## Model Architecture

Our final model combines four classifiers:

- 2 Histogram-Based Gradient Boosting classifiers
- 2 Logistic Regression classifiers

The Gradient Boosting models handle missing values directly, while the Logistic Regression models use quantile normalization and iterative imputation.

The final spam probability is calculated by averaging predictions from all four models.

## Methodology

- Analyzed **3,000 labeled emails** with 30 numerical features and approximately 20% missing values.
- Compared seven classification algorithms, including Naive Bayes, kNN, SVM, Random Forest, Extra Trees, Logistic Regression, and Histogram-Based Gradient Boosting.
- Evaluated missing-value imputation, feature selection, and normalization techniques.
- Used stratified 5-fold cross-validation to evaluate model performance.
- Optimized model hyperparameters using randomized search.

## Results

Our final ensemble achieved the following cross-validation performance:

| Metric | Result |
|---|---|
| ROC AUC | 0.9143 |
| True Positive Rate at 1% False Positive Rate | 50.78% |

We prioritized spam detection at a low false positive rate because incorrectly marking legitimate emails as spam can have more serious consequences than allowing some spam emails through.

These results are based on cross-validation during model development, rather than an independent final test set.

## Technical Report

For a detailed explanation of our methodology, preprocessing, model selection, and evaluation results, see our [Full Technical Report](Email_Spam_Classification_Report.pdf).

## Technologies

Python · scikit-learn · Machine Learning · Ensemble Learning · Data Preprocessing · Model Evaluation

## Authors

Sofia Popov and Phillip Kornberg

Case Western Reserve University | Fall 2026
