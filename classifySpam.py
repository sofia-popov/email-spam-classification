
"""
Spam Classification Model Using Ensemble Learning

Authors: Phillip Kornberg, Sofia Popov
Date: 11.1.26
"""

# Importing Libraries
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score, roc_curve

# Hiding Convergence Warnings
warnings.filterwarnings("ignore")

# Function to Train Models and Report Spam Scores
def predictTest(trainFeatures, trainLabels, testFeatures):

    # Helper Function to Create Four Models for Ensemble Learning
    def buildModels():

        # Building Boosted Trees Version A
        hgbA = HistGradientBoostingClassifier(l2_regularization = 0.6896073227486572, learning_rate = 0.03007289141770309, max_depth = 4,
                                              max_features = 0.6280834179040147, max_iter = 483, max_leaf_nodes = 27,
                                              min_samples_leaf = 10, random_state = 4)

        # Building Boosted Trees Version B
        hgbB = HistGradientBoostingClassifier(l2_regularization = 5.7135511848418, learning_rate = 0.1717082874174535, max_depth = 2,
                                              max_features = 0.3166657939188432, max_iter = 840, max_leaf_nodes = 3,
                                              min_samples_leaf = 11, random_state = 4)

        # Building Logistic Regression Version A
        logRegA = make_pipeline(QuantileTransformer(n_quantiles = 497, output_distribution = "normal", random_state = 4),
                                IterativeImputer(max_iter = 7, random_state = 4, tol = 0.013993707456361838),
                                LogisticRegression(C = 0.01325437558961157, max_iter = 5000, random_state = 4))

        # Building Logistic Regression Version B
        logRegB = make_pipeline(QuantileTransformer(n_quantiles = 385, output_distribution = "normal", random_state = 4),
                                IterativeImputer(max_iter = 9, random_state = 4, tol = 0.022426022487286996),
                                LogisticRegression(C = 0.013336310707556967, max_iter = 5000, random_state = 4))

        # Return Four Models
        return [hgbA, hgbB, logRegA, logRegB]

    # Preparing Data Copies
    xTrain = np.array(trainFeatures, dtype = float)
    xTest = np.array(testFeatures, dtype = float)

    # Converting Missing-Value Indicators to NaN
    xTrain[xTrain == -1] = np.nan
    xTest[xTest == -1] = np.nan

    yTrain = np.array(trainLabels).ravel().astype(int)

    # Initializing Model Scores
    modelScores = []

    # Looping Through Each Model
    for model in buildModels():

        # Fitting Model on All Training Data
        fittedModel = model.fit(xTrain, yTrain)

        # Predicting Test Email Probabilities
        testProbs = fittedModel.predict_proba(xTest)[:, 1]

        modelScores.append(testProbs)

    # Combining Model Scores by Averaging
    return np.mean(modelScores, axis = 0)
