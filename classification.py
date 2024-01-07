# -*- coding: utf-8 -*-
"""
Created on [Current Date]

@author: Sneha
"""

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

import zipfile

# Specify the path to your ZIP file
zip_path = "C:/Users/Sneha/Downloads/titanic.zip"

# Extract each CSV file from the ZIP archive
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Load gender_submission
    with zip_ref.open('gender_submission.csv') as file:
        gender_submission = pd.read_csv(file)

    # Load test
    with zip_ref.open('test.csv') as file:
        test = pd.read_csv(file)

    # Load train
    with zip_ref.open('train.csv') as file:
        train = pd.read_csv(file)

# Data Preprocessing
# For simplicity, let's drop some columns and handle missing values
df_train = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]
df_train = df_train.dropna()

df_test = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

# Convert categorical variables to numeric using .loc
df_train.loc[:, 'Sex'] = df_train['Sex'].map({'male': 0, 'female': 1})
df_test.loc[:, 'Sex'] = df_test['Sex'].map({'male': 0, 'female': 1})

# Separate features and labels
X_train = df_train.drop('Survived', axis=1).values
y_train = df_train['Survived'].values

X_test = df_test.values

# Naive Bayes Implementation
def calculate_prior_probability(y_train):
    unique_classes, counts = np.unique(y_train, return_counts=True)
    prior_probability = dict(zip(unique_classes, counts / len(y_train)))
    return prior_probability

def calculate_likelihood(x_train, y_train, feature_value, class_value):
    total_class_instances = np.sum(y_train == class_value)
    class_occurrences = x_train[y_train == class_value]
    feature_occurrences = class_occurrences[:, x_train[y_train == class_value][0] == feature_value]

    likelihood = (len(feature_occurrences) + 1) / (total_class_instances + len(np.unique(x_train)))
    return likelihood

def naive_bayes_predict(x_train, y_train, x_test):
    prior_probability = calculate_prior_probability(y_train)
    predictions = []

    for instance in x_test:
        class_probabilities = {}
        for class_value in np.unique(y_train):
            likelihood = 1
            for i, feature_value in enumerate(instance):
                likelihood *= calculate_likelihood(x_train[:, i], y_train, feature_value, class_value)
            class_probabilities[class_value] = prior_probability[class_value] * likelihood

        predicted_class = max(class_probabilities, key=class_probabilities.get)
        predictions.append(predicted_class)

    return np.array(predictions)

# K-Nearest Neighbours Implementation
def calculate_distance(point1, point2):
    point1 = np.array(point1, dtype=np.float64)  # Convert to float
    point2 = np.array(point2, dtype=np.float64)  # Convert to float
    return np.sqrt(np.sum((point1 - point2) ** 2))

def knn_predict(x_train, y_train, x_test, k=3):
    predictions = []

    for instance in x_test:
        distances = [(calculate_distance(instance, x_train[i]), y_train[i]) for i in range(len(x_train))]
        sorted_distances = sorted(distances, key=lambda x: x[0])
        k_nearest_neighbors = sorted_distances[:k]

        unique_classes, counts = np.unique(np.array(k_nearest_neighbors)[:, 1], return_counts=True)
        predicted_class = unique_classes[np.argmax(counts)]
        predictions.append(predicted_class)

    return np.array(predictions)

# Example Usage
predictions_nb = naive_bayes_predict(X_train, y_train, X_test)
predictions_knn = knn_predict(X_train, y_train, X_test)
