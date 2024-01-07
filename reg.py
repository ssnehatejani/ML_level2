# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 16:45:56 2024

@author: Sneha
"""

# Import necessary libraries
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("C:/Users/Sneha/Downloads/Medical Price Dataset.csv")

# Data Preprocessing
# In this example, we'll drop non-numeric columns and handle missing values
df = df.select_dtypes(include=[np.number]).fillna(0)

# Separate features and labels
X_train = df.drop('charges', axis=1)
y_train = df['charges']

# Linear Regression Implementation
def linear_regression(x_train, y_train, learning_rate=0.001, epochs=1000):
    # Normalize input features
    x_train = (x_train - x_train.mean()) / x_train.std()

    # Initialize weights and bias
    weights = np.zeros(x_train.shape[1])
    bias = 0

    # Gradient Descent
    for epoch in range(epochs):
        # Predictions
        predictions = np.dot(x_train, weights) + bias

        # Calculate errors
        errors = predictions - y_train

        # Update weights and bias
        weights -= learning_rate * (1/x_train.shape[0]) * np.dot(x_train.T, errors)
        bias -= learning_rate * (1/x_train.shape[0]) * np.sum(errors)

        # Optional: Print the loss to monitor training
        loss = np.mean(errors**2)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")

        # Check for NaN values in weights
        if np.isnan(weights).any():
            print("NaN values encountered in weights. Adjust the learning rate.")
            break

    return weights, bias

# Example Usage
weights, bias = linear_regression(X_train.values, y_train.values)
