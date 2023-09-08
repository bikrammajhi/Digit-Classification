# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Read digits
def read_digits():
    digits = datasets.load_digits()
    x = digits.images
    y = digits.target 
    return x, y

# Preprocess data
def preprocess_data(data):
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data

# Split data into train and test subsets
def split_data(X, y, test_size=0.5, random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    return X_train, X_test, y_train, y_test

# Create a classifier: a support vector classifier
def train_model(X, y, model_params, model_type='svm'):
    if model_type == 'svm':
        clf = svm.SVC(**model_params)
    clf.fit(X, y)
    return clf

# Function to split data into train, dev, and test sets
def split_train_dev_test(X, y, test_size, dev_size):
    # Ensure that test_size and dev_size sum up to less than 1.0
    assert test_size + dev_size < 1.0, "Test and dev sizes sum to more than 1.0"
    
    # First, split the data into training and the rest
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size + dev_size), random_state=42)
    
    # Calculate the ratio of the remaining data to be used for the development set
    dev_ratio = dev_size / (test_size + dev_size)
    
    # Now, split the remaining data into dev and test sets
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=dev_ratio, random_state=42)
    
    return X_train, X_dev, X_test, y_train, y_dev, y_test

# Function to predict and evaluate a model on the test data
def predict_and_eval(model, X_test, y_test):
    # Predict the labels on the test set
    predicted = model.predict(X_test)

    # Print the classification report
    print(
        f"Classification report for classifier {model}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )

    # Plot the confusion matrix
    disp = metrics.ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")

    plt.show()
    return predicted


def tune_hparams(X_train, y_train, X_dev, y_dev, list_of_all_param_combinations):
    best_accuracy = 0.0
    best_hparams = None
    best_model = None

    for param_combination in list_of_all_param_combinations:
        # Create a model with the current hyperparameter combination
        model = train_model(X_train, y_train, param_combination)
        
        # Evaluate the model on the development set
        y_dev_pred = model.predict(X_dev)
        accuracy = accuracy_score(y_dev, y_dev_pred)
        
        # Check if this hyperparameter combination resulted in a better model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_hparams = param_combination
            best_model = model

    return best_hparams, best_model, best_accuracy


