"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers, and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

# Function to preprocess data by flattening images
def preprocess_data(images):
    n_samples = len(images)
    data = images.reshape((n_samples, -1))
    return data

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

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

# 1. Get the data
digits = datasets.load_digits()

# Qualitative sanity check of the data
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.

# 2. Data Preprocessing
# Flatten the images
data = preprocess_data(digits.images)

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

# 3. Data splitting -- to create train, dev, and test sets
# Split data into train, dev, and test subsets
X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(data, digits.target, test_size=0.2, dev_size=0.1)

# 4. Model training on the training set
clf.fit(X_train, y_train)

# 5. Model Prediction on the test data and evaluation
predicted = predict_and_eval(clf, X_test, y_test)

###############################################################################
# Below we visualize the first 4 test samples and show their predicted
# digit value in the title.

# 6. Qualitative sanity check of prediction
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label, prediction in zip(axes, X_test, y_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"True: {label}\nPredicted: {prediction}")

plt.show()

###############################################################################
# :func:`~sklearn.metrics.classification_report` builds a text report showing
# the main classification metrics.

# 8. Evaluation 
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

###############################################################################
# We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# true digit values and the predicted digit values.

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()

###############################################################################
# If the results from evaluating a classifier are stored in the form of a
# :ref:`confusion matrix <confusion_matrix>` and not in terms of `y_true` and
# `y_pred`, one can still build a :func:`~sklearn.metrics.classification_report`
# as follows:


# The ground truth and predicted lists
y_true = []
y_pred = []
cm = disp.confusion_matrix

# For each cell in the confusion matrix, add the corresponding ground truths
# and predictions to the lists
for gt in range(len(cm)):
    for pred in range(len(cm)):
        y_true += [gt] * cm[gt][pred]
        y_pred += [pred] * cm[gt][pred]

print(
    "Classification report rebuilt from confusion matrix:\n"
    f"{metrics.classification_report(y_true, y_pred)}\n"
)
