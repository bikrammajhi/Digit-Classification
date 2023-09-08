# Import datasets, classifiers, and performance metrics
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from utils import preprocess_data, train_model, split_train_dev_test, predict_and_eval, tune_hparams, accuracy_score

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

disp = metrics.ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
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

# Define a list of hyperparameter combinations as dictionaries
param_combinations = [
    {'C': 1.0, 'kernel': 'linear'},
    {'C': 0.1, 'kernel': 'rbf'},
    {'C': 0.01, 'kernel': 'linear'},
    {'C': 0.001, 'kernel': 'rbf'},
    {'C': 10.0, 'kernel': 'linear'},
]

# Loop through different hyperparameter combinations
best_hparams = None
best_accuracy = 0.0

for param_combination in param_combinations:
    # Hyperparameter Tuning
    current_best_hparams, _, current_best_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, [param_combination])
    
    if current_best_accuracy > best_accuracy:
        best_hparams = current_best_hparams
        best_accuracy = current_best_accuracy

# Print the best hyperparameters after the hyperparameter tuning loop
print("Best Hyperparameters:", best_hparams)

# Define a list of test_size and dev_size values
test_size_values = [0.1, 0.2, 0.3]
dev_size_values = [0.1, 0.2, 0.3]

# Loop through different test_size and dev_size combinations
for test_size in test_size_values:
    for dev_size in dev_size_values:
        # Calculate train_size based on test_size and dev_size
        train_size = 1.0 - test_size - dev_size

        # Split the data into train, dev, and test sets
        X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(data, digits.target, test_size, dev_size)

        # Define model parameters
        model_params = best_hparams  # Use the best hyperparameters

        # Train a model using the training data
        model = train_model(X_train, y_train, model_params)

        # Evaluate the model on the training, dev, and test sets
        train_acc = accuracy_score(y_train, model.predict(X_train))
        dev_acc = accuracy_score(y_dev, model.predict(X_dev))
        test_acc = accuracy_score(y_test, model.predict(X_test))

        # Print the results
        print(f"Test Size={test_size}, Dev Size={dev_size}, Train Size={train_size:.2f}")
        print(f"Train Accuracy={train_acc:.2f}, Dev Accuracy={dev_acc:.2f}, Test Accuracy={test_acc:.2f}")
        print("="*50)  # Separate results for different combinations
