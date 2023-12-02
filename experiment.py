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

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from joblib import dump, load
from utils import *

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The `images` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The `target` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:matplotlib.pyplot.imread.

digits = datasets.load_digits()
# print the height , width
print(digits.images[0].shape)

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
# each 2-D array of grayscale values from shape `(8, 8)` into shape
# `(64,)`. Subsequently, the entire dataset will be of shape
# `(n_samples, n_features), where ``n_samples` is the number of images and
# `n_features` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.



num_runs  = 1

# Data preprocessing
def split_train_dev_test(X,y,test_size,dev_size):
    _ = test_size + dev_size
    X_train, _xtest, y_train, _ytest = train_test_split(
    X, y, test_size=_, shuffle=False)
    X_test, X_dev, y_test, y_dev = train_test_split(
    _xtest, _ytest, test_size=dev_size, shuffle=False)
    return X_train, X_test, X_dev , y_train, y_test, y_dev
    
    

# Predict the value of the digit on the test subset
def predict_and_eval(model, X_test, y_test):
    predicted = model.predict(X_test)
    ###############################################################################
    # Below we visualize the first 4 test samples and show their predicted
    # digit value in the title.

    # _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    # for ax, image, prediction in zip(axes, X_test, predicted):
    #     ax.set_axis_off()
    #     image = image.reshape(8, 8)
    #     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    #     ax.set_title(f"Prediction: {prediction}")


    ###############################################################################
    # We can also plot a :ref:confusion matrix <confusion_matrix> of the
    # true digit values and the predicted digit values.

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    # disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")

    # plt.show()

    ###############################################################################
    # If the results from evaluating a classifier are stored in the form of a
    # :ref:confusion matrix <confusion_matrix> and not in terms of y_true and
    # y_pred, one can still build a :func:~sklearn.metrics.classification_report
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
    return metrics.accuracy_score(y_test, predicted), metrics.f1_score(y_test, predicted, average="macro"), predicted
# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
X = data
y  = digits.target
# No. of samples in data
print(len(X))


# 2. Hyperparameter combinations
classifier_param_dict = {}
# 2.1. SVM
gamma_list = [0.0001, 0.0005, 0.001, 0.01, 0.1, 1]
C_list = [0.1, 1, 10, 100, 1000]
h_params={
    'gamma' : gamma_list,
    'C': C_list
    }
h_params_combinations = get_hyperparameter_combinations(h_params)
classifier_param_dict['svm'] = h_params_combinations

# 2.2 Decision Tree
max_depth_list = [5, 10, 15, 20, 50, 100]
h_params_tree = {
    'max_depth' :max_depth_list}

h_params_trees_combinations = get_hyperparameter_combinations(h_params_tree)
classifier_param_dict['tree'] = h_params_trees_combinations


solver = ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]
h_params_tree = {
    'solver' :solver}

h_params_trees_combinations = get_hyperparameter_combinations(h_params_tree)
classifier_param_dict['lr'] = h_params_trees_combinations


 
# param_groups = [{"gamma":i, "C":j} for i in gamma for j in C] 
# Create Train_test_dev size groups
test_sizes = [0.1] 
dev_sizes  = [0.1]
test_dev_size_combintion = [{"test_size":i, "dev_size":j} for i in test_sizes for j in dev_sizes] 

# Create a classifier: a support vector classifier
# model = svm.SVC
# for test_dev_size in test_dev_size_combintion:
#     X_train, X_test, X_dev , y_train, y_test, y_dev = split_train_dev_test(X,y,**test_dev_size)
#     train_acc, dev_acc, test_acc, optimal_param = tune_hparams(model,X_train, X_test, X_dev , y_train, y_test, y_dev,param_groups)
#     _ = 1 - (sum(test_dev_size.values()))
#     print(f'train_size: {_}, dev_size: {test_dev_size["dev_size"]}, test_size: {test_dev_size["test_size"]} , train_acc: {train_acc}, dev_acc: {dev_acc}, test_acc: {test_acc}, optimal_param: {optimal_param}')




results = []
test_sizes =  [0.2]
dev_sizes  =  [0.2]
for cur_run_i in range(num_runs):
    
    for test_size in test_sizes:
        for dev_size in dev_sizes:
            train_size = 1- test_size - dev_size
            # train_size = 1- test_size - dev_size
            # 3. Data splitting -- to create train and test sets                
            # X_train, X_test, X_dev, y_train, y_test, y_dev = train_test_dev_split(X, y, test_size=test_size, dev_size=dev_size)
            X_train, X_test, X_dev , y_train, y_test, y_dev = split_train_dev_test(X,y,test_size=test_size, dev_size=dev_size)

            transforms = Normalizer().fit(X_train)
            X_train = transforms.transform(X_train)
            X_test = transforms.transform(X_test)
            X_dev = transforms.transform(X_dev)

            dump(transforms,'./models/transforms.joblib')

            # # 4. Data preprocessing
            # X_train = preprocess_data(X_train)
            # X_test = preprocess_data(X_test)
            # X_dev = preprocess_data(X_dev)

            binary_preds = {}
            model_preds = {}
            for model_type in classifier_param_dict:
                current_hparams = classifier_param_dict[model_type]
                best_hparams, best_model_path, best_accuracy  = tune_hparams(model_type,X_train, X_test, X_dev , y_train, y_test, y_dev,current_hparams)        
                # train_acc, dev_acc, test_acc, best_hparams,_test_predicted, best_model_path
                # loading of model         
                best_model = load(best_model_path) 

                test_acc, test_f1, predicted_y = predict_and_eval(best_model, X_test, y_test)
                train_acc, train_f1, _ = predict_and_eval(best_model, X_train, y_train)
                dev_acc = best_accuracy

                print("{}\ttest_size={:.2f} dev_size={:.2f} train_size={:.2f} train_acc={:.2f} dev_acc={:.2f} test_acc={:.2f}, test_f1={:.2f}".format(model_type, test_size, dev_size, train_size, train_acc, dev_acc, test_acc, test_f1))
                cur_run_results = {'model_type': model_type, 'run_index': cur_run_i, 'train_acc' : train_acc, 'dev_acc': dev_acc, 'test_acc': test_acc}
                results.append(cur_run_results)
                binary_preds[model_type] = y_test == predicted_y
                model_preds[model_type] = predicted_y
                
                print("{}-GroundTruth Confusion metrics".format(model_type))
                print(metrics.confusion_matrix(y_test, predicted_y))


# print("svm-tree Confusion metrics".format())
# print(metrics.confusion_matrix(model_preds['svm'], model_preds['tree']))

# print("binarized predictions")
# print(metrics.confusion_matrix(binary_preds['svm'], binary_preds['tree'], labels=[True, False]))
# print("binarized predictions -- normalized over true labels")
# print(metrics.confusion_matrix(binary_preds['svm'], binary_preds['tree'], labels=[True, False] , normalize='true'))
# print("binarized predictions -- normalized over pred  labels")
# print(metrics.confusion_matrix(binary_preds['svm'], binary_preds['tree'], labels=[True, False] , normalize='pred'))
        
# print(pd.DataFrame(results).groupby('model_type').describe().T)