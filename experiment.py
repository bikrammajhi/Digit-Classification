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


digits = datasets.load_digits()
# print the height , width
print(digits.images[0].shape)

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)


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

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    # disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")


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



test_sizes = [0.1] 
dev_sizes  = [0.1]
test_dev_size_combintion = [{"test_size":i, "dev_size":j} for i in test_sizes for j in dev_sizes] 


results = []
test_sizes =  [0.2]
dev_sizes  =  [0.2]
for cur_run_i in range(num_runs):
    
    for test_size in test_sizes:
        for dev_size in dev_sizes:
            train_size = 1- test_size - dev_size
         
            X_train, X_test, X_dev , y_train, y_test, y_dev = split_train_dev_test(X,y,test_size=test_size, dev_size=dev_size)

            transforms = Normalizer().fit(X_train)
            X_train = transforms.transform(X_train)
            X_test = transforms.transform(X_test)
            X_dev = transforms.transform(X_dev)

            dump(transforms,'./models/transforms.joblib')


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

