from sklearn.model_selection import train_test_split
# Standard scientific Python imports
import matplotlib.pyplot as plt
from sklearn import svm, tree, datasets, metrics, linear_model
from joblib import dump, load

def find_acc(model, X_test, y_test):
    predicted = model.predict(X_test)
    return metrics.accuracy_score(y_test,predicted),predicted

def train_model(x, y, model_params, model_type="svm"):
    if model_type == "svm":

        clf = svm.SVC
    if model_type == "tree":

        clf = tree.DecisionTreeClassifier
    if model_type == "lr":

        clf = linear_model.LogisticRegression
    model = clf(**model_params)
    # train the model
    model.fit(x, y)
    return model



def tune_hparams(model_type,X_train, X_test, X_dev , y_train, y_test, y_dev,list_of_param_combination):
    best_acc = -1
    best_model_path = ""
    for param_group in list_of_param_combination:
        temp_model = train_model(X_train, y_train, param_group, model_type=model_type)
        # temp_model = model(**param_group)
        temp_model.fit(X_train,y_train)
        acc,_ = find_acc(temp_model,X_dev,y_dev)
        if acc > best_acc:
            best_acc = acc
            best_model_path = f'./models/{model_type}' +"".join(["{}:{}".format(k,v) for k,v in param_group.items()]) + ".joblib"
            best_model = temp_model
            optimal_param = param_group
        dev_acc,_ = find_acc(temp_model,X_dev,y_dev,)
        test_acc,_test_predicted =  find_acc(temp_model,X_test,y_test)
        print(f'model: {model_type}:  train_acc: {acc}, dev_acc: {dev_acc}, test_acc: {test_acc}, params: {param_group}')
        model_path = f'./models/{"M23CSA007_"}{model_type}' +"".join(["{}:{}".format(k,v) for k,v in param_group.items()]) + ".joblib"
        dump(temp_model, model_path)
    train_acc,_= find_acc(best_model,X_train,y_train) 
    dev_acc,_ = find_acc(best_model,X_dev,y_dev)
    test_acc,_test_predicted =  find_acc(best_model,X_test,y_test)
    # save the best_model    
    dump(best_model, best_model_path) 
    
    return optimal_param,best_model_path, best_acc


def get_combinations(param,values,combinations):    
    new_combinations = []
    for value in values:
        for combination in combinations:
            combination[param] = value
            new_combinations.append(combination.copy())    
    return new_combinations

def get_hyperparameter_combinations(dict_of_param_lists):    
    base_combinations = [{}]
    for param_name, param_values in dict_of_param_lists.items():
        base_combinations = get_combinations(param_name, param_values, base_combinations)
    return base_combinations