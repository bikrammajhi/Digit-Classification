from utils import get_hyperparameter_combinations
from sklearn.linear_model import LogisticRegression
from joblib import dump, load

def test_for_hparam_cominations_count():
    # a test case to check that all possible combinations of paramers are indeed generated
    gamma= [0.001, 0.01, 0.1, 1]
    C = [1, 10, 100, 1000]
    param_groups = {
        "gamma": gamma,
        "C": C
    }
    params_combinations = get_hyperparameter_combinations(param_groups)
    
    assert len(params_combinations) == len(gamma) * len(C)

def test_for_hparam_cominations_values():    
    gamma = [0.001, 0.01]
    C = [1]
    param_groups = {
        "gamma": gamma,
        "C": C
    }
    params_combinations = get_hyperparameter_combinations(param_groups)
    
    expected_param_combo_1 = {'gamma': 0.001, 'C': 1}
    expected_param_combo_2 = {'gamma': 0.01, 'C': 1}

    assert (expected_param_combo_1 in params_combinations) and (expected_param_combo_2 in params_combinations)

def test_LR():
    name = 'lr'
    parameters = ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]
    for params in parameters:
        model  = load(f'./models/M23CSA007_{name}_solver:{params}.joblib')
        model_params = model.get_params(deep=True)
        print(model_params)
        assert isinstance(model,LogisticRegression)
        
        assert model_params['solver']==params