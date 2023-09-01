e.g 100 smaples, 2-class/binary Classification : image of cat or dog;
    50 samples: cat
    50 samples: dog
        Data distribution: balanced/uniform
    x amount of data for training
    n-x amount of data for testing

    calculate some evaluation metric (train model(70 samples in training): (35 cats, 35 dogs),
    (30 samples in testing: 15,15)) == performance

In practise:
    train, developement, test

    train = training the model (model type, model hyperparameters, model iteration)
    dev = seleting model
    test = reporting the performance



system requirements:
OS
h/w --- may be skipped ---

How to run 
install conda

conda create -n digits python-3.9
conda activate digits
pip install -r requirement.txt


how to run:

python experiment.python

Meaning of failure:
- accuracy is poor
- coding runtime/compile error

- the model give bad prediction on the new test sample during test
 

feature
-vary model hyperparameters
# Digit-Classification
