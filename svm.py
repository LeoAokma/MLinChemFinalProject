"""
Hyperparam optimization: opt_model
SVM with selected features: main
author: Yichen Nie
System Environment
OS: MacOS Mojave 10.14.6
(No requirements of necessity)
Python Environment
python==3.8
numpy==1.18.5
sklearn-learn==0.23.2
hypopt==1.0.9
matplotlib==2.2.3
data_loader: see data_loader.py
"""

import numpy as np
import sklearn.svm as svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, PredefinedSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
from hypopt import GridSearch
import matplotlib.pyplot as plt

import data_loader
from data_keys import FeatNames


def load_preprocess():
    """
    Data loading/preprocessing
    """
    train_dl = data_loader.DataLoader('./data/train.csv')
    test_dl = data_loader.DataLoader('./data/test.csv')

    # data preprocessing
    # column(key): leak
    # 'no' -> 0, 'yes' -> 1
    train_dl.replace('leak', lambda x: 0 if x == 'no' else 1)
    test_dl.replace('leak', lambda x: 0 if x == 'no' else 1)

    # column(key): slowCool
    train_dl.replace('slowCool', lambda x: 0 if x == 'no' else 1)
    test_dl.replace('slowCool', lambda x: 0 if x == 'no' else 1)

    # column(key): outcome
    # 1, 2 -> 0 (failed)
    # 3, 4 -> 1 (successful)
    train_dl.replace('outcome', lambda x: 0 if x < 3 else 1)
    test_dl.replace('outcome (actual)', lambda x: 0 if x < 3 else 1)

    return train_dl, test_dl


def train_valid_test(feat):
    """
    Load, preprocess and normalize data from train.csv and test.csv.
    Use selected feature.
    train_valid_test() -> train_X, valid_X, train_y, valid_y, test_X, test_y
    Parameters
    -----
    feat : list
        list of features.
    """
    train_dl, test_dl = load_preprocess()

    # training/validation/test set split
    train_X, train_y = train_dl.generate_trainset(feat, include_first_column=False)
    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, train_size=0.9)
    test_X, test_y = test_dl.generate_trainset(feat, include_first_column=False)

    # normalization
    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    valid_X = scaler.transform(valid_X)
    test_X = scaler.transform(test_X)

    return train_X, valid_X, train_y, valid_y, test_X, test_y


def svm_evaluate(regular_num, train_X, train_y, valid_X, valid_y):
    """
    Calculate accuracy, precision, recall, confusion metrix \
        of svm model with regular number.
    """
    svm_model = svm.SVC(kernel='rbf', class_weight='balanced', C=regular_num)
    svm_model.fit(train_X, train_y)

    train_pred = svm_model.predict(train_X)
    valid_pred = svm_model.predict(valid_X)

    train_acc = accuracy_score(train_y, train_pred)
    valid_acc = accuracy_score(valid_y, valid_pred)
    train_prec = precision_score(train_y, train_pred)
    valid_prec = precision_score(valid_y, valid_pred)
    train_rec = recall_score(train_y, train_pred)
    valid_rec = recall_score(valid_y, valid_pred)
    train_cm = confusion_matrix(train_y, train_pred)
    valid_cm = confusion_matrix(valid_y, valid_pred)

    datalst = [
        train_acc, valid_acc, train_prec, valid_prec,
        train_rec, valid_rec, train_cm, valid_cm,
    ]
    return datalst


# Optimization of Regularization number C.
def test_svm(train_X, train_y, valid_X, valid_y):
    train_lst = []
    valid_lst = []
    test_regu = np.logspace(-3, 3, num=25)

    for i in test_regu:
        train_acc, valid_acc, *others = svm_evaluate(i, train_X, train_y, valid_X, valid_y)
        train_lst.append(train_acc)
        valid_lst.append(valid_acc)

    plt.plot(test_regu, train_lst)
    plt.plot(test_regu, valid_lst)
    plt.xlabel('Regularization number')
    plt.ylabel('Accuracy')
    plt.axis([min(test_regu), max(test_regu), 0, 1])
    plt.xscale('log')
    plt.legend(['Training set', 'Validation set'])
    plt.savefig('SVM.png', dpi=600)


def opt_model(model, param_grid, train_X, train_y, valid_X, valid_y):
    """
    Optimization of Regularization number C using GridSearchCV.
    opt_svm(train_X, train_y, valid_X, valid_y) \
        -> model, best_params, results
    Parameters
    -----
    model : sklearn model
        model to optimize
    param_grid : list of dict
        parameters grid
    train_X, train_y, valid_X, valid_y : np.ndarray
        data
    """
    # Using GridSearchCV
    # all_X = np.append(train_X, valid_X, axis=0)
    # all_y = np.append(train_y, valid_y, axis=0)
    # Create a list where train data indices are -1 and validation data indices are 0
    # For details, see
    # https://stackoverflow.com/questions/31948879
    # split_index = [-1]*len(train_X) + [0]*(len(valid_X))
    # pds = PredefinedSplit(test_fold=split_index)
    # clf = GridSearchCV(model, param_grid, verbose=1,
    #     scoring='accuracy', cv=pds, n_jobs=-1)
    # clf.fit(all_X, all_y)

    clf = GridSearch(model, param_grid)
    clf.fit(train_X, train_y, valid_X, valid_y)

    best_params = clf.best_params
    results = list(zip(clf.params, clf.scores))

    return clf, best_params, results


def main():
    train_X, valid_X, train_y, valid_y, test_X, test_y = train_valid_test(FeatNames.feat_top9)
    param_grid = [{'C': np.logspace(-3, 4.5, num=10)}]
    svm_model = svm.SVC(kernel='rbf', class_weight='balanced')
    optimized_svm, best_regu, results = opt_model(
        svm_model, param_grid, train_X, train_y, valid_X, valid_y)

    train_score = optimized_svm.score(train_X, train_y)
    valid_score = optimized_svm.score(valid_X, valid_y)
    test_score = optimized_svm.score(test_X, test_y)

    print("Best parameters: %s" % best_regu)
    print("Accuracy for train/valid/test model: %.3f, %.3f, %.3f" %
          (train_score, valid_score, test_score))


if __name__ == "__main__":
    main()