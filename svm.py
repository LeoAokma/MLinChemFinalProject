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
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from hypopt import GridSearch
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

import data_loader
from data_keys import FeatNames


def load_preprocess():
    """
    Data loading/preprocessing
    """
    dataset_dl = data_loader.DataLoader('./data/dataset.csv')

    # data preprocessing
    # column(key): leak
    # 'no' -> 0, 'yes' -> 1
    dataset_dl.replace('leak', lambda x: 0 if x == 'no' else 1)

    # column(key): slowCool
    dataset_dl.replace('slowCool', lambda x: 0 if x == 'no' else 1)

    # column(key): outcome
    # 1, 2 -> 0 (failed)
    # 3, 4 -> 1 (successful)
    dataset_dl.replace('outcome', lambda x: 0 if x < 3 else 1)
    # for the second csv
    # test_dl.replace('outcome (actual)', lambda x: 0 if x < 3 else 1)

    return dataset_dl


def prepare_dataset(feature_lst, create_validation_set=False, valid_ratio=None):
    """
    Load, preprocess and standardize data from train.csv and test.csv.
    Use selected feature.
    train_valid_test(feature_lst, create_validation_set, valid_ratio) \
        -> train_X, train_y, (valid_X, valid_y), test_X, test_y
    Parameters
    -----
    feature_lst : List
        List of feature names. See FeatNames.
    create_validation_set : Bool
        Split validation set from training set.
    valid_ratio : float
        Ratio of validation set
    """
    dataset_dl = load_preprocess()
    dataset_dl.split_test(0.25)
    train_X, train_y = dataset_dl.generate_trainset(
        feature_lst, include_first_column=False, is_test=False)
    test_X, test_y = dataset_dl.generate_trainset(
        feature_lst, include_first_column=False, is_test=True)
    # Loading dataset
    if create_validation_set:
        train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y)

    # standardization
    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)

    if create_validation_set:
        valid_X = scaler.transform(valid_X)
        return train_X, train_y, valid_X, valid_y, test_X, test_y
    else:
        return train_X, train_y, test_X, test_y


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
        train_acc, valid_acc, *others = svm_evaluate(
            i, train_X, train_y, valid_X, valid_y)
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
    Optimization of param grid using GridSearch.
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
    clf = GridSearch(model, param_grid)
    clf.fit(train_X, train_y, valid_X, valid_y,
            scoring='accuracy')

    best_params = clf.best_params
    results = list(zip(clf.params, clf.scores))

    return clf, best_params, results


def opt_model_gridsearchcv(model, param_grid, train_X, train_y, cv):
    """
    Optimization of param grid using GridSearchCV.
    opt_svm(train_X, train_y, valid_X, valid_y) \
        -> best model
    Parameters
    -----
    model : sklearn model
        model to optimize
    param_grid : list of dict
        parameters grid
    train_X, train_y : np.ndarray
        data
    cv : int
        'K'-fold cross validation.
    """
    clf = GridSearchCV(model, param_grid, verbose=3, return_train_score=True,
                       scoring='accuracy', cv=cv, n_jobs=-1)
    clf.fit(train_X, train_y)
    return clf


def evaluate(model, param_grid, train_X, train_y, valid_X, valid_y, test_X, test_y):
    """
    Training on training set.
    Optimizing hyperparameters on validation set.
    Evaluate model on test set.
    """
    optimized_model, best_param, results = opt_model(
        model, param_grid, train_X, train_y, valid_X, valid_y)
    # Best params: {'C': 10.0, 'gamma': 0.07742636826811278}
    # Accuracy in train/valid/test set: 0.910, 0.781, 0.616

    train_score = optimized_model.score(train_X, train_y)
    valid_score = optimized_model.score(valid_X, valid_y)
    test_score = optimized_model.score(test_X, test_y)

    print("Best params: %s" % best_param)
    print("Accuracy in train/valid/test set: %.3f, %.3f, %.3f" %
          (train_score, valid_score, test_score))


def evaluate_cross_valid(model, param_grid, train_X, train_y, test_X, test_y, cv):
    """
    Evaluate model using K-fold cross validation.
    """
    optimized_model = opt_model_gridsearchcv(
        model, param_grid, train_X, train_y, cv=cv)

    train_score = accuracy_score(train_y, optimized_model.predict(train_X))
    test_score = accuracy_score(test_y, optimized_model.predict(test_X))
    print(confusion_matrix(test_y, optimized_model.predict(test_X)))

    print("Best parameters: %s" % optimized_model.best_params_)
    index = np.argwhere(optimized_model.cv_results_['rank_test_score'] == 1)
    print("Test mean/std of accuracy in cross validation %.3f ± %.3f" % (
        optimized_model.cv_results_['mean_test_score'][index],
        optimized_model.cv_results_['std_test_score'][index]))
    print("Accuracy for train/test model: %.3f, %.3f" %
          (train_score, test_score))


def main():
    # train/validation/test split
    """train_X, train_y, valid_X, valid_y, test_X, test_y = prepare_dataset(
        FeatNames.feat_first, True, 0.25)
    params_opt = [[{'gamma': np.logspace(-5, 5, 10), 'C': np.logspace(2, 5, num=3)}]
    svm_model = svm.SVC(kernel='rbf', class_weight='balanced')
    evaluate(svm_model, params_opt, train_X, train_y,
        valid_X, valid_y, test_X, test_y)"""

    # cross validation
    train_X, train_y, test_X, test_y = prepare_dataset(
        FeatNames.feat_first, False)
    params_opt = [{'gamma': np.logspace(-5, 5, 5), 'C': np.logspace(-5, 5, num=5)}]
    svm_model = svm.SVC(kernel='rbf', class_weight='balanced')
    evaluate_cross_valid(svm_model, params_opt, train_X, train_y, test_X, test_y, 15)


if __name__ == "__main__":
    main()
