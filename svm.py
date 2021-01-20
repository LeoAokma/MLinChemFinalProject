#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

!!!!!!!!!!!!!!!!!!!!    
!     Caution      !    
!!!!!!!!!!!!!!!!!!!!    
Change metrics.scorer to metrics._scorer in model_selection.py in hypopt package \
    due to incompatibility between hypopt and old version of sklearn.    
For more information, see:    
    https://stackoverflow.com/questions/65156471
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, PredefinedSplit
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from hypopt import GridSearch
import matplotlib.pyplot as plt

import data_loader

class FeatNames:
    """
    Feature names.
    For more information, see Supporting Material of reference.
    Ref: Raccuglia et al. Nature 2016, 533, 73. doi: 10.1038/nature17439
    """

    # Attributes selected by CfsSubsetEval best-first selection.
    feat_first = ['orgvanderwaalsMin', 'orgASA+Min', 'orghbdamsdonGeomAvg',
                  'PaulingElectronegMean', 'hardnessMaxWeighted', 'AtomicRadiusMeanWeighted']

    # Attributes selected by CfsSubsetEval greedy stepwise selection.
    feat_top6 = ['time', 'hardnessMinWeighted', 'orgASA_HGeomAvg', 'leak', 
                'inorg-water-moleratio', 'orgvanderwaalsArithAvg']

    feat_top9 = ['time', 'hardnessMinWeighted', 'orgASA_HGeomAvg', 'leak', 
                 'inorg-water-moleratio', 'orgvanderwaalsArithAvg', 'orgvanderwaalsArithAvg',
                 'orghbdamsaccMax', 'temp', 'EAMinWeighted']

    # See table S1
    feat_inorg = ['orgavgpolMax', 'IonizationMax', 'IonizationMin', 'IonizationMean', 'IonizationGeom', 
                  'IonizationMaxWeighted', 'IonizationMinWeighted', 'IonizationMeanWeighted', 'IonizationGeomWeighted', 
                  'EAMax', 'EAMin', 'EAMean', 'EAGeom', 
                  'EAMaxWeighted', 'EAMinWeighted', 'EAMeanWeighted', 'EAGeomWeighted', 
                  'PaulingElectronegMax', 'PaulingElectronegMin', 'PaulingElectronegMean', 'PaulingElectronegGeom', 
                  'PaulingElectronegMaxWeighted', 'PaulingElectronegMinWeighted', 'PaulingElectronegMeanWeighted', 'PaulingElectronegGeomWeighted', 
                  'PearsonElectronegMax', 'PearsonElectronegMin', 'PearsonElectronegMean', 'PearsonElectronegGeom',
                  'PearsonElectronegMaxWeighted', 'PearsonElectronegMinWeighted', 'PearsonElectronegMeanWeighted', 'PearsonElectronegGeomWeighted', 
                  'hardnessMax', 'hardnessMin','hardnessMean', 'hardnessGeom', 
                  'hardnessMaxWeighted', 'hardnessMinWeighted', 'hardnessMeanWeighted', 'hardnessGeomWeighted', 
                  'AtomicRadiusMax', 'AtomicRadiusMin', 'AtomicRadiusMean', 'AtomicRadiusGeom', 
                  'AtomicRadiusMaxWeighted', 'AtomicRadiusMinWeighted', 'AtomicRadiusMeanWeighted', 'AtomicRadiusGeomWeighted', 'AtomicRadiusGeom']
    
    # See table S2
    feat_stoichio = ['inorg-water-moleratio', 'inorg-org-moleratio', 'org-water-moleratio', 
                     'orgacc-waterdonratio', 'orgdon-wateraccratio', 'notwater-water-moleratio']

    # See table S3
    feat_condition = ['temp', 'time', 'slowCool', 'pH', 'leak']

    # See table S4
    feat_org = ['orgavgpolMax', 'orgavgpolMin', 'orgavgpolArithAvg', 'orgavgpolGeomAvg', 
                'orgavgpol_pHdependentMax', 'orgavgpol_pHdependentMin', 'orgavgpol_pHdependentArithAvg', 'orgavgpol_pHdependentGeomAvg', 
                'orgrefractivityMax', 'orgrefractivityMin', 'orgrefractivityArithAvg', 'orgrefractivityGeomAvg', 
                'orgmaximalprojectionareaMax', 'orgmaximalprojectionareaMin', 'orgmaximalprojectionareaArithAvg', 'orgmaximalprojectionareaGeomAvg', 
                'orgmaximalprojectionradiusMax', 'orgmaximalprojectionradiusMin', 'orgmaximalprojectionradiusArithAvg', 'orgmaximalprojectionradiusGeomAvg', 
                'orgmaximalprojectionsizeMax', 'orgmaximalprojectionsizeMin', 'orgmaximalprojectionsizeArithAvg', 'orgmaximalprojectionsizeGeomAvg', 
                'orgminimalprojectionareaMax', 'orgminimalprojectionareaMin', 'orgminimalprojectionareaArithAvg', 'orgminimalprojectionareaGeomAvg', 
                'orgminimalprojectionradiusMax', 'orgminimalprojectionradiusMin', 'orgminimalprojectionradiusArithAvg', 'orgminimalprojectionradiusGeomAvg', 
                'orgminimalprojectionsizeMax', 'orgminimalprojectionsizeMin', 'orgminimalprojectionsizeArithAvg', 'orgminimalprojectionsizeGeomAvg', 
                'orgmolpolMax', 'orgmolpolMin', 'orgmolpolArithAvg', 'orgmolpolGeomAvg', 
                'orgvanderwaalsMax', 'orgvanderwaalsMin', 'orgvanderwaalsArithAvg', 'orgvanderwaalsGeomAvg', 
                'orgASAMax', 'orgASAMin', 'orgASAArithAvg', 'orgASAGeomAvg', 
                'orgASA+Max', 'orgASA+Min', 'orgASA+ArithAvg', 'orgASA+GeomAvg', 
                'orgASA-Max', 'orgASA-Min', 'orgASA-ArithAvg', 'orgASA-GeomAvg', 
                'orgASA_HMax', 'orgASA_HMin', 'orgASA_HArithAvg', 'orgASA_HGeomAvg', 
                'orgASA_PMax', 'orgASA_PMin', 'orgASA_PArithAvg', 'orgASA_PGeomAvg', 
                'orgpolarsurfaceareaMax', 'orgpolarsurfaceareaMin', 'orgpolarsurfaceareaArithAvg', 'orgpolarsurfaceareaGeomAvg', 
                'orghbdamsaccMax', 'orghbdamsaccMin', 'orghbdamsaccArithAvg', 'orghbdamsaccGeomAvg', 
                'orghbdamsdonMax', 'orghbdamsdonMin', 'orghbdamsdonArithAvg', 'orghbdamsdonGeomAvg']
    
    feat_misc = feat_stoichio + feat_condition
    feat_all = feat_inorg + feat_org + feat_misc


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


def calc_metrics(y_true, y_pred, name='given dataset'):
    """
    Calculate accuracy, recall, precision and confusion matrix from given data.    
    
    Parameters
    -----
    y_true, y_pred : np.array
        true value, predicted value
    name : str
        name of dataset, such as 'training set', 'test set'.
    """
    print("Accuracy of %s : %.3f" % (name, accuracy_score(y_true, y_pred)))
    print("Recall of %s : %.3f" % (name, recall_score(y_true, y_pred)))
    print("Precision of %s : %.3f" % (name, precision_score(y_true, y_pred)))
    print("Confusion matrix of %s:" % name)
    print(confusion_matrix(y_true, y_pred))
    print("\n")
    
    
def prepare_dataset(feature_lst, create_validation_set=False, test_ratio=1/3, valid_ratio=None):
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
    test_ratio : float
        Ratio of test set
    valid_ratio : float
        Ratio of validation set.
    """
    # Loading dataset
    dataset_dl = load_preprocess()
    dataset_dl.split_test(test_ratio)
    train_X, train_y = dataset_dl.generate_trainset(
            feature_lst, include_first_column=False, is_test=False)
    test_X, test_y = dataset_dl.generate_trainset(
            feature_lst, include_first_column=False, is_test=True)
     
    if create_validation_set: 
        act_valid_ratio = valid_ratio/(1 - test_ratio)
        train_X, valid_X, train_y, valid_y = train_test_split(
            train_X, train_y, test_size=act_valid_ratio
        )

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


def opt_evaluate(model, param_grid, train_X, train_y, valid_X, valid_y, test_X, test_y):
    """
    Training on training set.
    Optimizing hyperparameters on validation set.
    Evaluate model on test set.

    evaluate(model, param_grid, train_X, train_y, valid_X, valid_y, test_X, test_y) \
        -> optimized model    
    Print out best parameters and evaluation results.

    Parameters
    -----
    model : sklearn model
        input model need to be optimized
    param_grid : list of dict
        hyperparameters grid
    train_X, train_y, valid_X, valid_y, test_X, test_y : np.ndarrays
        dataset
    """
    optimized_model = GridSearch(model, param_grid)
    optimized_model.fit(train_X, train_y, valid_X, valid_y, 
        scoring='accuracy')

    print("\n")
    print("            |------------------|")
    print("            | Model evaluation |")
    print("            |------------------|\n")

    best_params = optimized_model.best_params
    print("Best params: %s\n" % best_params)
    
    train_pred = optimized_model.predict(train_X)
    calc_metrics(train_pred, train_y, "training set")
    valid_pred = optimized_model.predict(valid_X)
    calc_metrics(valid_pred, valid_y, "validation set")
    test_pred = optimized_model.predict(test_X)
    calc_metrics(test_pred, test_y, "test set")
        
    return optimized_model


def opt_evaluate_cross_valid(model, param_grid, train_X, train_y, test_X, test_y, cv, log=True):
    """
    Hyperparam optimization using K-FOLD CROSS VALIDATION.
    Evaluate model on test set.

    evaluate(model, param_grid, train_X, train_y, test_X, test_y, cv) \
        -> optimized model  
    Print out best parameters and evaluation results.

    Parameters
    -----
    model : sklearn model
        input model need to be optimized
    param_grid : list of dict
        hyperparameters grid
    train_X, train_y, test_X, test_y : np.ndarrays
        dataset
    """
    optimized_model = GridSearchCV(model, param_grid, verbose=3, return_train_score=True, 
                    scoring='accuracy', cv=cv, n_jobs=-1)
    optimized_model.fit(train_X, train_y)

    print("\n")
    print("            |------------------|")
    print("            | Model evaluation |")
    print("            |------------------|\n")

    print("Best parameters: %s" % optimized_model.best_params_)
    index = np.argwhere(optimized_model.cv_results_['rank_test_score']==1)
    if len(index):
        print("There are more than one best model.")
    for i in index:
        print("Test mean ± std of accuracy in cross validation %.3f ± %.3f." % (
            optimized_model.cv_results_['mean_test_score'][i], 
            optimized_model.cv_results_['std_test_score'][i]))
    print("\n")
    if log:
        with open('data/output.log', 'a+') as f:
            f.writelines("\n")
            f.writelines("            |------------------|\n")
            f.writelines("            | Model evaluation |\n")
            f.writelines("            |------------------|\n")

            f.writelines("Best parameters: %s\n" % optimized_model.best_params_)
            index = np.argwhere(optimized_model.cv_results_['rank_test_score'] == 1)
            if len(index):
                f.writelines("There are more than one best model.\n")
            for i in index:
                f.writelines("Test mean ± std of accuracy in cross validation %.3f ± %.3f.\n" % (
                    optimized_model.cv_results_['mean_test_score'][i],
                    optimized_model.cv_results_['std_test_score'][i]))
            f.writelines("\n")
    
    train_pred = optimized_model.predict(train_X)
    calc_metrics(train_pred, train_y, "training set")
    test_pred = optimized_model.predict(test_X)
    calc_metrics(test_pred, test_y, "test set")
    
    return optimized_model


def main():    
    # train/validation/test split
    train_X, train_y, valid_X, valid_y, test_X, test_y = prepare_dataset(
        FeatNames.feat_first, True, 0.25, 0.25)
    params_opt = [{'gamma': np.logspace(-5, 5, 1), 'C': np.logspace(2, 5, num=3)}]
    svm_model = SVC(kernel='rbf', class_weight='balanced')
    opt_evaluate(svm_model, params_opt, 
        train_X, train_y, valid_X, valid_y, test_X, test_y)
    
    # cross validation
    train_X, train_y, test_X, test_y = prepare_dataset(
        FeatNames.feat_first, False, 1/3)
    params_opt = [{'gamma': np.logspace(-5, 5, num=5), 'C': np.logspace(-5, 5, num=5)}]
    svm_model = SVC(kernel='rbf', class_weight='balanced')
    opt_evaluate_cross_valid(svm_model, params_opt, train_X, train_y, test_X, test_y, 15)


if __name__ == "__main__":
    main()
