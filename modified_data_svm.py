#!/usr/bin/env python3
# -*- coding: utf-8 -*-"""
"""
SVM with modified Data

Procedure:
    1. Inorg/Org/Misc descriptors -standardize-> \ 
        -PCA-> Dim-reduced features (respectively, dim=6+7+8)
    2. Dim-reduced features -SVM hyperopt-> Predicted results

Author: Yichen Nie

System Environment
OS: MacOS Mojave 10.14.6
(No requirements of necessity)

Python Environment
python==3.8
numpy==1.18.5
sklearn-learn==0.23.2
data_loader: see data_loader.py
svm: see svm.py
"""

import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing, model_selection
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import data_loader
from SVM import FeatNames, load_preprocess, opt_evaluate_cross_valid

# Successful or failed
NCLASS = 2
# Color for each category
category_colors = plt.get_cmap('tab10')(np.linspace(0., 1., 2))
digit_styles = {'weight': 'bold', 'size': 8}


def load_data(dataloader, is_test):
    """
    Dataloading and spliting.
    Data -> inorg + org + misc
    """
    inorg_X, y = dataloader.generate_trainset(
        FeatNames.feat_inorg, include_first_column=False, is_test=is_test)
    org_X, y = dataloader.generate_trainset(
        FeatNames.feat_org, include_first_column=False, is_test=is_test)
    misc_X, y = dataloader.generate_trainset(
        FeatNames.feat_misc, include_first_column=False, is_test=is_test)
    return inorg_X, org_X, misc_X, y


def step_one(n_components, *dataset):
    """
    Do PCA analysis with n_components to keep.    
    step_one(*dataset, n_components) -> \
        proj_dataset
    """
    # stardardization
    train_X, *other_sets_X = dataset
    scaler = preprocessing.StandardScaler()
    train_X = scaler.fit_transform(train_X)
    for i in range(len(other_sets_X)):
        other_sets_X[i] = scaler.transform(other_sets_X[i])
    
    # PCA for step 1
    # For optimization of n_components, see DimReduce.py
    PCA_model = PCA(n_components=n_components, copy=False, whiten=False)
    proj_train_X = PCA_model.fit_transform(train_X)
    proj_other_sets_X = []
    for i in range(len(other_sets_X)):
        proj_other_sets_X.append(
            PCA_model.transform(other_sets_X[i]))

    return [proj_train_X] + proj_other_sets_X


def pca_steps(test_ratio=0.25):
    """
    Data loading and PCA steps were included.

    Parameters
    -----
    include_valid : Bool
        Split validation set from training set.
    """
    dataset_dl = load_preprocess()

    # Loading training/validation/test set 
    dataset_dl.split_test(test_ratio)
    inorg_train_X, org_train_X, misc_train_X, train_y = load_data(dataset_dl, False)
    inorg_test_X, org_test_X, misc_test_X, test_y = load_data(dataset_dl, True)
    
    # Do step one
    inorg_X = step_one(6, inorg_train_X, inorg_test_X)
    org_X = step_one(7, org_train_X, org_test_X)
    misc_X = step_one(8, misc_train_X, misc_test_X)

    # Do step two
    # [[inorg_train_X, org_train_X, misc_train_X], [inorg_test_X, org_test_X, misc_test_X]]
    # -concatenate-> train_X, test_X
    train_Xs, test_Xs = zip(inorg_X, org_X, misc_X)
    train_X = np.concatenate(train_Xs, axis=1)
    test_X = np.concatenate(test_Xs, axis=1)

    return train_X, train_y, test_X, test_y


def main():
    # PCA
    train_X, train_y, test_X, test_y = pca_steps(0.25)

    # svm
    param_grid = [{'gamma': np.logspace(-5, 5, 5), 'C': np.logspace(-5, 5, num=5)}]
    svm_model = SVC(kernel='rbf', class_weight='balanced')
    opt_evaluate_cross_valid(svm_model, param_grid, train_X, train_y, test_X, test_y, 15)


if __name__ == "__main__":
    main()
