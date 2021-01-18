"""
SVM with modified Data
Procedure:
    1. Inorg/Org/Misc descriptors -Normalize->
        -PCA-> Dim-reduced features (respectively, dim=10+5+3)
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
SVM: see SVM.py
"""

import numpy as np
import sklearn.svm as svm
from sklearn import preprocessing, model_selection
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import data_loader
from SVM import FeatNames, opt_model, load_preprocess

# Successful or failed
NCLASS = 2
# Color for each category
category_colors = plt.get_cmap('tab10')(np.linspace(0., 1., 2))
digit_styles = {'weight': 'bold', 'size': 8}


def load_data(dataloader, is_validation):
    """
    Dataloading and spliting.
    Data -> inorg + org + misc
    """
    inorg_X, y = dataloader.generate_trainset(
        FeatNames.feat_inorg, include_first_column=False, is_validation=is_validation)
    org_X, y = dataloader.generate_trainset(
        FeatNames.feat_org, include_first_column=False, is_validation=is_validation)
    misc_X, y = dataloader.generate_trainset(
        FeatNames.feat_misc, include_first_column=False, is_validation=is_validation)
    return inorg_X, org_X, misc_X, y


def step_one(train_X, valid_X, test_X, n_components):
    """
    Do PCA analysis with n_components to keep.
    step_one(train_X, valid_X, test_X, n_components) -> \
        proj_train_X, proj_valid_X, proj_test_X
    """
    scaler = preprocessing.StandardScaler()
    train_X = scaler.fit_transform(train_X)
    valid_X = scaler.transform(valid_X)
    test_X = scaler.transform(test_X)

    # PCA for step 1
    # For optimization of n_components, see DimReduce.py
    PCA_model = PCA(n_components=n_components, copy=False, whiten=False)
    proj_train_X = PCA_model.fit_transform(train_X)
    proj_valid_X = PCA_model.transform(valid_X)
    proj_test_X = PCA_model.transform(test_X)

    return proj_train_X, proj_valid_X, proj_test_X


def main():
    train_dl, test_dl = load_preprocess()

    # Loading training/validation set
    train_dl.split_validation()
    inorg_train_X, org_train_X, misc_train_X, train_y = load_data(train_dl, False)
    inorg_valid_X, org_valid_X, misc_valid_X, valid_y = load_data(train_dl, True)
    inorg_test_X, org_test_X, misc_test_X, test_y = load_data(test_dl, False)

    # Do step one
    inorg_X = step_one(inorg_train_X, inorg_valid_X, inorg_test_X, 6)
    org_X = step_one(org_train_X, org_valid_X, org_test_X, 7)
    misc_X = step_one(misc_train_X, misc_valid_X, misc_test_X, 8)

    # Do step two
    train_Xs, valid_Xs, test_Xs = zip(inorg_X, org_X, misc_X)
    train_X = np.concatenate(train_Xs, axis=1)
    valid_X = np.concatenate(valid_Xs, axis=1)
    test_X = np.concatenate(test_Xs, axis=1)

    param_grid = [{'C': np.logspace(-3, 5, num=10)}]
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