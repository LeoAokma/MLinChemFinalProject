import numpy as np
import pandas as pd
import sklearn.svm as svm
from sklearn import preprocessing, model_selection
from sklearn.metrics import confusion_matrix, accuracy_score, silhouette_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# from sklearn.feature_selection import RFE

import data_loader

train_dl = data_loader.DataLoader('../data/history.csv')
test_dl = data_loader.DataLoader('../data/new.csv')

feat_first = ['orgvanderwaalsMin', 'orgASA+Min', 'orghbdamsdonGeomAvg',
              'PaulingElectronegMean', 'hardnessMaxWeighted', 'AtomicRadiusMeanWeighted']

feat_top6 = ['time', 'hardnessMinWeighted', 'orgASA_HGeomAvg', 'leak', 
             'inorg-water-moleratio', 'orgvanderwaalsArithAvg']

feat_top9 = ['time', 'hardnessMinWeighted', 'orgASA_HGeomAvg', 'leak', 
             'inorg-water-moleratio', 'orgvanderwaalsArithAvg', 'orgvanderwaalsArithAvg',
             'orghbdamsaccMax', 'temp', 'EAMinWeighted']

# column(key): leak
# 'no' -> 0, 'yes' -> 1
train_dl.replace('leak', lambda x: 0 if x == 'no' else 1)
test_dl.replace('leak', lambda x: 0 if x == 'no' else 1)


# column(key): outcome
# 1, 2 -> 0 (failed)
# 3, 4 -> 1 (successful)
train_dl.replace('outcome', lambda x: 0 if x < 3 else 1)
test_dl.replace('outcome (actual)', lambda x: 0 if x < 3 else 1)


# training/validation/test set split
train_X, train_y = train_dl.generate_trainset(feat_first, include_first_column=False)
train_X, valid_X, train_y, valid_y = model_selection.train_test_split(train_X, train_y, train_size=0.9)
test_X, test_y = test_dl.generate_trainset(feat_first, include_first_column=False)

# normalization
scaler = preprocessing.StandardScaler()
scaler.fit(train_X)
train_X = scaler.transform(train_X)
valid_X = scaler.transform(valid_X)
test_X = scaler.transform(test_X)


# best: C=210, kernel='rbf'
# train_acc = 0.7794
# valid_acc = 0.7500
def svm_valuate(regular_num):
    svm_model = svm.SVC(kernel='rbf', class_weight='balanced', C=regular_num)
    svm_model.fit(train_X, train_y)

    train_pred = svm_model.predict(train_X)
    valid_pred = svm_model.predict(valid_X)

    train_cm = confusion_matrix(train_y, train_pred)
    valid_cm = confusion_matrix(valid_y, valid_pred)
    train_acc = accuracy_score(train_y, train_pred)
    valid_acc = accuracy_score(valid_y, valid_pred)

    return train_acc, valid_acc


def test_svm():
    train_lst = []
    valid_lst = []
    test_regu = np.logspace(-3, 3, num=25)
    for i in test_regu:
        train_acc, valid_acc = svm_valuate(i)
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


def test2_svm():
    param_grid = [{'C': np.logspace(-3, 3, num=10)}]
    svm_model = svm.SVC(kernel='rbf', class_weight='balanced')
    grid_search = GridSearchCV(svm_model, param_grid, scoring='accuracy', cv=5, n_jobs=-1)
    grid_search.fit(train_X, train_y)
    
    best_regu = grid_search.best_params_
    print(best_regu)

    train_pred = grid_search.predict(train_X)
    valid_pred = grid_search.predict(valid_X)

    train_cm = confusion_matrix(train_y, train_pred)
    valid_cm = confusion_matrix(valid_y, valid_pred)
    train_acc = accuracy_score(train_y, train_pred)
    valid_acc = accuracy_score(valid_y, valid_pred)

    print(train_cm, valid_cm, train_acc, valid_acc, sep='\n\n')


test2_svm()