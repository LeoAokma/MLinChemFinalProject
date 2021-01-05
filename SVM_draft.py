import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
import sklearn.svm as svm
from sklearn.metrics import confusion_matrix, accuracy_score
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

train_dl.replace('leak', lambda x: 0 if x == 'no' else 1)
test_dl.replace('leak', lambda x: 0 if x == 'no' else 1)

train_X, train_y = train_dl.generate_trainset(feat_first, include_first_column=False)
train_X, valid_X, train_y, valid_y = model_selection.train_test_split(train_X, train_y, train_size=0.9)

# test_X, test_y = test_dl.generate_trainset(feat_first, include_first_column=False)

# normalization
scaler = preprocessing.StandardScaler()
scaler.fit(train_X)
train_X = scaler.transform(train_X)
valid_X = scaler.transform(valid_X)

# C = 580
for i in np.linspace(1, 1000, num=20):
    svm_model = svm.SVC(kernel='rbf', class_weight='balanced', C=i)
    svm_model.fit(train_X, train_y)

    train_pred = svm_model.predict(train_X)
    valid_pred = svm_model.predict(valid_X)

    train_cm = confusion_matrix(train_y, train_pred)
    valid_cm = confusion_matrix(valid_y, valid_pred)

    train_acc = accuracy_score(train_y, train_pred)
    valid_acc = accuracy_score(valid_y, valid_pred)

    # print(train_cm)
    # print(test_cm)
    print(i)
    print(train_cm)
    print(valid_cm)
    print(train_acc)
    print(valid_acc)
    print('\n')

    del svm_model
