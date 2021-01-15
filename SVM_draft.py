import numpy as np
from sklearn import preprocessing, model_selection
import sklearn.svm as svm
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.feature_selection import SelectFromModel, RFE
import threadpoolctl

# importing own codes
from data_loader import DataLoader
import visualization as vz
import data_keys


# Dataset paths
train_dl = DataLoader('data/train.csv')
test_dl = DataLoader('data/test.csv')

keys = data_keys.all_keys

# make all discontinuous data a binary plot
train_dl.binarize_all_data()
test_dl.binarize_all_data()

test_dl.binarize('XXX-Intuition', 'FALSE', data_type='string')
test_dl.binarize('outcome (actual)', 3)
train_dl.binarize('outcome', 3)


def hyper_coefficient(X, y, test_X, test_y):
    """
    A function designed for finding best hyper parameters
    :return: the best hyper parameter
    """
    # C = 580
    train_accs = []
    train_rrs = []
    test_accs = []
    test_rrs = []
    cs = []
    for i in np.linspace(1, 1500, num=200):
        cs.append(i)
        svm_model = svm.SVC(kernel='rbf', class_weight='balanced', C=i)
        svm_model.fit(X, y)

        # Statistical Features
        train_pred = svm_model.predict(X)
        valid_pred = svm_model.predict(test_X)

        train_cm = confusion_matrix(y, train_pred)
        valid_cm = confusion_matrix(test_y, valid_pred)

        train_acc = accuracy_score(y, train_pred)
        valid_acc = accuracy_score(test_y, valid_pred)

        train_recall = recall_score(y, train_pred)
        valid_recall = recall_score(test_y, valid_pred)

        train_accs.append(train_acc)
        train_rrs.append(train_recall)
        test_accs.append(valid_acc)
        test_rrs.append(valid_recall)

        # print(train_cm)
        # print(test_cm)
        print(i)
        print("Confusion Matrix")
        vz.print_matrix(train_cm, title='Training Set')
        vz.print_matrix(valid_cm, title='Testing Set')
        print("Train Accuracy:{:.4f}".format(train_acc))
        print("Test Accuracy:{:.4f}".format(valid_acc))
        print("Train Recall:{:.4f}".format(train_recall))
        print("Test Recall:{:.4f}".format(valid_recall))
        print('\n')

        del svm_model

    best_c = cs[train_accs.index(max(train_accs))]
    print('Best Hyperparameter C: {:.4f}'.format(best_c))
    print('Acc: {:.4f}, Recall: {:.4f}'.format(max(test_accs), test_rrs[test_accs.index(max(test_accs))]))
    vz.hyper_learning_plot(test_acc=test_accs,
                           train_acc=train_accs,
                           coef=np.linspace(1, 1500, num=200),
                           filename='Hyper_Learning_acc.png',
                           title='Accuracy dependency on regularization',
                           xscale='linear')
    vz.hyper_learning_plot(test_acc=test_rrs,
                           train_acc=train_rrs,
                           coef=np.linspace(1, 1500, num=200),
                           y_name='Recall Rate',
                           filename='Hyper_Learning_rr.png',
                           title='Recall-rate dependency on regularization',
                           xscale='linear')
    vz.acc_recall_plot(accs=test_accs,
                       rrs=test_rrs,
                       filename='acc_recall_test.png',
                       title='ROC Curve of test set',
                       xscale='linear')
    vz.acc_recall_plot(accs=train_accs,
                       rrs=train_rrs,
                       filename='acc_recall_train.png',
                       title='ROC Curve of train set',
                       xscale='linear')
    return best_c


def feature_selection(input_keys, features=10):
    """
    Select best features by using sklearn.SelectFromModel
    :param input_keys:
    :param features:
    :return:
    """
    # generate the dataset and binarize the outcome
    train_X, train_y = train_dl.generate_trainset(keys[0], include_first_column=False, binarize=True)
    valid_X, valid_y = test_dl.generate_trainset(keys[0])
    # normalization
    scaler = preprocessing.StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    valid_X = scaler.transform(valid_X)
    # Evaluating features(Using linear kernels, not rbf)
    """
    selection = SelectFromModel(svm.SVC(kernel='linear', class_weight='balanced', C=hyper_c),
                                max_features=features,
                                ).fit(train_X, train_y)
    """
    selection = SelectFromModel(RFC(),
                                max_features=features,
                                ).fit(train_X, train_y)
    print('Selected Features: {} in {}.'.format(features, len(input_keys[0])))
    print('Feature name\t Translation')
    features_selected = []
    for _ in range(len(input_keys[1])):
        if selection.get_support()[_]:
            print(input_keys[0][_], input_keys[1][_], sep='\t')
            features_selected.append(input_keys[0][_])
    return features_selected


human_acc = accuracy_score(test_dl.get_value_array('outcome (actual)'), test_dl.get_value_array('XXX-Intuition'))
print('Human test accuracy: {:.4f}'.format(human_acc))

selected_features = feature_selection(keys, features=20)
# generate the train and valid dataset and binarize the outcome
tr_X, tr_y = train_dl.generate_trainset(selected_features, include_first_column=False, binarize=True)
va_X, va_y = test_dl.generate_trainset(selected_features)
# normalization
scaler = preprocessing.StandardScaler()
scaler.fit(tr_X)
tr_X = scaler.transform(tr_X)
va_X = scaler.transform(va_X)
hyper_c = hyper_coefficient(tr_X, tr_y, va_X, va_y)

# train_dl.identification_features(number_serial=[0, 18])
# print(len(train_dl.features()))
