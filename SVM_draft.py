import numpy as np
from sklearn import preprocessing, model_selection
import sklearn.svm as svm
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
from sklearn.feature_selection import SelectFromModel, RFE

# importing own codes
from data_loader import DataLoader
import visualization as vz
import data_keys


# Dataset paths
train_dl = DataLoader('data/train.csv')
test_dl = DataLoader('data/test.csv')

keys = data_keys.test_key

# make all discontinuous data a binary plot
train_dl.binarize_all_data()
test_dl.binarize_all_data()

test_dl.binarize('outcome (actual)', 3)
train_dl.binarize('outcome', 3)

# generate the dataset and binarize the outcome
train_X, train_y = train_dl.generate_trainset(keys[0], include_first_column=False, binarize=True)
train_X, valid_X, train_y, valid_y = model_selection.train_test_split(train_X, train_y, train_size=0.9)

# test_X, test_y = test_dl.generate_trainset(feat_first, include_first_column=False)

# normalization
scaler = preprocessing.StandardScaler()
scaler.fit(train_X)
train_X = scaler.transform(train_X)
valid_X = scaler.transform(valid_X)


def hyper_coefficient():
    """
    A function designed for finding best hyper parameters
    :return: the best hyper parameter
    """
    # C = 580
    test_accs = []
    test_rrs = []
    cs = []
    for i in np.linspace(1, 1000, num=20):
        cs.append(i)
        svm_model = svm.SVC(kernel='rbf', class_weight='balanced', C=i)
        svm_model.fit(train_X, train_y)

        # Statistical Features
        train_pred = svm_model.predict(train_X)
        valid_pred = svm_model.predict(valid_X)

        train_cm = confusion_matrix(train_y, train_pred)
        valid_cm = confusion_matrix(valid_y, valid_pred)

        train_acc = accuracy_score(train_y, train_pred)
        valid_acc = accuracy_score(valid_y, valid_pred)

        train_recall = recall_score(train_y, train_pred)
        valid_recall = recall_score(valid_y, valid_pred)

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

    best_c = cs[test_accs.index(max(test_accs))]
    print('Best Hyperparameter C: {:.4f}'.format(best_c))
    print('Acc: {:.4f}, Recall: {:.4f}'.format(max(test_accs), test_rrs[test_accs.index(max(test_accs))]))
    return best_c


human_acc = train_dl.get_value_array('Intuition')
hyper_c = hyper_coefficient()
features = 10
# Evaluating features(Using linear kernels, not rbf)
selection = SelectFromModel(svm.SVC(kernel='linear', class_weight='balanced', C=hyper_c),
                            max_features=features,
                            ).fit(train_X, train_y)
print('Selected Features: {} in {}.'.format(features, len(keys[0])))
print('Feature name\t Translation')
features_selected = []
for _ in range(len(keys[1])):
    if selection.get_support()[_]:
        print(keys[0][_], keys[1][_], sep='\t')
        features_selected.append(keys[0][_])
