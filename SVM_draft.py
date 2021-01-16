import numpy as np
from sklearn import preprocessing, model_selection
import sklearn.svm as svm
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.feature_selection import SelectFromModel, RFE
import matplotlib.pyplot as plt
# modules for multi thread processing:
from time import time
import threadpoolctl

# importing own codes
from data_loader import DataLoader
import visualization as vz
import data_keys
from decision_tree import DecisionTree


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


def hyper_coefficient(X, y, test_X, test_y, plot_str, print_det=False):
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
    cnt = 1
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
        if cnt % 20 == 0:
            print("\r{}/200".format(cnt), end='', flush=True)
        if print_det:
            print("Confusion Matrix")
            vz.print_matrix(train_cm, title='Training Set')
            vz.print_matrix(valid_cm, title='Testing Set')
            print("Train Accuracy:{:.4f}".format(train_acc))
            print("Test Accuracy:{:.4f}".format(valid_acc))
            print("Train Recall:{:.4f}".format(train_recall))
            print("Test Recall:{:.4f}".format(valid_recall))
            print('\n')

        del svm_model
        cnt += 1

    best_c = cs[train_accs.index(max(train_accs))]
    print('\nBest Hyperparameter C: {:.4f}'.format(best_c))
    print('Hyper Acc: {:.4f}, Hyper Recall: {:.4f}'.format(max(train_accs), train_rrs[train_accs.index(max(train_accs))]))
    vz.hyper_learning_plot(test_acc=test_accs,
                           train_acc=train_accs,
                           coef=np.linspace(1, 1500, num=200),
                           filename='Hyper_Learning_acc_{}.png'.format(plot_str),
                           title='Accuracy dependency on regularization',
                           xscale='linear')
    vz.hyper_learning_plot(test_acc=test_rrs,
                           train_acc=train_rrs,
                           coef=np.linspace(1, 1500, num=200),
                           y_name='Recall Rate',
                           filename='Hyper_Learning_rr_{}.png'.format(plot_str),
                           title='Recall-rate dependency on regularization',
                           xscale='linear')
    # generating final svm model:
    svm_final = svm.SVC(kernel='rbf', class_weight='balanced', C=best_c, probability=True)
    return best_c, svm_final


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
    translation = []
    for _ in range(len(input_keys[1])):
        if selection.get_support()[_]:
            print(input_keys[0][_], input_keys[1][_], sep='\t')
            features_selected.append(input_keys[0][_])
            translation.append(input_keys[1][_])
    return features_selected, translation


human_acc = accuracy_score(test_dl.get_value_array('outcome (actual)'), test_dl.get_value_array('XXX-Intuition'))
print('Human test accuracy: {:.4f}'.format(human_acc))

accs = []
rrs = []
fets = []
for fet_num in np.linspace(1, 50, 50):
    fet = round(fet_num)
    fets.append(fet)
    selected_features, trans = feature_selection(keys, features=fet)
    # generate the train and valid dataset and binarize the outcome
    tr_X_origin, tr_y = train_dl.generate_trainset(feature_list=selected_features, include_first_column=False, binarize=True)
    va_X_origin, va_y = test_dl.generate_trainset(feature_list=selected_features)
    # normalization
    scaler = preprocessing.StandardScaler()
    scaler.fit(tr_X_origin)
    tr_X = scaler.transform(tr_X_origin)
    va_X = scaler.transform(va_X_origin)
    # hyper learning
    _, hyper_tr_X = model_selection.train_test_split(tr_X, test_size=0.2, random_state=114514)
    _, hyper_tr_y = model_selection.train_test_split(tr_y, test_size=0.2, random_state=114514)
    # _, hyper_va_X = model_selection.train_test_split(va_X, test_size=0.2, random_state=114514)
    # _, hyper_va_y = model_selection.train_test_split(va_y, test_size=0.2, random_state=114514)
    hyper_c, svm_fn = hyper_coefficient(hyper_tr_X, hyper_tr_y, va_X, va_y, plot_str=str(fet))

    # Generate the instance according to hyper learning
    svm_main = svm.SVC(kernel='rbf', class_weight='balanced', C=hyper_c)
    svm_main.fit(tr_X, tr_y)
    # Statistical Features
    train_pred = svm_main.predict(tr_X)
    valid_pred = svm_main.predict(va_X)

    train_cm = confusion_matrix(tr_y, train_pred)
    valid_cm = confusion_matrix(va_y, valid_pred)

    train_acc = accuracy_score(tr_y, train_pred)
    valid_acc = accuracy_score(va_y, valid_pred)

    train_recall = recall_score(tr_y, train_pred)
    valid_recall = recall_score(va_y, valid_pred)
    print('Test Acc: {:.4f}, Test Recall: {:.4f}'.format(valid_acc, valid_recall))
    # saving data to lists
    accs.append(valid_acc)
    rrs.append(valid_recall)
    # Instance used for generating ROC or PR figs
    svm_fn.fit(tr_X, tr_y)
    # p-r curve of the model
    train_prpba = svm_fn.predict_proba(tr_X)[:, 1]
    valid_prpba = svm_fn.predict_proba(va_X)[:, 1]
    train_p, train_r, _ = precision_recall_curve(tr_y, train_prpba)
    test_p, test_r, _ = precision_recall_curve(va_y, valid_prpba)
    vz.acc_recall_plot(train_p, train_r, title='P-R Curve of train set', filename='pr_train_{}.png'.format(str(fet)))
    vz.acc_recall_plot(test_p, test_r, title='P-R Curve of test set', filename='pr_test_{}.png'.format(str(fet)))


    # generating decision tree
    dcx_tree = DecisionTree(max_depth=round(np.log2(fet)+1),
                            splitter='best',
                            class_weight='balanced')

    dcx_tree.fit(tr_X_origin, train_pred)
    graph = dcx_tree.plot(trans, class_name=['失败', '成功'])
    graph.render('data/tree_{}'.format(str(fet)), view=False)

    # train_dl.identification_features(number_serial=[0, 18])
    # print(len(train_dl.features()))

plt.plot(fets, accs, label='Accuracy', color='blue')
plt.plot(fets, rrs, label='Recall rate', color='black')
plt.xlabel('Numbers of feature')
plt.ylabel('Score')
plt.legend()
plt.title('The feature selection results')
plt.xscale('linear')
plt.savefig('data/{}'.format('feature_selection.png'))
plt.close()
