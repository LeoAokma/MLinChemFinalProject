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
from SVM import opt_evaluate_cross_valid


# Dataset paths
train_dl = DataLoader('data/dataset.csv')
test_dl = DataLoader('data/test.csv')

keys = data_keys.all_keys

# make all discontinuous data a binary plot
train_dl.binarize_all_data()
test_dl.binarize_all_data()

test_dl.binarize('XXX-Intuition', 'FALSE', data_type='string')
test_dl.binarize('outcome (actual)', 3)
train_dl.binarize('outcome', 3)


def hyper_coefficient(valid_X, valid_y, test_X, test_y, plot_str, print_det=False):
    """
    A function designed for finding best hyper parameters in regularization
    :param: valid_X: list or nd-array, the validation dataset
    :param: valid_y: list or nd-array, the outcome of validation dataset
    :param: test_X: list or nd-array, the test dataset used for plotting
    :param: test_y: list or nd-array, the outcome of test dataset used for plotting
    :param: plot_str: str, the caption of the plot and its filename
    :param: print_det: bool, if you want to print the details during the process. Default=False
    :return: tuple, the best hyper parameter and the svm model with probability on.
    (best_param, svm_model)
    """
    train_accs = []
    train_rrs = []
    test_accs = []
    test_rrs = []
    cs = []
    cnt = 1
    for i in np.linspace(1, 1500, num=200):
        cs.append(i)
        svm_model = svm.SVC(kernel='rbf', class_weight='balanced', C=i)
        svm_model.fit(valid_X, valid_y)

        # Statistical Features
        train_pred = svm_model.predict(valid_X)
        test_pred = svm_model.predict(test_X)

        train_cm = confusion_matrix(valid_y, train_pred)
        test_cm = confusion_matrix(test_y, test_pred)

        train_acc = accuracy_score(valid_y, train_pred)
        test_acc = accuracy_score(test_y, test_pred)

        train_recall = recall_score(valid_y, train_pred)
        test_recall = recall_score(test_y, test_pred)

        train_accs.append(train_acc)
        train_rrs.append(train_recall)
        test_accs.append(test_acc)
        test_rrs.append(test_recall)

        # print(train_cm)
        # print(test_cm)
        if cnt % 20 == 0:
            print("\r{}/200".format(cnt), end='', flush=True)
        if print_det:
            print("Confusion Matrix")
            vz.print_matrix(train_cm, title='Training Set')
            vz.print_matrix(test_cm, title='Testing Set')
            print("Train Accuracy:{:.4f}".format(train_acc))
            print("Test Accuracy:{:.4f}".format(test_acc))
            print("Train Recall:{:.4f}".format(train_recall))
            print("Test Recall:{:.4f}".format(test_recall))
            print('\n')

        del svm_model
        cnt += 1

    best_c = cs[train_accs.index(max(train_accs))]
    print('\nBest Hyperparameter C: {:.4f}'.format(best_c))
    print('Hyper Acc: {:.4f}, Hyper Recall: {:.4f}'.format(
                                                            max(train_accs),
                                                            train_rrs[train_accs.index(max(train_accs))]
                                                            ))
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
    svm_proba = svm.SVC(kernel='rbf', class_weight='balanced', C=best_c, probability=True)
    return best_c, svm_proba


def feature_selection(input_keys, features=10, model='random_forest'):
    """
    Select best features by using sklearn.SelectFromModel
    :param input_keys: List, the feature list you want to use
    :param features: Int, the number of features to choose from selected features. Default=10
    :param model: str, the model used in model selection, random forest or linear svm. Default='random_forest'
    :return: array, containing the features selected by the algorithm and its corresponding translation:
    (selected_list, translation_list)
    """
    # generate the dataset and binarize the outcome
    tr_X, tr_y = train_dl.generate_trainset(keys[0], include_first_column=False, binarize=True)
    train_X, test_X = model_selection.train_test_split(tr_X, test_size=0.33, random_state=114514)
    train_y, test_y = model_selection.train_test_split(tr_y, test_size=0.33, random_state=114514)
    # normalization
    scaler = preprocessing.StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    valid_X = scaler.transform(test_X)
    # Evaluating features(Using linear kernels, not rbf)
    if model == 'random_forest':
        selection = SelectFromModel(RFC(),
                                    max_features=features,
                                    ).fit(train_X, train_y)
    elif model == 'svm':
        selection = SelectFromModel(svm.SVC(kernel='linear', class_weight='balanced'),
                                    max_features=features,
                                    ).fit(train_X, train_y)
    print('Selected Features: {} in {}.'.format(features, len(input_keys[0])))
    print('Feature name\t Translation')
    features_selected = []
    translation = []
    with open('data/output.log', 'a+') as f:
        f.writelines('Selected Features: {} in {}.\n'.format(features, len(input_keys[0])))
        f.writelines('Feature name\t Translation\n')
        for _ in range(len(input_keys[1])):
            if selection.get_support()[_]:
                print(input_keys[0][_], input_keys[1][_], sep='\t')
                f.writelines("{}\t{}\n".format(input_keys[0][_], input_keys[1][_]))
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
    selected_features, trans = feature_selection(keys, features=fet, model='svm')
    # generate the train and valid dataset and binarize the outcome
    tr_X_origin, tr_y = train_dl.generate_trainset(feature_list=selected_features,
                                                   include_first_column=False,
                                                   binarize=True)
    # normalization
    scaler = preprocessing.StandardScaler()
    scaler.fit(tr_X_origin)
    tr_X = scaler.transform(tr_X_origin)
    # splitting testing data set
    train_valid_X, test_X = model_selection.train_test_split(tr_X, test_size=0.33, random_state=114514)
    train_valid_y, test_y = model_selection.train_test_split(tr_y, test_size=0.33, random_state=114514)
    # splitting train and valid set
    # train_X, valid_X = model_selection.train_test_split(train_valid_X, test_size=0.25, random_state=114514)
    # train_y, valid_y = model_selection.train_test_split(train_valid_y, test_size=0.25, random_state=114514)
    # hyper_c, svm_fn = hyper_coefficient(valid_X, valid_y, test_X, test_y, plot_str=str(fet))
    params_opt = [{'gamma': np.logspace(-5, 5, 8), 'C': np.logspace(-5, 5, num=8)}]
    svm_model = svm.SVC(kernel='rbf', class_weight='balanced')
    best_params = opt_evaluate_cross_valid(svm_model, params_opt, train_valid_X, train_valid_y, test_X, test_y, 15)

    # Generate the instance according to hyper learning
    svm_main = svm.SVC(kernel='rbf',
                       class_weight='balanced',
                       C=best_params.best_params_['C'],
                       gamma=best_params.best_params_['gamma'])
    svm_main.fit(tr_X, tr_y)
    # Statistical Features
    train_pred = svm_main.predict(tr_X)
    test_pred = svm_main.predict(test_X)

    train_cm = confusion_matrix(tr_y, train_pred)
    test_cm = confusion_matrix(test_y, test_pred)

    train_acc = accuracy_score(tr_y, train_pred)
    test_acc = accuracy_score(test_y, test_pred)

    train_recall = recall_score(tr_y, train_pred)
    test_recall = recall_score(test_y, test_pred)
    with open('data/output.log', 'a+') as f:
        f.writelines('Test Acc: {:.4f}, Test Recall: {:.4f}\n'.format(test_acc, test_recall))
    # saving data to lists
    accs.append(test_acc)
    rrs.append(test_recall)
    # Instance used for generating ROC or PR figs
    svm_proba = svm.SVC(kernel='rbf',
                        class_weight='balanced',
                        C=best_params.best_params_['C'],
                        gamma=best_params.best_params_['gamma'],
                        probability=True)
    svm_proba.fit(tr_X, tr_y)
    # p-r curve of the model
    train_prpba = svm_proba.predict_proba(tr_X)[:, 1]
    test_prpba = svm_proba.predict_proba(test_X)[:, 1]
    train_p, train_r, _ = precision_recall_curve(tr_y, train_prpba)
    test_p, test_r, _ = precision_recall_curve(test_y, test_prpba)
    vz.acc_recall_plot(
        train_p,
        train_r,
        title='{} features P-R Curve of train set'.format(fet),
        filename='pr_train_{}.png'.format(str(fet)))
    vz.acc_recall_plot(
        test_p,
        test_r,
        title='{} features P-R Curve of test set'.format(fet),
        filename='pr_test_{}.png'.format(str(fet)))

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
plt.title('The feature selection result in {} features'.format(max(fets)))
plt.xscale('linear')
plt.savefig('data/{}'.format('feature_selection.png'), dpi=300)
plt.close()
