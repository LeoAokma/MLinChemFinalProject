"""
author: Whale Song
System Environment
OS: Microsoft Windows 10 Professional x64, WSL with Ubuntu 16.0.4 LTS or MacOS Big Sur 11.0 above
(No requirements of necessity)
Python Environment
python==3.8
"""
import matplotlib.pyplot as plt
import numpy as np
import unittest


def print_matrix(mtx, title=None):
    """
    Printing matrix in the cmd lines in a relatively clear way.
    :param mtx: The input matrix, could be ndarray or list.
    :param title: The caption of the matrix, default=None.
    :return: None
    """
    if title != None:
        print(title)
    for line in mtx:
        for one in line:
            print("{}\t".format(one), end='')
        print('\n', end='')


def testing_status_plot(x,
                        acc, 
                        rrs, 
                        xscale='log',
                        x_name='Regularization Number',
                        title='', 
                        filename='testing_status_plot.png',
                        test=False
                        ):
    """
    Generate the plot of accuracy and recall rate by the progress of training.
    1x2 subplots, spline
    :param: acc: list or array.
    :param: rrs: list or array.
    :param: xscale: str. 'log', 'linear' or others. Default='log'. Check the docs of plt.plot for more info.
    :param: title: string. Caption of the plot. Default=''.
    :param: filename: str. The file name of pic to be saved, make sure includes the format postfix (e.g. 'pic.png').
            Default='testing_status_plot.png'.
    :param: test: bool. Won't generate file if true. Only used for unittest. Default=False
    :return: None
    """
    # figure initialize
    fig1, (accplot, rrplot) = plt.subplots(1, 2, figsize=(9., 4.5), dpi=320)
    
    # acc subplot
    accplot.plot(x, acc)
    accplot.set_xscale(xscale)
    accplot.set_xlabel(x_label)
    accplot.set_ylabel('Accuracy')
    
    # rr subplot
    rrplot.plot(x, rrs)
    rrplot.set_xscale(xscale)
    rrplot.set_xlabel(x_name)
    rrplot.set_ylabel('Recall Rate')
    
    # show plot and save figure
    fig1.show()
    if test == False:
        plt.savefig('data/{}'.format(filename), dpi=300)


def cm_heat_plot(cm, 
                 title='confusion_matrix', 
                 filename='cm_heat_plot.png',
                 test=False
                 ):
    """
    Generate the heat-point plot of confusion matrix.
    :param: cm: ndarray, confusion matrix
    :param: title: string. Caption of the picture, Default='confusion_matrix'
    :param: filename: str. The file name of pic to be saved, make sure includes the format postfix (e.g. 'pic.png').
            Default='cm_heat_plot.png'.
    :return: None
    """
    # plot initialize
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)
    plt.title(title)
    plt.colorbar()
    
    #
    elements = np.reshape([[[i, j] for j in range(2)] for i in range(2)], (cm.size, 2))
    for i, j in elements:
        plt.text(j, i, format(cm[i, j]))
    
    # set labels
    plt.ylabel('Real label')
    plt.xlabel('Prediction')
    plt.tight_layout()
    
    # show plot and save figure
    plt.show()
    if not test:
        plt.savefig('data/{}'.format(filename), dpi=300)


def hyper_learning_plot(test_acc,
                        train_acc,
                        coef, 
                        title='',
                        xscale='log',
                        x_name='Regularization Coefficient',
                        y_name='Accuracy',
                        filename='hyper_learning_plot.png', 
                        test=False):
    """
    Generate the accuracy plot by the progress of regularization coefficient
    Binary plot, which needs two sets of curve, correspond to test set and train set.
    :param: test_acc: list or nd-array.
    :param: train_acc:
    :param: coef:
    :param: title: str. The caption of the plot. Default=''
    :param: xscale: str. 'log', 'linear' or others, Default='log'. Check the docs of plt.plot for more info.
    :param: x_name: str. The name of x axis. Default='Regularization Coefficient'
    :param: y_name: str. The name of y axis. Default='Accuracy'
    :param: filename: str. The file name of pic to be saved, make sure includes the format postfix (e.g. 'pic.png').
            Default='hyper_learning_plot.png'.
    :param: test: bool. Won't generate file if true. Only used for unittest. Default=False
    :return: None
    """
    # TODO
    plt.plot(coef, test_acc, label='Test', color='blue')
    plt.plot(coef, train_acc, label='Train', color='black')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.legend()
    plt.title(title)
    plt.xscale(xscale)
    plt.show()
    if not test:
        plt.savefig('data/{}'.format(filename), dpi=300)
    plt.close()


def acc_recall_plot(pres,
                    rrs,
                    title='',
                    xscale='linear',
                    x_name='Recall Rate',
                    y_name='Accuracy',
                    filename='pr_curve.png',
                    test=False):
    """
    Generate the accuracy plot by the progress of regularization coefficient
    :param: pres: list or array.
    :param: rrs: list or array.
    :param: xscale: str. 'log', 'linear' or others. Default='linear'. Check the docs of plt.plot for more info.
    :param: title: string. Caption of the plot. Default=''.
    :param: filename: str. The file name of pic to be saved, make sure includes the format postfix (e.g. 'pic.png').
            Default='pr_curve.png'.
    :param: test: bool. Won't generate file if true. Only used for unittest. Default=False
    :return: None
    """
    # plot initialize
    plt.step(rrs, pres, color='blue')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
    plt.xscale(xscale)
    
    # save figure
    if not test:
        plt.savefig('data/{}'.format(filename), dpi=300)
    plt.close()

def dim_reduce_plot(
        len_feat,
        var_ratio,
        cum_var_ratio,
        x_name='Number of features',
        y_name='Scores',
        title='PCA dimension-reduce plot',
        filename='',
        test=False):
    """
    Generate the plot PCA dimension reduction by the progress of len_feat
    :param: len_feat: integer.
    :param: var_ratio: list or array.
    :param: cum_var_ratio: list or array.
    :param: x_name: str. The name of x axis. Default='Number of features'
    :param: y_name: str. The name of y axis. Default='Scores'
    :param: title: string. Caption of the plot. Default='PCA dimension-reduce plot'.
    :param: filename: str. The file name of pic to be saved, make sure includes the format postfix (e.g. 'pic.png').
            Default=''.
    :param: test: bool. Won't generate file if true. Only used for unittest. Default=False
    :return: None
    """
    fig = plt.figure(figsize=(6, 4), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(range(1, len_feat + 1), var_ratio, linewidth=2, label='ratio')
    plt.plot(range(1, len_feat + 1), cum_var_ratio, linewidth=2, label='accumulated')
    plt.title(title)
    plt.legend()
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.xlim(1, len_feat)
    plt.ylim(0, 1)
    # plt.grid()
    # plt.xticks(np.arange(0, len_feat, 5))
    plt.yticks(np.linspace(0, 1, num=11))
    if test == False:
        plt.savefig('data/pca_num_%s.png' % filename, dpi=300)

class svm:
    def svm_plot(test_regu,
                 train_lst,
                 valid_lst,
                 x_name='Regularization number',
                 y_name='Accuracy',
                 title='Accuracy-Regulation plot of SVM',
                 filename='SVM.png',
                 test=False):
        """
        Generate the plot PCA dimension reduction by the progress of len_feat
        :param: test_regu: list or array.
        :param: train_ls: list or array.
        :param: valid_lst: list or array.
        :param: x_name: str. The name of x axis. Default='Regularization number'.
        :param: y_name: str. The name of y axis. Default='Accuracy'.
        :param: title: string. Caption of the plot. Default='Accuracy-Regulation plot of SVM'.
        :param: filename: str. The file name of pic to be saved, make sure includes the format postfix (e.g. 'pic.png').
                Default='SVM.png'.
        :param: test: bool. Won't generate file if true. Only used for unittest. Default=False
        :return: None
        """    
        plt.plot(test_regu, train_lst)
        plt.plot(test_regu, valid_lst)
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.axis([min(test_regu), max(test_regu), 0, 1])
        plt.xscale('log')
        plt.title(title)
        plt.legend(['Training set', 'Validation set'])
        if test == False:
            plt.savefig('./data/{}'.format(filename), dpi=600)

class svm_draft:
    def randomforest_selection_plot(
            fets,
            accs,
            rrs,
            x_name='Numbers of feature',
            y_name='Score',
            title='The feature selection result in {} features',
            filename='feature_selection.png',
            test=False):
        """
        Generate the plot PCA dimension reduction by the progress of len_feat
        :param: fets: list or array.
        :param: accs: list or array.
        :param: rrs: list or array.
        :param: x_name: str. The name of x axis. Default='Number of features'.
        :param: y_name: str. The name of y axis. Default='Scores'.
        :param: title: string. Caption of the plot,'{}'should be included to add number of features with function format(). 
            Default='The feature selection result in {} features'.
        :param: filename: str. The file name of pic to be saved, make sure includes the format postfix (e.g. 'pic.png').
                Default='feature_selection.png'.
        :param: test: bool. Won't generate file if true. Only used for unittest. Default=False
        :return: None
        """    
        plt.plot(fets, accs, label='Accuracy', color='blue')
        plt.plot(fets, rrs, label='Recall rate', color='black')
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.legend()
        plt.title(title.format(max(fets)))
        plt.xscale('linear')
        if test == False:
            plt.savefig('./data/{}'.format(filename), dpi=300)
        plt.close()

class VisualizationTest(unittest.TestCase):
    """
    Class for testing
    """
    def get_test_data(self):
        self.test_pro = np.logspace(-3, 3, num=25)
        self.test_acc = np.arange(25)
        self.test_rr = np.arange(25)
        mtx = np.random.rand(2, 2)
        print(mtx)
        self.test_cm = mtx
        self.test_accoef = np.arange(25)
        self.test_coef = np.arange(25)
        self.test_coe = range(25,50)

    def test_testing_status_plot(self):
        self.get_test_data()
        testing_status_plot(self.test_pro, 
                            self.test_pro, 
                            self.test_pro, 
                            test=True)
    
    def test_cm_heat_plot(self):
        self.get_test_data()
        cm_heat_plot(self.test_cm,
                     test=True)

    def test_hyper_learning_plot(self):
        self.get_test_data()
        hyper_learning_plot(self.test_accoef,
                            self.test_coe,
                            self.test_coef,
                            test=True)


if __name__ == '__main__':
    unittest.main()

