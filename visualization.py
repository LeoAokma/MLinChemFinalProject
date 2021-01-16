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
                        x_label='Regularization Number',
                        title='', 
                        filename='testing_status_plot.png',
                        test=False
                        ):
    """
    Generate the plot of accuracy and recall rate by the progress of training.
    1x2 subplots, spline
    :return: None
    """
    # fig
    fig1, (accplot, rrplot) = plt.subplots(1, 2, figsize=(9.,4.5), dpi=320)
    
    # acc subplot
    accplot.plot(x, acc)
    accplot.set_xscale(xscale)
    accplot.set_xlabel(x_label)
    accplot.set_ylabel('Accuracy')
    
    # rr subplot
    rrplot.plot(x, rrs)
    rrplot.set_xscale(xscale)
    rrplot.set_xlabel(x_label)
    rrplot.set_ylabel('Recall Rate')

    fig1.show()
    if test == False:
        plt.savefig('data/{}'.format(filename))


def cm_heat_plot(cm, 
                 title='confusion_matrix', 
                 filename='cm_heat_plot.png',
                 test=False
                 ):
    """
    Generate the heat-point plot of confusion matrix.
    :param: cm: ndarray, confusion matrix
    :param: title: string, the title of the picture, Default='confusion_matrix'
    :return: None
    """
    # TODO
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)
    plt.title(title)
    plt.colorbar()
    
    #thresh = cm.max() / 2
    elements = np.reshape([[[i, j] for j in range(2)] for i in range(2)], (cm.size, 2))
    for i, j in elements:
        plt.text(j, i, format(cm[i, j]))
    
    plt.ylabel('Real label')
    plt.xlabel('Prediction')
    plt.tight_layout()
    plt.show()
    if not test:
        plt.savefig('data/{}'.format(filename))


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
            Default='hyper_learning_plot.png'
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
    if test == False:
        plt.savefig('data/{}'.format(filename))
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
    :return:
    """
    # TODO
    plt.step(rrs, pres, color='blue')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
    plt.xscale(xscale)
    if test == False:
        plt.savefig('data/{}'.format(filename))
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

    def test_testing_status_plot(self):
        # TODO
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
                            self.test_coef,
                            test=True)


if __name__ == '__main__':
    unittest.main()
