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
import SVM_draft as svm


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
                        title='', 
                        filename='testing_status_plot.png',
                        test=False
                        ):
    """
    Generate the plot of accuracy and recall rate by the progress of training.
    1x2 subplots, spline
    :return:
    """
    # TODO
    fig1, (accplot, rrplot) = plt.subplots(1, 2, figsize=(9.,4.5), dpi=320)
    
    # acc subplot
    accplot.plot(x, acc)
    accplot.set_xscale(xscale)
    accplot.set_xlabel('Regularization Number')
    accplot.set_ylabel('Accuracy')
    
    # rr subplot
    rrplot.plot(x, rrs)
    rrplot.set_xscale(xscale)
    rrplot.set_xlabel('Regularization Number')
    rrplot.set_ylabel('Recall Rate')

    fig1.show()
    if test == False:
        plt.savefig('./data/{}'.format(filename))


def cm_heat_plot(cm, 
                 title='confusion_matrix', 
                 filename='cm_heat_plot.png',
                 test=False
                 ):
    """
    Generate the heat-point plot of confusion matrix.
    :return:
    """
    # TODO
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)
    plt.title(title)
    plt.colorbar()
    
    thresh = cm.max() / 2
    elements = np.reshape([[[i, j] for j in range(2)] for i in range(2)], (cm.size, 2))
    for i, j in elements:
        plt.text(j, i, format(cm[i, j]))
    
    plt.ylabel('Real label')
    plt.xlabel('Prediction')
    plt.tight_layout()
    plt.show()
    if not test:
        plt.savefig('./data/{}'.format(filename))


def hyper_learning_plot(acc, 
                        coef, 
                        title='',
                        xscale='log', 
                        filename='hyper_learning_plot.png', 
                        test=False):
    """
    Generate the accuracy plot by the progress of regularization coefficient
    :return:
    """
    # TODO
    plt.plot(coef, acc)
    plt.xlabel('Regularization Coefficient')
    plt.ylabel('Accuracy')
    plt.xscale(xscale)
    if test == False:
        plt.savefig('./data/{}'.format(filename))


class VisualizationTest(unittest.TestCase):
    """
    Class for testing
    """
    def __init__(self, 
                 test_acc=None, 
                 test_rr=None, 
                 test_pro=None,
                 test_cm=None, 
                 test_accoef=None,
                 test_coef=None
                 ):
        self.test_acc = test_acc
        self.test_rr = test_rr
        self.test_pro = test_pro
        self.test_cm = test_cm
        self.test_accoef = test_accoef
        self.test_coef = test_coef

    def test_testing_status_plot(self):
        # TODO
        if self.test_acc != None and self.test_rr != None and self.test_pro != None:
            print('testing testing_status_plot..')
            testing_status_plot(self.test_pro, 
                                self.test_acc,
                                self.test_rr,
                                test=True)
            print('test pass.')
    
    def test_cm_heat_plot(self):
        if self.cm != None:
            print('testing cm_heat_plot..')
            testing_status_plot(self.test_cm,
                                test=True)
            print('test pass.')
        
    def test_hyper_learning_plot(self):
        if self.test_accoef != None and self.test_coef != None:
            print('testing hyper_learning_plot..')
            testing_status_plot(self.test_accoef,
                                self.test_coef,
                                test=True)
            print('test pass.')


if __name__ == '__main__':
    unittest.main()
    # cm = svm.svm_valuate(10)
    # print(cm)
    # cm_heat_plot(cm)
