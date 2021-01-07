"""
author: Leo Aokma

System Environment
OS: Microsoft Windows 10 Professional x64, WSL with Ubuntu 16.0.4 LTS or MacOS Big Sur 11.0 above
(No requirements of necessity)

Python Environment
python==3.8
"""
import matplotlib.pyplot as plt
import numpy as np


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


def testing_status_plot():
    """
    Generate the plot of accuracy and recall rate by the progress of training.
    :return:
    """
    # TODO
    pass


def cm_heat_plot():
    """
    Generate the heat-point plot of confusion matrix.
    :return:
    """
    # TODO
    pass


def hyper_learning_plot():
    """
    Generate the accuracy plot by the progress of regularization coefficient
    :return:
    """
    # TODO
    pass
