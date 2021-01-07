"""
author: Leo Aokma

System Environment
OS: Microsoft Windows 10 Professional x64, WSL with Ubuntu 16.0.4 LTS or MacOS Big Sur 11.0 above
(No requirments of necessity)

Python Environment
python==3.8
"""


def print_matrix(mtx, title=None):
    """
    Printing matrix in the cmd lines in a relatively clear way.
    :param mtx: The input matrix, could be ndarray or list.
    :param title: The caption of the matrix
    :return: None
    """
    if title != None:
        print(title)
    for line in mtx:
        for one in line:
            print("{}\t".format(one), end='')
        print('\n', end='')
