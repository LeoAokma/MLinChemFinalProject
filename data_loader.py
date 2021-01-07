"""
author: Shenchen Joe Zhong

System Environment
OS: Microsoft Windows 10 Professional x64, WSL with Ubuntu 16.0.4 LTS or MacOS Big Sur 11.0 above
(No requirements of necessity)

Python Environment
python==3.8
"""

import numpy as np
import collections
import pandas as pd
import unittest


class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.dataset = pd.read_csv(data_path)

    def features(self):
        """
        return all features of the dataset as a list
        :return: all features of the dataset as a list
        """
        return list(self.dataset.columns.values)

    def data(self):
        """
        return current dataset
        :return: current dataset
        """
        return self.dataset

    def get_value_list(self, feature):
        """
        return a list of all values of a certain feature
        :param feature:
        :return:
        """
        return self.dataset[feature].values.tolist()

    def get_value_array(self, feature):
        """
        return values as a 1d array
        :param feature:
        :return:
        """
        lst = self.get_value_list(feature)
        return np.array(lst)

    def value_counter(self, feature):
        """
        return values as a dict:{'value name': number of the value}
        :param feature:
        :return:
        """
        return collections.Counter(self.get_value_list(feature))

    def value_numbers(self, feature, value):
        """
        return numbers of a certain value
        :param feature:
        :param value:
        :return:
        """
        counter = self.value_counter(feature)
        return counter[value]

    def generate_trainset(self, feature_list=None, include_first_column=True, binarize=False):
        """
        generate train dataset based on selected features, data_loader.generate_trainset()[0] for X,
        data_loader.generate_trainset()[1] for Y
        last column for outcomes as default
        :param feature_list: list of selected features
        :param include_first_column: bool, Default True.
        Whether first column used for dataset(if feature_list == None), default = True
        :param binarize: bool, Default=False.
        Turn the outcome column into binary value with 0 and 1 if True.
        :return: train dataset based on selected features
        """
        if binarize:
            self.binarize('outcome', 3)
        if feature_list == None:
            train = np.array(self.dataset)
            if include_first_column:
                feature_Y = self.features()[-1]
                train_Y = self.get_value_array(feature_Y)
                train_X = self.dataset.drop([feature_Y], axis=1)
                train_X = np.array(train_X)
            else: 
                feature_first_column = self.features()[0]
                feature_Y = self.features()[-1]
                train_Y = self.get_value_array(feature_Y)
                train_X = self.dataset.drop([feature_first_column, feature_Y], axis=1)
                train_X = np.array(train_X)
        else:
            trainset = []
            for item in feature_list:
                trainset.append(self.get_value_list(item))
            train_X = np.array(trainset).T
            feature_Y = self.features()[-1]
            train_Y = self.get_value_array(feature_Y)
        train = [train_X, train_Y]
        return train

    def replace(self, key, func):
        """
        apply function 'func' to column 'key'
        :param key: the key of column
        :param func: callable, a function that you want to apply.
        :return:
        """
        self.dataset[key] = self.dataset[key].apply(func)

    def binarize(self, key, criteria, data_type='figure'):
        """
        Designed for turning a multiple discontinuous data into a binary dataset by
        :param: key: str.
        The key of the dataframe to be processed
        :param: criteria: int or float.
        The criteria to binarize the dataset. The criteria refers to:
        1. The inferior edge of the value if using figure as data type.
        2. The criteria turning data into 0(or state False) if using
        sting as data type.

        :param: data_type: 'figure' or 'string', Default='figure'
        :return: Processed data
        Examples:
        binarize('outcome', 3)
        """
        if data_type == 'figure':
            self.replace(key, lambda x: 0 if x < criteria else 1)
        elif data_type == 'string':
            self.replace(key, lambda x: 0 if x == criteria else 1)


'''
functions to be implemented:self.dataset editing(include new feature, delete useless features, etc.),
 loading trainset and test set via one DataLoader(class data_loader(self, train_data_path, test_data_path)) ...
'''


# testing code
class DataLoaderTest(unittest.TestCase):
    """
    Class for testing
    """
    def test_data_loader(self):
        # input your own path of origin dataset(.csv only)
        dl = DataLoader('./data/train.csv')
        # print(dl.features())
        # print(dl.dataset)
        # print(dl.get_value_list('XXXinorg'))
        # print(dl.get_value_list('XXXinorg1'))
        # print(dl.get_value_array('XXXinorg'))
        # print(dl.value_numbers('XXXinorg1', 'potassium vanadium trioxide'))
        print(dl.generate_trainset(['XXXinorg1', 'XXXinorg2', 'XXXinorg3'], include_first_column=False)[0])
        print(dl.generate_trainset(include_first_column=False)[1])
        # print(dl.dataset)


if __name__ == '__main__':
    unittest.main()
