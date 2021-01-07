"""
author: Shenchen Joe Zhong

System Environment
OS: Microsoft Windows 10 Professional x64, WSL with Ubuntu 16.0.4 LTS or MacOS Big Sur 11.0 above
(No requirments of necessity)

Python Environment
python==3.8
"""

import numpy as np
import collections
import pandas as pd


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

    def generate_trainset(self, feature_list=None, include_first_column=True):
        """
        generate train dataset based on selected features, data_loader.generate_trainset()[0] for X,
        data_loader.generate_trainset()[1] for Y
        last column for outcomes as default
        :param feature_list: list of selected features
        :param include_first_column: whether first column used for dataset(if feature_list == None), default = True
        :return: train dataset based on selected features
        """
        if feature_list == None:
            train = np.array(self.dataset)
            if include_first_column:
                feature_Y = self.features()[-1]
                train_Y = self.get_value_array(feature_Y)
                train_X = self.dataset.drop([feature_Y], axis = 1)
                train_X = np.array(train_X)
            else: 
                feature_first_column = self.features()[0]
                feature_Y = self.features()[-1]
                train_Y = self.get_value_array(feature_Y)
                train_X = self.dataset.drop([feature_first_column, feature_Y], axis = 1)
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

'''
functions to be implemented:self.dataset editing(inclued new feature, delete useless features, etc.), loading trainset and test set via one data_loader(class data_loader(self, train_data_path, test_data_path)) ...
'''

# imput your own path of origin dataset(.csv only)
dl = DataLoader('./data/train.csv')

# testing code
# print(dl.features())
# print(dl.dataset)
# print(dl.get_value_list('XXXinorg'))
# print(dl.get_value_list('XXXinorg1'))
# print(dl.get_value_array('XXXinorg'))
# print(dl.value_numbers('XXXinorg1', 'potassium vanadium trioxide'))
print(dl.generate_trainset(['XXXinorg1', 'XXXinorg2', 'XXXinorg3'], include_first_column=False)[0])
print(dl.generate_trainset(include_first_column=False)[1])
# print(dl.dataset)
