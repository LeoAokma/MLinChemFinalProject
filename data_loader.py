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
import json
import google.protobuf as protobuf


class DataLoader:
    def __init__(self, file_path):
        self.data_path = file_path
        self.dataset = pd.read_csv(file_path)
        self.identity = None
        self.sample_number = len(self.dataset.index)
        self.validation = None

        # initialize the status of whether the discontinuous data in data loader is normalized to {0,1}
        self.is_binary = False

    def sample_numbers(self):
        return self.sample_number

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

    def generate_trainset(self, feature_list=None, include_first_column=True, binarize=False, is_validation = False):
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
        if not is_validation:
            try:
                feature_Y = 'outcome (actual)'
                train_Y = self.get_value_array(feature_Y)
            except Exception:
                feature_Y = 'outcome'
                train_Y = self.get_value_array(feature_Y)
            if binarize:
                self.binarize_all_data()
            if feature_list == None:
                train = np.array(self.dataset)
                if include_first_column:
                    train_X = self.dataset.drop([feature_Y], axis=1)
                    train_X = np.array(train_X)
                else: 
                    feature_first_column = self.features()[0]
                    train_X = self.dataset.drop([feature_first_column, feature_Y], axis=1)
                    train_X = np.array(train_X)
            else:
                trainset = []
                for item in feature_list:
                    trainset.append(self.get_value_list(item))
                train_X = np.array(trainset).T
            train = [train_X, train_Y]
        else:
            try:
                feature_Y = 'outcome (actual)'
                train_Y = np.array(self.validation[feature_Y].values.tolist())
            except Exception:
                feature_Y = 'outcome'
                train_Y = np.array(self.validation[feature_Y].values.tolist())
            if binarize:
                self.binarize_all_data()
            if feature_list == None:
                train = np.array(self.validation)
                if include_first_column:
                    train_X = self.validation.drop([feature_Y], axis=1)
                    train_X = np.array(train_X)
                else: 
                    feature_first_column = self.validation.features()[0]
                    train_X = self.validation.drop([feature_first_column, feature_Y], axis=1)
                    train_X = np.array(train_X)
            else:
                trainset = []
                for item in feature_list:
                    trainset.append(self.validation[item].values.tolist())
                train_X = np.array(trainset).T
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

    def binarize_all_data(self):
        """
        Make all data into binary {0,1} data set except outcome.
        :return:
        """
        if not self.is_binary:
            self.binarize('leak', 'no', data_type='string')
            self.binarize('slowCool', 'no', data_type='string')
            atoms = ['Na', 'Li', 'Te', 'Br', 'K',
                     'C', 'F', 'I', 'Mo', 'O',
                     'N', 'P', 'S', 'V', 'Se',
                     'Zn', 'Co', 'Cl', 'Ga', 'Cs',
                     'Cr', 'Cu', 'Actinide', 'AlkaliMetal', 'Lanthanide',
                     'P1', 'P2', 'P3', 'P4', 'P5',
                     'P6', 'P7', 'G1', 'G2', 'G3',
                     'G4', 'G5', 'G6', 'G7', 'G8',
                     'G9', 'G10', 'G11', 'G12', 'G13',
                     'G14', 'G15', 'G16', 'G17', 'G18',
                     'V0', 'V1', 'V2', 'V3', 'V4',
                     'V5', 'V6', 'V7',
                    ]
            for atom in atoms:
                self.binarize(atom, 'no', data_type='string')
            self.is_binary = True
    
    def remove_feature(self, feature_list):
        """
        remove certain feature from dataset.self.dataest would be CHANGED permanently! 
        :param: featuer_list: list
        Features to be removed
        """
        for item in feature_list:
            self.dataset = self.dataset.drop([item], axis=1)
        return 0

    def identification_features(self, feature_list = None, number_serial = None):
        """
        remove certain feature from dataset to a new DataFrame as 'identifications' of the initial dataset.self.dataest would be CHANGED permanently!
        :param: feature_list: list
        Features for identifications
        :param: number_serial: list
        lines between number_serial[0] and number_serial[1] are for identifications(including both),both number are required
        first column = 0
        this method execute according to the feature_list if feature_list and ignore serial numbers(even if serial numbers are included)
        """
        if feature_list:
            self.identity = pd.DataFrame(self.dataset.loc[:, feature_list])
            for item in feature_list:
                self.dataset = self.dataset.drop([item], axis=1)
        else:
            start_number = number_serial[0]
            stop_number = number_serial[1]
            i = start_number
            feature_list = [self.features()[start_number]]
            while i < stop_number:
                i = i + 1
                feature_list.append(self.features()[i])
            self.identity = pd.DataFrame(self.dataset.loc[:, feature_list])
            for item in feature_list:
                self.dataset = self.dataset.drop([item], axis=1)
        return 0

    def identifications(self):
        """
        return identifications, if not identities, return 0
        """
        if self.identity.empty:
            return 0
        else:
            return list(self.identityset.columns.values)

    def identities(self):
        """
        return identifications DataFrame, if no identities, return 0
        """
        if self.identity.empty:
            return 0
        else:
            return self.identity
    
    def split_validation(self, ratio = 0.25):
        """
        generate a validation set from original dataset,return new dataset and validation dataset.Validation dataset will be expelled from self.dataset and 
        possibly called as self.validation. self.dataest would be CHANGED permanently!
        :param:ratio: float
        decides the ratio of validation samples
        """
        np.random.seed(1)
        shuffled_indices = np.random.permutation(self.sample_number)
        valid_size = int(self.sample_number * ratio)
        valid_indices = shuffled_indices[:valid_size]
        train_indices = shuffled_indices[valid_size:]
        self.validation = self.dataset.iloc[valid_indices]
        self.dataset = self.dataset.iloc[train_indices]
        return self.dataset,self.validation


'''
functions to be implemented:self.dataset editing(include new feature, delete useless features, etc.),
 loading trainset and test set via one DataLoader(class data_loader(self, train_data_path, test_data_path)) ...
'''


class JsonDataLoader:
    """
    A class designed for loading databases in json form
    """
    def __init__(self, file_path):
        with open(file_path, 'r') as file:
            self.content = json.load(file)
        print(self.content)

class protobufDataLoader:
    """
    A class designed for loading databases in protobuf form
    """
    def __init__(self, file_path):
        self.data_path = file_path
        self.dataset = None
        


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
        train, valid = dl.split_validation()
        print(train)
        print(valid)
        lst1 = ['XXXinorg1',  'XXXinorg1mass',  'XXXinorg1moles']
        print(dl.generate_trainset(include_first_column=True)[0])
        print(dl.generate_trainset(include_first_column=True)[1])
        print(dl.generate_trainset(include_first_column=True, feature_list = lst1, is_validation = True)[0])
        print(dl.generate_trainset(include_first_column=True, feature_list = lst1, is_validation = True)[1])
        # print(dl.dataset)


if __name__ == '__main__':
    unittest.main()
