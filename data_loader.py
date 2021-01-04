"""
author: Shenchen Joe Zhong

System Environment
OS: Microsoft Windows 10 Professional x64
CPU: Intel(R) Core(TM) i5-8300h @ 2.30GHz
GPU: GeForce GTX 1060 with Max-Q Design
Memory: 32 GiB

Python Environment
python==3.6
numpy==1.19.2
scipy==1.5.2
pandas==1.1.3
scikit-learn==0.23.2
matplotlib==3.3.2
"""

import numpy as np
import collections
import matplotlib.pyplot as plt
import pandas as pd


class data_loader():
    def __init__(self, data_path):
        self.data_path = data_path
        self.dataset = pd.read_csv(data_path)
        self.trainset = []
    
    # return current dataset
    def data(self):
        return self.dataset
     
    # return a list of all values of a certain feature
    def get_value_list(self, feature):
        return self.dataset[feature].values.tolist()

    # return values as a 1d array
    def get_value_array(self, feature):
        lst = self.get_value_list(feature)
        return np.array(lst)
    
    # return all features of the dataset as a list
    def features(self):
        return self.dataset.columns.values

    # return values as a dict:{'value name': number of the value}
    def value_counter(self, feature):
        return collections.Counter(self.get_value_list(feature))
    
    # return numbers of a certain value
    def value_numbers(self, feature, value):
        counter = self.value_counter(feature)
        return counter[value]

    # generate train dataset based on selected features
    # feature_list: list of selected features
    def generate_trainset(self, feature_list = None, include_id = True):
        if feature_list ==[]:
            train = np.array(self.dataset)
            if include_id == True:
                return train
            else: 

                pass

'''
functions to be implemented: generate dataset with selected features(generate_trainset), self.dataset editing(inclued new feature, delete useless features, etc.)
, loading trainset and test set via one data_loader(class data_loader(self, train_data_path, test_data_path)) ...
'''

# imput your own path of origin dataset(.csv only)
dl = data_loader('./data/41586_2016_BFnature17439_MOESM231_ESM.csv')

# testing code
print(dl.features())
print(dl.dataset)
print(dl.get_value_list('XXXinorg1'))
print(dl.get_value_array('XXXinorg1'))
print(dl.value_numbers('XXXinorg1', 'potassium vanadium trioxide'))
