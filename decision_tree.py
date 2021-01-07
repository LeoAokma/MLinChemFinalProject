import numpy as np
from sklearn import tree
import unittest


class DecisionTree(tree.DecisionTreeClassifier):
    """
    A class of decision tree succeeded from the sklearn.tree.DecisionTreeClassifier.
    """

    def plot(self):
        tree.plot_tree(self)


class DecisionTreeTest(unittest.TestCase):
    """
    Class for testing
    """
    def test_decision_tree(self):
        from data_loader import DataLoader
        train_dl = DataLoader('data/train.csv')
        test_dl = DataLoader('data/test.csv')
        train_dl.replace('leak', lambda x: 0 if x == 'no' else 1)
        test_dl.replace('leak', lambda x: 0 if x == 'no' else 1)
        test_dl.binarize('outcome (actual)', 3)
        train_x, train_y = train_dl.generate_trainset(include_first_column=False, binarize=True)
        dcx_tree = DecisionTree(
            max_depth=9,
            splitter='random',
            class_weight='balanced',
        )
        dcx_tree.fit(train_x, train_y)
        dcx_tree.plot()


if __name__ == '__main__':
    unittest.main()
