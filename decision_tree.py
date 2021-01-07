import numpy as np
from sklearn import tree


class DecisionTree(tree.DecisionTreeClassifier):
    """
    A class of decision tree succeeded from the sklearn.tree.DecisionTreeClassifier.
    """

    def plot(self):
        tree.plot_tree(self)


def test():
    from data_loader import DataLoader
    train_dl = DataLoader('data/train.csv')
    test_dl = DataLoader('data/test.csv')
    train_dl.replace('leak', lambda x: 0 if x == 'no' else 1)
    test_dl.replace('leak', lambda x: 0 if x == 'no' else 1)
    train_x, train_y = train_dl.generate_trainset(include_first_column=False)
    dcx_tree = DecisionTree(
                            max_depth=9,
                            splitter='random',
                            class_weight='balanced',
                            )
    dcx_tree.fit(train_x, train_y)
    dcx_tree.plot()


test()
