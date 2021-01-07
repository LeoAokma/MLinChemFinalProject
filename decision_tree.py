import numpy as np
from sklearn import tree
import graphviz
import unittest
import data_keys


class DecisionTree(tree.DecisionTreeClassifier):
    """
    A class of decision tree succeeded from the sklearn.tree.DecisionTreeClassifier.
    """

    def plot(self, feature_name, class_name=None):
        dot_data = tree.export_graphviz(self,
                                        out_file=None,
                                        feature_names=feature_name,
                                        class_names=class_name,
                                        filled=True,
                                        rounded=True,
                                        )
        new_dot = dot_data.replace('helvetica', '"Microsoft YaHei"')
        return graphviz.Source(new_dot, format='png', encoding='utf-8')


class DecisionTreeTest(unittest.TestCase):
    """
    Class for testing
    """
    def test_decision_tree(self):
        keys = data_keys.feat_top9

        from data_loader import DataLoader

        train_dl = DataLoader('data/train.csv')
        test_dl = DataLoader('data/test.csv')

        # make all discontinuous data a binary plot
        train_dl.binarize_all_data()
        test_dl.binarize_all_data()

        test_dl.binarize('outcome (actual)', 3)
        train_dl.binarize('outcome', 3)

        train_x, train_y = train_dl.generate_trainset(
            feature_list=keys[0],
            include_first_column=False,
            binarize=True,
            )

        dcx_tree = DecisionTree(
            max_depth=4,
            splitter='random',
            class_weight='balanced',
        )

        dcx_tree.fit(train_x, train_y)
        graph = dcx_tree.plot(keys[1])
        graph.render('data/tree', view=True)


if __name__ == '__main__':
    unittest.main()
