import numpy as np
from sklearn import tree
import graphviz
import unittest


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
        # print(tree.export_text(self, feature_name))
        return graphviz.Source(dot_data, format='png')


class DecisionTreeTest(unittest.TestCase):
    """
    Class for testing
    """
    def test_decision_tree(self):
        test_key = ['slowCool', 'pH', 'leak', 'numberInorg', 'numberOrg',
                    'numberOxlike', 'numberComponents', 'orgavgpolMax',
                    'orgrefractivityMax', 'orgmaximalprojectionareaMax',
                    'orgmaximalprojectionradiusMax', 'orgmaximalprojectionsizeMax',
                    'orgminimalprojectionareaMax', 'orgminimalprojectionradiusMax',
                    'orgminimalprojectionsizeMax', 'orgavgpol_pHdependentMax',
                    'orgmolpolMax', 'orgvanderwaalsMax', 'orgASAMax', 'orgASA+Max',
                    'orgASA-Max', 'orgASA_HMax', 'orgASA_PMax', 'orgpolarsurfaceareaMax',
                    'orghbdamsaccMax', 'orghbdamsdonMax', 'orgavgpolMin', 'orgrefractivityMin',
                    'orgmaximalprojectionareaMin', 'orgmaximalprojectionradiusMin',
                    'orgmaximalprojectionsizeMin', 'orgminimalprojectionareaMin',
                    ]
        from data_loader import DataLoader
        train_dl = DataLoader('data/train.csv')
        test_dl = DataLoader('data/test.csv')
        train_dl.binarize('leak', 'no', data_type='string')

        test_dl.binarize('leak', 'no', data_type='string')
        train_dl.binarize('slowCool', 'no', data_type='string')
        train_dl.binarize('outcome', 3)
        test_dl.binarize('slowCool', 'no', data_type='string')
        test_dl.binarize('outcome (actual)', 3)
        train_x, train_y = train_dl.generate_trainset(feature_list=test_key, include_first_column=False, binarize=True)
        dcx_tree = DecisionTree(
            max_depth=9,
            splitter='random',
            class_weight='balanced',
        )
        dcx_tree.fit(train_x, train_y)
        graph = dcx_tree.plot(test_key)
        graph.render('data/tree.png', view=True)


if __name__ == '__main__':
    unittest.main()
