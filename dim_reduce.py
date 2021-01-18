"""
Dimension reduce.
Python Environment
python==3.8
numpy==1.18.5
sklearn-learn==0.23.2
matplotlib==2.2.3
data_loader: see data_loader.py
SVM: see SVM.py
"""
import pandas as pd
import numpy as np
import sklearn.svm as svm
from sklearn import preprocessing, model_selection
from sklearn.metrics import confusion_matrix, accuracy_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import data_loader
from SVM import FeatNames, load_preprocess


def plot2D(X, labels, title="", save="./2D-plot.png"):
    """
    Draw 2-dim data.
    """
    # Color for each category
    category_colors = plt.get_cmap('tab10')(np.linspace(0., 1., 2))
    digit_styles = {'weight': 'bold', 'size': 8}

    fig = plt.figure(figsize=(6, 6), dpi=600)
    ax = fig.add_subplot(1, 1, 1)
    X_std = preprocessing.MinMaxScaler().fit_transform(X)
    for xy, l in zip(X_std, labels):
        ax.text(*xy, str(l), color=category_colors[l], **digit_styles)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save)
    plt.cla()


def pca_data(X, y, title=None, save="", n_comp=2, draw=False):
    """
    Do PCA
    """
    pca_model = PCA(n_components=n_comp, copy=False, whiten=False)
    X_projected = pca_model.fit_transform(X)

    if draw == True:
        plot2D(X_projected, y, title, save)

    return X_projected


def tsne_data(X, y, title, save="", n_comp=2):
    """
    Do tSNE
    """
    tsne_model = TSNE(n_components=n_comp)
    X_projected = tsne_model.fit_transform(X)

    plot2D(X_projected, y, title, save)

    return X_projected


def pca_test(X, len_feat, name):
    """
    Test best number of components to keep.
    """
    pca_model = PCA(n_components=len_feat, copy=False, whiten=False)
    pca_model.fit_transform(X)

    fig = plt.figure(figsize=(6, 6), dpi=600)
    ax = fig.add_subplot(1, 1, 1)
    var_ratio = pca_model.explained_variance_ratio_
    cum_var_ratio = np.cumsum(var_ratio)
    plt.plot(range(1, len_feat + 1), var_ratio, linewidth=2)
    plt.plot(range(1, len_feat + 1), cum_var_ratio, linewidth=2)
    plt.legend(['ratio', 'accumulated'])
    plt.xlim(1, len_feat)
    plt.ylim(0, 1)
    plt.xticks(np.arange(0, len_feat, 5))
    plt.yticks(np.linspace(0, 1, num=21))
    plt.savefig('pca_num_%s.png' % name, dpi=600, grid=True)

    # components to keep
    for i in range(len(cum_var_ratio)):
        if cum_var_ratio[i] >= 0.95:
            comp_to_keep = i + 1
            break
    return comp_to_keep


def pca_dim_reduce_test(train_dl, feat_lst_names, feats_lst):
    """
    Do PCA analysis on feats.
    """
    for i in range(len(feats_lst)):
        train_X, train_y = train_dl.generate_trainset(feats_lst[i], include_first_column=False)

        # normalization
        scaler = preprocessing.StandardScaler()
        train_X = scaler.fit_transform(train_X)

        # draw a picture of cumulated variances ~ number of features
        # left 95% variance
        # number of features for feat_inorg = 5
        # number of features for feat_org = 10
        comp_to_keep = pca_test(train_X, len(feats_lst[i]), feat_lst_names[i])
        print("Number of components to keep for feature list %s: %s." %
              (feat_lst_names[i], comp_to_keep))


def main():
    train_dl, test_dl = load_preprocess()
    feat_lst_names = ['feat_inorg', 'feat_org', 'feat_misc']
    feats = [FeatNames.feat_inorg, FeatNames.feat_org, FeatNames.feat_misc]
    pca_dim_reduce_test(train_dl, feat_lst_names, feats)


if __name__ == "__main__":
    main()