import numpy as np
import pandas as pd
import sklearn.svm as svm
from sklearn import preprocessing, model_selection
from sklearn.metrics import confusion_matrix, accuracy_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


import data_loader


train_dl = data_loader.DataLoader('../data/history.csv')
test_dl = data_loader.DataLoader('../data/new.csv')

feat_inorg = ['IonizationMax', 'IonizationMin', 'IonizationMean', 'IonizationGeom', 
              'IonizationMaxWeighted', 'IonizationMinWeighted', 'IonizationMeanWeighted', 'IonizationGeomWeighted', 
              'EAMax', 'EAMin', 'EAMean', 'EAGeom', 
              'EAMaxWeighted', 'EAMinWeighted', 'EAMeanWeighted', 'EAGeomWeighted', 
              'PaulingElectronegMax', 'PaulingElectronegMin', 'PaulingElectronegMean', 'PaulingElectronegGeom', 
              'PaulingElectronegMaxWeighted', 'PaulingElectronegMinWeighted', 'PaulingElectronegMeanWeighted', 'PaulingElectronegGeomWeighted', 
              'PearsonElectronegMax', 'PearsonElectronegMin', 'PearsonElectronegMean', 'PearsonElectronegGeom',
              'PearsonElectronegMaxWeighted', 'PearsonElectronegMinWeighted', 'PearsonElectronegMeanWeighted', 'PearsonElectronegGeomWeighted', 
              'hardnessMax', 'hardnessMin','hardnessMean', 'hardnessGeom', 
              'hardnessMaxWeighted', 'hardnessMinWeighted', 'hardnessMeanWeighted', 'hardnessGeomWeighted', 
              'AtomicRadiusMax', 'AtomicRadiusMin', 'AtomicRadiusMean', 'AtomicRadiusGeom', 
              'AtomicRadiusMaxWeighted', 'AtomicRadiusMinWeighted', 'AtomicRadiusMeanWeighted', 'AtomicRadiusGeomWeighted', 'AtomicRadiusGeom']
            
feat_stoichio = ['inorg-water-moleratio', 'inorg-org-moleratio', 'org-water-moleratio', 
                 'orgacc-waterdonratio', 'orgdon-wateraccratio', 'notwater-water-moleratio']

feat_condition = ['Temp_max', 'time', 'slowcool', 'pH', 'leak']

feat_org = ['orgavgpolMax', 'orgavgpolMin', 'orgavgpolArithAvg', 'orgavgpolGeomAvg', 
            'orgavgpol_pHdependentMax', 'orgavgpol_pHdependentMin', 'orgavgpol_pHdependentArithAvg', 'orgavgpol_pHdependentGeomAvg', 
            'orgrefractivityMax', 'orgrefractivityMin', 'orgrefractivityArithAvg', 'orgrefractivityGeomAvg', 
            'orgmaximalprojectionareaMax', 'orgmaximalprojectionareaMin', 'orgmaximalprojectionareaArithAvg', 'orgmaximalprojectionareaGeomAvg', 
            'orgmaximalprojectionradiusMax', 'orgmaximalprojectionradiusMin', 'orgmaximalprojectionradiusArithAvg', 'orgmaximalprojectionradiusGeomAvg', 
            'orgmaximalprojectionsizeMax', 'orgmaximalprojectionsizeMin', 'orgmaximalprojectionsizeArithAvg', 'orgmaximalprojectionsizeGeomAvg', 
            'orgminimalprojectionareaMax', 'orgminimalprojectionareaMin', 'orgminimalprojectionareaArithAvg', 'orgminimalprojectionareaGeomAvg', 
            'orgminimalprojectionradiusMax', 'orgminimalprojectionradiusMin', 'orgminimalprojectionradiusArithAvg', 'orgminimalprojectionradiusGeomAvg', 
            'orgminimalprojectionsizeMax', 'orgminimalprojectionsizeMin', 'orgminimalprojectionsizeArithAvg', 'orgminimalprojectionsizeGeomAvg', 
            'orgmolpolMax', 'orgmolpolMin', 'orgmolpolArithAvg', 'orgmolpolGeomAvg', 
            'orgvanderwaalsMax', 'orgvanderwaalsMin', 'orgvanderwaalsArithAvg', 'orgvanderwaalsGeomAvg', 
            'orgASAMax', 'orgASAMin', 'orgASAArithAvg', 'orgASAGeomAvg', 
            'orgASA+Max', 'orgASA+Min', 'orgASA+ArithAvg', 'orgASA+GeomAvg', 
            'orgASA-Max', 'orgASA-Min', 'orgASA-ArithAvg', 'orgASA-GeomAvg', 
            'orgASA_HMax', 'orgASA_HMin', 'orgASA_HArithAvg', 'orgASA_HGeomAvg', 
            'orgASA_PMax', 'orgASA_PMin', 'orgASA_PArithAvg', 'orgASA_PGeomAvg', 
            'orgpolarsurfaceareaMax', 'orgpolarsurfaceareaMin', 'orgpolarsurfaceareaArithAvg', 'orgpolarsurfaceareaGeomAvg', 
            'orghbdamsaccMax', 'orghbdamsaccMin', 'orghbdamsaccArithAvg', 'orghbdamsaccGeomAvg', 
            'orghbdamsdonMax', 'orghbdamsdonMin', 'orghbdamsdonArithAvg', 'orghbdamsdonGeomAvg']


NCLASS = 10
# Color for each category
category_colors = plt.get_cmap('tab10')(np.linspace(0., 1., NCLASS))
digit_styles = {'weight': 'bold', 'size': 8}

# Prepocessing
# column(key): leak
# 'no' -> 0, 'yes' -> 1
train_dl.replace('leak', lambda x: 0 if x == 'no' else 1)
test_dl.replace('leak', lambda x: 0 if x == 'no' else 1)


# column(key): outcome
# 1, 2 -> 0 (failed)
# 3, 4 -> 1 (successful)
train_dl.replace('outcome', lambda x: 0 if x < 3 else 1)
test_dl.replace('outcome (actual)', lambda x: 0 if x < 3 else 1)


# Plot
def plot2D(X, labels, title="", save="./2D-plot.png"):
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

# PCA
def pca_data(X, y, title, save="", n_comp=2):
    pca_model = PCA(n_components=n_comp, copy=False, whiten=True)
    X_projected = pca_model.fit_transform(X)
    
    plot2D(X_projected, y, title, save)

    return X_projected


def pca_test(X, len_feat, name):
    pca_model = PCA(n_components=len_feat, copy=False, whiten=True)
    X_projected = pca_model.fit_transform(X)

    # test best number of components
    fig = plt.figure(figsize=(6, 6), dpi=600)
    ax = fig.add_subplot(1, 1, 1)
    var_ratio = pca_model.explained_variance_ratio_
    plt.plot(range(1, len_feat+1), var_ratio, linewidth=0.3)
    plt.plot(range(1, len_feat+1), np.cumsum(var_ratio), linewidth=0.3)
    plt.legend(['ratio', 'accumulated'])
    plt.xlim(1, len_feat)
    plt.ylim(0, 1)
    plt.xticks(np.arange(0, len_feat, 5))
    plt.yticks(np.linspace(0, 1, num=21))
    plt.savefig('pca_num_%s.png'%name, dpi=600, grid=True)


# tSNE
def tsne_data(X, y, title, save="", n_comp=2):
    tsne_model = TSNE(n_components=n_comp)
    X_projected = tsne_model.fit_transform(X)
    
    plot2D(X_projected, y, title, save)

    return X_projected


def main():
    feat_lst_names = ['feat_inorg', 'feat_org']
    for feat_name in feat_lst_names:
        feat = eval(feat_name)
        train_X, train_y = train_dl.generate_trainset(feat, include_first_column=False)

        # normalization
        scaler = preprocessing.StandardScaler()
        train_X = scaler.fit_transform(train_X)

        # pca_data(train_X, train_y, feat_name, "%s.png"%feat_name)
        # tsne_data(train_X, train_y, feat_name, "%s.png"%feat_name)
        pca_test(train_X, len(feat), feat_name)


if __name__ == "__main__":
    main()
