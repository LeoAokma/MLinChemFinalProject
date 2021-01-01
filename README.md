# MLinChemFinalProject
the final project of ML in Chem course.

# Project Chooosing
### 1.Finding MOFs using SVM
https://doi.org/10.1038/nature17439
### 2.Peptides
https://doi.org/10.1038/s41467-018-07717-6
### 3.Playing Atari game by using deep reinforced learning.
https://arxiv.org/abs/1312.5602
### 4.Predicting BTC by using beyesian.
https://github.com/Rachnog/Deep-Trading/tree/master/bayesian

##From Databases:
###CSD Cambridge Database
With open APIs in python 2.

###Reaxys Database
http://reaxys.com

Commertial, close source.

###RSC API: RESTful APIs

###Chem Spider

###Blue Obelisk
Opensource, multiple data. The datas of this API 
(O'Boyle N. M. *et. al.*, *Journal of Cheminformatics*, **2011**, 3:37)

| Name | Licence/Waiver | Description |
| ----- | -------   | -------   |
|Chempedia [98]	|CC0	|Crowd-sourced chemical names (project discontinued but data still available)|
|CrystalEye	|PPDL	|Crystal structures from primary literature|
|ONS Solubility	|CC0	|Solubility data for various solvents |
|Reaction Attempts	|CC0	|Data on successful and unsuccessful reactions |

###NIST PubChem Database
https://pubchem.ncbi.nlm.nih.gov

###ChEMBL database
https://www.ebi.ac.uk/chembl/ws

With apis.

###IBM RXN Database API
https://rxn.res.ibm.com/

Scroll to the bottom of the website and you will see the api entrance.

###Several Other APIs

|API Name	|Description	|Category	|Followers	|Versions|
|----   |----   |----   |----   |----   |
|Materials Platform for Data Science	|This API presents the curated materials data of the PAULING FILE database, suitable for automated processing, materials simulations, discovery, and scientific design.	|Science	|2	|REST v0|
|Chemcaster	|Chemcaster is a REST-based API offering services for managing cheminformatics resources, including substance registration, structure imaging, and substructure/exact-structure seach. Chemcaster is a...	|Reference	|6	|REST|


## Problems
### 1.Python3.6以上Django==1.8出现的问题
https://blog.csdn.net/grace666/article/details/103568674
### 3.缺少库crispy_forms, plotly
自行安装之
### 2.No module name "django.urls" 报错于crispy_forms中的helper.py
https://www.cnblogs.com/timest/archive/2012/03/26/2417364.html
django版本较新，已经弃用urls库，改用django.core.urlresolvers

## Steps
### 1.安装并部署MySQL服务器，配置/etc/my.cnf文件如下：

```
[mysqld]
socket=/tmp/mysql.sock
[mysql]
socket=/tmp/mysql.sock
```

### 2.初始化MySQL安装，空白密码进入程序：

```
sudo mysql_secure_installation
```

按照如下步骤部署：

https://blog.csdn.net/liang19890820/article/details/105071479

