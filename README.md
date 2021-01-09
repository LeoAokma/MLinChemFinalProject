# MLinChemFinalProject
the final project of ML in Chem course.

## Environment Configuration
### 1.Configuration of conda environment.
#### MacOS Users
For macOS user, use homebrew would be the best choice to install
anaconda.
```
brew install anaconda
```
After installation, you will have to export the path to .bash_profile:
```
export 
source ~/.bash_profile
```

#### Windows(PC) Users
If you are win user, it is recommended to use chocolatey to install
anaconda:
```
choco install anaconda
```

### 2.The installation of RDKit
RDKit must be installed into your virtual environment of conda.
Make sure you have full access to your working directory and avoid
entering the following commands in a public directory such as "/usr/lib"
or "C:\Windows\".
```
conda create -c conda-forge -n my-rdkit-env rdkit
```

Check more info at the github repo of rdkit. 
https://github.com/rdkit/rdkit/blob/master/Docs/Book/Install.md
### 3.Download other essential module required
Install the module as requirements.txt says.
```
pip install -r requirements.txt
```
Then install a non-distributive module "ord-schema", please
download the source codes at github repo:
https://github.com/open-reaction-database/ord-schema

Change the directory where your source code of ord-schema locates
and enter the following commands:
```
python setup.py install
```
Make sure you are in the same virtual environment you just created
for installing rdkit, so that your environment would be set up 
correctly. And be careful about your python version, for conda
might include multiple python versions. You can use the command 
below if necessary.
```
python3 setup.py install
```
or
```
python3.9 setup.py install
```