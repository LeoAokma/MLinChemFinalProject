# MLinChemFinalProject
the final project of ML in Chem course.

## Installation
### 1.Installation of conda environment.
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

### 3.Download other essential module required
```
pip install -r requirements.txt
```