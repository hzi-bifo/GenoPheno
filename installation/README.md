# GenoPheno Installation

# Installation instruction using miniconda on Linux and MacOSx

Here we have provided the detailed installation instructions of Geno2Pheno on Linux 64 and MacOSx 64 using conda virtual environment.

## (1) Miniconda installation

The first step is to install the latest version of conda on your system.

### Linux
```
cd ~
curl -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh
```

### MacOS
```
cd ~
curl -O https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash ./Miniconda3-latest-MacOSX-x86_64.sh
```


## (2) General configurations of conda envinronment

Then you need to add the conda to your path. Please modify the following path if you changed in the installation

```
export PATH="~/miniconda3/bin:$PATH"
```

Then you need to add conda channels:

```
conda config --add channels conda-forge
conda config --add channels bioconda
```


## (3) Installation of dependencies in the virtual environment

The next step would be installation of the dependencies:

### Linux
```
conda create --name genopheno --file installations/requirements_linux.yaml
pip install joblib
pip install xlwt
export QT_QPA_PLATFORM='offscreen'
```

### MacOS
```
conda create --name genopheno --file installations/requirements_osx.yaml

```

### Linux and MacOS


Then you need to activate the genopheno virtual environment:

```
source activate genopheno
```



## (4) Testing the installation

For a quick test you may run the following command:

```
python test/run_test.py
```

The test config exist at:

```
data/genyml_examples/test_run_config.yml
```


## (5) Server setting example

For a quick test you may run the following command, "data/server_example/toy_data_in" is the content of the zip file uploaded on the server.

```
python data/server_example/run_experiment.py
```
