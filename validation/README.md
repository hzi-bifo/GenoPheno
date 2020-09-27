# GenoPheno Data Validation

GenoPheno part of the Seq2Geno2Pheno is supposed to perform the machine learning predictive modeling and biomarker extraction
and generate the publication quality figures after following the seq2geno part. However, this part is not limited to the features created by Seq2Geno and
you have already created your data features you may run your data on our server application. However, in order to save your time and avoid incorrect submissions to the server you can verify
your data and config prior to the submission using a validator code and modify your config.


## Run the validation and zip generation


In order to create the environment in anaconda for the config/data verification create an anaconda environment:

```
conda env create -n YOUR_ENV_NAME -f environment.yml
conda activate YOUR_ENV_NAME
```

To check the config file and create the zip file for the GenoPheno server run the following command:

```
python validator.py --config data/config.yml --log log.txt --zip myproj.zip
```

where:
`data/config.yml` is the path to the config file in which every path is explicit.
`log.txt` is the path to the log file generated in checking. Please read this file carefully after generation to ensure the project is created according to your wishes.
`myproj.zip` is the generated zip file which you can submit to the GenoPheno server.


## Config modification

The config of GenoPheno has three main parts: (1) metadata, (2) genotype tables, (3) prediction block

### metadata

In this part the metadata of the project are provided, which are the followings:

  `project`: \<string\> the project name <br/>
  `phylogenetic_tree`: \<string\> the path to the phylogenetic tree<br/>
  `phenotype_table`: \<string\> the path to the phenotype table in tabular format,please see the example files)<br/>

  NOTE:
  ```
  Please note that the phenotype table can contain multiple schemes for the phenotype.
  The first column is the name of instances, the headers are the phenotypes, and the labels can be any string

  To learn more please see the example:
  validation/data/phenotypes.txt

  ```


  `output_directory`: \<string\> `results/` please do not change this path.<br/>
  `number_of_cores`: \<int\> 20<br/> the integer number of cores (<20)<br/>

### genotype tables

In this part, the genotype tables are provided, including the tabular features and the k-mers needed to be extracted from the sequence contigs.

```
genotype_tables:
  tables:
    - table:
          name: 'genexp'            --------> define a name for this feature
          path: "data/genexp.csv"   --------> tabular file of features where the first column is the instance IDs
          preprocessing: none       --------> preprocessing on the matrix of features, possible options are ['l1','l2','percent','zero2one','none','std','binary']
          delimiter: ","            --------> delimiter for parsing the table
          datatype: numerical       --------> the data type, possible options are: ['numerical','text']
    ..
    ..
    ..
    - sequence:                   -------->  the second type is sequence features
          name: '6mer'            -------->  a name for the feature
          path: "data/sequences/" -------->  a directory where each file follows the pattern of instanceID.fasata
          preprocessing: l1       --------> preprocessing on the matrix of features, possible options are ['l1','l2','percent','zero2one','none','std','binary']
          k_value: 6              --------> k of k-mer feature (1<k<9)
```


### prediction block

```
predictions:
  - prediction:
        name: "amr_pred"
        label_mapping:
          1: 1
          0: 0
        optimized_for: "f1_macro"
        reporting: ['accuracy', 'f1_pos', 'f1_macro']
        features:
          - feature: "GenExp"
            list: ['genexp']
            validation_tuning:
              name: "cv_tree"
              train:
                method: "treebased"
                folds: 10
              test:
                method: "treebased"
                ratio: 0.1
              inner_cv: 10
          - feature: "K-mer"
            list: ["6mer"]
            use_validation_tuning: "cv_tree"
          - feature: "GPA"
            list: ["gpa"]
            use_validation_tuning: "cv_tree"
        classifiers:
          - lsvm
          - svm
          - lr
          - rf
```
