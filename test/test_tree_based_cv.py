import os
GENO2PHENO_DIR = os.path.dirname(__file__) + "/../"
import sys
sys.path.append(GENO2PHENO_DIR)
from src.geno2pheno.pipeline import Geno2PhenoPipeline
DATA_DIR = os.path.dirname(__file__) + "/../data"
from src.utility.file_utility import FileUtility
import numpy as np
from numpy import linalg as LA


def test_ps_tob():
    geno2pheno = Geno2PhenoPipeline(F"{DATA_DIR}/genyml_examples/test_pseudomona_TOB.yml", overwrite=False, cores=4,
                                    replace_list=[('%(PROJDIR)%', f"{DATA_DIR}/test_data/"),('%(config)%', f"{DATA_DIR}/configs/")])


def test_predefined_cv():
    geno2pheno = Geno2PhenoPipeline(F"{DATA_DIR}/genyml_examples/test_cv_predefined_toydata.yml", overwrite=False, cores=4,
                                    replace_list=[('%(PROJDIR)%', f"{DATA_DIR}/test_data/"),('%(config)%', f"{DATA_DIR}/configs/")])

def test_cv():
    geno2pheno = Geno2PhenoPipeline(F"{DATA_DIR}/genyml_examples/test_cv_toydata.yml", overwrite=True, cores=4,
                                    replace_list=[('%(PROJDIR)%', f"{DATA_DIR}/test_data/")])

def test_normal():
    geno2pheno = Geno2PhenoPipeline(F"{DATA_DIR}/genyml_examples/test_toy_data.yml",overwrite=True, cores=4,replace_list = [('%(PROJDIR)%',f"{DATA_DIR}/test_data/")])

def test_mismatches():
    geno2pheno = Geno2PhenoPipeline(F"{DATA_DIR}/genyml_examples/test_mismatch_toydataset.yml",overwrite=True, cores=4,replace_list = [('%(PROJDIR)%',f"{DATA_DIR}/test_data/")])

def test_preprocessings():
    geno2pheno = Geno2PhenoPipeline(F"{DATA_DIR}/genyml_examples/test_toy_data.yml",overwrite=True, cores=4,replace_list = [('%(PROJDIR)%',f"{DATA_DIR}/test_data/")])
    geno2pheno.list_of_primitive_tables.sort()
    for table in geno2pheno.list_of_primitive_tables:
        mat = FileUtility.load_sparse_csr(F"{geno2pheno.output_directory}/intermediate_representations/{table}_feature_data.npz")
        # TODO: add assert
        print(table, np.min(mat), np.max(mat), np.mean(mat) , mat.shape)

test_predefined_cv()