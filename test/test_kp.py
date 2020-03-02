import os
GENO2PHENO_DIR = os.path.dirname(__file__) + "/../"
import sys
sys.path.append(GENO2PHENO_DIR)
from src.geno2pheno.pipeline import Geno2PhenoPipeline
DATA_DIR = os.path.dirname(__file__) + "/../data"

def test_kp():
    geno2pheno = Geno2PhenoPipeline(F"{DATA_DIR}/genyml_examples/test_kp_dataset.yml", overwrite=False, cores=20,
                                    replace_list=[('%(config)%', f"{DATA_DIR}/configs/")])


test_kp()