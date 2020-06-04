import os
GENO2PHENO_DIR = os.path.dirname(__file__) + "/../"
import sys
sys.path.append(GENO2PHENO_DIR)
from src.geno2pheno.pipeline import Geno2PhenoPipeline
DATA_DIR = os.path.dirname(__file__) + "/../data"

def run_test():
    geno2pheno = Geno2PhenoPipeline(F"{DATA_DIR}/genyml_examples/test_run_config_predefined.yml", overwrite=True, cores=30,
                                    replace_list=[('%(config)%', f"{DATA_DIR}/configs/"),('%(PROJDIR)%',f"{DATA_DIR}/test_data/toy_data_in/"),
                                                  ('%(OUTDIR)%', f"{DATA_DIR}/../../output/")
                                                  ])

run_test()
