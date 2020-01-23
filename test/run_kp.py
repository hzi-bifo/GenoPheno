import os
GENO2PHENO_DIR = os.path.dirname(__file__) + "/../"
import sys
sys.path.append(GENO2PHENO_DIR)
from src.geno2pheno.pipeline import Geno2PhenoPipeline
DATA_DIR = os.path.dirname(__file__) + "/../data"

def run_kp():
    geno2pheno = Geno2PhenoPipeline(F"{DATA_DIR}/genyml_examples/classify_kp_dataset.yml", overwrite=False, cores=30,
                                    replace_list=[('%(config)%', f"{DATA_DIR}/configs/"),('%(PROJDIR)%','/net/sgi/metagenomics/projects/pseudo_genomics/results/PackageTesting/'),
                                                  ('%(OUTDIR)%', '/net/sgi/metagenomics/projects/pseudo_genomics/results/GenoPheno/KP_related/')
                                                  ])


run_kp()
