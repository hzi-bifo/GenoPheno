import os
import sys

GENO2PHENO_DIR = os.path.dirname(__file__) + "/../"
sys.path.append(GENO2PHENO_DIR)
from src.geno2pheno.pipeline import Geno2PhenoPipeline

def run_test():
    geno2pheno = Geno2PhenoPipeline(F"config.yaml", overwrite=True, cores=30, replace_list=[])

run_test()
