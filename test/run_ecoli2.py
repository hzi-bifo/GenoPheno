import os
GENO2PHENO_DIR = os.path.dirname(__file__) + "/../"
import sys
sys.path.append(GENO2PHENO_DIR)
from src.geno2pheno.pipeline import Geno2PhenoPipeline
DATA_DIR = os.path.dirname(__file__) + "/../data"


def run_ecoli():
    for phenotype in ['aztreonam', 'cefotaxime', 'ceftazidime', 'ceftriaxone', 'cefuroxime']:
        geno2pheno = Geno2PhenoPipeline(F"/net/sgi/metagenomics/nobackup/prot/revisit_Geno2Pheno_2021/EColi_Sep/datasets/template_EColi.yaml", overwrite=True, cores=40,
                                    replace_list=[('%(phenotype)%', phenotype)])

run_ecoli()
