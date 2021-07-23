import os
GENO2PHENO_DIR = os.path.dirname(__file__) + "/../"
import sys
sys.path.append(GENO2PHENO_DIR)
from src.geno2pheno.pipeline import Geno2PhenoPipeline
DATA_DIR = os.path.dirname(__file__) + "/../data"

geno2pheno = Geno2PhenoPipeline(F"/net/sgi/metagenomics/nobackup/prot/revisit_Geno2Pheno_2021/biofilm/config.yaml", overwrite=False, cores=25,replace_list=[('%(phenotype)%', 'A')])

geno2pheno = Geno2PhenoPipeline(F"/net/sgi/metagenomics/nobackup/prot/revisit_Geno2Pheno_2021/biofilm/config.yaml", overwrite=False, cores=25,replace_list=[('%(phenotype)%', 'B')])

geno2pheno = Geno2PhenoPipeline(F"/net/sgi/metagenomics/nobackup/prot/revisit_Geno2Pheno_2021/biofilm/config.yaml", overwrite=False, cores=25,replace_list=[('%(phenotype)%', 'C')])

geno2pheno = Geno2PhenoPipeline(F"/net/sgi/metagenomics/nobackup/prot/revisit_Geno2Pheno_2021/biofilm/config.yaml", overwrite=False, cores=25,replace_list=[('%(phenotype)%', 'D')])

geno2pheno = Geno2PhenoPipeline(F"/net/sgi/metagenomics/nobackup/prot/revisit_Geno2Pheno_2021/biofilm/config.yaml", overwrite=False, cores=25,replace_list=[('%(phenotype)%', 'E')])

