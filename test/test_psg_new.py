import os
GENO2PHENO_DIR = os.path.dirname(__file__) + "/../"
import sys
sys.path.append(GENO2PHENO_DIR)
from src.geno2pheno.pipeline import Geno2PhenoPipeline
DATA_DIR = os.path.dirname(__file__) + "/../data"

INP_DIR = '/net/sgi/metagenomics/nobackup/prot/projects/verify_genopheno/'

def test_ps_tob():
    geno2pheno = Geno2PhenoPipeline(F"{DATA_DIR}/genyml_examples/classify_psg_datasets_generic.yml", overwrite=True, cores=25,
                                    replace_list=[('%(PROJDIR)%', f"{INP_DIR}"),('%(config)%', f"{DATA_DIR}/configs/"),('%(DRUG)%','TOB'),('%(drug_full)%','Tobramycin'),('%(OUTDIR)%',f"{INP_DIR}/results/")])

def test_ps_mer():
    geno2pheno = Geno2PhenoPipeline(F"{DATA_DIR}/genyml_examples/classify_psg_datasets_generic.yml", overwrite=True, cores=25,
                                    replace_list=[('%(PROJDIR)%', f"{INP_DIR}"),('%(config)%', f"{DATA_DIR}/configs/"),('%(DRUG)%','MER'),('%(drug_full)%','Meropenem'),('%(OUTDIR)%', f"{INP_DIR}/results/")])


def test_ps_cip():
    geno2pheno = Geno2PhenoPipeline(F"{DATA_DIR}/genyml_examples/classify_psg_datasets_generic.yml", overwrite=True, cores=25,
                                    replace_list=[('%(PROJDIR)%', f"{INP_DIR}"),('%(config)%', f"{DATA_DIR}/configs/"),('%(DRUG)%','CIP'),('%(drug_full)%','Ciprofloxacin'),('%(OUTDIR)%',f"{INP_DIR}/results/")])

def test_ps_cez():
    geno2pheno = Geno2PhenoPipeline(F"{DATA_DIR}/genyml_examples/classify_psg_datasets_generic.yml", overwrite=True, cores=25,
                                    replace_list=[('%(PROJDIR)%', f"{INP_DIR}"),('%(config)%', f"{DATA_DIR}/configs/"),('%(DRUG)%','CEZ'),('%(drug_full)%','Ceftazidim'),('%(OUTDIR)%',f"{INP_DIR}/results/")])

def test_ps_col():
    geno2pheno = Geno2PhenoPipeline(F"{DATA_DIR}/genyml_examples/classify_psg_datasets_generic.yml", overwrite=True, cores=25,
                                    replace_list=[('%(PROJDIR)%', f"{INP_DIR}"),('%(config)%', f"{DATA_DIR}/configs/"),('%(DRUG)%','COL'),('%(drug_full)%','Colistin'),('%(OUTDIR)%',f"{INP_DIR}/results/")])


test_ps_tob()

