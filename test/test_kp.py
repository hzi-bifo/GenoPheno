import os
GENO2PHENO_DIR = os.path.dirname(__file__) + "/../"
import sys
sys.path.append(GENO2PHENO_DIR)
from src.geno2pheno.pipeline import Geno2PhenoPipeline
DATA_DIR = os.path.dirname(__file__) + "/../data"

INP_DIR = '/net/sgi/metagenomics/nobackup/prot/projects/verify_genopheno/'


def test_kp():
    geno2pheno = Geno2PhenoPipeline(F"{DATA_DIR}/genyml_examples/test_kp_dataset.yml", overwrite=False, cores=20,
                                    replace_list=[('%(PROJDIR)%', f"{INP_DIR}"),('%(OUTDIR)%',f"{INP_DIR}/kp_comprehensive/"),('%(config)%', f"{DATA_DIR}/configs/")])


test_kp()
