import os
GENO2PHENO_DIR = os.path.dirname(__file__) + "/../"
import sys
sys.path.append(GENO2PHENO_DIR)
from src.geno2pheno.pipeline import Geno2PhenoPipeline
DATA_DIR = os.path.dirname(__file__) + "/../data"

def test_ps_tob():
    geno2pheno = Geno2PhenoPipeline(F"{DATA_DIR}/genyml_examples/classify_psg_dataset_tob.yml", overwrite=True, cores=4,
                                    replace_list=[('%(PROJDIR)%', f"/Users/vwr33vv/Documents/techday/pseudomona/final/input_data/"),('%(config)%', f"{DATA_DIR}/configs/"),('%(DRUG)%','TOB'),('%(drug_full)%','Tobramycin'),('%(OUTDIR)%','/Users/vwr33vv/Documents/techday/pseudomona/final/input_data/results/')])



# def test_ps_mer():
#     geno2pheno = Geno2PhenoPipeline(F"{DATA_DIR}/genyml_examples/pseudomonas_dir/test_pseudomona_MER.yml", overwrite=False, cores=4,
#                                     replace_list=[('%(PROJDIR)%', f"{DATA_DIR}/test_data/pseudogenomics/"),('%(config)%', f"{DATA_DIR}/configs/")])
#
#
# def test_ps_cip():
#     geno2pheno = Geno2PhenoPipeline(F"{DATA_DIR}/genyml_examples/pseudomonas_dir/test_pseudomona_CIP.yml", overwrite=False, cores=4,
#                                     replace_list=[('%(PROJDIR)%', f"{DATA_DIR}/test_data/pseudogenomics/"),('%(config)%', f"{DATA_DIR}/configs/")])
#
# def test_ps_col():
#     geno2pheno = Geno2PhenoPipeline(F"{DATA_DIR}/genyml_examples/pseudomonas_dir/test_pseudomona_COL.yml", overwrite=False, cores=4,
#                                     replace_list=[('%(PROJDIR)%', f"{DATA_DIR}/test_data/pseudogenomics/"),('%(config)%', f"{DATA_DIR}/configs/")])
#
# def test_ps_cez():
#     geno2pheno = Geno2PhenoPipeline(F"{DATA_DIR}/genyml_examples/pseudomonas_dir/test_pseudomona_CEZ.yml", overwrite=False, cores=4,
#                                     replace_list=[('%(PROJDIR)%', f"{DATA_DIR}/test_data/pseudogenomics/"),('%(config)%', f"{DATA_DIR}/configs/")])
#
test_ps_tob()

