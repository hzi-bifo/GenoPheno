import os
GENO2PHENO_DIR = os.path.dirname(__file__) + "/../../"
import sys
sys.path.append(GENO2PHENO_DIR)
from src.geno2pheno.pipeline import Geno2PhenoPipeline

DATA_DIR = os.path.dirname(__file__) + "/../../data"

def run_test(path_to_in_dir, path_to_out_dir, cores=30):
    geno2pheno = Geno2PhenoPipeline(F"{path_to_in_dir}/config.yml", overwrite=True, cores=cores,
                                    replace_list=[('%(config)%', f"{DATA_DIR}/configs/"),('%(PROJDIR)%',f"{path_to_in_dir}"),
                                                  ('%(OUTDIR)%', f"{path_to_out_dir}")
                                                  ])


run_test('data/server_example/toy_data_in','./project_output/', 30)
