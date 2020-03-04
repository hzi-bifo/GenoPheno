import os
GENO2PHENO_DIR = os.path.dirname(__file__) + "/../"
import sys
sys.path.append(GENO2PHENO_DIR)
import pandas as pd
import json
from src.utility.math_utility import *
from src.utility.file_utility import FileUtility

class PhenotypeTable(object):

    def __init__(self, path, save_path=None, overwrite=True,  logger=None):

        self.phenotype_table = pd.read_table(path, delimiter='\s+')
        self.phenotypes = self.phenotype_table.columns.tolist()[1::]
        self.instances = self.phenotype_table[self.phenotype_table.columns.tolist()[0]].tolist()[1::]
        self.matrix = self.phenotype_table[self.phenotypes]

        # phenotype dictionary preparation
        self.phenotype_dict = dict()
        for phenotype in self.phenotypes:
            # in case of having no title for the isolates
            if len(self.phenotype_table[phenotype].tolist()) != len(self.instances):
                self.instances = self.phenotype_table[self.phenotype_table.columns.tolist()[0]].tolist()[0::]
            self.phenotype_dict[phenotype] = pd.Series(self.phenotype_table[phenotype].tolist(), index=self.instances).to_dict()

        # TODO: add phenotype vector for tree visualization

        if save_path and (not FileUtility.exists(save_path) or overwrite):
            with open(save_path, 'w') as fp:
                json.dump(self.phenotype_dict, fp)
            if logger:
                logger.info(F"the phenotype json file is created: {save_path}")
        elif FileUtility.exists(save_path) and save_path:
            logger.info(F"the phenotype json file exists in: {save_path} ")


    def get_phenotype_kl_divergence(self):
        '''
        :return: kl-div between phenotypes
        '''
        KL=get_kl_rows(self.matrix.T)
        KL=KL/np.max(KL)
        return KL

    def get_instance_kl_divergence(self):
        '''
        :return: kl-div between samples
        '''
        KL=get_kl_rows(self.matrix)
        KL=KL/np.max(KL)
        return KL

    def get_phenotype_correlation_coefficient(self):
        '''
        :return: Return Pearson product-moment correlation coefficients
        '''
        return np.corrcoef(self.matrix.T)

    def get_instance_correlation_coefficient(self):
        '''
        :return: Return Pearson product-moment correlation coefficients
        '''
        return np.corrcoef(self.matrix)
