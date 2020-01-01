import os

GENO2PHENO_DIR = os.path.dirname(__file__) + "/../"
import sys
sys.path.append(GENO2PHENO_DIR)
import numpy as np
from src.utility.file_utility import FileUtility
from src.utility.list_set_util import get_intersection_of_list
from scipy import sparse

class Genotype_data_load(object):

    @staticmethod
    def load_single_feature(output_directory, feature_title):
        mat = FileUtility.load_sparse_csr(F"{output_directory}/intermediate_representations/{feature_title}_feature_data.npz")
        feature_names = FileUtility.load_list(F"{output_directory}/intermediate_representations/{feature_title}_feature_names.txt")
        instances = FileUtility.load_list(F"{output_directory}/intermediate_representations/{feature_title}_instances.txt")
        return mat, feature_names, instances

    @staticmethod
    def load_aggregated_data(output_directory, feature_list, phenotype_table, mapping=None, logger = None):

        # dictionary of feature types
        X_dict = dict()
        # dictionary of feature names
        feature_names_dict = dict()
        # dictionary of isolates
        instance_dict = dict()

        if logger:
            logger.debug(F"aggregation of {'-'.join(feature_list)} is in progress")

        for feature in feature_list:
            X_dict[feature], feature_names_dict[feature], instance_dict[feature] = Genotype_data_load.load_single_feature(output_directory, feature)

            if logger:
                logger.debug(F"the shape of {feature} is {X_dict[feature].shape}")

        for phenotype, instance_to_label in phenotype_table.phenotype_dict.items():

            if logger:
                logger.debug(F"start aggregation for phenotype {phenotype}")

            instance_final = get_intersection_of_list(list(instance_dict.values())+[list(instance_to_label.keys())])
            instance_final.sort()

            if logger:
                logger.debug(F"{len(instance_final)} instances are selected")


            X_final = []
            feature_names_final = []
            for feature_title, mat in X_dict.items():
                instances_in_mat = instance_dict[feature_title]
                idx_common_instances_in_mat = [instances_in_mat.index(instance) for instance in instance_final]
                X_final.append(mat[idx_common_instances_in_mat,:].toarray())
                feature_names_final = feature_names_final + feature_names_dict[feature_title]

            X_final = np.concatenate(tuple(X_final), axis=1)
            X_final = sparse.csr_matrix(X_final)
            Y_final = [mapping[instance_to_label[instance]] if instance_to_label[instance]  in mapping else  instance_to_label[instance] for instance in instance_final]

            if logger:
                logger.debug(F"the shape of feature matrix for phenotype {phenotype} of {'-'.join(feature_list)} is {X_final.shape}")

            yield phenotype, (X_final, Y_final, feature_names_final, instance_final)

