import os
GENO2PHENO_DIR = os.path.dirname(__file__) + "/../"
MODEL_CONFIG_DIR = os.path.dirname(__file__) + "/../../data/configs/scikit_models/"
import sys
sys.path.append(GENO2PHENO_DIR)
from sklearn.svm import SVC,LinearSVC
from src.utility.file_utility import FileUtility
import numpy as np
import codecs
from src.utility.list_set_util import argsort
from src.validation_tuning.tune_eval import *

class SVM:
    '''
        Support vector machine classifier
    '''
    # TODO: to make it as a parameter
    TOP_K_features = 1000

    def __init__(self, X, Y, instances, feature_names=[], svm_param_config_file = None, logger=None, linear=False):

        if linear:
            self.model = LinearSVC(C=1.0)
        else:
            self.model = SVC(C=1.0)
        self.X = X
        self.Y = Y
        self.instances = instances
        self.feature_names = feature_names

        if not svm_param_config_file:
            if linear:
                svm_param_config_file = F"{MODEL_CONFIG_DIR}/svm/linear_svm_config.json"

            else:
                svm_param_config_file = F"{MODEL_CONFIG_DIR}/svm/svm_config.json"
        self.params_to_tune = FileUtility.load_json(svm_param_config_file, logger=logger)


    def tune_and_eval(self, save_path, inner_cv_number_of_folds, outer_cv , optimizing_score='f1_macro', n_jobs=4, overwrite=True,
                  logger=None):

        np.save('X',self.X)
        FileUtility.save_list('Y.txt', self.Y)
        FileUtility.save_list('features.txt', self.feature_names)
        FileUtility.save_list('instances.txt', self.instances)
        tune_eval = TuneEvalSteps(self.X, self.Y, self.instances, outer_cv, inner_cv_number_of_folds)
        best_estimator = tune_eval.run(save_path, self.model, self.params_to_tune, optimizing_score=optimizing_score, n_jobs=n_jobs, overwrite=overwrite,
                      logger=logger)

        if len(self.feature_names) > 0:
            if type(best_estimator)==SVC:
                if best_estimator.kernel == 'linear':
                    coefficients = best_estimator.coef_.toarray().tolist()[0]
                    sorted_idxs_of_coefficients = argsort(np.abs(coefficients).tolist(), rev=True)[0:SVM.TOP_K_features]
                    FileUtility.save_list(F"{save_path.replace(save_path.split('/')[-3], 'feature_selection')}.txt",
                                          ['\t'.join(['feature', 'score'])] + [
                                              '\t'.join([self.feature_names[idx], str(coefficients[idx])]) for idx in
                                              sorted_idxs_of_coefficients], overwrite=overwrite, logger=logger)

            else:
                coefficients = best_estimator.coef_.tolist()[0]
                sorted_idxs_of_coefficients = argsort(np.abs(coefficients).tolist(), rev=True)[0:SVM.TOP_K_features]
                FileUtility.save_list(F"{save_path.replace(save_path.split('/')[-3], 'feature_selection')}.txt",
                                          ['\t'.join(['feature', 'score'])] + [
                                              '\t'.join([self.feature_names[idx], str(coefficients[idx])]) for idx in
                                              sorted_idxs_of_coefficients], overwrite=overwrite, logger=logger)

