import os
GENO2PHENO_DIR = os.path.dirname(__file__) + "/../"
MODEL_CONFIG_DIR = os.path.dirname(__file__) + "/../../data/configs/scikit_models/"
import sys
sys.path.append(GENO2PHENO_DIR)
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from src.utility.file_utility import FileUtility
import numpy as np
import codecs
from src.utility.list_set_util import argsort
from src.validation_tuning.tune_eval import *
import math
import operator
import warnings

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

        warnings.filterwarnings('ignore')
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



class RandomForest:
    '''
        Random Forest classifier
    '''
    # TODO: to make it as a parameter
    TOP_K_features = 1000

    def __init__(self, X, Y, instances, feature_names=[], rf_param_config_file = None, logger=None, linear=False):

        self.model = RandomForestClassifier(bootstrap=True, criterion='gini',
                                            min_samples_split=2, max_features='auto', min_samples_leaf=1,
                                            n_estimators=1000)
        self.X = X
        self.Y = Y
        self.instances = instances
        self.feature_names = feature_names

        if not rf_param_config_file:
            rf_param_config_file = F"{MODEL_CONFIG_DIR}/rf/rf_config.json"

        self.params_to_tune = FileUtility.load_json(rf_param_config_file, logger=logger)


    def tune_and_eval(self, save_path, inner_cv_number_of_folds, outer_cv , optimizing_score='f1_macro', n_jobs=4, overwrite=True,
                  logger=None):
        warnings.filterwarnings('ignore')
        tune_eval = TuneEvalSteps(self.X, self.Y, self.instances, outer_cv, inner_cv_number_of_folds)
        best_estimator = tune_eval.run(save_path, self.model, self.params_to_tune, optimizing_score=optimizing_score, n_jobs=n_jobs, overwrite=overwrite,
                      logger=logger)

        if len(self.feature_names) > 0:
            self.generate_RF_important_features(save_path, best_estimator, self.feature_names)

    def generate_RF_important_features(self, save_path, clf_random_forest, feature_names):
        '''
        :param clf_random_forest:
        :param feature_names:
        :param results_file:
        :param N:
        :return:
        '''
        std = np.std([tree.feature_importances_ for tree in clf_random_forest.estimators_], axis=0)

        scores = {feature_names[i]: (s, std[i]) for i, s in enumerate(list(clf_random_forest.feature_importances_)) if
                  not math.isnan(s)}
        scores = sorted(scores.items(), key=operator.itemgetter([1][0]), reverse=True)[0:RandomForest.TOP_K_features]

        f = codecs.open(F"{save_path.replace(save_path.split('/')[-3], 'feature_selection')}.txt", 'w')
        f.write('\t'.join(['feature', 'score']) + '\n')
        for w, score in scores:
            f.write('\t'.join([str(w), str(score[0])]) + '\n')
        f.close()


class LogisticRegressionClassifier:
    '''
        Logistic Regression classifier
    '''
    # TODO: to make it as a parameter
    TOP_K_features = 1000

    def __init__(self, X, Y, instances, feature_names=[], lr_param_config_file = None, logger=None, linear=False):

        self.model = LogisticRegression(C=1.0)
        self.X = X
        self.Y = Y
        self.instances = instances
        self.feature_names = feature_names

        if not lr_param_config_file:
            if linear:
                lr_param_config_file = F"{MODEL_CONFIG_DIR}/lr/lr_config.json"

        self.params_to_tune = FileUtility.load_json(lr_param_config_file, logger=logger)


    def tune_and_eval(self, save_path, inner_cv_number_of_folds, outer_cv , optimizing_score='f1_macro', n_jobs=4, overwrite=True,
                  logger=None):
        warnings.filterwarnings('ignore')
        tune_eval = TuneEvalSteps(self.X, self.Y, self.instances, outer_cv, inner_cv_number_of_folds)
        best_estimator = tune_eval.run(save_path, self.model, self.params_to_tune, optimizing_score=optimizing_score, n_jobs=n_jobs, overwrite=overwrite,
                      logger=logger)

        if len(self.feature_names) > 0:
            try:
                coefficients = best_estimator.coef_.toarray().tolist()[0]
            except:
                coefficients = best_estimator.coef_.tolist()[0]
            sorted_idxs_of_coefficients = argsort(np.abs(coefficients).tolist(), rev=True)[0:LogisticRegressionClassifier.TOP_K_features]
            FileUtility.save_list(F"{save_path.replace(save_path.split('/')[-3], 'feature_selection')}.txt",
                                  ['\t'.join(['feature', 'score'])] + [
                                      '\t'.join([self.feature_names[idx], str(coefficients[idx])]) for idx in
                                      sorted_idxs_of_coefficients], overwrite=overwrite, logger=logger)

