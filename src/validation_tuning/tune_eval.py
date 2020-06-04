import os
GENO2PHENO_DIR = os.path.dirname(__file__) + "/../"
import sys
sys.path.append(GENO2PHENO_DIR)
from src.validation_tuning.metrics import Scoring
from src.utility.file_utility import FileUtility
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_val_score
import itertools
import copy
import numpy as np
import sklearn.metrics as skmetrics
import warnings

class TuneEvalSteps(object):
    '''
        Tuning and evaluation
    '''
    def __init__(self, X, Y, instances_list, cv_outer, cv_inner_fold_num, random_state = 0):
        self.X = X
        self.Y = Y
        self.instance_list = instances_list
        self.cv_outer = cv_outer

        self.X_train = self.X[self.cv_outer.train_indices,:]
        self.Y_train = [self.Y[idx] for idx in self.cv_outer.train_indices]
        self.instance_list_train = [self.instance_list[idx] for idx in self.cv_outer.train_indices]

        self.X_test =None
        self.Y_test = None
        self.instance_list_test = None

        if len(self.cv_outer.test_indices)>0:
            self.X_test = self.X[self.cv_outer.test_indices, :]
            self.Y_test = [self.Y[idx] for idx in self.cv_outer.test_indices]

        self.cv_inner = StratifiedKFold(n_splits=cv_inner_fold_num, shuffle=True, random_state=random_state)

    def run(self, save_path, model, model_param, optimizing_score='f1_macro', n_jobs=4, overwrite=False, logger=None):

        warnings.filterwarnings('ignore')


        if logger:
            logger.info(F"fine tuning is getting started..")

        if optimizing_score not in Scoring.metrics or optimizing_score not in Scoring.functions:
            if logger:
                logger.error(F"the score {optimizing_score} is not defined in Scoring class")
            exit()

        eval_fun = Scoring.functions[optimizing_score]

        # inner CV
        clf_inner = GridSearchCV(estimator=model, param_grid=model_param, cv=self.cv_inner, refit=Scoring.metrics[optimizing_score], n_jobs=n_jobs)

        # (1) nested CV
        ## outer CV
        nested_cross_val_pred = cross_val_predict(clf_inner, X=self.X_train, y=self.Y_train, cv=self.cv_outer)
        ## (1.1) the results of nested cv
        nested_cv_true_pred_list = [([self.Y_train[idx] for idx in valid_idxs], [nested_cross_val_pred[idx] for idx in valid_idxs]) for valid_idxs in
         self.cv_outer.folds_list]
        ## (1.2) the order of instance
        instances_in_nested_cv_res = [[self.instance_list_train[idx] for idx in valid_idxs]  for valid_idxs in self.cv_outer.folds_list]

        if logger:
            score_res=[eval_fun(y_t,y_p) for y_t,y_p in nested_cv_true_pred_list]
            logger.info(F"nested CV is done and the unbiased {optimizing_score} score is {np.round(np.mean(score_res), 2)} ± {np.round(np.std(score_res),2)} ")


        # (2) parameter tuning on the whole train
        clf_inner.fit(self.X_train, self.Y_train)
        tunned_classifier_on_train = clf_inner.best_estimator_
        tunned_parameters_on_train = clf_inner.best_params_

        if logger:
            logger.info(F"the parameter are tuned over the whole training set")


        ## (2.1) the results of non-nested cv
        # TODO: make sure the split of begging remains after the second
        non_nested_split = list(self.cv_inner.split(self.X_train, self.Y_train))
        non_nested_cross_val_pred = cross_val_predict(clf_inner.best_estimator_, self.X_train, self.Y_train, cv=self.cv_inner)
        nonnested_cv_true_pred_list = [
            ([self.Y_train[idy] for idy in train_valid_idxs], [non_nested_cross_val_pred[idy] for idy in train_valid_idxs]) for x, train_valid_idxs
            in non_nested_split]
        instances_in_nonnested_cv_res = [[self.instance_list_train[idx] for idx in valid_idxs]  for x, valid_idxs in non_nested_split]

        if logger:
            score_res = [eval_fun(y_t, y_p) for y_t, y_p in nonnested_cv_true_pred_list]
            logger.info(F"the non-nested cv is also done and the {optimizing_score} score is {np.round(np.mean(score_res), 2)} ± {np.round(np.std(score_res),2)} ")

        # TODO: handle when there is no test
        ## (2.2) test performance
        if len(self.cv_outer.test_indices)>0:
            clf_inner.best_estimator_.fit(self.X_train, self.Y_train)
            test_true_pred_list = [self.Y_test, clf_inner.best_estimator_.predict(self.X_test)]
            instances_in_test_res = [self.instance_list[idx]  for idx in self.cv_outer.test_indices]

            if logger:
                score_res = eval_fun(test_true_pred_list[0], test_true_pred_list[1])
                logger.info(F"the evaluated {optimizing_score} score is on test set is {np.round(np.mean(score_res), 2)} ")

        else:
            test_true_pred_list = None
            instances_in_test_res = None

        #  (3) final model
        best_model_aggregation_test_train = copy.deepcopy(clf_inner.best_estimator_)
        best_model_aggregation_test_train.fit(self.X, self.Y)

        FileUtility.save_obj(save_path, [nested_cv_true_pred_list, instances_in_nested_cv_res, tunned_classifier_on_train, tunned_parameters_on_train, nonnested_cv_true_pred_list, instances_in_nonnested_cv_res, test_true_pred_list, instances_in_test_res, best_model_aggregation_test_train],overwrite=overwrite, logger=logger)

        return best_model_aggregation_test_train

    @staticmethod
    def load_tune_eval_res(file_path, logger=None):
        [nested_cv_true_pred_list, instances_in_nested_cv_res, tunned_classifier_on_train, tunned_parameters_on_train, nonnested_cv_true_pred_list, instances_in_nonnested_cv_res, test_true_pred_list, instances_in_test_res, best_model_aggregation_test_train] = FileUtility.load_obj(file_path, logger=logger)
        return {'nested_cv_true_pred_list':nested_cv_true_pred_list, 'instances_in_nested_cv_res':instances_in_nested_cv_res,'tunned_classifier_on_train':tunned_classifier_on_train,'tunned_parameters_on_train':tunned_parameters_on_train,'nonnested_cv_true_pred_list':nonnested_cv_true_pred_list,'instances_in_nonnested_cv_res': instances_in_nonnested_cv_res, 'test_true_pred_list':test_true_pred_list, 'instances_in_test_res':instances_in_test_res, 'best_model_aggregation_test_train':best_model_aggregation_test_train}


