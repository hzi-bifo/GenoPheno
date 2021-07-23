import os
GENO2PHENO_DIR = os.path.dirname(__file__) + "/../"
import sys
sys.path.append(GENO2PHENO_DIR)

from src.utility.file_utility import FileUtility
from src.visualization_reporting.tree_visualization import VisualizeCircularTree
from src.utility.color_utility import ColorUtility
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils.validation import _num_samples
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, StratifiedKFold, KFold
import itertools
import numpy as np
from enum import Enum
from sklearn.preprocessing import OneHotEncoder

class Split(Enum):
    TreeBased = 1
    Random = 2

# TODO: Get logger

class TreeBasedKFold(_BaseKFold):
    """ TreeBasedK-Folds cross-validator extended from SCIKIT-LEARN BaseClass

    Provides train/test indices to split data in train/test sets. Split
    dataset into k consecutive folds given the phylogenetic tree of instance.

    Parameters
    ----------
    phylogenetic_tree : newick format

    n_splits : int, default=5
        Number of folds. Must be at least 2.

        .. versionchanged:: 0.22
            ``n_splits`` default value changed from 3 to 5.

    shuffle : boolean, optional
        Whether to shuffle the data before splitting into batches.

    random_state : int, RandomState instance or None, optional, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Only used when ``shuffle`` is True. This should be left
        to None if ``shuffle`` is False.

    Examples
    --------


    See also
    --------
    StratifiedKFold
        Takes group information into account to avoid building folds with
        imbalanced class distributions (for binary or multiclass
        classification tasks).

    GroupKFold: K-fold iterator variant with non-overlapping groups.

    RepeatedKFold: Repeats K-Fold n times.
    """

    def __init__(self, n_splits=5, shuffle=False,
                 random_state=None):
        super().__init__(n_splits, shuffle, random_state)
        self.folds_list = []
        self.instance_list = []
        self.train_indices = []
        self.test_instances = []
        self.test_indices = []
        self.tree_newick = ''
        self.mapping_idx_to_redidx = dict()
        self.cv_instance_fold_mapping = dict()
        self.phenotype_mapping = dict()

    def reset(self):
        self.folds_list = []
        self.instance_list = []
        self.test_instances = []
        self.train_indices = [] # original indices
        self.test_indices = [] # original indices
        self.tree_newick = ''
        self.mapping_idx_to_redidx = dict()
        self.cv_instance_fold_mapping = dict()
        self.phenotype_mapping = dict()


    def make_partition(self):
        self.train_indices = list(itertools.chain(*self.folds_list))
        self.train_indices.sort()
        self.mapping_idx_to_redidx = itertools.chain(*[[(idx,self.train_indices.index(idx)) for idx in fold] for fold in self.folds_list])
        self.folds_list = [[self.train_indices.index(idx) for idx in fold] for fold in self.folds_list]

    def load_predefined_folds(self, tree_newick, instance_list, train_folds_file_path, test_instance_file_path=None, y=None, save_directory=None, tag_setting='', save_tree=True, overwrite=False, logger=None):

        self.reset()
        self.train_indices = list(range(len(instance_list)))
        self.folds_list = [[instance_list.index(instance) for instance in fold.split() if instance in instance_list] for fold in FileUtility.load_list(train_folds_file_path)]
        self.instance_list = instance_list
        self.tree_newick = tree_newick
        if test_instance_file_path:
            self.test_indices = [instance_list.index(instance) for instance in FileUtility.load_list(test_instance_file_path) if instance in instance_list]
            self.test_instances = [self.instance_list[idx] for idx in self.test_indices]
            self.cv_instance_fold_mapping = {instance: "Test set" for instance in self.test_instances}
        if y:
            self.phenotype_mapping = {instance:y[idx] for idx,instance in enumerate(self.instance_list)}

        cv_instances = [[self.instance_list[idx] for idx in fold] for fold in self.folds_list]
        self.cv_instance_fold_mapping.update(dict(list(itertools.chain(
            *[[(instance, "Fold_{:02d}".format(idx+1)) for instance in fold_list] for idx, fold_list in
              enumerate(cv_instances)]))))

        if save_directory:
            if FileUtility.exists(save_directory):
                train_description =  'predefined'
                test_description = ''
                FileUtility.save_list(f"{save_directory}/cv_folds.txt",['\t'.join(fold) for fold in cv_instances], overwrite=overwrite, logger=logger)
                if len(self.test_instances)>0:
                    test_description = 'predefined'
                    FileUtility.save_list(f"{save_directory}/test_fold.txt",
                                          self.test_instances, overwrite=overwrite, logger=logger)
                else:
                    # if test instances are removed the old file needs to be removed
                    FileUtility.remove(f"{save_directory}/test_fold.txt")
            else:
                raise ValueError("The provided path to save the tree-based CV does not exist")

            if save_tree:
                try:
                    self.save_tree(f"{save_directory}/tree_cv_train_{train_description}{'_test_'+test_description if test_description else ''}{'_'+tag_setting if tag_setting else ''}", "Visualization of Cross-validation Folds over the Phylogenetic Tree")
                    if logger:
                        logger.info(
                            f"the annotated tree is saved at: {save_directory}/tree_cv_train_{train_description}{'_test_' + test_description if test_description else ''}{'_' + tag_setting if tag_setting else ''}")
                except:
                    if logger:
                        logger.info(f"problem with saving the tree")
        self.make_partition()


    def set_tree_instances_testratio_list(self, tree_newick, instance_list, y=None, test_ratio=0, train_split_method = Split.TreeBased, test_split_method = Split.TreeBased, save_directory=None, tag_setting='', save_tree=True, random_state=1, overwrite=False, logger=None):
        '''
        :param tree_newick:
        :param instance_list:
        :param test_ratio: should start with _
        :param save_directory:
        :param tag_setting:
        :param save_tree:
        :return:
        '''

        # TODO: handle prefix instances too ISO1 ISO11
        self.reset()
        print(instance_list)                           
        self.instance_list = instance_list
        self.tree_newick = tree_newick
        self.train_indices = list(range(len(instance_list)))
        modified_instance_list = self.instance_list
        modified_tree = self.tree_newick

        if y:
            self.phenotype_mapping = {instance:y[idx] for idx,instance in enumerate(self.instance_list)}

        if test_ratio > 0:
            if test_ratio >= 1:
                raise ValueError("The test ratio must be less than 1")
            else:

                if test_split_method == Split.TreeBased:

                    temp_n_split_for_test =  int(1/test_ratio)
                    location_of_instance_in_tree_line = [self.tree_newick.find(instance) for instance in self.instance_list]
                    sorted_instance_idx = np.argsort(location_of_instance_in_tree_line)
                    folds_list = TreeBasedKFold.chunk_for_cv(sorted_instance_idx, temp_n_split_for_test)
                    location_of_instance_in_tree_line.sort()
                    sort_locs = TreeBasedKFold.chunk_for_cv(location_of_instance_in_tree_line, temp_n_split_for_test)
                    try:
                        idx_best_isolated_fold = np.argmin(
                            [np.abs(self.tree_newick[x[0]:x[-1]].count('(') - self.tree_newick[x[0]:x[-1]].count(')')) for x in
                             sort_locs])
                    except:
                        idx_best_isolated_fold=0   
                    modified_tree = self.tree_newick.replace(
                        self.tree_newick[sort_locs[idx_best_isolated_fold][0]:sort_locs[idx_best_isolated_fold][-1]], '')

                    self.test_indices = folds_list[idx_best_isolated_fold]


                elif test_split_method == Split.Random:
                    # if y is given stratified splitting
                    if y:
                        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_state)
                        try:
                            self.test_indices = list(sss.split(self.instance_list, y))[0][1].tolist()
                        except:
                            raise ValueError('The test ratio does not match the number of instance')
                    # else completely random
                    else:
                        sss = ShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_state)
                        try:                            
                            self.test_indices = list(sss.split(self.instance_list, y))[0][1].tolist()
                        except:
                            raise ValueError('The test ratio does not match the number of instance')

                self.test_instances = [self.instance_list[idx] for idx in self.test_indices]
                self.cv_instance_fold_mapping = {instance:"Test set" for instance in self.test_instances}
                modified_instance_list = [x for x in self.instance_list if x not in self.test_instances]
        if y:
            modified_y = [y[self.instance_list.index(instance)] for instance in modified_instance_list]

        # TODO: Extend tree splitting to multiple schemes
        if train_split_method==Split.TreeBased:
            # create train folds
            location_of_instance_in_tree_line = [modified_tree.find(instance) for instance in modified_instance_list]
            sorted_instance_idx = np.argsort(location_of_instance_in_tree_line)
            temp_folds_list = TreeBasedKFold.chunk_for_cv(sorted_instance_idx, self.n_splits)

        if train_split_method == Split.Random:
            # if y is given stratified splitting
            if y:
                sss = StratifiedKFold(n_splits=self.n_splits,shuffle=True,random_state=random_state)
                try:
                    temp_folds_list = [folding[1].tolist() for folding in sss.split(modified_instance_list, modified_y)]
                except:
                    raise ValueError('The split number does not match the number of instance')
            # else completely random
            else:
                sss = KFold(n_splits=self.n_splits, shuffle=True, random_state=random_state)
                try:
                    temp_folds_list = [folding[1].tolist() for folding in sss.split(modified_instance_list)]
                except:
                    raise ValueError('The split number does not match the number of instance')

        cv_instances = [[modified_instance_list[idx] for idx in fold] for fold in temp_folds_list]
        self.cv_instance_fold_mapping.update(dict(list(itertools.chain(
            *[[(instance, "Fold_{:02d}".format(idx+1)) for instance in fold_list] for idx, fold_list in
              enumerate(cv_instances)]))))
        # reconstruct the original indices
        self.folds_list = [[self.instance_list.index(modified_instance_list[idx]) for idx in idx_list] for idx_list in temp_folds_list]


        if save_directory:
            if FileUtility.exists(save_directory):
                train_description =  'treebased' if train_split_method==Split.TreeBased else 'random' if y!=None else 'stratified'
                test_description = ''
                FileUtility.save_list(f"{save_directory}/cv_folds.txt",['\t'.join(fold) for fold in cv_instances], overwrite=overwrite, logger=logger)

                if len(self.test_instances)>0:
                    test_description = 'treebased' if test_split_method==Split.TreeBased else 'random' if y!=None else 'stratified'
                    FileUtility.save_list(f"{save_directory}/test_fold.txt",
                                          self.test_instances, overwrite=overwrite, logger=logger)
                else:
                    try:
                        # if test instances are removed the old file needs to be removed
                        FileUtility.remove(f"{save_directory}/test_fold.txt")
                    except:
                        if logger:
                            logger.warning(f"file to remove doesn't exist: {save_directory}/test_fold.txt")
            else:
                raise ValueError("The provided path to save the tree-based CV does not exist")
                if logger:
                    logger.error(f"The provided path to save the tree-based CV does not exist")

            if save_tree:
                try:
                    self.save_tree(f"{save_directory}/tree_cv_train_{train_description}{'_test_'+test_description if test_description else ''}{'_'+tag_setting if tag_setting else ''}", "Visualization of Cross-validation Folds over the Phylogenetic Tree")
                    if logger:
                        logger.info(f"the annotated tree is saved at: {save_directory}/tree_cv_train_{train_description}{'_test_'+test_description if test_description else ''}{'_'+tag_setting if tag_setting else ''}")
                except:
                    if logger:
                        logger.info(f"problem with saving the tree")

        self.make_partition()

    def save_tree(self, path, title):
        vis_tree = VisualizeCircularTree(self.tree_newick)
        distinct_folds = list(set(self.cv_instance_fold_mapping.values()))
        color_mapping = {value: (ColorUtility.thirdy_distinct_colors[idx] if self.n_splits<30 else ColorUtility.color_schemes['large'][idx]) for idx, value in enumerate(distinct_folds)}

        self.vector_mapping = None

        if len(self.phenotype_mapping)>0:
            # prepare the encoding
            enc = OneHotEncoder(handle_unknown='ignore')
            all_possible_labels = list(set(self.phenotype_mapping.values()))
            all_possible_labels.sort()
            Y = [[l] for l in all_possible_labels]
            enc.fit(Y)
            phenotypes = enc.categories_[0].tolist()
            title = title + F"\nPhenotype is depicted as the an ordered heatmap vector at each node.\nThe order from the inner to the outer is {', '.join([str(x) for x in phenotypes])} (red means presence)"
            self.vector_mapping = {instance:enc.transform([[phenotype]]).toarray()[0] for instance,phenotype in self.phenotype_mapping.items()}


        vis_tree.create_circle(path, title, name2color=None,
                               name2class_dic=self.cv_instance_fold_mapping, class2color_dic=color_mapping,
                               ignore_branch_length=True, vector=self.vector_mapping)

    def _iter_test_indices(self, X, y=None, groups=None):

        for e in self.folds_list:
            yield e

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        n_samples = _num_samples(X)
        if self.n_splits > n_samples:
            raise ValueError(
                ("Cannot have number of splits n_splits={0} greater"
                 " than the number of samples: n_samples={1}.")
                    .format(self.n_splits, n_samples))

        for idx in range(len(self.folds_list)):
            test = self.folds_list[idx]
            train = list(itertools.chain(*[fold for idy, fold in enumerate(self.folds_list) if idy != idx]))
            yield train, test

    @staticmethod
    def chunk_for_cv(x, n_split):
        '''
        :param l: the list
        :param n: num
        :return:
        '''
        return [l.tolist() for l in np.array_split(x, n_split)]