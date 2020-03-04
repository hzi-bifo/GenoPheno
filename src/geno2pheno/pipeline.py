import os
import sys
import tqdm
import itertools
import pandas as pd
import copy
GENO2PHENO_DIR = os.path.dirname(__file__) + "/../"
sys.path.append(GENO2PHENO_DIR)
import logging
from src.phenotype_util.phenotype_table import PhenotypeTable
from src.genotyoe_util.genotype_create_utility import *
from src.genotyoe_util.genotype_load_utility import *
from src.validation_tuning.tree_based_kfold import *
from src.discriminative_models.scikit_classification_models import *
from src.validation_tuning.tune_eval import TuneEvalSteps
from src.validation_tuning.metrics import Scoring
from src.utility.color_utility import ColorUtility
from src.utility.list_set_util import *
import plotly.graph_objects as go

class Geno2PhenoPipeline(object):

    def __init__(self, pipeline_yaml, overwrite=True, cores=4, replace_list=[]):

        self.pipeline = FileUtility.load_yaml(pipeline_yaml, replace_list=replace_list)
        self.number_of_cores = cores
        self.overwrite = overwrite
        self.pipeline_yaml = pipeline_yaml

        # logging setting
        self.logger = logging.getLogger('Geno2Pheno')
        self.logger.setLevel(logging.DEBUG)
        self.log_handler = logging.FileHandler("geno2pheno_log.txt")
        self.log_handler.setFormatter(logging.Formatter("%(levelname)s %(asctime)s - %(message)s "))
        self.logger.addHandler(self.log_handler)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

        # object of all defined k-folds
        self.kfold_settings = dict()
        # track all requested results in a dict
        self.requested_results = dict()

        # read metadata
        self.parse_project_metadata()
        # prepare the outline
        self.prepare_output_directory_structure()
        # prepare the basic reports
        self.prepare_metadata_reports()
        # load the feature tables
        self.list_of_primitive_tables = []
        self.load_the_feature_tables()
        # run the prediction blocks
        self.run_prediction_block()
        # generate report
        self.prepare_report()
        # TODO: add misclassification to the report
        # TODO: generate HTML report

    def parse_project_metadata(self):
        # parse the metadata and descriptions
        self.project_name = self.pipeline['metadata']['project']

        self.phylogenetic_tree = FileUtility.load_list(self.pipeline['metadata']['phylogenetic_tree'])[0]

        self.phenotype_table_path = self.pipeline['metadata']['phenotype_table']

        self.output_directory = self.pipeline['metadata']['output_directory']
        self.output_directory = self.output_directory + ('/' if self.output_directory[-1] != '/' else '')

        if 'number_of_cores' in self.pipeline['metadata']:
            self.number_of_cores = self.pipeline['metadata']['number_of_cores']

        self.logger.info('Meta data has been parsed')

    def prepare_output_directory_structure(self):
        # make sure the directory exist and handle overriding
        if self.overwrite and FileUtility.exists(self.output_directory):
            removed = FileUtility.force_remove_dir(self.output_directory)
            if removed:
                self.logger.info(
                    F"directory removal was successful: {self.output_directory}")
            else:
                self.logger.warning(
                    F"directory removal was NOT successful: {self.output_directory}")

        # move the log file to the output dir
        FileUtility.ensure_dir(self.output_directory, self.logger)
        temp_logs = FileUtility.load_list('geno2pheno_log.txt')
        FileUtility.save_list(F"{self.output_directory}geno2pheno_log.txt", temp_logs)
        self.logger.removeHandler(self.log_handler)
        FileUtility.remove('geno2pheno_log.txt')

        # setup the logger
        self.log_handler = logging.FileHandler(F"{self.output_directory}geno2pheno_log.txt")
        self.log_handler.setFormatter(logging.Formatter("%(levelname)s %(asctime)s - %(message)s "))
        self.logger.addHandler(self.log_handler)

        # create the layout
        FileUtility.ensure_dir(F"{self.output_directory}metadata/", self.logger)
        FileUtility.ensure_dir(F"{self.output_directory}intermediate_representations/", self.logger)
        FileUtility.ensure_dir(F"{self.output_directory}classification/", self.logger)
        FileUtility.ensure_dir(F"{self.output_directory}classification/cv/", self.logger)
        FileUtility.ensure_dir(F"{self.output_directory}feature_selection/", self.logger)
        FileUtility.ensure_dir(F"{self.output_directory}reports/", self.logger)
        FileUtility.ensure_dir(F"{self.output_directory}reports/metadata_related/", self.logger)
        FileUtility.ensure_dir(F"{self.output_directory}reports/classification/", self.logger)

        # copy config file
        FileUtility.save_list(F"{self.output_directory}config.yaml", FileUtility.load_list(self.pipeline_yaml), overwrite=self.overwrite, logger=self.logger)
        self.pipeline_yaml = F"{self.output_directory}config.yaml"

        # load phylogenetic_tree
        FileUtility.save_list(F"{self.output_directory}metadata/phylogenetic_tree.txt", [self.phylogenetic_tree], overwrite=self.overwrite, logger=self.logger)

        # load phenotype table
        self.phenotype_table = PhenotypeTable(self.phenotype_table_path, overwrite=self.overwrite, save_path=F"{self.output_directory}metadata/phenotype.json", logger=self.logger)


    def prepare_metadata_reports(self):
        pass
        # create_phylogenetic tree_visualization
        # TODO: UNCOMMENT
        #vcct=VisualizeCircularTree(self.phylogenetic_tree, overwrite=self.overwrite, logger=self.logger)
        #vcct.create_circle(F"{self.output_directory}reports/metadata_related/phylogenetic_tree", F"Phylogenetic Tree -- {self.project_name}")

        # TODO: create phenotype relations
        # TODO: create phenotype stats
        # TODO: add log to console

    def load_the_feature_tables(self):

        preprocessing_map = {'l1':Preprocessing.L1, 'l2':Preprocessing.L2, 'percent':Preprocessing.PERCENT, 'zero2one':Preprocessing.ZERO2ONE, 'none':Preprocessing.NONE, 'std':Preprocessing.STANDARDIZED, 'binary':Preprocessing.BINARY}
        data_type_map = {'text': DataType.TEXT, 'numerical':DataType.NUMERICAL}

        self.logger.info('loading the genotype tables')

        for table in tqdm.tqdm(self.pipeline['genotype_tables']['tables']):

            delimeter = None

            if 'table' not in table:
                if 'sequence' in table:
                    # TODO: add preprocessing for k-mers
                    # TODO: add bpe unit
                    if self.overwrite or (not FileUtility.exists(F"{self.output_directory}intermediate_representations/{table['sequence']}_feature_data.npz") or not FileUtility.exists(F"{self.output_directory}intermediate_representations/{table['sequence']}_feature_names.txt") or not FileUtility.exists
                        (F"{self.output_directory}intermediate_representations/{table['sequence']}_instances.txt")):
                        GenotypeVectorizer.create_seq_kmer_table(F"{self.output_directory}intermediate_representations/",
                                                             table['path'], table['sequence'],
                                                             kvalue=table['k_value'], n_cores=self.number_of_cores,
                                                             logger=self.logger,overwrite=self.overwrite)
                    self.list_of_primitive_tables.append(table['sequence'])
                else:
                    logging.error('The name of table needs to be specified')
                    break
            else:
                self.list_of_primitive_tables.append(table['table'])

                if 'preprocessing' not in table or table['preprocessing'] not in preprocessing_map:
                    preprocessing = Preprocessing.NONE
                    logging.warning(F"No preprocessing is selected for table {table['table']}")
                else:
                    preprocessing = preprocessing_map[table['preprocessing']]

                if  table['datatype'] not in data_type_map:
                    logging.error("The datatype of table needs to be 'text' or 'numerical'")
                    break
                else:
                    datatype = data_type_map[table['datatype']]

                GenotypeVectorizer.create_genotype_table(F"{self.output_directory}intermediate_representations/", table['path'], table['table'], delimeter=table['delimeter'] if 'delimeter' in table else None, preprocessing=preprocessing, datatype=datatype, logger=self.logger, overwrite=self.overwrite)

    def run_prediction_block(self):
        self.logger.info('begin parsing the prediction block')
        split_method_map = {'random':Split.Random, 'treebased':Split.TreeBased}

        for prediction in self.pipeline['predictions']:

            self.requested_results[prediction['prediction']]=dict()

            self.logger.info(F" * begin with the prediction of {prediction['prediction']}")
            # label mapping
            label_mapping = prediction['label_mapping'] if 'label_mapping' in prediction else None
            # optimizing for
            optimized_for = prediction['optimized_for']
            # TODO: make reporting specialized
            # final reporting metrics
            self.reporting = prediction['reporting']
            # classifiers
            classifiers = prediction['classifiers']

            print('prediction')
            for feature in prediction['features']:
                print('hereee')

                feature_list = feature['list']
                feature_title =  feature['feature']

                if feature_title not in self.requested_results[prediction['prediction']]:
                    self.requested_results[prediction['prediction']][feature_title] = dict()

                for phenotype, (X, Y, feature_names, instances) in Genotype_data_load.load_aggregated_data(self.output_directory, feature_list, self.phenotype_table, mapping=label_mapping, logger = self.logger):
                    self.logger.info(F" ** begin with phenotype {phenotype} prediction of {prediction['prediction']}")

                    if phenotype not in self.requested_results[prediction['prediction']][feature_title]:
                        self.requested_results[prediction['prediction']][feature_title][phenotype] = dict()

                    print(feature)
                    if 'validation_tuning' in feature:
                        print('here')
                        # read the tune/eval setting
                        validation_tuning_setting = feature['validation_tuning']
                        cv_name = validation_tuning_setting['name']
                        current_cv = F"{validation_tuning_setting['name']}>{phenotype}"
                        self.kfold_settings[current_cv] = TreeBasedKFold(validation_tuning_setting['train']['folds'])
                        try:
                            test_ratio = validation_tuning_setting['test']['ratio'] if 'test' in validation_tuning_setting else 0
                            self.logger.info(f"for the feature {feature_title} the test/ratio {test_ratio}")

                        except:
                            self.logger.error(f"for the feature {feature_title} the test/ratio is not properly defined")
                            exit()

                        if validation_tuning_setting['train']['method'] in split_method_map:
                            train_split_method = split_method_map[validation_tuning_setting['train']['method']]
                            self.logger.info(f"for the feature {feature_title} the train cv method is {train_split_method}")
                        else:
                            train_split_method = Split.TreeBased
                            self.logger.warning(f"the method of train split for the feature {feature_title} was not properly specified and tree based is used.")

                        if test_ratio>0:
                            if validation_tuning_setting['test']['method'] in split_method_map:
                                test_split_method = split_method_map[validation_tuning_setting['test']['method']]
                                self.logger.info(f"for the feature {feature_title} the test cv method is {test_split_method}")
                            else:
                                test_split_method = Split.TreeBased
                                self.logger.warning(f"the method of test split for the feature {feature_title} was not properly specified and tree based is used.")
                        else:
                            test_split_method = None
                        FileUtility.ensure_dir(F"{self.output_directory}classification/cv/{prediction['prediction']}/", self.logger)
                        FileUtility.ensure_dir(F"{self.output_directory}classification/cv/{prediction['prediction']}/{validation_tuning_setting['name']}/",
                                               self.logger)
                        FileUtility.ensure_dir(
                            F"{self.output_directory}classification/cv/{prediction['prediction']}/{validation_tuning_setting['name']}/{feature_title}_{phenotype}/",
                            self.logger)

                        save_directory = F"{self.output_directory}classification/cv/{prediction['prediction']}/{validation_tuning_setting['name']}/{feature_title}_{phenotype}/"
                        tag_setting = validation_tuning_setting['name']

                        # TODO: add these two also as input to the GenYML
                        save_tree = True
                        random_state = 1
                        inner_cv = validation_tuning_setting['inner_cv']
                        self.kfold_settings[F"{validation_tuning_setting['name']}>{phenotype}"].set_tree_instances_testratio_list(self.phylogenetic_tree, instances, y=Y, test_ratio=test_ratio, random_state= random_state, save_tree=save_tree,save_directory=save_directory,tag_setting=tag_setting,test_split_method=test_split_method, train_split_method=train_split_method, overwrite=self.overwrite, logger=self.logger)

                    elif 'predefined_cv' in feature:
                        # read the tune/eval setting
                        predefined_cv = feature['predefined_cv']
                        cv_name = predefined_cv['name']
                        train_path = predefined_cv['train']
                        inner_cv = predefined_cv['inner_cv']

                        if FileUtility.exists(train_path):
                            k_fold = len(FileUtility.load_list(train_path))
                        else:
                            self.logger.error("the predefined fold does not exist")
                            exit()

                        current_cv = F"{cv_name}"
                        self.kfold_settings[current_cv] = TreeBasedKFold(k_fold)

                        if 'test' in predefined_cv:
                            test_path = predefined_cv['test']
                            if not FileUtility.exists(test_path):
                                self.logger.error("the test fold does not exist")
                                exit()
                        else:
                            test_path = None

                        FileUtility.ensure_dir(F"{self.output_directory}classification/cv/{prediction['prediction']}/",
                                               self.logger)
                        FileUtility.ensure_dir(
                            F"{self.output_directory}classification/cv/{prediction['prediction']}/{cv_name}/",
                            self.logger)
                        FileUtility.ensure_dir(
                            F"{self.output_directory}classification/cv/{prediction['prediction']}/{cv_name}/{feature_title}_{phenotype}/",
                            self.logger)

                        save_directory = F"{self.output_directory}classification/cv/{prediction['prediction']}/{cv_name}/{feature_title}_{phenotype}/"
                        tag_setting = predefined_cv['name']

                        self.kfold_settings[current_cv].load_predefined_folds(
                            self.phylogenetic_tree, instances, train_path, test_path, y=Y, save_directory=save_directory, tag_setting=tag_setting, save_tree=True, overwrite=self.overwrite, logger=self.logger)


                    if cv_name not in self.requested_results[prediction['prediction']][feature_title][phenotype]:
                        self.requested_results[prediction['prediction']][feature_title][phenotype][cv_name] = []

                    for classifier, config in tqdm.tqdm(classifiers.items()):

                        self.logger.info(
                            F" *** begin with classifier {classifier} for phenotype {phenotype} prediction of {prediction['prediction']}")
                        FileUtility.ensure_dir(F"{self.output_directory}classification/{prediction['prediction']}/", self.logger)
                        FileUtility.ensure_dir(F"{self.output_directory}feature_selection/{prediction['prediction']}/", self.logger)

                        if classifier == 'lsvm':
                            if self.overwrite or not FileUtility.exists(
                                    F"{self.output_directory}classification/{prediction['prediction']}/{feature_title}_{phenotype}_{cv_name}_{classifier}.pickle"):
                                model = SVM(X, Y, instances, feature_names=feature_names, svm_param_config_file=config['param_config'], logger=self.logger, linear=True)
                                model.tune_and_eval(F"{self.output_directory}classification/{prediction['prediction']}/{feature_title}_{phenotype}_{cv_name}_{classifier}", inner_cv, self.kfold_settings[current_cv], optimizing_score=optimized_for, n_jobs=self.number_of_cores,overwrite=self.overwrite,logger=self.logger)

                        if classifier == 'svm':
                            if self.overwrite or not FileUtility.exists(F"{self.output_directory}classification/{prediction['prediction']}/{feature_title}_{phenotype}_{cv_name}_{classifier}.pickle"):

                                model = SVM(X, Y, instances, feature_names=feature_names, svm_param_config_file=config['param_config'], logger=self.logger)
                                model.tune_and_eval(F"{self.output_directory}classification/{prediction['prediction']}/{feature_title}_{phenotype}_{cv_name}_{classifier}", inner_cv, self.kfold_settings[current_cv], optimizing_score=optimized_for, n_jobs=self.number_of_cores,overwrite=self.overwrite,logger=self.logger)

                        self.requested_results[prediction['prediction']][feature_title][phenotype][cv_name].append(classifier)

        FileUtility.save_json(F"{self.output_directory}classification/results_track.json", self.requested_results, overwrite=self.overwrite, logger=self.logger)

    def prepare_report(self):
        self.logger.info(f'begin generating reports at {self.output_directory}reports/classification/*')
        results_track_dic = FileUtility.load_json(F"{self.output_directory}classification/results_track.json")
        results_files = [F"{self.output_directory}classification/{res_instance[0]}/{'_'.join(res_instance[1::])}.pickle" for res_instance in
                         list(traverse_dictionary(results_track_dic, path=None))]
        results_rows = list(traverse_dictionary(results_track_dic, path=None))

        df_rows_table = [['prediction', 'feature', 'phenotype', 'cv', 'classifier'] + list(
            itertools.chain(*[['nested ' + metric, 'non_nested ' + metric, 'test ' + metric] for metric in self.reporting]))]
        df_rows_table_full = copy.deepcopy(df_rows_table)

        # prepare results to export excel file as well as charts

        for row_idx, row in enumerate(results_rows):

            full_res_dict = TuneEvalSteps.load_tune_eval_res(results_files[row_idx])

            row_table = copy.deepcopy(row)

            full_row = []

            for metric in self.reporting:
                nested_performance = [Scoring.functions[metric](x, y) for x, y in
                                      full_res_dict['nested_cv_true_pred_list']]
                nonnested_performance = [Scoring.functions[metric](x, y) for x, y in
                                         full_res_dict['nonnested_cv_true_pred_list']]

                test_performance = Scoring.functions[metric](full_res_dict['test_true_pred_list'][0],full_res_dict['test_true_pred_list'][1])

                row_table.append(
                    F"{np.round(np.mean(nested_performance), 2)} ± {np.round(np.std(nested_performance), 2)}")
                row_table.append(
                    F"{np.round(np.mean(nonnested_performance), 2)} ± {np.round(np.std(nonnested_performance), 2)}")
                row_table.append(F"{np.round(np.mean(test_performance), 2)}")

                if len(nested_performance) == len(nonnested_performance):
                    all_cv_rows = [list(l) for l in list(
                        zip(nested_performance, nonnested_performance, [test_performance] * len(nested_performance)))]
                    full_row.append(np.array(all_cv_rows))
            for cv_row in np.concatenate(full_row, axis=1).tolist():
                df_rows_table_full.append(row + cv_row)

            df_rows_table.append(row_table)

        df = pd.DataFrame.from_records(df_rows_table[1::], columns=df_rows_table[0])
        df.to_excel(F"{self.output_directory}reports/classification/final_results.xls", index=False)

        df_full = pd.DataFrame.from_records(df_rows_table_full[1::], columns=df_rows_table_full[0])

        classifiers = df_full.classifier.unique().tolist()
        for metric in self.reporting:
            for classifier in classifiers:
                # for each classifier make a new diagram
                df_classifier = df_full[df_full.classifier == classifier]
                phenotypes = df_classifier.phenotype.unique().tolist()
                for phentype in phenotypes:
                    fig = go.Figure()
                    df_phenotype = df_classifier[df_classifier.phenotype == phentype]
                    y = df_phenotype.feature.tolist()

                    for idx, col in enumerate(['nested ' + metric, 'non_nested ' + metric, 'test ' + metric]):
                        fig.add_trace(go.Box(
                            x=df_phenotype[col].tolist(),
                            y=y,
                            name=col,
                            marker_color=ColorUtility.five_distinct_colors[idx]
                        ))
                    fig.update_layout(
                        yaxis=dict(title=F'Features'),
                        xaxis=dict(title=F'{metric} for Phenotype: {phentype} - Classifier: {classifier}',
                                   zeroline=True, range=[0, 1]),

                        boxmode='group')

                    fig.update_traces(orientation='h')  # horizontal box plots
                    fig.write_image(F"{self.output_directory}reports/classification/{phentype}_{classifier}_{metric}.pdf")

