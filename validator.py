from cerberus import Validator
import yaml
import newick
import codecs
from Bio import Phylo
from io import StringIO
import pandas as pd
import itertools
import os
import copy

class ValidateGenML(object):
    '''
        Developed by Ehsaneddin Asgari
    
    
    '''
    
    def __init__(self, config_file):
        self.genotype_tables = dict()
        self.sequence_dir = dict()        
        self.__load_schema()
        self.config = self.__load_config(config_file)
        self.v = Validator()
        print (self.v.validate(self.config, self.schema_general))
        print (self.v.errors)
        
    @staticmethod   
    def exists(file_path):
        return os.path.exists(file_path)        
        
    def check_tree(self, field, value, error):
        self.tree_path = value.replace('%(PROJDIR)%','/net/sgi/metagenomics/nobackup/prot/projects/verify_genopheno/kp_res/KP_results/metadata')        
        try:
            if not ValidateGenML.exists(self.tree_path):
                error(field, "The phylogenetic tree file does not exist!")
        except:    
            try:
                self.newick = Phylo.read(codecs.open(self.tree_path,'rb','utf-8'), "newick")
                self.nodes_in_tree = [n.name for n in self.newick.get_terminals()]
            except:
                error(field, "The phylogenetic tree does not follow standard newick format!")
        return True

    def check_phenotype(self, field, value, error):
        self.phenotype_path = '/net/sgi/metagenomics/nobackup/prot/projects/verify_genopheno/kp_res/phenotype_list'
        try:
            if not ValidateGenML.exists(self.phenotype_path):
                error(field, "The phenotype file does not exist!")
        except:            
            try:
                self.phenotype_table = pd.read_table(self.phenotype_path, delimiter='\s+',index_col=0)
                self.phenotypes_list = self.phenotype_table.columns.tolist()

                if self.phenotype_table.index.name in self.phenotypes_list:
                    self.phenotypes_list.remove(self.phenotype_table.index.name)

                self.nodes_in_phenotype = self.phenotype_table.index.tolist()
                temp = self.phenotype_table[self.phenotypes_list]
                self.phentype_unique_values = list(set(itertools.chain(*temp.values.tolist())))

                if len(set(self.nodes_in_phenotype).intersection(self.nodes_in_tree)) ==0:
                        error(field, "No overlap between phenotype instances and the instances in the tree")
                else:
                    print(F"{len(set(self.nodes_in_phenotype).intersection(self.nodes_in_tree))} instances in common between the tree (#{len(self.nodes_in_tree)})  and phenotype table (#{len(self.nodes_in_phenotype)})")
            except:
                error(field, "The phenotype table does not follow the correct format")
        return True


    def check_tables(self, field, value, error):
        vtemp = Validator()
        
        for idx, table in enumerate(value):
            if 'table' in table:
                res = vtemp.validate(table, self.schema_table)
                if vtemp.errors:
                    error(field, F"in Table {idx+1} {str(vtemp.errors)}")
                else:
                    if not ValidateGenML.exists(table['path']):
                        error(field, F"The path {table['path']} does not exist!")
                    else:
                        self.genotype_tables[table['name']] = table['path']

                    
            if 'sequence' in table:     
                res = vtemp.validate(table, self.schema_sequences)
                if vtemp.errors:
                    error(field, F"in Table {idx+1} {str(vtemp.errors)}")
                else:
                    if not ValidateGenML.exists(table['path']):
                        error(field, F"The directory {table['path']} does not exist!")
                    else:
                        self.sequence_dir[table['name']] = table['path']
        return True
                      
    def check_predictions(self, field, value, error):
        return True

    def __load_config(self, config_file):
        with open(config_file, 'r') as stream:
            try:
                return yaml.load(stream)
            except yaml.YAMLError as exception:
                raise exception
                
    def __load_schema(self):
        
        self.schema_meta = {
                'project': {
                    'required': True,
                    'type': 'string'
                },
                'phylogenetic_tree': {
                    'required': True,
                    'type': 'string',
                    'validator': self.check_tree
                },
                'phenotype_table': {
                    'required': True,
                    'type': 'string',
                    'validator': self.check_phenotype
                },
                'output_directory': {
                    'required': True,
                    'type': 'string'
                },
                'number_of_cores': {
                    'required': True,
                    'type': 'number',
                    'min': 1,
                    'max': 100
                }
            }

        self.schema_prediction = {
            'prediction': {
                        'required': True,
                        'type': 'dict',
                        'schema': {
                                'name': 
                                       {
                                            'required': True,
                                            'type': 'string'  
                                        },                           
                                'label_mapping': 
                                       {
                                            'required': True,                                        
                                            'type': 'dict'  
                                        },
                                'optimized_for': 
                                       {
                                            'required': True,
                                            'minlength': 1,
                                            'type': 'string'  
                                        },
                                'reporting': 
                                       {
                                            'required': True,
                                            'minlength': 1,
                                            'type': 'dict'  
                                        },
                                'features': 
                                       {
                                            'required': True,
                                            'minlength': 1,
                                            'type': 'dict'  
                                        },
                                'classifiers': 
                                       {
                                            'required': True,
                                            'minlength': 1,
                                            'type': 'list'  
                                        }                       
                        }
                    }
            }


        self.schema_general = {
                'metadata': {
                    'required': True,
                    'type': 'dict',
                    'schema': self.schema_meta
                },
                'genotype_tables': {
                    'required': True,
                    'type': 'dict',
                    'schema': {
                        'tables': {
                                    'required': True,
                                    'minlength': 1,
                                    'type': 'list',
                                    'validator': self.check_tables,
                                }
                              }
                },
                'predictions': {
                    'required': True,
                    'type': 'list',
                    'validator': self.check_predictions
                }
            }         
        self.schema_table = {'table': {
                            'required': False,
                            'type': 'dict',
                            'schema': {
                                        'name': {
                                            'required': True,
                                            'type': 'string',
                                            },
                                        'path': {
                                            'required': True,
                                            'type': 'string',
                                            },
                                         'preprocessing': {
                                            'required': True,
                                            'type': 'string',
                                            'allowed': ['l1','l2','percent','zero2one','none','std','binary']
                                            },  
                                         'delimiter': {
                                            'required': True,
                                            'type': 'string',
                                            }, 
                                         'datatype': {
                                            'required': False,
                                            'type': 'string',
                                            'allowed': ['numerical','text']
                                            } 
                                       }
                            }}
        self.schema_sequences = {'sequence': {
                            'required': False,
                            'type': 'dict',
                            'schema': {
                                        'name': {
                                            'required': True,
                                            'type': 'string',
                                            },
                                        'path': {
                                            'required': True,
                                            'type': 'string',
                                            },
                                         'preprocessing': {
                                            'required': True,
                                            'type': 'string',
                                            'allowed': ['l1','l2','percent','zero2one','none','std','binary']
                                            },  
                                         'k_value': {
                                            'required': True,
                                            'type': 'number',
                                            'allowed': [1,2,3,4,5,6,7,8]
                                            }, 
                                       }
                            }}        

