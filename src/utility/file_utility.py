__author__ = "Ehsaneddin Asgari"
__license__ = "To be added -- for now all rights are reserved for the author"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "ehsan.asgari@gmail.com"
__source__ = "from personal repo"

import sys
sys.path.append('../')
import _pickle as pickle
import codecs
import fnmatch
import os
import numpy as np
from Bio import SeqIO
from Bio.Alphabet import generic_dna
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from scipy import sparse
import yaml
import shutil
import json

class FileUtility(object):

    def __init__(self):
        pass

    @staticmethod
    def create_fasta_file(file_address, corpus, label):
        seq_id_pairs = [('.'.join([str(idx + 1), label[idx]]), x) for idx, x in enumerate(corpus)]
        seq_recs = [SeqRecord(Seq(seq, generic_dna), id=id, description='') for id, seq in seq_id_pairs]
        SeqIO.write(seq_recs, file_address, "fasta")


    @staticmethod
    def read_sequence_file(file_name_sample):
        '''
        :param file_name_sample:
        :return:
        '''
        corpus = []
        if file_name_sample[-1] == 'q':
            for cur_record in SeqIO.parse(file_name_sample, "fastq"):
                corpus.append(str(cur_record.seq).lower())
        else:
            for cur_record in SeqIO.parse(file_name_sample, "fasta"):
                corpus.append(str(cur_record.seq).lower())
        return file_name_sample.split('/')[-1], corpus

    @staticmethod
    def read_sequence_file_length(file_name_sample):
        '''
        :param file_name_sample:
        :return:
        '''
        corpus = []
        if file_name_sample[-1] == 'q':
            for cur_record in SeqIO.parse(file_name_sample, "fastq"):
                corpus.append(str(cur_record.seq).lower())
        else:
            for cur_record in SeqIO.parse(file_name_sample, "fasta"):
                corpus.append(str(cur_record.seq).lower())
        return file_name_sample.split('/')[-1], len(corpus)

    @staticmethod
    def read_fasta_directory(file_directory, file_extenstion, only_files=[]):
        '''
        :param file_directory:
        :param file_extenstion:
        :param only_files:
        :return: list of fasta files, and a dic to map file to index
        '''
        if len(only_files) > 0:
            fasta_files = [x for x in FileUtility.recursive_glob(file_directory, '*.' + file_extenstion) if
                           x.split('/')[-1] in only_files]
        else:
            fasta_files = [x for x in FileUtility.recursive_glob(file_directory, '*.' + file_extenstion)]

        fasta_files.sort()
        mapping = {v: k for k, v in enumerate(fasta_files)}
        return fasta_files, mapping

    @staticmethod
    def save_obj(filename, value, overwrite=True, logger=None):
        if not FileUtility.exists(filename) or overwrite:
            with open(filename + '.pickle', 'wb') as f:
                pickle.dump(value, f)
            if logger:
                logger.info(F"file created: {filename} ")
        elif logger:
            logger.info(F"file existed and remained unchanged: {filename}")

    @staticmethod
    def load_obj(filename, logger=None):
        if not FileUtility.exists(filename):
            logger.info(F"file does not exist: {filename}")
            return None
        else:
            if logger:
                logger.info(F"file is loading: {filename} ")
            return pickle.load(open(filename, "rb"))


    @staticmethod
    def load_json(filename, logger=None):
        if not FileUtility.exists(filename):
            logger.info(F"json file does not exist: {filename}")
            return None
        else:
            if logger:
                logger.info(F"json file is loading..: {filename} ")
            with open(filename, 'r') as f:
                return json.load(f)

    @staticmethod
    def save_list(filename, list_names, overwrite=True, logger=None):
        if not FileUtility.exists(filename) or overwrite:
            f = codecs.open(filename, 'w', 'utf-8')
            for x in list_names:
                f.write(x + '\n')
            f.close()
            if logger:
                logger.info(F"file created: {filename} ")
        elif logger:
            logger.info(F"file existed and remained unchanged: {filename}")

    @staticmethod
    def save_json(filename, dictionary, overwrite=True, logger=None):
        if not FileUtility.exists(filename) or overwrite:
            with open(filename, 'w') as fp:
                json.dump(dictionary, fp, sort_keys=True, indent=4)
            if logger:
                logger.info(F"file created: {filename} ")
        elif logger:
            logger.info(F"file existed and remained unchanged: {filename}")


    @staticmethod
    def load_list(filename):
        return [line.rstrip() for line in codecs.open(filename, 'r', 'utf-8').readlines()]

    @staticmethod
    def save_sparse_csr(filename, array):
        np.savez(filename, data=array.data, indices=array.indices,
                 indptr=array.indptr, shape=array.shape)

    @staticmethod
    def load_sparse_csr(filename):
        loader = np.load(filename)
        return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

    @staticmethod
    def _float_or_zero(value):
        try:
            return float(value)
        except:
            return 0.0

    @staticmethod
    def recursive_glob(treeroot, pattern):
        '''
        :param treeroot: the path to the directory
        :param pattern:  the pattern of files
        :return:
        '''
        results = []
        for base, dirs, files in os.walk(treeroot):
            good_files = fnmatch.filter(files, pattern)
            results.extend(os.path.join(base, f) for f in good_files)
        return results

    @staticmethod
    def read_fasta_sequences(file_name):
        corpus=[]
        for cur_record in SeqIO.parse(file_name, "fasta"):
                corpus.append(str(cur_record.seq).lower())
        return corpus

    @staticmethod
    def read_fasta_sequences_ids(file_name):
        corpus=dict()
        for cur_record in SeqIO.parse(file_name, "fasta"):
                corpus[str(cur_record.id)]=(str(cur_record.seq).lower(),str(cur_record.description))
        return corpus


    @staticmethod
    def ensure_dir(file_path, logger = None):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            if logger:
                logger.info(F"directory has been created: {file_path} ")
            os.makedirs(directory)
        elif logger:
            logger.info(F"directory already existed: {file_path}")

    @staticmethod
    def exists(file_path):
        return os.path.exists(file_path)

    @staticmethod
    def remove(file_path):
        if FileUtility.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def load_yaml(file_path, replace_list=[]):
        file = open(file_path).read()
        for x,y in replace_list:
            file = str(file).replace(x,y)
        return yaml.safe_load(file)

    @staticmethod
    def force_remove_dir(dir_path):
        try:
            shutil.rmtree(dir_path, ignore_errors=True)
            return True
        except:
            return False