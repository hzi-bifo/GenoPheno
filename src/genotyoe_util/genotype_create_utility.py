import os

GENO2PHENO_DIR = os.path.dirname(__file__) + "/../"
import sys

sys.path.append(GENO2PHENO_DIR)
import codecs
import numpy as np
from scipy import sparse
from sklearn.preprocessing import MaxAbsScaler
from utility.file_utility import FileUtility
import sklearn
from sklearn.preprocessing import normalize
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
from enum import Enum
import pandas as pd
import tqdm
from scipy import sparse
from multiprocessing import Pool

class Preprocessing(Enum):
    NONE = 0
    BINARY = 1
    ZERO2ONE = 2
    PERCENT = 3
    STANDARDIZED = 4
    L1 = 5
    L2 = 6


class DataType(Enum):
    NUMERICAL = 0
    TEXT = 1


class GenotypeVectorizer(object):
    '''
        Class for vectorization of the genotypes
    '''

    def __init__(self):
        print('Genotype reader object created..')

    @staticmethod
    def create_genotype_table(save_path, load_path, feature_title='_', datatype=DataType.TEXT,
                              preprocessing=Preprocessing.NONE, delimiter=None, overwrite=False, logger=None):

        if overwrite or (not os.path.exists(F"{save_path}{feature_title}_feature_data.npz") or not os.path.exists(
                F"{save_path}{feature_title}_feature_name.txt") or not os.path.exists(
                F"{save_path}{feature_title}_instances.txt")):

            if datatype == DataType.TEXT:
                feature_lines = [' '.join(l.split()[1::]) if not delimiter else ' '.join(l.split(delimiter)[1::]) for l in FileUtility.load_list(load_path)]
                instances = [l.split()[0] if not delimiter else l.split(delimiter)[0] for l in FileUtility.load_list(load_path)]

                vectorizer = TfidfVectorizer(use_idf=False, analyzer='word', ngram_range=(1, 1),
                                             norm='l1' if preprocessing == Preprocessing.L1 else 'l2' if preprocessing == Preprocessing.L2 else None,
                                             stop_words=[], tokenizer=str.split, lowercase=True,
                                             binary=True if preprocessing == Preprocessing.BINARY else False)
                matrix_representation = vectorizer.fit_transform(feature_lines)
                feature_names = vectorizer.get_feature_names()

                FileUtility.save_obj(F"{save_path}{feature_title}_feature_vectorizer", vectorizer)

            if datatype == DataType.NUMERICAL:

                df = pd.read_table(load_path, delimiter='\s+' if not delimiter else delimiter)

                feature_names = df.columns.tolist()
                instances = df.index.tolist()
                matrix_representation = sparse.csr_matrix(df[feature_names].values)

                # TODO: Needs to be fixed
                if not matrix_representation.shape[0]==len(set(instances)):
                    feature_names = df.columns.tolist()[1::]
                    instances = df[df.columns.tolist()[0]].tolist()
                    matrix_representation = sparse.csr_matrix(df[feature_names].values)

                if preprocessing == Preprocessing.BINARY:
                    matrix_representation = np.round(MaxAbsScaler().fit_transform(matrix_representation))
                elif preprocessing == Preprocessing.L1:
                    matrix_representation = normalize(matrix_representation, norm='l1')
                elif preprocessing == Preprocessing.L2:
                    matrix_representation = normalize(matrix_representation, norm='l2')

            if preprocessing == Preprocessing.ZERO2ONE:
                matrix_representation = MaxAbsScaler().fit_transform(matrix_representation)
            elif preprocessing == Preprocessing.PERCENT:
                matrix_representation = np.round(MaxAbsScaler().fit_transform(matrix_representation) * 100)
            elif preprocessing == Preprocessing.STANDARDIZED:
                matrix_representation = sparse.csr_matrix(
                    sklearn.preprocessing.StandardScaler().fit_transform(matrix_representation.toarray()))

            FileUtility.save_sparse_csr(F"{save_path}{feature_title}_feature_data.npz", matrix_representation)
            FileUtility.save_list(F"{save_path}{feature_title}_feature_names.txt", feature_names)
            FileUtility.save_list(F"{save_path}{feature_title}_instances.txt", instances)

            if logger:
                logger.info(F"The files for {feature_title} created: {save_path}{feature_title}_*")

        elif logger:
            logger.info(F"The feature table {feature_title} already exist: {save_path}{feature_title}_*")

    @staticmethod
    def create_seq_kmer_table(save_path, load_path, feature_title='_', kvalue=6, n_cores=4, overwrite=False, logger=None):

        if overwrite or (not os.path.exists(F"{save_path}{feature_title}_feature_data.npz") or not os.path.exists(
                F"{save_path}{feature_title}_feature_name.txt") or not os.path.exists(
                F"{save_path}{feature_title}_instances.txt")):

            # prepare files for multicore sequence k-mer generation
            files_in_assembly_directory = FileUtility.recursive_glob(load_path, '*')
            files_in_assembly_directory.sort()
            input_tuples = []
            for file_path in files_in_assembly_directory:
                instance = os.path.splitext(os.path.basename(file_path))[0]
                input_tuples.append((instance, file_path, kvalue))

            instances = []
            mat = []
            kmers = []
            pool = Pool(processes=n_cores)
            for instance, vec, vocab in tqdm.tqdm(pool.imap_unordered(GenotypeVectorizer._get_kmer_rep, input_tuples, chunksize=n_cores),
                                                total=len(input_tuples)):
                instances.append(instance)
                mat.append(vec)
                kmers = vocab
            pool.close()
            mat = sparse.csr_matrix(mat)

            FileUtility.save_sparse_csr(F"{save_path}{feature_title}_feature_data.npz", mat)
            FileUtility.save_list(F"{save_path}{feature_title}_feature_names.txt", kmers)
            FileUtility.save_list(F"{save_path}{feature_title}_instances.txt", instances)

            if logger:
                logger.info(F"The files for {feature_title} created: {save_path}{feature_title}_*")

        elif logger:
            logger.info(F"The feature table {feature_title} already exist: {save_path}{feature_title}_*")

    @staticmethod
    def _get_kmer_rep(input_tuple):
        strain, seq_file, k=input_tuple
        seq=FileUtility.read_fasta_sequences(seq_file)
        vec, vocab = GenotypeVectorizer.get_nuc_kmer_distribution(seq, k)
        return strain, vec, vocab


    @staticmethod
    def get_nuc_kmer_distribution(sequences, k):
        vocab = [''.join(xs) for xs in itertools.product('atcg', repeat=k)]
        vocab.sort()
        vectorizer = TfidfVectorizer(use_idf=False, vocabulary=vocab, analyzer='char', ngram_range=(k, k),
                                     norm='l1', stop_words=[], lowercase=True, binary=False)
        return (np.mean(vectorizer.fit_transform([seq.lower() for seq in sequences]).toarray(), axis=0), vocab)


