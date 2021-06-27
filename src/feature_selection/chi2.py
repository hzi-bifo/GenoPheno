import pandas as pd
import codecs
import math
import operator
import numpy as np
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import chi2
from scipy.sparse import csr_matrix

class Chi2Analysis(object):
    # X^2 is statistically significant at the p-value level
    def __init__(self, X, Y, feature_names):
        '''
        :param X: data matrix
        :param Y: 1 or 0
        :param feature_names: list of X columns
        '''
        self.X = X
        self.Y = Y
        self.feature_names = feature_names


    def extract_features_fdr(self, file_name, N=-1, alpha=5e-2, direction=True, allow_subseq=True, binarization=True):
        '''
        :param file_name: output Chi2
        :param N: Top-N significant features
        :param alpha: the min p-value
        :param direction: if true the score would have sign
        :param allow_subseq:
        :param binarization: if the data is not binary ==> 'binary','median'
        :return:
        '''
        '''
            Feature extraction with fdr-correction
        '''
        # https://brainder.org/2011/09/05/fdr-corrected-fdr-adjusted-p-values/
        # Filter: Select the p-values for an estimated false discovery rate
        # This uses the Benjamini-Hochberg procedure. alpha is an upper bound on the expected false discovery rate.
        X = self.X
        selector = SelectFdr(sklearn.feature_selection.chi2, alpha=alpha)
        selector.fit(self.X, self.Y)
        scores = {self.feature_names[i]: (s, selector.pvalues_[i]) for i, s in enumerate(list(selector.scores_)) if
                  not math.isnan(s)}
        if N==-1:
            scores = sorted(scores.items(), key=operator.itemgetter([1][0]), reverse=True)
        else:
            scores = sorted(scores.items(), key=operator.itemgetter([1][0]), reverse=True)[0:N]

        f = codecs.open(file_name, 'w')
        c_1 = np.sum(self.Y)
        c_0 =np.sum([1 for x in self.Y if x==0])
        f.write('\t'.join(['feature', 'score', 'p-value', 'mean+-pos', 'mean+-neg']) + '\n')
        X = X.toarray()
        pos_scores = []

        extracted_features=[]
        for w, score in scores:
            feature_array = X[:, self.feature_names.index(w)]
            pos = [feature_array[idx] for idx, x in enumerate(self.Y) if x == 1]
            neg = [feature_array[idx] for idx, x in enumerate(self.Y) if x == 0]
            m_pos=np.mean(pos)
            s_pos=np.std(pos)
            m_neg=np.mean(neg)
            s_neg=np.std(neg)

            c11 = np.sum(pos)
            c01 = c_1 - c11
            c10 = np.sum(neg)
            c00 = c_0 - c10
            s=score[0]
            if direction and c11 > ((1.0 * c11) * c00 - (c10 * 1.0) * c01):
                s=-s
            s=np.round(s,2)

            if allow_subseq:
                pos_scores.append([str(w), s, score[1], m_pos, m_neg])
                #if m_pos> m_neg:
                f.write('\t'.join([str(w), str(s), str(score[1])] + [str(x) for x in [m_pos, s_pos, m_neg, s_neg]]) + '\n')
            else:
                flag=False
                for feature in extracted_features:
                    if w in feature:
                        flag=True
                if not flag:
                    pos_scores.append([str(w), s, score[1], m_pos, m_neg])
                    f.write('\t'.join([str(w), str(s), str(score[1])] + [str(x) for x in [m_pos, s_pos, m_neg, s_neg]]) + '\n')

        f.close()
        return pos_scores
