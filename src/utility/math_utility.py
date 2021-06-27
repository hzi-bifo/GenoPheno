__author__ = "Ehsaneddin Asgari"
__license__ = "To be added -- for now all rights are reserved for the author"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "ehsan.asgari@gmail.com"
__source__ = "from personal repo"

from scipy import stats
from sklearn.preprocessing import normalize
import numpy as np

from scipy.special import rel_entr

def entropy_custom(pk, qk=None, base=None, axis=0):
    """custom version of entropy without shape requirements"""
    pk = np.asarray(pk)
    pk = 1.0*pk / np.sum(pk, axis=axis, keepdims=True)
    if qk is None:
        vec = stats.entropy(pk)
    else:
        qk = np.asarray(qk)
        # if qk.shape != pk.shape:
        #     raise ValueError("qk and pk must have same shape.")
        qk = 1.0*qk / np.sum(qk, axis=axis, keepdims=True)
        vec = rel_entr(pk, qk)
    S = np.sum(vec, axis=axis)
    if base is not None:
        S /= np.log(base)
    return S

def get_kl_rows(A):
    '''
    :param A: matrix A
    :return: Efficient implementation to calculate kl-divergence between rows in A
    '''
    norm_A = normalize(A + 1e-100, norm='l1')
    return entropy_custom(norm_A.T[:,:,None], norm_A.T[:,None,:])

def generate_binary(N):
    '''
    generate the binary representation of input N
    :param N:
    :return:
    '''
    return [''.join(list(b)[::-1]) for b in  ['{:0{}b}'.format(i, N) for i in range(2**N)][1::]]

def get_sym_kl_rows(A):
    '''
    :param A: matrix A
    :return: Efficient implementation to calculate kl-divergence between rows in A
    '''
    norm_A=normalize(A+np.finfo(np.float64).eps, norm='l1')
    a=entropy_custom(norm_A.T[:,:,None], norm_A.T[:,None,:])
    return a+a.T

def get_kl_rows(A):
    '''
    :param A: matrix A
    :return: Efficient implementation to calculate kl-divergence between rows in A
    '''
    norm_A=normalize(A+np.finfo(np.float64).eps, norm='l1')
    return entropy_custom(norm_A.T[:,:,None], norm_A.T[:,None,:])

def normalize_mat(A,norm ='l1', axis=1):
    '''

    :param A:
    :param norm:
    :param axis: 0 colum
    :return:
    '''

    return normalize(A, norm=norm, axis=axis)


def seriation(Z,N,cur_index):
    '''
        got the code from: https://gmarti.gitlab.io/ml/2017/09/07/how-to-sort-distance-matrix.html
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z

        seriation computes the order implied by a hierarchical tree (dendrogram)
    '''
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index-N,0])
        right = int(Z[cur_index-N,1])
        return (seriation(Z,N,left) + seriation(Z,N,right))

def compute_serial_matrix(dist_mat,method="ward"):
    '''
        got the code from: https://gmarti.gitlab.io/ml/2017/09/07/how-to-sort-distance-matrix.html
        input:
            - dist_mat is a distance matrix
            - method = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)

        compute_serial_matrix transforms a distance matrix into
        a sorted distance matrix according to the order implied
        by the hierarchical tree (dendrogram)
    '''
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method,preserve_input=True)
    res_order = seriation(res_linkage, N, N + N-2)
    seriated_dist = np.zeros((N,N))
    a,b = np.triu_indices(N,k=1)
    seriated_dist[a,b] = dist_mat[ [res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b,a] = seriated_dist[a,b]

    return seriated_dist, res_order, res_linkage

def get_borders(mylist):
    val=0
    borders=[]
    for i,v in enumerate(mylist):
        if i==0:
            val=v
        else:
            if not v==val:
                borders.append(i)
                val=v
    return borders

def jensen_shannon(query, matrix):
    """
    This function implements a Jensen-Shannon similarity
    between the input query (an LDA topic distribution for a document)
    and the entire corpus of topic distributions.
    It returns an array of length M where M is the number of documents in the corpus
    """
    # lets keep with the p,q notation above
    p = query[None,:].T + np.zeros(matrix.T.shape()) # take transpose
    q = matrix.T # transpose matrix
    m = 0.5*(p + q)
    return np.sqrt(0.5*(stats.entropy(p,m) + stats.entropy(q,m)))
