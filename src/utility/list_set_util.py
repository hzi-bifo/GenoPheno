__author__ = "Ehsaneddin Asgari"
__license__ = "To be added -- for now all rights are reserved for the author"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "ehsan.asgari@gmail.com"
__source__ = "from personal repo"

import operator

def get_intersection_of_list(list_of_list_features):
    return list(set.intersection(*map(set, list_of_list_features)))

def get_max_of_dict(inp):
    return max(inp.items(), key=operator.itemgetter(1))[0]

def argsort(seq, rev=False):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__, reverse=rev)

def traverse_dictionary(dic, path=None):
    if not path:
        path=[]
    if isinstance(dic,dict):
        for x in dic.keys():
            local_path = path[:]
            local_path.append(x)
            for b in traverse_dictionary(dic[x], local_path):
                 yield b
    else:
        for x in dic:
            yield path + [x]