'''
These are functions to do fuzzy matching
'''

import pandas as pd
from polyleven import levenshtein
import networkx as nx
import numpy as np

# Normalized levenshtein
def norm_leven(x):
    a, b = x.iloc[0], x.iloc[1]
    ldist = levenshtein(a,b)
    # This is to prevent matching very tiny strings
    # min length is 4 characters
    la, lb = max(len(a),4), max(len(b),4)
    min_diff = np.abs(la - lb)
    max_diff = max(la,lb)
    return (ldist - min_diff)/(max_diff - min_diff)


# Uses connected components to reduce ids
def conn_comp(nodes,edges):
    G = nx.Graph() 
    G.add_nodes_from(nodes)
    G.add_edges_from(edges.values)
    cc = nx.connected_components(G)
    rep_di = {}
    for i,c in enumerate(cc):
        for sub in c:
            rep_di[sub] = str(i+1)
    return rep_di

# Function to create pairs
def res_map(data,field,thresh=0.2,func=norm_leven,res_map=False):
    fin_pairs = []
    d2 = pd.unique(data[field])
    d2.sort()
    n = len(d2)
    tot_pairs = 0
    for i in range(n-1):
        dl = d2[i+1:]
        dfl = pd.DataFrame(dl,columns=['a'])
        dfl['b'] = d2[i]
        test = dfl.apply(func,axis=1) <= thresh
        if test.sum() > 0:
            tot_pairs += 1
            fin_pairs.append(dfl[test].copy())
    if tot_pairs > 0:
        edges = pd.concat(fin_pairs,axis=0,ignore_index=True)
        nodes = d2.tolist()
        res_map = conn_comp(nodes,edges)
    else:
        res_map = {n:str(i+1) for i,n in enumerate(d2)}
    if res_map:
        return res_map
    else:
        return data[field].replace(res_map)


#import duckdb
#
#d2 = data[[field]].copy()
#d2['n'] = 1
#d2.columns = ['field','n']
#d2 = d2.groupby(field,as_index=False)['n'].size()
#duckdb.sql("CREATE TABLE d AS SELECT * FROM d2")
#
#match_query = '''
#SELECT
# l.field AS lf,
# r.field as rf,
# damerau_levenshtein(l.field,r.field) AS dl
#FROM d AS l
#CROSS JOIN d AS r
#WHERE l.field < r.field
#  AND damerau_levenshtein(l.field,r.field) < 3
#'''
#
# by the time you get to 3, it is pretty 
#res_pairs = duckdb.sql(match_query).df()