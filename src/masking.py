'''
Set of functions to illustrate
text sanitization based on 
provided model
'''

import pandas as pd
from . import fuzzy
from itertools import chain
import torch
from transformers import pipeline

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Import the classifier, should do it one time and then cache the model
classifier = pipeline("token-classification",model="StanfordAIMI/stanford-deidentifier-base",device=device)

# I collapse some of the Stanford AIML categories
lab_map = {'PATIENT': 'PersonName',
           'HCW': 'PersonName',
           'ID': 'IdentNumber',
           'SSN': 'IdentNumber',
           'PHONE': 'Contact',
           'FAX': 'Contact',
           'EMAIL': 'Contact',
           'DATE': 'Date',
           'GEO': 'Geo',
           'HOSPITAL': 'Geo',
           'WEB': 'Web'}

lab_keep = list(lab_map.keys())
fin_tokens = list(set(lab_map.values()))
fin_dict = {k: [] for k in fin_tokens}

# Doing exact matches for now
def ord_unique(series):
    un = pd.unique(series)
    rep = {u:str(i+1) for i,u in enumerate(un)}
    return series.replace(rep)


# This is to make the tokenized entities in a nicer wide format
def token_wide(text, lab_map=lab_map, lab_keep=lab_keep, fin_dict=fin_dict, classifier=classifier, thresh=0.9):
    res = classifier(text,aggregation_strategy="simple")  # could also use max or average
    fd2 = fin_dict.copy()
    if len(res) > 0:
        res_pd = pd.DataFrame(res)
        res_pd = res_pd[res_pd['entity_group'].isin(lab_keep)].copy()
        res_pd = res_pd[res_pd['score'] >= thresh].copy()
        res_pd['entity_group'] = res_pd['entity_group'].replace(lab_map)
        ent = pd.unique(res_pd['entity_group'])
        for e in ent:
            lpd = res_pd[res_pd['entity_group'] == e].reset_index(drop=True)
            ordval = ord_unique(lpd['word'])
            lpd['entity_group'] = lpd['entity_group'] + ordval
            fd2[e] = lpd.to_dict(orient='records')
    return fd2

# This takes input dataframe and returns entities wide format
def ner_data(data,text_field,lab_map=lab_map, lab_keep=lab_keep, fin_dict=fin_dict):
    res = []
    for txt in data[text_field]:
        res.append(token_wide(txt,lab_map,lab_keep,fin_dict))
    token_text = pd.DataFrame(res,index=data.index)
    return token_text

# Specialized function for applying to text in dataframe
# assumes first column is text, and the rest are dictionaries
def mask_input(li):
    res = li[0]
    dis = li[1:]
    # getting all of the begin/end parts in one go
    be = {}
    for d in dis:
        for mask in d:
            be[(mask['start'],mask['end'])] = mask['entity_group']
    be = dict(sorted(be.items(), reverse=True))
    txt = res
    for slice,rep in be.items():
        txt = txt[0:slice[0]] + rep + txt[slice[1]:]
    return txt

# This uses the fuzzy logic to replace
# matched strings with similar matches
def rep_match(data,field):
    # if empty, don't do anything
    fl = data[field].apply(len)
    if fl.max() == 0:
        pass
    else:
        res = data[field].values
        res_df = pd.DataFrame(chain(*res))
        res_map = fuzzy.res_map(res_df,'word',res_map=True)
        # Now replacing the initial words
        for row in res:
            for m in row:
                m['entity_group'] = field + res_map[m['word']]

# Final function, takes in input dataframe and does the masking
# plus returns the entities
def mask_dataframe(data,
                   text_field,
                   mask_fields=['Contact', 'Geo', 'IdentNumber', 'PersonName'],
                   lab_map=lab_map,
                   classifier=classifier):
    lab_keep = list(lab_map.keys())
    fin_tokens = list(set(lab_map.values()))
    fin_dict = {k: [] for k in fin_tokens}
    rt = ner_data(data,text_field,lab_map,lab_keep,fin_dict)
    # doing the fuzzy matching
    for v in mask_fields:
        rep_match(rt,v)
    # now doing the name masking
    rt[text_field] = data[text_field]
    rt = rt[[text_field] + mask_fields]
    rt[text_field] = rt.apply(mask_input,axis=1)
    return rt

##############################################
## Illustrative single dataset to check
#
#t1 = "Andy Wheeler is a birder 190682540 where I live 100 Main St Kansas with Joe Schmo and andy wheeler"
#t2 = "Scott Jacques is an interesting fellow, his check number 18887623597 is a good one."
#t3 = "lol, what a noob, Atlanta GA is on fire"
#t4 = "so what, andrew wheeler @ 100 main st kansas is not so bad"
#text_li = [t1,t2,t3,t4]
#id = [1,2,3,4]
#
#test_df = pd.DataFrame(zip(id,text_li),columns=['ID','Text'], index=['a','b','c','e'])
#mask_dataframe(test_df,'Text')
##############################################