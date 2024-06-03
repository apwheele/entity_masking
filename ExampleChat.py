'''
Example of fuzzing the Decker/Wright files
'''


import os
import pandas as pd
from src.masking import mask_dataframe

data = pd.read_csv('./data/chat.zip')
data['msg'] = data['msg'].fillna('')
fuzz_file = './data/ChatFuzzed.pkl'

if os.path.exists(fuzz_file):
    res = pd.read_pickle(fuzz_file)
else:
    res = mask_dataframe(data,'msg') # pass in dataframe, and the field that has the text
    res['OrigText'] = data['msg']
    res['_id'] = data['_id']
    res.to_pickle(fuzz_file)

# Lets identify
print(res.head())

#rese = res.explode('PersonName')
#rese = rese[~rese['PersonName'].isna()].copy()



# sample 1500 no IDs, assign A=500, S=500, B=500
id_flag = res['PersonName'].apply(len) > 0
no_id = res[~id_flag].sample(1500,replace=False,random_state=10)
no_id['type'] = ['A','S','B']*500
no_id['Name'] = ''
no_id['Contact'] = ''
no_id['Geo'] = ''
no_id['IdentNumber'] = ''
no_id.reset_index(inplace=True)

var_keep = ['index','OrigText','Name','Contact','Geo','IdentNumber']

scott = no_id[no_id['type'].isin(['S','B'])]
scott = scott[var_keep].copy()
scott = scott.sample(1000,replace=False,random_state=10) # another shuffle
scott.to_csv('./data/ScottCheckFalseNegatives.csv',index=False)

andy = no_id[no_id['type'].isin(['A','B'])]
andy = andy[var_keep].copy()
andy = andy.sample(1000,replace=False,random_state=10) # another shuffle
andy.to_csv('./data/AndyCheckFalseNegatives.csv',index=False)


# sample 1500 with name IDs
one_id = res[id_flag].sample(1500)
one_id['type'] = ['A','S','B']*500
mult_id = one_id.explode('PersonName')
mult_id['index'] = list(range(mult_id.shape[0]))
mult_id['False'] = ''

var_keep = ['index','msg','OrigText','PersonName','False']

scott = mult_id[mult_id['type'].isin(['S','B'])]
scott = scott[var_keep].copy()
scott = scott.sample(scott.shape[0],replace=False,random_state=10) # another shuffle
scott.to_csv('./data/ScottCheckFalsePositives.csv',index=False)

andy = mult_id[mult_id['type'].isin(['A','B'])]
andy = andy[var_keep].copy()
andy = andy.sample(andy.shape[0],replace=False,random_state=10) # another shuffle
andy.to_csv('./data/AndyCheckFalsePositives.csv.csv',index=False)



