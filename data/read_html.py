'''
Reading in webpage data
returning ?dataframes?
with masked information

run from root directory
'''

from bs4 import BeautifulSoup
import pandas as pd
from src.masking import mask_dataframe


files = ['./data/OTplugExclusiveDistro/messages.html',
         './data/Gunstore71_tactical/messages.html',
         './data/F R E S H G R U B S/messages.html']

def mask_html(file,classes=["text","from_name","initials","details"]):
    # open with UTF
    with open(file,'r',encoding='utf8') as fi:
       fi = fi.read()
    
    soup = BeautifulSoup(fi,'html.parser')
    all_items = soup.find_all("div", class_=classes)
    res = []
    for i,a in enumerate(all_items):
        res.append([i,a,a['class'],a.text])
    
    res_df = pd.DataFrame(res,columns=['ID','item','classes','text'])
    mask_df = mask_dataframe(res_df,'text')
    mask_df['classes'] = res_df['classes']
    mask_df['orig_text'] = res_df['text']
    mask_df['html'] = res_df['item']
    mask_df['ID'] = res_df['ID']
    return mask_df

fname = ['OTPlug','GunStore','FreshGrubs']

for n,f in zip(fname,files):
    f1 = mask_html(f)
    f1.to_csv(f'./data/{n}.csv',index=False)



