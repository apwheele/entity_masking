'''
This is the analysis of the agreement
between the coding in Scott/Andy
samples

python analysis_agreement.py > agreement.txt
'''

import pandas as pd
import numpy as np
from scipy.stats import beta

# Exact Clopper Pearson binomial CI
def binom_int(num,den, confint=0.95):
    quant = (1 - confint)/ 2.
    low = beta.ppf(quant, num, den - num + 1)
    high = beta.ppf(1 - quant, num + 1, den - num)
    return (np.nan_to_num(low), np.where(np.isnan(high), 1, high))

# ANALYSIS OF FALSE POSITIVES
SFP = pd.read_csv("./data/ScottCheckFalsePositives (1).csv")
AFP = pd.read_csv("./data/AndyCheckFalsePositives.csv")

SFP['FALSE'] = SFP['FALSE'].fillna(0.0)
AFP['FALSE'] = AFP['FALSE'].fillna(0.0)

# SEEING THE TOTAL ESTIMATED FALSE POSITIVE RATE FOR EACH
print('')
print('Scott False Positive')
sfp = SFP['FALSE'].sum()
tot = SFP.shape[0]
print(f'{sfp}/{tot} = {sfp/tot:,.2f}')
print('')

print('')
print('Andy False Positive')
afp = AFP['FALSE'].sum()
tot = AFP.shape[0]
print(f'{afp}/{tot} = {afp/tot:,.2f}')
print('')

SFP.rename(columns={"FALSE":"FSP"},inplace=True)
all_FP = pd.merge(AFP,SFP,how='outer',on='index')


# SEEING THE AGREEMENT
# Getting overlap in the two
cross_FP = pd.crosstab(all_FP['FALSE'],all_FP['FSP'])
agree = np.diag(cross_FP).sum()
tot = cross_FP.sum().sum()
print('')
print('Agreement in False Positives')
print(f'{agree}/{tot} = {agree/tot:,.2f}')
print('')
print(cross_FP)
print('')

# save to CSV to pull out a few examples
dis_FP = (all_FP['FALSE'] != all_FP['FSP']) & (~all_FP['FALSE'].isna()) & (~all_FP['FSP'].isna())
all_FP[dis_FP].to_csv("./data/CheckDiff_FP.csv")


# GETTING BOUNDS FOR THE FALSE POSITIVE SAMPLE ESTIMATE
all_FP['FALSE2'] = all_FP['FALSE'].fillna(-1).astype(int)
all_FP['FSP2'] = all_FP['FSP'].fillna(-1).astype(int)

def alt_vals(x):
    x0, x1 = x.iloc[0], x.iloc[1]
    if x0 == -1:
        low, high = x1, x1
    elif x1 == -1:
        low, high = x0, x0
    elif (x0 == 0) & (x1 == 0):
        low, high = 0, 0
    elif (x0 == 1) & (x1 == 1):
        low, high = 1, 0
    else:
        low, high = 0, 1
    return [low, high]

all_FP[['FL','FH']] = all_FP[['FALSE2','FSP2']].apply(alt_vals,axis=1,result_type='expand')
n = all_FP.shape[0]
ln = all_FP['FL'].sum()
hn = all_FP['FH'].sum()

low, high = binom_int(np.array([ln,hn]),n,confint=0.99)
print('')
print(f'Low estimate {ln}/{n} = {ln/n:,.2f}, 99% CI [{low[0]:.2f},{high[0]:.2f}]')
print(f'High estimate {hn}/{n} = {hn/n:,.2f}, 99% CI [{low[1]:.2f},{high[1]:.2f}]')
print('')

# ANALYSIS OF FALSE NEGATIVES
AFN = pd.read_csv("./data/AndyCheckFalseNegatives.csv")
SFN = pd.read_csv("./data/ScottCheckFalseNegatives (1).csv")

SFN['Name'] = SFN['Name'].fillna(0.0)
SFN['Name'] = 1*(SFN['Name'] != 0.0)

AFN['Name'] = AFN['Name'].fillna(0.0)

# SEEING THE TOTAL ESTIMATED FALSE POSITIVE RATE FOR EACH
print('')
print('Scott False Negative')
sfn = SFN['Name'].sum()
tot = SFN.shape[0]
print(f'{sfn}/{tot} = {sfn/tot:,.2f}')
print('')

print('')
print('Andy False Negative')
afn = AFN['Name'].sum()
tot = AFN.shape[0]
print(f'{afn}/{tot} = {afn/tot:,.2f}')
print('')

SFN.rename(columns={"Name":"NameS"},inplace=True)
all_FN = pd.merge(AFN,SFN,how='outer',on='index')


# SEEING THE AGREEMENT
# Getting overlap in the two
cross_FN = pd.crosstab(all_FN['Name'],all_FN['NameS'])
agree = np.diag(cross_FN).sum()
tot = cross_FN.sum().sum()
print('')
print('Agreement in False Negatives')
print(f'{agree}/{tot} = {agree/tot:,.2f}')
print('')
print(cross_FN)
print('')

# Get a few examples
dis_FN = (all_FN['Name'] != all_FN['NameS']) & (~all_FN['Name'].isna()) & (~all_FN['NameS'].isna())
all_FN[dis_FN].to_csv("./data/CheckDiff_FN.csv")


# GETTING BOUNDS FOR THE FALSE POSITIVE SAMPLE ESTIMATE
all_FN['FALSE2'] = all_FN['Name'].fillna(-1).astype(int)
all_FN['FSP2'] = all_FN['NameS'].fillna(-1).astype(int)

all_FN[['FL','FH']] = all_FN[['FALSE2','FSP2']].apply(alt_vals,axis=1,result_type='expand')
n = all_FN.shape[0]
ln = all_FN['FL'].sum()
hn = all_FN['FH'].sum()

low, high = binom_int(np.array([ln,hn]),n,confint=0.99)
print('')
print(f'Low estimate {ln}/{n} = {ln/n:,.2f}, 99% CI [{low[0]:.2f},{high[0]:.2f}]')
print(f'High estimate {hn}/{n} = {hn/n:,.2f}, 99% CI [{low[1]:.2f},{high[1]:.2f}]')
print('')