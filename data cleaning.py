#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 18:31:41 2017
"""

import numpy as np
import pandas as pd

# Looking at categorical values
def cat_exploration(column):
    return houseprice[column].value_counts()
    
# Imputing the missing values
def cat_imputation(column, value):
    houseprice.loc[houseprice[column].isnull(),column] = value

def show_missing():
    missing = houseprice.columns[houseprice.isnull().any()].tolist()
    return missing

houseprice=pd.read_csv('/Users/VK/Documents/!_Kaggle_Competitions/House Prices/Input/train.csv')
houseprice.head()
    
houseprice[show_missing()].isnull().sum()

#=========== LotFrontage 

houseprice['kDep2Front'] = houseprice['LotArea']/np.square(houseprice['LotFrontage'])
kDFmean = np.mean(houseprice['kDep2Front'])
houseprice['kDep2Front'].hist(bins=25)

df=houseprice['kDep2Front'].to_frame()  # convert to dataframe to replace NaN with means
df = df.fillna(kDFmean)
houseprice['kDep2Front'] = df.T.squeeze() # convert back to series

houseprice['NewFront']=np.sqrt(houseprice['LotArea']/houseprice['kDep2Front'])

"""  compare distributions

np.min(houseprice['NewFront'])
np.max(houseprice['NewFront'])
houseprice['NewFront'].hist(bins=25) 

np.min(houseprice['LotFrontage'])
np.max(houseprice['LotFrontage'])
houseprice['LotFrontage'].hist(bins=25) 

"""

cond = houseprice['LotFrontage'].isnull()
houseprice.LotFrontage[cond]=houseprice.NewFront[cond]

del houseprice['NewFront']
del houseprice['SqrtLotArea']
del houseprice['kDep2Front']

# ========== Alley

cat_exploration('Alley')
# I assume empty fields here mean no alley access
cat_imputation('Alley','None')

houseprice[['MasVnrType','MasVnrArea']][houseprice['MasVnrType'].isnull()==True]
cat_exploration('MasVnrType')
cat_imputation('MasVnrType', 'None')
cat_imputation('MasVnrArea', 0.0)

#=== Basement 
basement_cols=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2']
houseprice[basement_cols][houseprice['BsmtQual'].isnull()==True]

for cols in basement_cols:
    if 'FinSF'not in cols:
        cat_imputation(cols,'None')
        

cat_exploration('Electrical')
# Impute most frequent value
cat_imputation('Electrical','SBrkr')

cat_exploration('FireplaceQu')

# VERY cool feature describe - gives stdev mean etc
houseprice['Fireplaces'][houseprice['FireplaceQu'].isnull()==True].describe()
cat_imputation('FireplaceQu','None')

# Cool way to print out tables for two features
pd.crosstab(houseprice.Fireplaces, houseprice.FireplaceQu)

garage_cols=['GarageType','GarageQual','GarageCond','GarageYrBlt','GarageFinish','GarageCars','GarageArea']
houseprice[garage_cols][houseprice['GarageType'].isnull()==True]

#Garage Imputation
for cols in garage_cols:
    if houseprice[cols].dtype==np.object:
        cat_imputation(cols,'None')
    else:
        cat_imputation(cols, 0)

cat_exploration('PoolQC')
houseprice['PoolArea'][houseprice['PoolQC'].isnull()==True].describe()
cat_imputation('PoolQC', 'None')

cat_imputation('Fence', 'None')
cat_imputation('MiscFeature', 'None')
houseprice[show_missing()].isnull().sum()

        