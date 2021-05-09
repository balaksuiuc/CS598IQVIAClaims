#%%
import pandas, numpy
import urllib.request
import os, datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import sklearn

import common.utils.timerUtils as tu

os.chdir("C:\\Drive\\UIUC\\CS598_DLHealthcare\\HW3_2\\release\\HW3_RNN-lib\\data\\")
DATA_DIR = "C:\\Drive\\UIUC\\CS598_DLHealthcare\\Project\\iqvia_data\\"

import os
import pickle
import random
import numpy  as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

#####################################################
## load raw data
ENROLL_FILE = DATA_DIR + 'enroll_synth.dat'
CLAIMS_2019 = DATA_DIR + 'claims_2019.dat'
CLAIMS_2018 = DATA_DIR + 'claims_2018.dat'
CLAIMS_2017 = DATA_DIR + 'claims_2017.dat'
CLAIMS_2016 = DATA_DIR + 'claims_2016.dat'
CLAIMS_2015 = DATA_DIR + 'claims_2015.dat'

df_enroll = pd.read_csv(ENROLL_FILE, sep='|', low_memory=False)

df_claims2019 = pd.read_csv(CLAIMS_2019, sep='|', low_memory=False)
df_claims2018 = pd.read_csv(CLAIMS_2018, sep='|', low_memory=False)
df_claims2017 = pd.read_csv(CLAIMS_2017, sep='|', low_memory=False)
df_claims2016 = pd.read_csv(CLAIMS_2016, sep='|', low_memory=False)
df_claims2015 = pd.read_csv(CLAIMS_2015, sep='|', low_memory=False)

## Add year and create a single dataset for claims
df_claims2015["year"] = 2015
df_claims2016["year"] = 2016
df_claims2017["year"] = 2017
df_claims2018["year"] = 2018
df_claims2019["year"] = 2019

list_of_claims = [df_claims2015, df_claims2016, df_claims2017, df_claims2018, df_claims2019]
df_claims = pd.concat(list_of_claims)

#####################################################
## 
df_claims["quarter"] = pd.PeriodIndex(pd.to_datetime(df_claims["to_dt"]), freq = 'Q')
df_claims['to_dt'] = pd.to_datetime(df_claims['to_dt'])

#find the length of claims for each patient
df_claims_length = df_claims[['pat_id','to_dt']].groupby('pat_id').agg({'to_dt':['max','min']}).reset_index() 
df_claims_length.columns = ['pat_id','max','min']
df_claims_length['length'] = df_claims_length['max'] - df_claims_length['min']
df_claims_length['length'] = df_claims_length['length'].dt.days
avg_claim_length = df_claims_length['length'].mean()
min_claim_length = df_claims_length['length'].min()
max_claim_length = df_claims_length['length'].max()
print(min_claim_length,avg_claim_length,max_claim_length)
rd = df_claims_length['length'].plot(kind='hist', bins=15)
rd.set_title("Frequency of Length")
#most patients's length is less than three years, 12 quarters, so chooce an observation window of 12 quarters

#calculate index date = last claim - 270 days (180days as prediction window + 90days as last quarter)
claim_indx_date = df_claims[['pat_id','to_dt']].groupby('pat_id').agg({'to_dt':['max']}).reset_index()
claim_indx_date.columns = ['pat_id','max']
claim_indx_date['indx_date'] = claim_indx_date['max'] - pd.to_timedelta(270,unit='d')
#filter claims, 
filterred_claims = pd.merge(df_claims,claim_indx_date,how = 'left',on=['pat_id'])
#observation includes 1000 days(three years) before index date
#prediction window is 180 days (two quarters), last 90 days(last quarter) is reserved as target
filterred_observation = filterred_claims.loc[filterred_claims.to_dt<=filterred_claims.indx_date]
filterred_observation = filterred_observation.loc[filterred_observation.to_dt>=filterred_observation.indx_date-pd.to_timedelta(1095,unit='d')]
filterred_observation = filterred_observation.loc[filterred_observation.to_dt>=pd.to_datetime('2015-10-1')] #code changed after October 1, 2015
filterred_target = filterred_claims.loc[filterred_claims.to_dt>=filterred_claims.indx_date+pd.to_timedelta(180,unit='d')]

#find the number of claims after filter
pat_claimsnumb = filterred_observation[['pat_id']].groupby('pat_id').agg({'pat_id':['count']}).reset_index()
pat_claimsnumb.columns = ['pat_id','count']
print(pat_claimsnumb.shape,pat_claimsnumb['count'].max(),pat_claimsnumb['count'].mean(),pat_claimsnumb['count'].min())

##
diag_cols = ["diag1", "diag2", "diag3", "diag4", "diag5", "diag6", "diag7", "diag8", "diag9", "diag10", "diag11", "diag12"]
icdprc_cols=["icdprc1", "icdprc2", "icdprc3", "icdprc4", "icdprc5", "icdprc6", "icdprc7", "icdprc8", "icdprc9", "icdprc10", "icdprc11", "icdprc12"]

#number of unique diag codes and prc codes after filter
diag = []
for colname in diag_cols:
    diag.extend(pd.unique(filterred_observation[colname]))
diag_dict = np.unique(diag)
print(len(np.unique(diag)))

prc = []
for colname in icdprc_cols:
    prc.extend(pd.unique(filterred_observation[colname]))
prc_dict = np.unique(prc)
print(len(np.unique(prc)))

#number of unique record type, procedure code and revenue code(high-level description of services)
filterred_observation["rectype"] = filterred_observation["rectype"].astype('str')
filterred_observation["proc_cde"] = filterred_observation["proc_cde"].astype('str')
filterred_observation["rev_code"] = filterred_observation["rev_code"].astype('str')
print(len(np.unique(filterred_observation["rectype"])))
print(len(np.unique(filterred_observation["proc_cde"])))
print(len(np.unique(filterred_observation["rev_code"])))
rectype_dict = np.unique(filterred_observation["rectype"])
proc_cde_dict = np.unique(filterred_observation["proc_cde"])
rev_code_dict = np.unique(filterred_observation["rev_code"])

#extract interested columns
filtered_features = filterred_observation[["pat_id","paid","charge","quarter","rectype","proc_cde","rev_code","diag1", "diag2", "diag3", 
                                           "diag4", "diag5", "diag6", "diag7", "diag8", "diag9", "diag10", "diag11","diag12","icdprc1", 
                                           "icdprc2", "icdprc3", "icdprc4", "icdprc5", "icdprc6", "icdprc7", "icdprc8", "icdprc9","icdprc10",
                                           "icdprc11", "icdprc12"]]

##
#investigate unqiue codes after filter
proc_cde_valuecount = filtered_features["proc_cde"].value_counts()
rev_code_valuecount = filtered_features["rev_code"].value_counts()
diag_valuecount = filtered_features["diag1"].value_counts()
prc_valuecount = filtered_features["icdprc1"].value_counts()

proc_cde_valuecount = proc_cde_valuecount.to_frame()
rd = np.log10(proc_cde_valuecount).plot(kind='hist', bins=15)
rd.set_title("frequency of proc_cde")
#among 6114, less than 500 proc codes appear more than 100 times
rev_code_valuecount = rev_code_valuecount.to_frame()
rd = np.log10(rev_code_valuecount).plot(kind='hist', bins=15)
rd.set_title("frequency of revenue code")
#among 718, less than 300 revenue codes appear more than 100 times
diag_valuecount = diag_valuecount.to_frame()
rd = np.log10(diag_valuecount).plot(kind='hist', bins=15)
rd.set_title("frequency of diag code")
#among 12952, about 1000 diag codes appear more than 100 times
prc_valuecount = prc_valuecount.to_frame()
rd = np.log10(prc_valuecount).plot(kind='hist', bins=15)
rd.set_title("frequency of prc code")
#among 420, less than 50 appear more than 100 times

proc_cde_valuecount = filtered_features["proc_cde"].value_counts().reset_index()
rev_code_valuecount = filtered_features["rev_code"].value_counts().reset_index()
diag_valuecount = filtered_features["diag1"].value_counts().reset_index()
prc_valuecount = filtered_features["icdprc1"].value_counts().reset_index()

#apply thresholds on codes to extract dictionary for high frequency codes
proc_cde_dict = set(proc_cde_valuecount.apply(lambda x: x['index'] if x['proc_cde']>=1000 else 0,axis=1).to_list())
rev_code_dict = set(rev_code_valuecount.apply(lambda x: x['index'] if x['rev_code']>=1000 else 0,axis=1).to_list())
diag_dict = set(diag_valuecount.apply(lambda x: x['index'] if x['diag1']>=1000 else 0,axis=1).to_list())
prc_dict = set(prc_valuecount.apply(lambda x: x['index'] if x['icdprc1']>=100 else 0,axis=1).to_list())
proc_cde_dict.remove(0)
rev_code_dict.remove(0)
diag_dict.remove(0)
prc_dict.remove(0)

print(len(proc_cde_dict),len(rev_code_dict),len(diag_dict),len(prc_dict))
#total code features are reduced to less than 300 after filtering

#filter out code features that not appears in the dictionary
def combine_columns_into_list(x,cols,dict_set):
    out = list()
    for col in cols:
        if x[col] in dict_set:
            out.append(x[col])
    return out

filtered_features['diags'] = filtered_features.apply(lambda x: combine_columns_into_list(x,["diag1","diag2","diag3","diag4","diag5","diag6","diag7","diag8","diag9","diag10"],diag_dict),axis=1)
filtered_features['rev_code'] = filtered_features.apply(lambda x: x['rev_code'] if x['rev_code'] in rev_code_dict else 0,axis=1)
filtered_features['icdprc'] = filtered_features.apply(lambda x: x['icdprc1'] if x['icdprc1'] in prc_dict else 0,axis=1)
filtered_features['proc_cde'] = filtered_features.apply(lambda x: x['proc_cde'] if x['proc_cde'] in proc_cde_dict else 0,axis=1)

#drop original code columns
dropped_features = ["diag1", "diag2", "diag3", "diag4", "diag5", "diag6", "diag7", "diag8", "diag9", "diag10", "diag11", 
                "diag12","icdprc1", "icdprc2", "icdprc3", "icdprc4", "icdprc5", "icdprc6", "icdprc7", "icdprc8", "icdprc9",
                "icdprc10", "icdprc11", "icdprc12"]
filtered_features = filtered_features.drop(dropped_features,axis=1)

##
filtered_features = filtered_features.drop(['charge'],axis=1)
#build multi-hot for rectype, proc_cde, rev_code, recvtype and icdprc
filtered_features = pd.get_dummies(filtered_features,columns = ['rectype','proc_cde','rev_code','icdprc'])
print(filtered_features.shape)
filtered_dict_features = filtered_features.copy(deep=True)

#build multi-hot for diags list feature
v = filtered_dict_features.diags.values
l = [len(x) for x in v.tolist()]
f, u = pd.factorize(np.concatenate(v))
n, m = len(v), u.size
i = np.arange(n).repeat(l)

dummies = pd.DataFrame(
    np.bincount(i * m + f, minlength=n * m).reshape(n, m),
    filtered_dict_features.index, u
)

filtered_dict_features = filtered_dict_features.drop('diags', 1).join(dummies)

#filtered_dict_features.to_pickle('./filterd_feature_dict_281.pkl')
filtered_dict_features = filtered_dict_features.groupby(['pat_id','quarter']).sum().reset_index()
#sort the pat_id and quarter in ascending order
filtered_dict_features = filtered_dict_features.sort_values(by=['pat_id','quarter'],ascending=False)

#build pat_id list to extract data from filtered_dict_features
pat_ids_dict = set(np.unique(filtered_dict_features['pat_id']))
pat_ids_dict = sorted(list(pat_ids_dict),reverse=True)

##
y_paid = filterred_target[['pat_id','paid']].groupby('pat_id').agg({'paid':['sum']}).reset_index()
y_paid.columns = ['pat_id','paid']
y_paid['miss'] = y_paid.apply(lambda x: 0 if x['pat_id'] in pat_ids_dict else 1,axis=1)
y_paid = y_paid.loc[y_paid['miss']!=1]
y_paid = y_paid.drop(['miss'],axis=1)
y_paid = y_paid.sort_values(by=['pat_id'],ascending = False)

#########################################################################
## create x and y variables similar to RNN in class, write to pkl file
filterred_observation = filterred_observation.sort_values(['pat_id', 'from_dt'])
cols = ['diag1', 'diag2', 'diag3', 'diag4', 'diag5', 'diag6', \
         'diag7', 'diag8', 'diag9', 'diag10', 'diag11', 'diag12', \
         'icdprc1', 'icdprc2', 'icdprc3', 'icdprc4', 'icdprc5', 'icdprc6', \
         'icdprc7', 'icdprc8', 'icdprc9', 'icdprc10', 'icdprc11', 'icdprc12']
filterred_observation['allcols'] = filterred_observation[cols].values.tolist()

xpd = filterred_observation[['pat_id', 'from_dt', 'allcols']]
ypd = y_paid
    
## get rid of nan in allcols list
## then, get rid of 0 length entries
## then, drop duplicates    
def getridofnan(yy):
    return([z for z in yy if isinstance(z, str)])
xpd['allcolsmod'] = xpd.allcols.apply(getridofnan)

# remove zero length
xpd['colLENGTH'] = xpd.allcols.apply(lambda zz : len(zz))
xpd['colLENGTH'] = xpd.allcolsmod.apply(lambda zz : len(zz))
xpd = xpd[xpd.colLENGTH > 0]

# remove duplicate rows
def joinfunc(yy):
    return '-'.join(yy)
xpd['diagstr'] = xpd.allcolsmod.apply(joinfunc)
xpd = xpd.drop_duplicates(['pat_id', 'from_dt', 'diagstr'])

# aggregate
myX = xpd.groupby('pat_id').allcolsmod.agg(lambda yy: list(yy)).reset_index()
myY = y_paid
myTbl = pandas.merge(myX, myY, how='inner', on='pat_id')

# save to pickle files
pids = myTbl.pat_id.tolist()
pids = list(range(0,len(pids)))
pids = [1 + xx for xx in pids]
pandas.to_pickle(pids, f'{DATA_DIR}/pids.pkl')

# seqs
import copy, itertools
seqs = myTbl.allcolsmod.tolist()

seqs2 = copy.deepcopy(seqs)
# make the dictionary
unqdiags = numpy.unique(list(itertools.chain(*list(itertools.chain(*seqs)))))
vals = list(range(0, len(unqdiags)))
dmap = {unqdiags[i] : vals[i] for i in range(0, len(vals))}

# map diagnosis codes to ints
for i, x in enumerate(seqs):
    for j, y in enumerate(x):
        for k, z in enumerate(y):
            seqs2[i][j][k] = 1 + dmap[z]
pandas.to_pickle(seqs2, f'{DATA_DIR}/seqs.pkl')

# paid
pandas.to_pickle(myTbl.paid.tolist(), f'{DATA_DIR}/morts.pkl')


#%% check if values match what Wei had in this code
pat_id = 's154AAAAARGHNMPH'
print('----filterred_target----')
print(filterred_target.query(f"pat_id=='{pat_id}'"))
print('----filterred_observation----')
print(filterred_observation.query(f"pat_id=='{pat_id}'"))
print('----y_paid----')
print(y_paid.query(f"pat_id=='{pat_id}'"))
print('----y from target----')
print(filterred_target.query(f"pat_id=='{pat_id}'").paid.sum())

## make sure for each patient, filterred_target.from_dt.min is always greater than
#  filterred_observation.to_dt.max
#  results in one patient with some overlap. Let's not worry about it. 
t1 = filterred_target.groupby('pat_id', as_index=False).from_dt.min()
t2 = filterred_observation.groupby('pat_id', as_index=False).to_dt.max()
t = pandas.merge(t1, t2, how='inner')


