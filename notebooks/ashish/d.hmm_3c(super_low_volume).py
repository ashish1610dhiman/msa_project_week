#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().system('pwd')


# In[2]:


import sys
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from joblib import Parallel, delayed

print(sys.version)


# In[3]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[4]:


UNIQUE_CNT = 10
VERSION = "v2"


# ### Read data

# In[5]:


train_test = pd.read_csv("../../data/train_validation_marker.csv")
train_test["SALES_DATE"] = pd.to_datetime(train_test["SALES_DATE"])
print (train_test.shape)


# #### Find SKUs with very low volume

# In[6]:


#clean train/test
train = train_test[(train_test.validation==False) & (train_test.validation_clean==True)]
validation = train_test[(train_test.validation==True) & (train_test.validation_clean==True)]
train.shape,validation.shape


# In[7]:


sku_vals = train[train.SALES_DATE>="2022-03-01"].groupby("Encoded_SKU_ID")["DAILY_UNITS"].nunique()


# In[8]:


(sku_vals<=UNIQUE_CNT).sum()


# In[9]:


sku_vals.count()


# ### Fit HMM on these

# In[10]:


sku_in_scope = sku_vals[(sku_vals<=UNIQUE_CNT)]


# In[11]:


import sys
sys.path.append("../../")

from src.ad_hmm import sku_predict


# In[12]:


pd.options.mode.chained_assignment = None


# In[13]:


from tqdm import tqdm


# In[14]:


sku_in_scope.index[:1]


# In[15]:


from IPython.utils import io


# In[ ]:

def parallel_hmm_function(sku_id):
    n_comps = sku_in_scope[sku_id]
    sku_pred_model = sku_predict(train_test, sku_id)
    print(f"predicting for {sku_id}")
    try:
        with io.capture_output() as captured:
            feats_sku = sku_pred_model.get_features(n_lags=2)
            train1, valid1 = sku_pred_model.split_train_test("2022-07-25")
            sku_pred_model.fit_hmm(train1, "2022-03-01", n_components1=n_comps)
            if n_comps<=5:
                sku_pred_15 = sku_pred_model.predict(valid1, 15)
            sku_pred_30 = sku_pred_model.predict(valid1, 30)
            sku_pred_45 = sku_pred_model.predict(valid1, 45)
            sku_pred_60 = sku_pred_model.predict(valid1, 60)
            sku_pred_90 = sku_pred_model.predict(valid1, 90)
            sku_pred_max = sku_pred_model.predict(valid1, sku_pred_model.X.shape[0])
        merge0 = sku_pred_15[["predicted"]].merge(sku_pred_30[["predicted"]], left_index=True, \
                                                  right_index=True, suffixes=("", "_30"))
        merge1 = merge0.merge(sku_pred_45[["predicted"]], left_index=True, \
                              right_index=True, suffixes=("", "_45"))
        merge2 = merge1.merge(sku_pred_60[["predicted"]], left_index=True, \
                              right_index=True, suffixes=("", "_60"))
        merge3 = merge2.merge(sku_pred_90[["predicted"]], left_index=True, \
                              right_index=True, suffixes=("", "_90"))
        merge4 = merge3.merge(sku_pred_max[["predicted"]], left_index=True, \
                              right_index=True, suffixes=("", "_max"))
        merge4["Encoded_SKU_ID"] = [sku_id] * merge4.shape[0]
        return(merge4)
    except:
        print(f"Error for {sku_id}")


all_preds = Parallel(n_jobs=6)(delayed(parallel_hmm_function)(sku_id) \
                               for sku_id in tqdm(sku_in_scope.index))

# for sku_id in tqdm(sku_in_scope.index):
#     n_comps = sku_in_scope[sku_id]
#     sku_pred_model = sku_predict(train_test,sku_id)
#     try:
#         with io.capture_output() as captured:
#             feats_sku = sku_pred_model.get_features(n_lags=2)
#             train1,valid1 = sku_pred_model.split_train_test("2022-07-25")
#             sku_pred_model.fit_hmm(train1,"2022-03-01", n_components1 = n_comps)
#             sku_pred_15 = sku_pred_model.predict(valid1, 15)
#             sku_pred_30 = sku_pred_model.predict(valid1, 30)
#             sku_pred_45 = sku_pred_model.predict(valid1, 45)
#             sku_pred_60 = sku_pred_model.predict(valid1, 60)
#             sku_pred_90 = sku_pred_model.predict(valid1, 90)
#             sku_pred_max = sku_pred_model.predict(valid1, sku_pred_model.X.shape[0])
#         merge0 = sku_pred_15[["predicted"]].merge(sku_pred_30[["predicted"]],left_index = True,\
#                               right_index = True, suffixes=("","_30"))
#         merge1 = merge0.merge(sku_pred_45[["predicted"]],left_index = True,\
#                               right_index = True, suffixes=("","_45"))
#         merge2 = merge1.merge(sku_pred_60[["predicted"]],left_index = True,\
#                               right_index = True, suffixes=("","_60"))
#         merge3 = merge2.merge(sku_pred_90[["predicted"]],left_index = True,\
#                               right_index = True, suffixes=("","_90"))
#         merge4 = merge3.merge(sku_pred_max[["predicted"]],left_index = True,\
#                               right_index = True, suffixes=("","_max"))
#         merge4["Encoded_SKU_ID"] =[sku_id]*merge4.shape[0]
#         all_preds.append(merge4)
#     except:
#         print (f"Error for {sku_id}")


# In[ ]:


hmm_result = pd.concat(all_preds)
hmm_result.to_csv(f"../../data/hmm_result_{VERSION}.csv")


# In[ ]:




