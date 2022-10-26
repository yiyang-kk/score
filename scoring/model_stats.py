"""
Home Credit Python Scoring Library and Workflow
Copyright © 2017-2019, Pavel Sůva, Marek Teller, Martin Kotek, Jan Zeller, 
Marek Mukenšnabl, Kirill Odintsov, Elena Kuchina, Jan Coufalík, Jan Hynek and
Home Credit & Finance Bank Limited Liability Company, Moscow, Russia.
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import copy
from matplotlib import pyplot as plt
from sklearn.calibration import CalibratedClassifierCV

from scoring.metrics import gini, lift

from scoring.ROC_curve import *
import scoring.gm_uplift_custom as cgmu


# In[11]:


def cgm_model(data, base_columns, logit_columns,
              extra_nans={}, target='def_6_60', base='base_6_60', c=1.66810054e-03, data_type_column=None):
    """Returns fitted ClientGM3.0 model"""
    if data_type_column:
        data = data[(data[data_type_column] == 'train') & (data[base] == 1)]
    
    estimator_mix_reg = cgmu.gm_uplift_custom(data,
                                              columns=base_columns,
                                              C=c,
                                              logit_columns=logit_columns,
                                              extra_nans=extra_nans,
                                              target_col=target, 
                                              random_state=241, n_jobs=4, verbose=1)
    model = CalibratedClassifierCV(estimator_mix_reg, cv='prefit')
    model = model.fit(data[base_columns], data[target])
    
    return model


# In[12]:


def acq_model(data, base_columns, logit_columns,
              extra_nans={}, target='def_6_60', base='base_6_60', c = 0.00428133239872, data_type_column=None):
    """Returns fitted AcquisitionGM3.0 model"""
    
    if data_type_column:
        data = data.loc[(data[data_type_column] == 'train') & (data[base] == 1)]
              
    final_model = cgmu.gm_uplift_custom(data,
                                        columns=base_columns,
                                        C=c, 
                                        logit_columns = logit_columns,
                                        extra_nans=extra_nans,
                                        target_col = target, 
                                        random_state = 241, n_jobs = 4, verbose = 1)

    model = CalibratedClassifierCV(final_model, cv='prefit')
    model = model.fit(data[base_columns], data[target])
    
    return model


# In[15]:


def model_score(model, input_columns, test_df):
    """
    Returns model score
    """
    return model.predict_proba(test_df[input_columns])[:, 0]


# In[16]:


def model_gini(model, input_columns, test_df, target='def_6_60'):
    """
    Returns model score gini
    """
    return gini_score(1 - test_df[target], model_score(model, input_columns, test_df))


# In[17]:


def model_lift(model, input_columns, test_df, perc, target='def_6_60'):
    """
    Return model score perc percents lift
    """
    return lift(test_df[target], model_score(model, input_columns, test_df), perc)


# In[21]:


def models_statistics_df(models, input_columns, test_df, perc, target='def_6_60', models_names=[]):
    """
    Returns dataframe with models gini and lift statistics

    Args:
        models (array like): array of models for which gini and lift statistics will be calculated
        input_columns (array like): array of input columns array for each model (order should correspond to models order)
        test_df (pandas.DataFrame): dataframe on which model statistics will be calculated, should contain every column from every input columns array
        perc (array like): list of lift percentages for each model (order should correspond to models order)
        target (str, optiona): default 'def_6_60', target on which models will be tested
        models_names(list of str, optional): names of models for output dataframe index (order should correspond to models order)
    """
    ginis = []
    lifts = []
    for i in range(len(models)):
        ginis.append(model_gini(models[i], input_columns[i], test_df, target))
        lifts.append(model_lift(models[i], input_columns[i], test_df, perc, target))
    
    res = pd.DataFrame()
    res['ginis_%s' % target[4:]] = ginis
    res['lift_%s' % perc] = lifts
    
    if (len(models_names) > 0) & (len(models_names) == len(models)):
        res.index = models_names
    
    return res


# In[ ]:




