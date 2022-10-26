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
from scoring.ROC_curve import draw_ROC_curve, gini_score
from matplotlib import pyplot as plt


# In[5]:


import numpy as np
#from sklearn.model_selection import BaseShuffleSplit
from sklearn.utils import check_random_state

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline

from sklearn.metrics import roc_auc_score, make_scorer

#from sklearn.base import clone
#from sklearn.externals.joblib import Parallel, delayed
import pandas as pd
import re
from scoring.transformation import ScoreImputer, Logit, Range
from scoring.metrics import gini, lift, lift_grid_search_wrapper
from scoring.plot import plot_calib
from sklearn.calibration import CalibratedClassifierCV


# In[6]:


def gm_uplift_custom(t, columns, C, logit_columns=[], extra_nans={},  target_col='def_6_60',
random_state=241, n_jobs=-1, verbose=1, pre_dispatch='2*n_jobs'):

    logit_columns=[columns.index(c) for c in logit_columns]
    
    extra_nans={columns.index(c): v for c, v in extra_nans.items()}
    #print(columns, logit_columns)
    pipe=Pipeline([
        ('imp', ScoreImputer(extra_nans = extra_nans)),
        ('logit', Logit(columns=logit_columns)),
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(C=C)),   
    ])

    pipe.fit(t[columns], t[target_col])
    
    return pipe

