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

# -*- coding: utf8 -*-

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
from .transformation import ScoreImputer, Logit, Range
from .metrics import gini, lift_grid_search_wrapper


def gm_uplift(t, columns, lift_perc, logit_columns=[], extra_nans={}, n_folds=15,  target_col='def_6_60',
random_state=241, n_jobs=-1, verbose=1, pre_dispatch='2*n_jobs'):
    """
    Вычисление кросс-валидационных статистик gini, lift

    Args:
        t (pd.DataFrame):
        columns (list of str):
        lift_perc (float):
        logit_columns (list of str, optional):
            Defaults to [].
        extr_nans (dict, optional):
            Defaults to {}.
        n_folds (int, optional):
            Defaults to 15.
        target_col (str, optional):
            Defaults to 'def_6_60'.
        random_state (int, optional):
            Defaults to 241.
        n_jobs (int, optional):
            Defaults to -1.
        verbose (int, optional):
            Defaults to 1.
        pre_dispatch (str, optional):
            Defaults to '2*n_jobs'.
    
    Returns:
        pd.DataFrame: датафрейм со статистиками, модель
    """
    logit_columns=[columns.index(c) for c in logit_columns]

    extra_nans={columns.index(c): v for c, v in extra_nans.items()}
    #print(columns, logit_columns)
    pipe=Pipeline([
        ('imp', ScoreImputer(extra_nans = extra_nans)),
        #('imp', ScoreImputer()),
        #('minmax', MinMaxScaler((0.001, 0.999))),
        ('logit', Logit(columns=logit_columns)),
        ('scale', StandardScaler()),
        ('clf', LogisticRegression()),   
    ])



    param_grid={
        'clf__C': np.logspace(-3,3,7)   
    }

    # search for best parameters
    #cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=241)
    cv=StratifiedShuffleSplit(n_splits=n_folds, test_size=0.33, random_state=random_state)
    grid=GridSearchCV(estimator=pipe, param_grid=param_grid, scoring={
                    'gini': make_scorer(gini, greater_is_better=True, needs_threshold=True),
                    'lift': make_scorer(lift_grid_search_wrapper, greater_is_better=True, needs_threshold=True, lift_perc=lift_perc),
                    }, refit='gini', cv=cv, n_jobs=n_jobs)
    grid.fit(t[columns], t[target_col])
    
    #parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
    #                    pre_dispatch=pre_dispatch)
    #scores = parallel(delayed(_fit_and_score)(clone(grid.best_estimator_), t.iloc[train], t.iloc[test], columns, target_col, lift_perc)
    #                  for train, test in cv.split(t[columns], t[target_col])) 

    #scores = (_fit_and_score(clone(grid.best_estimator_), t.iloc[train], t.iloc[test], columns, target_col, lift_perc)
    #                  for train, test in cv.split(t[columns], t[target_col])) 
    
    gini_keys=sorted([key for key in  grid.cv_results_ if re.match('split\\d+_test_gini', key)])
    lift_keys=sorted([key for key in  grid.cv_results_ if re.match('split\\d+_test_lift', key)])    
    ginis=[]
    for k in gini_keys:
        ginis.append(grid.cv_results_[k][grid.best_index_])
    lifts=[]
    for k in lift_keys:
        lifts.append(grid.cv_results_[k][grid.best_index_])
    #print(gini_keys, lift_keys, ginis, lifts, grid.best_index_, grid.cv_results_)
    return pd.DataFrame(list(zip(ginis, lifts)), columns=['gini', 'lift']), grid.best_estimator_

        
        
"""        
def _fit_and_score(estimator, train_t, test_t, columns, target_col, lift_perc):
    estimator.fit(train_t[columns], train_t[target_col])
    pred=estimator.predict_proba(test_t[columns])[:, 0]
    gini=roc_auc_score(test_t[target_col], 1-pred)*2-1
    cutoff=np.percentile(pred, lift_perc)       
    lift=test_t[pred<=cutoff][target_col].mean()/test_t[target_col].mean()
    return gini, lift
"""

       
        
