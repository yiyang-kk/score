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

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
import math
from tqdm.notebook import tqdm


def kolmogorov_smirnov(y_true, y_pred):
    """
    Returns a results of Kolmogorov-smirnov test on goodness of fit using scipy.ks_2sample test.

    Args:
        y_true (np.array): true values of target
        y_pred (np.array): predicted score

    Returns:
        float: float value of test

    """
    return ks_2samp(y_pred[y_true == 1], y_pred[y_true == 0]).statistic

    
def gini(y_true, y_pred, sample_weight=None):
    """
    Returns Gini coefficient (linear transformation of Area Under Curve)

    Args:
        y_true (np.array): true values of target
        y_pred (np.array): predicted score
        sample_weight (np.array, optional): weights of the observations (default: None)

    Returns:
        float: float value of gini

    """
    return 2*roc_auc_score(y_true, y_pred, sample_weight=sample_weight)-1


def lift(y_true, y_pred, lift_perc = 10):
    """
    Returns Lift of prediction

    Args:
        y_true (np.array): true values of target
        y_pred (np.array): predicted score
        lift_perc (float): lift level (i.e. for 10%-Lift, set lift_perc = 10) (default: 10)

    Returns:
        float: float value of lift

    """
    cutoff=np.percentile(y_pred, lift_perc)       
    return y_true[y_pred<=cutoff].mean()/y_true.mean()
    

def eval_performance_wrapper(data, masks, col_target, col_score, col_weight=None, lift_perc=10):
    """Evaluates performance (Gini, KS, Lift) of score in relation to target on subsets of data defined by dict of masks.

    Args:
        data (pd.DataFrame): data
        masks (dict {str: array}): Masks (defining subsets of data) where socre performance should be evaluated on.
        col_target (str): Name of target column.
        col_score (str or list of str): Name of column with score (its performance is evaluated). If list is passed, all scores in list are evaluated
        col_weight (str, optional): Name of wieght column. Defaults to None.
        lift_perc (int, optional): Level on which cumulative lift should be calculated (between 1 and 99). Defaults to 10.

    Raises:
        TypeError: col_score must be either string or list of strings

    Returns:
        pd.DataFrame: table with performance of score on each data subset.
    """
    if isinstance(col_score,str):
        col_score = [col_score]
    elif isinstance(col_score,list):
        pass
    else:
        raise TypeError('col_score must be either string or list of strings')
    results = []
    for score in col_score:
        if col_weight is not None:
            print('Only Gini statistics supports using weights. Other performance metrics use unweighted sample.')
        for mask_name, mask in masks.items():
            if col_weight is not None:
                weights = data[mask][col_weight]
            else:
                weights = None
            results.append({
                'gini':gini(data[mask][col_target], data[mask][score], sample_weight=weights),
                f'lift_{lift_perc}':lift(data[mask][col_target], -data[mask][score], lift_perc),
                'KS':kolmogorov_smirnov(data[mask][col_target], data[mask][score]),
                'score':score,
                'sample':mask_name
            })
    results = pd.DataFrame(results).pivot(index='sample',columns='score',values=['gini', f'lift_{lift_perc}', 'KS'])
    return results


def lift_grid_search_wrapper(y_true, y_pred, lift_perc):
    return lift(y_true, -y_pred, lift_perc)


def iv(y_true, x):
    """
    Returns Information Value of a binned predictor

    Args:
        y_true (np.array): true values of target
        x (np.array): binned predictor

    Returns:
        float: float Information Value

    """
    woe={}
    lin={}
    iv=0
    for v in np.unique(x):
        woe[v]=(1.*(len(x[(x==v)&(y_true==0)])+1)/(len(x[y_true==0])+1))/(1.*(len(x[(x==v)&(y_true==1)])+1)/(len(x[y_true==1])+1))
        woe[v]=math.log(woe[v])
        lin[v]=(1.*(len(x[(x==v)&(y_true==0)])+1)/(len(x[y_true==0])+1))-(1.*(len(x[(x==v)&(y_true==1)])+1)/(len(x[y_true==1])+1))
        iv=iv+woe[v]*lin[v]
    return iv
	
	
def predictors_power(y_true, X):
    """Measures Gini and Information value of all variables in dataframe X in relation with target in series y_true.

    Args:
        y_true (pd.Series): true values of target variable
        X (pd.DataFrame): predictors for which power should be measured (must be numerical and without empty values, e.g. WOE transformed)

    Returns:
        pd.DataFrame: table with predictor power values for each predictor
    """
    n_cols = X.shape[1]
    power_tab = []
    for j in range(0,n_cols):
        x=X[X.columns[j]]
        power_tab.append({'Name':X.columns[j],'IV':iv(y_true,x),'Gini':gini(y_true,-x)})
    power_out = pd.DataFrame.from_records(power_tab)
    power_out = power_out.set_index('Name')
    power_out = power_out.sort_values('Gini',ascending=False)
    return power_out
    
    
def bootstrap_gini(data, col_target, col_score, col_weight=None, n_iter = 100, ci_range = 5, col_score_ref = None, use_tqdm=True, random_seed=None):
    """
    Calculates Gini Coefficient confidence intervals using bootstrapping.

    Args:
        data (pandas.DataFrame): the data frame with prediction and target
        col_target (str): name of the target column in data
        col_score (str): name of the score (prediction) column in data
        col_weight (str, optional): name of weight column in the data (default: None)
        col_score_ref (str, optional): name of baseline score column in data. If filled in, the function will return results based on difference in gini of col_score and col_score_ref. (default: None)
        n_iter (int, optional): number of iterations to be performed in bootstrapping. The more iterations, the more precise results. (default: 100)
        ci_range (float, optional): confidence interval range in percent. E.g. if we want to know the 90% confidence interval, we set ci_range = 5, and the confidence interval will be (5%,95%). (default: 5)
        use_tqdm (boolean, optional): whether graphical progress bar should be displayed (default: True)
        random_seed (int, optional): random seed for data sampling (default: None)

    Returns:
        mean, std, [lower bound cofidence, higher bound confidence]: mean, standard deviation and a list of 2 confidence interval bounds of Gini
    """

    gini_bootstrap = [] #all of n_iter gini values will be stored here
    
    # bootstrapping iterations
    loop_iterator = range(n_iter)
    if use_tqdm:
        loop_iterator = tqdm(loop_iterator, leave=False)

    for _ in loop_iterator:
        sampled_data = data.sample(frac = 1, replace = True, random_state = random_seed) #sampling with replacement, i.e. bootstrapping
        if col_weight is None:
            sample_weight = None
        else:
            sample_weight = sampled_data[col_weight]
        gini_value = gini(sampled_data[col_target], sampled_data[col_score], sample_weight=sample_weight) #calculate gini for this sample using AUC
        if col_score_ref:
            gini_value = gini_value - gini(sampled_data[col_target], sampled_data[col_score_ref], sample_weight=sample_weight)
        gini_bootstrap += [gini_value]
        if random_seed is not None:
            random_seed += 1
    
    ci = [np.percentile(gini_bootstrap, ci_range), np.percentile(gini_bootstrap, 100 - ci_range)] #confidence intervals

    return np.mean(gini_bootstrap), np.std(gini_bootstrap), ci #return mean, standard deviation and confidence intervals
