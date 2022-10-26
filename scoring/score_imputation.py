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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import logit, expit

class DiscreteImputer():
    """Missing score value imputation using a discrete approach:
     - the score is grouped to a certain number of quantiles (or, if it has only limited number of distinct values, left as it is)
     - for each such value, mean target is calculated
     - mean target of non hits is caluclated
     - group with closest mean target to the non hit mean target is found
     - non hit score is imputed by mean score of the group found in previous step
    
    Args:
        subgroups (int, optional): max number of discrete groups the hit score should be split into (default: 50)
    """
    
    def __init__(self, subgroups=50):
        """Initalization
        """
        self.subgroups = subgroups
    
    def fit(self, x_hit, y_hit, y_impute, weight_hit=None, weight_impute=None):
        """Calculates the value the missing score should be imputed with
        
        Args:
            x_hit (np.array): array of score values of hit observations
            y_hit (np.array): array of target values of hit observations
            y_impute (np.array): array of target values of to-be-imputed observations
            weight_hit (np.array, optional): array of weights of hit obs. (default: None)
            weight_impute (np.array, optional): array of weights of to-be-imputed obs. (default: None)

        Returns:
            float: value the missing score should be imputed with
        """
        
        if weight_hit is None:
            weight_hit = np.ones(len(y_hit))
        if weight_impute is None:
            weight_impute = np.ones(len(y_impute))
        
        data_hit = pd.DataFrame({'score': x_hit, 'target': y_hit, 'weight': weight_hit})
        data_hit['weighted_score'] = data_hit['score'] * data_hit['weight']
        data_hit['weighted_target'] = data_hit['target'] * data_hit['weight']
        
        if data_hit['score'].nunique() > self.subgroups:
            data_hit['score_cat'] = pd.qcut(x=data_hit['score'], q=self.subgroups)
        else:
            data_hit['score_cat'] = data_hit['score'].copy()
            
        cat_defrates = data_hit.groupby(['score_cat']).sum()
        
        cat_defrates.loc[cat_defrates['weight'] > 0, 'score'] = \
            cat_defrates[cat_defrates['weight'] > 0]['weighted_score'] / cat_defrates[cat_defrates['weight'] > 0]['weight']
        cat_defrates.loc[~(cat_defrates['weight'] > 0), 'score'] = np.nan
        
        cat_defrates.loc[cat_defrates['weight'] > 0, 'target'] = \
            cat_defrates[cat_defrates['weight'] > 0]['weighted_target'] / cat_defrates[cat_defrates['weight'] > 0]['weight']
        cat_defrates.loc[~(cat_defrates['weight'] > 0), 'target'] = np.nan

        if sum(weight_impute) > 0:
            self.y_impute_value = sum(y_impute * weight_impute) / sum(weight_impute)
        else:
            self.y_impute_value = np.nan

        print('Imputation target value:', self.y_impute_value)
            
        cat_defrates['nonhit_diff'] = (cat_defrates['target'] - self.y_impute_value).abs()
        self.x_impute_value = cat_defrates.loc[cat_defrates['nonhit_diff'].idxmin()]['score']
        self.cat_defrates = cat_defrates[['score','target','nonhit_diff']]
        
        print('Imputation score value:', self.x_impute_value)

        return self.x_impute_value
            
    def draw(self):
        """
        Shows a chart how the default rate and score look and what value was used for imputation.
        """
        
        self.cat_defrates.index = self.cat_defrates['score']
        self.cat_defrates.plot(style='o-')
        plt.axvline(x=self.x_impute_value, color='k')
        plt.axhline(y=self.y_impute_value, color='k')
        
    def transform(self, x, x_imputation_mask):
        """Transforms array of score values - special (to be imputed) values are changed to calculated
        imputation values.
        
        Arguments:
            x (np.array): array with score, including the values that should be imputed
            x_imputation_mask (np.array): mask what should be imputed
        
        Returns:
            np.array: x where the values corresponding to x_imputation mask were imputed
        """
        
        x_imputed = x.copy()
        x_imputed[x_imputation_mask] = self.x_impute_value
        
        return x_imputed


def missing_value(sample_hit, sample_nohit, score, target, weight = None, shift = 0, scale = 1, ispd = False):
    '''Calculates imputation value for score that is missing for some observations.
    Based on default rate in sample_hit, average score in sample_hit and default rate in sample_nohit,
    it caluclates score for sample_nohit (using simple proportion - "trojčlenka")

    Arguments:
        sample_hit (pd.DataFrame): data sample with hits
        sample_nohit (pd.DataFrame): data sample with nonhits
        score (str): name of the score (probability of default)
        target (str): name of the target
        weight (str, optional): name of the weight (default: None)
        shift (float, optional): value the score should be added to (default: 0) 
        scale (float, optional): value the score should be multiplied by  (default: 1)
        ispd (bool), optional): indicator if score is in form of PD (if so, logit transformation is used before the interpolation) (default: False)

    Returns:
        float: value to be imputed to no hit sample
    '''
    if weight is not None:
        sample_hit = sample_hit[[score, target, weight]].copy()
        sample_nohit = sample_nohit[[score, target, weight]].copy()
    else:
        sample_hit = sample_hit[[score, target]].copy()
        sample_nohit = sample_nohit[[score, target]].copy()
        weight = '_w'
        sample_hit[weight] = 1
        sample_nohit[weight] = 1

    if ispd:
        avg_score_hit = (expit(shift + scale * logit(sample_hit[score])) * sample_hit[weight]).sum() / sample_hit[weight].sum()
    else:  
        avg_score_hit = (expit(shift + scale * sample_hit[score]) * sample_hit[weight]).sum() / sample_hit[weight].sum()

    def_rate_hit = (sample_hit[target] * sample_hit[weight]).sum() / sample_hit[weight].sum()
    def_rate_nohit = (sample_nohit[target] * sample_nohit[weight]).sum() / sample_nohit[weight].sum()

    imputation_value = avg_score_hit * def_rate_nohit / def_rate_hit

    if ispd:
        return imputation_value
    else:
        return logit(imputation_value)

def quantile_imputer(sample_hit, sample_nohit, score, target, weight = None, quantiles = 100, def_by_score_ascending = None):
    '''Calculates imputation value for score that is missing for some observations.
    It divides sample_hit into certain number of quantiles (sorted by score) and smoothens them (by joining neighbors together)
    to be monotonic in average target value.
    Then, it finds the quantile whose average target value is closest to average target of sample_nohit.
    Average score of this quantile is then outputted as desired imputation value for score in sample_nohit.

    Args:
        sample_hit (pd.DataFrame): data sample with hits
        sample_nohit (pd.DataFrame): data sample with nonhits
        score (str): name of the score (probability of default)
        target (str): name of the target
        weight (str, optional): name of the weight (default: None)
        quantiles (int, optional): number of quantiles the score should be divided to at the beginning, before smoothing (default: 100)
        def_by_score_ascending (boolean, optional): True if the score grows with probability default. False if the score decreases with PD. None for the function to determine this automatically. (default: None)

    Returns:
        float: value to be imputed to no hit sample
    '''

    if weight is not None:
        sample_hit = sample_hit[[score, target, weight]].copy()
        sample_nohit = sample_nohit[[score, target, weight]].copy()
    else:
        sample_hit = sample_hit[[score, target]].copy()
        sample_nohit = sample_nohit[[score, target]].copy()
        weight = '_w'
        sample_hit[weight] = 1
        sample_nohit[weight] = 1

    if def_by_score_ascending is None:
        correlation_default_score = np.corrcoef(sample_hit[score], sample_hit[target])[0][1]
        if correlation_default_score >= 0: 
            def_by_score_ascending = True
        else:
            def_by_score_ascending = False

    sample_hit.sort_values(score, ascending = def_by_score_ascending, inplace=True)
    sample_hit['cumulative_weight'] = sample_hit[weight].cumsum()

    sample_hit['quantile'] = sample_hit['cumulative_weight'] / sample_hit[weight].sum() * quantiles
    sample_hit['quantile'] = np.floor(sample_hit['quantile'])
    sample_hit.loc[sample_hit['quantile']==quantiles, 'quantile'] = quantiles-1

    sample_hit['weighted_target'] = sample_hit[weight] * sample_hit[target]
    sample_hit['weighted_score'] = sample_hit[weight] * sample_hit[score]

    aggregated_hit = sample_hit.groupby(['quantile']).aggregate({'weighted_score':'sum',
                                                                 'weighted_target':'sum',
                                                                 weight:'sum',})
    aggregated_hit[target] = aggregated_hit['weighted_target'] / aggregated_hit[weight]
    aggregated_hit[score] = aggregated_hit['weighted_score'] / aggregated_hit[weight]
    
    while True:
        previous_stage_quantiles = len(aggregated_hit)
        for prev_quantile, quantile in zip(aggregated_hit.index[:-1], aggregated_hit.index[1:]):
            # prev_quantile = aggregated_hit.iloc[aggregated_hit.index.get_loc(quantile)-1].name
            if aggregated_hit.loc[prev_quantile, target] >= aggregated_hit.loc[quantile, target]:
                aggregated_hit.loc[quantile, 'weighted_score'] += aggregated_hit.loc[prev_quantile, 'weighted_score']
                aggregated_hit.loc[quantile, 'weighted_target'] += aggregated_hit.loc[prev_quantile, 'weighted_target']
                aggregated_hit.loc[quantile, weight] += aggregated_hit.loc[prev_quantile, weight]
                aggregated_hit.loc[quantile, score] = aggregated_hit.loc[quantile, 'weighted_score'] / aggregated_hit.loc[quantile, weight]
                aggregated_hit.loc[quantile, target] = aggregated_hit.loc[quantile, 'weighted_target'] / aggregated_hit.loc[quantile, weight]
                aggregated_hit.drop(prev_quantile, axis=0, inplace=True)
        if previous_stage_quantiles == len(aggregated_hit):
            break
    
    def_rate_nohit = (sample_nohit[target] * sample_nohit[weight]).sum() / sample_nohit[weight].sum()

    aggregated_hit['nohit_default_diff'] = np.abs(aggregated_hit[target] - def_rate_nohit)

    best_match_quantile = aggregated_hit['nohit_default_diff'].idxmin()

    return aggregated_hit.loc[best_match_quantile, score]
    
