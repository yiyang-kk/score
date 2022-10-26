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


import pandas as pd
import numpy as np
import copy
from IPython.display import display, HTML


# TO DO add sufix as an atribute to grouping so we can use it to map variables from data to models


class ScoreCard:
    """Scorecard object holding information about scorecard and methods to export to various formats

    Args:
        predictors (list of str): name of predictors that enter the model. for full logit model, it should contain the dummy variables (not the original variables before dummy transformation)
        coefficients (list of float): beta coefficients of predictors. should be list of the same length as predictors, keeping variable order.
        intercept (int): intercept of the model (default: 0)
        grouping (Grouping or InteractiveGrouping): Grouping/InteractiveGrouping object that was used to transform the predictors if WOE or dummy transformation was used (default: None)
    """
    
    def __init__(self, predictors, coefficients, intercept = 0, grouping = None):

        self.bins_data = grouping.bins_data_

        if grouping is not None:
            self.grouping = grouping
            self.full_logit_constraints = grouping.get_dummy_names()
        
        if isinstance(intercept, list) or isinstance(intercept, np.ndarray):
            self.intercept = intercept[0]
        else:
            self.intercept = intercept

        self.suffix_woe = '_WOE'
        suffix_length = len(self.suffix_woe)
        
        dummy_backsearch_dict = {}
        if self.full_logit_constraints is not None:
            for variable_name, variable_meta in self.full_logit_constraints.items():
                for dummy_name in variable_meta:
                    dummy_backsearch_dict[dummy_name] = variable_name

        self.predictors_woe = []
        self.coefficients_woe = {}
        self.predictors_dummy = []
        self.coefficients_dummy = {}
        self.predictors_linear = []
        self.coefficients_linear = {}

        if type(coefficients[0]) == np.ndarray:
            coefficients = coefficients[0]

        for pred, coef in zip(predictors, coefficients):
            if pred[-suffix_length:] == self.suffix_woe:
                self.predictors_woe.append(pred[:-suffix_length])
                self.coefficients_woe[pred[:-suffix_length]] = coef
            elif pred in dummy_backsearch_dict.keys():
                original_predictor_name = dummy_backsearch_dict[pred]
                if original_predictor_name not in self.predictors_dummy:
                    self.predictors_dummy.append(original_predictor_name)
                    self.coefficients_dummy[original_predictor_name] = []
                    for dummy_name in self.full_logit_constraints[original_predictor_name]:
                        coef = 0
                        if dummy_name in predictors:
                            coef = coefficients[predictors.index(dummy_name)]
                        self.coefficients_dummy[original_predictor_name].append(coef)
            else:
                self.predictors_linear.append(pred)
                self.coefficients_linear[pred] = coef

        self.predictors = self.predictors_woe + self.predictors_dummy + self.predictors_linear

        self._transformed_data = None

    def scorecard_table_simple(self):
        """Outputs tabular version of the scorecard. Each rows represents a category of a variable, and includes information about its borders (for numerical) or unique values (for categorical), WOE, Beta coefficient, and BiXi (product WOE*Beta).

        Returns:
            pd.DataFrame: scorecard table
        """
        
        self.scorecard_table = list()

        for variable in self.predictors_woe:
            bin_data = self.bins_data[variable]
            if variable in self.grouping.cat_columns:
                for bin_var, bin_number in bin_data['bins'].items():
                    self.scorecard_table.append({'Variable': variable,
                                                 'Treatment': 'WOE Categorical',
                                                 'Min': np.nan,
                                                 'Max': np.nan,
                                                 'Value': bin_var if not pd.isnull(bin_var) else 'null',
                                                 'WOE': bin_data['woes'][bin_number],
                                                 'Beta': self.coefficients_woe[variable],
                                                 'BiXi': bin_data['woes'][bin_number] * self.coefficients_woe[variable]})

                self.scorecard_table.append({'Variable': variable,
                                             'Treatment': 'WOE Categorical',
                                             'Min': np.nan,
                                             'Max': np.nan,
                                             'Value': 'else',
                                             'WOE': bin_data['unknown_woe'],
                                             'Beta': self.coefficients_woe[variable],
                                             'BiXi': bin_data['unknown_woe'] * self.coefficients_woe[variable]})
            else:
                for i in range(len(bin_data['bins']) - 1):
                    lower_bound = bin_data['bins'][i]
                    upper_bound = bin_data['bins'][i + 1]
                    self.scorecard_table.append({'Variable': variable,
                                                 'Treatment': 'WOE Numerical',
                                                 'Min': lower_bound,
                                                 'Max': upper_bound,
                                                 'Value': np.nan,
                                                 'WOE': bin_data['woes'][i],
                                                 'Beta': self.coefficients_woe[variable],
                                                 'BiXi': bin_data['woes'][i] * self.coefficients_woe[variable]})
                self.scorecard_table.append({'Variable': variable,
                                             'Treatment': 'WOE Numerical',
                                             'Min': np.nan,
                                             'Max': np.nan,
                                             'Value': 'null',
                                             'WOE': bin_data['nan_woe'],
                                             'Beta': self.coefficients_woe[variable],
                                             'BiXi': bin_data['nan_woe'] * self.coefficients_woe[variable]})

        for variable in self.predictors_dummy:
            bin_data = self.bins_data[variable]
            if variable in self.grouping.cat_columns:
                for bin_var, bin_number in bin_data['bins'].items():
                    self.scorecard_table.append({'Variable': variable,
                                                 'Treatment': 'Dummy Categorical',
                                                 'Min': np.nan,
                                                 'Max': np.nan,
                                                 'Value': bin_var if not pd.isnull(bin_var) else 'null',
                                                 'WOE': bin_data['woes'][bin_number],
                                                 'Beta': self.coefficients_dummy[variable][bin_number],
                                                 'BiXi': self.coefficients_dummy[variable][bin_number]})

                self.scorecard_table.append({'Variable': variable,
                                             'Treatment': 'Dummy Categorical',
                                             'Min': np.nan,
                                             'Max': np.nan,
                                             'Value': 'else',
                                             'WOE': bin_data['unknown_woe'],
                                             'Beta': self.coefficients_dummy[variable][len(self.coefficients_dummy[variable])-1],
                                             'BiXi': self.coefficients_dummy[variable][len(self.coefficients_dummy[variable])-1]})
            else:
                for i in range(len(bin_data['bins']) - 1):
                    lower_bound = bin_data['bins'][i]
                    upper_bound = bin_data['bins'][i + 1]
                    self.scorecard_table.append({'Variable': variable,
                                                 'Treatment': 'Dummy Numerical',
                                                 'Min': lower_bound,
                                                 'Max': upper_bound,
                                                 'Value': np.nan,
                                                 'WOE': bin_data['woes'][i],
                                                 'Beta': self.coefficients_dummy[variable][i],
                                                 'BiXi': self.coefficients_dummy[variable][i]})
                self.scorecard_table.append({'Variable': variable,
                                             'Treatment': 'Dummy Numerical',
                                             'Min': np.nan,
                                             'Max': np.nan,
                                             'Value': 'null',
                                             'WOE': bin_data['nan_woe'],
                                             'Beta': self.coefficients_dummy[variable][len(self.coefficients_dummy[variable])-1],
                                             'BiXi': self.coefficients_dummy[variable][len(self.coefficients_dummy[variable])-1]})
            
        for variable in self.predictors_linear:
            self.scorecard_table.append({'Variable': variable,
                                         'Treatment': 'Linear Numerical',
                                         'Min': np.nan,
                                         'Max': np.nan,
                                         'Value': np.nan,
                                         'WOE': np.nan,
                                         'Beta': self.coefficients_linear[variable],
                                         'BiXi': self.coefficients_linear[variable]})


        self.scorecard_table.append({'Variable': '_Intercept',
                                     'Treatment': 'Intercept',
                                     'Min': np.nan,
                                     'Max': np.nan,
                                     'Value': np.nan,
                                     'WOE': np.nan,
                                     'Beta': self.intercept,
                                     'BiXi': self.intercept})

        return pd.DataFrame.from_records(self.scorecard_table)[['Variable', 'Treatment', 'Min', 'Max', 'Value', 'WOE', 'Beta', 'BiXi']]

    def scorecard_table_full(self, data, mask, target, weightcol=None):
        """Outputs enriched tabular version of the scorecard. Each rows represents a category of a variable, and includes information about its borders (for numerical) or unique values (for categorical), WOE, Beta coefficient, and BiXi (product WOE*Beta).

        Based on data that are passed as argument, this adds also columns with distribution of observations and target within the category.

        Args:
            data (pd.DataFrame): data to calculate distribution statistics on
            mask (array like): mask to be applied onto the data before statistics are calculate
            target (str): name of target column within the data
            weightcol (str, optional): name of observation weight column within the data. Defaults to None.

        Returns:
            pd.DataFrame: scorecard table with distribution statistics
        """

        # self._transformed_data = self.grouping.transform(data, columns_to_transform=self.predictors, suffix='')
        # self._transformed_data[target] = data[target]
        if weightcol is None:
            self.data = data[mask][self.predictors + [target]]
            weight = 'weight'
            self.data[weight] = 1.0
        else:
            self.data = data[mask][self.predictors + [target, weightcol]]
            weight = weightcol
        self.scorecard_table = list()

        for variable in self.predictors_woe:
            bin_data = self.bins_data[variable]
            if variable in self.grouping.cat_columns:
                bins = {i: set() for i in range(len(bin_data['woes'] + 1))}
                known_values = list(bin_data['bins'].keys())
                sub_data = self.data[[variable, target, weight]]
                # data_grouped = self._transformed_data[[variable, target, 'weight']].groupby(variable)
                for value, bin_num in bin_data['bins'].items():
                    bins[bin_num].add('null' if pd.isnull(value) else value)

                for bin_number, bin_vars in bins.items():
                    if 'null' in bin_vars:
                        sub_set = sub_data[(sub_data[variable].isin(bin_vars)) | (pd.isnull(sub_data[variable]))]
                        known_values.append('null')
                    else:
                        sub_set = sub_data[sub_data[variable].isin(bin_vars)]
                    observations = sub_set[target].count()
                    bads = sub_set[target].sum()
                    observations_weight = sub_set[weight].sum()
                    bads_weight = sub_set[sub_set[target] == 1][weight].sum()
                    self.scorecard_table.append({'Variable': variable,
                                                 'Treatment': 'WOE Categorical',
                                                 'Min': np.nan,
                                                 'Max': np.nan,
                                                 'Value': ','.join([str(i) for i in bin_vars]),
                                                 'WOE': bin_data['woes'][bin_number],
                                                 'Beta': self.coefficients_woe[variable],
                                                 'BiXi': bin_data['woes'][bin_number] * self.coefficients_woe[variable],
                                                 'Observations': observations,
                                                 'Bads': bads,
                                                 'Goods': observations - bads,
                                                 'Weighted Observations': observations_weight,
                                                 'Weighted Bads': bads_weight,
                                                 'Weighted Goods': observations_weight - bads_weight})
                if 'null' in known_values:
                    sub_set = sub_data[(~sub_data[variable].isin(known_values)) & (pd.notnull(sub_data[variable]))]
                else:
                    sub_set = sub_data[~sub_data[variable].isin(known_values)]
                observations = sub_set[target].count()
                bads = sub_set[target].sum()
                observations_weight = sub_set[weight].sum()
                bads_weight = sub_set[sub_set[target] == 1][weight].sum()
                self.scorecard_table.append({'Variable': variable,
                                             'Treatment': 'WOE Categorical',
                                             'Min': np.nan,
                                             'Max': np.nan,
                                             'Value': 'else',
                                             'WOE': bin_data['unknown_woe'],
                                             'Beta': self.coefficients_woe[variable],
                                             'BiXi': bin_data['unknown_woe'] * self.coefficients_woe[variable],
                                             'Observations': observations,
                                             'Bads': bads,
                                             'Goods': observations - bads,
                                             'Weighted Observations': observations_weight,
                                             'Weighted Bads': bads_weight,
                                             'Weighted Goods': observations_weight - bads_weight})
            else:
                sub_data = self.data[[variable, target, weight]]
                for i in range(len(bin_data['bins']) - 1):
                    lower_bound = bin_data['bins'][i]
                    upper_bound = bin_data['bins'][i + 1]
                    sub_set = sub_data[(sub_data[variable] >= lower_bound) & (sub_data[variable] < upper_bound)]
                    observations = sub_set[target].count()
                    bads = sub_set[target].sum()
                    observations_weight = sub_set[weight].sum()
                    bads_weight = sub_set[sub_set[target] == 1][weight].sum()
                    self.scorecard_table.append({'Variable': variable,
                                                 'Treatment': 'WOE Numerical',
                                                 'Min': lower_bound,
                                                 'Max': upper_bound,
                                                 'Value': np.nan,
                                                 'WOE': bin_data['woes'][i],
                                                 'Beta': self.coefficients_woe[variable],
                                                 'BiXi': bin_data['woes'][i] * self.coefficients_woe[variable],
                                                 'Observations': observations,
                                                 'Bads': bads,
                                                 'Goods': observations - bads,
                                                 'Weighted Observations': observations_weight,
                                                 'Weighted Bads': bads_weight,
                                                 'Weighted Goods': observations_weight - bads_weight})
                sub_set = sub_data[sub_data[variable].isnull()]
                observations = sub_set[target].count()
                bads = sub_set[target].sum()
                observations_weight = sub_set[weight].sum()
                bads_weight = sub_set[sub_set[target] == 1][weight].sum()
                self.scorecard_table.append({'Variable': variable,
                                             'Treatment': 'WOE Numerical',
                                             'Min': np.nan,
                                             'Max': np.nan,
                                             'Value': 'null',
                                             'WOE': bin_data['nan_woe'],
                                             'Beta': self.coefficients_woe[variable],
                                             'BiXi': bin_data['nan_woe'] * self.coefficients_woe[variable],
                                             'Observations': observations,
                                             'Bads': bads,
                                             'Goods': observations - bads,
                                             'Weighted Observations': observations_weight,
                                             'Weighted Bads': bads_weight,
                                             'Weighted Goods': observations_weight - bads_weight})
        
        for variable in self.predictors_dummy:
            bin_data = self.bins_data[variable]
            if variable in self.grouping.cat_columns:
                bins = {i: set() for i in range(len(bin_data['woes'] + 1))}
                known_values = list(bin_data['bins'].keys())
                sub_data = self.data[[variable, target, weight]]
                # data_grouped = self._transformed_data[[variable, target, 'weight']].groupby(variable)
                for value, bin_num in bin_data['bins'].items():
                    bins[bin_num].add('null' if pd.isnull(value) else value)

                for bin_number, bin_vars in bins.items():
                    if 'null' in bin_vars:
                        sub_set = sub_data[(sub_data[variable].isin(bin_vars)) | (pd.isnull(sub_data[variable]))]
                        known_values.append('null')
                    else:
                        sub_set = sub_data[sub_data[variable].isin(bin_vars)]
                    observations = sub_set[target].count()
                    bads = sub_set[target].sum()
                    observations_weight = sub_set[weight].sum()
                    bads_weight = sub_set[sub_set[target] == 1][weight].sum()
                    self.scorecard_table.append({'Variable': variable,
                                                 'Treatment': 'Dummy Categorical',
                                                 'Min': np.nan,
                                                 'Max': np.nan,
                                                 'Value': ','.join([str(i) for i in bin_vars]),
                                                 'WOE': bin_data['woes'][bin_number],
                                                 'Beta': self.coefficients_dummy[variable][bin_number],
                                                 'BiXi': self.coefficients_dummy[variable][bin_number],
                                                 'Observations': observations,
                                                 'Bads': bads,
                                                 'Goods': observations - bads,
                                                 'Weighted Observations': observations_weight,
                                                 'Weighted Bads': bads_weight,
                                                 'Weighted Goods': observations_weight - bads_weight})
                if 'null' in known_values:
                    sub_set = sub_data[(~sub_data[variable].isin(known_values)) & (pd.notnull(sub_data[variable]))]
                else:
                    sub_set = sub_data[~sub_data[variable].isin(known_values)]
                observations = sub_set[target].count()
                bads = sub_set[target].sum()
                observations_weight = sub_set[weight].sum()
                bads_weight = sub_set[sub_set[target] == 1][weight].sum()
                self.scorecard_table.append({'Variable': variable,
                                             'Treatment': 'Dummy Categorical',
                                             'Min': np.nan,
                                             'Max': np.nan,
                                             'Value': 'else',
                                             'WOE': bin_data['unknown_woe'],
                                             'Beta': self.coefficients_dummy[variable][len(self.coefficients_dummy[variable])-1],
                                             'BiXi': self.coefficients_dummy[variable][len(self.coefficients_dummy[variable])-1],
                                             'Observations': observations,
                                             'Bads': bads,
                                             'Goods': observations - bads,
                                             'Weighted Observations': observations_weight,
                                             'Weighted Bads': bads_weight,
                                             'Weighted Goods': observations_weight - bads_weight})
            else:
                sub_data = self.data[[variable, target, weight]]
                for i in range(len(bin_data['bins']) - 1):
                    lower_bound = bin_data['bins'][i]
                    upper_bound = bin_data['bins'][i + 1]
                    sub_set = sub_data[(sub_data[variable] >= lower_bound) & (sub_data[variable] < upper_bound)]
                    observations = sub_set[target].count()
                    bads = sub_set[target].sum()
                    observations_weight = sub_set[weight].sum()
                    bads_weight = sub_set[sub_set[target] == 1][weight].sum()
                    self.scorecard_table.append({'Variable': variable,
                                                 'Treatment': 'Dummy Numerical',
                                                 'Min': lower_bound,
                                                 'Max': upper_bound,
                                                 'Value': np.nan,
                                                 'WOE': bin_data['woes'][i],
                                                 'Beta': self.coefficients_dummy[variable][i],
                                                 'BiXi': self.coefficients_dummy[variable][i],
                                                 'Observations': observations,
                                                 'Bads': bads,
                                                 'Goods': observations - bads,
                                                 'Weighted Observations': observations_weight,
                                                 'Weighted Bads': bads_weight,
                                                 'Weighted Goods': observations_weight - bads_weight})
                sub_set = sub_data[sub_data[variable].isnull()]
                observations = sub_set[target].count()
                bads = sub_set[target].sum()
                observations_weight = sub_set[weight].sum()
                bads_weight = sub_set[sub_set[target] == 1][weight].sum()
                self.scorecard_table.append({'Variable': variable,
                                             'Treatment': 'Dummy Numerical',
                                             'Min': np.nan,
                                             'Max': np.nan,
                                             'Value': 'null',
                                             'WOE': bin_data['nan_woe'],
                                             'Beta': self.coefficients_dummy[variable][len(self.coefficients_dummy[variable])-1],
                                             'BiXi': self.coefficients_dummy[variable][len(self.coefficients_dummy[variable])-1],
                                             'Observations': observations,
                                             'Bads': bads,
                                             'Goods': observations - bads,
                                             'Weighted Observations': observations_weight,
                                             'Weighted Bads': bads_weight,
                                             'Weighted Goods': observations_weight - bads_weight})

        observations = self.data[target].count()
        bads = self.data[target].sum()
        observations_weight = self.data[weight].sum()
        bads_weight = self.data[self.data[target] == 1][weight].sum()

        for variable in self.predictors_linear:
            self.scorecard_table.append({'Variable': variable,
                                         'Treatment': 'Linear Numerical',
                                         'Min': np.nan,
                                         'Max': np.nan,
                                         'Value': 'else',
                                         'WOE': np.nan,
                                         'Beta': self.coefficients_linear[variable],
                                         'BiXi': self.coefficients_linear[variable],
                                         'Observations': observations,
                                         'Bads': bads,
                                         'Goods': observations - bads,
                                         'Weighted Observations': observations_weight,
                                         'Weighted Bads': bads_weight,
                                         'Weighted Goods': observations_weight - bads_weight})

        self.scorecard_table.append({'Variable': '_Intercept',
                                     'Treatment': 'Intercept',
                                     'Min': np.nan,
                                     'Max': np.nan,
                                     'Value': np.nan,
                                     'WOE': np.nan,
                                     'Beta': self.intercept,
                                     'BiXi': self.intercept,
                                     'Observations': observations,
                                     'Bads': bads,
                                     'Goods': observations - bads,
                                     'Weighted Observations': observations_weight,
                                     'Weighted Bads': bads_weight,
                                     'Weighted Goods': observations_weight - bads_weight})
        if weightcol is None:
            scorecard_table_pd = pd.DataFrame.from_records(self.scorecard_table)[['Variable', 'Treatment', 'Min', 'Max', 'Value', 'WOE', 'Beta', 'BiXi', 'Observations', 'Bads', 'Goods']]
            scorecard_table_pd['Bad Rate'] = scorecard_table_pd['Bads'] / scorecard_table_pd['Observations']
            observations_total = self.data[target].count()
            bads_total = self.data[target].sum()
            scorecard_table_pd['Bad Rate relative to population'] = scorecard_table_pd['Bad Rate'] / (bads_total / observations_total)
            scorecard_table_pd['% Observations'] = scorecard_table_pd['Observations'] / observations_total
            scorecard_table_pd['% Bads'] = scorecard_table_pd['Bads'] / bads_total
            scorecard_table_pd['% Goods'] = scorecard_table_pd['Goods'] / (observations_total - bads_total)
            scorecard_table_pd['Lift'] = scorecard_table_pd['% Bads'] / scorecard_table_pd['% Goods']
        else:
            scorecard_table_pd = pd.DataFrame.from_records(self.scorecard_table)[['Variable', 'Treatment', 'Min', 'Max', 'Value', 'WOE', 'Beta', 'BiXi', 'Observations', 'Weighted Observations', 'Weighted Bads', 'Weighted Goods']]
            scorecard_table_pd = scorecard_table_pd.rename({'Observations': 'Raw Obs.', 'Weighted Observations': 'Observations', 'Weighted Bads': 'Bads', 'Weighted Goods': 'Goods'}, axis='columns')
            scorecard_table_pd['Bad Rate'] = scorecard_table_pd['Bads'] / scorecard_table_pd['Observations']
            observations_total = self.data[target].count()
            bads_total = self.data[target].sum()
            scorecard_table_pd['Bad Rate relative to population'] = scorecard_table_pd['Bad Rate'] / (bads_total / observations_total)
            scorecard_table_pd['% Observations'] = scorecard_table_pd['Observations'] / observations_total
            scorecard_table_pd['% Bads'] = scorecard_table_pd['Bads'] / bads_total
            scorecard_table_pd['% Goods'] = scorecard_table_pd['Goods'] / (observations_total - bads_total)
            scorecard_table_pd['Lift'] = scorecard_table_pd['% Bads'] / scorecard_table_pd['% Goods']
        return scorecard_table_pd

    def _transform(self, data):
        # check if in cache
            # compute transformation on needed columns and save to cache
        if self._transformed_data is None:
            self._transformed_data = self.grouping.transform(data, columns_to_transform=self.predictors)
        return self._transformed_data

    def to_blaze(self, ntbOut=True, file=None, output_folder=None):
        """Creates table with the scorecard which can be later imported in Blaze Advisor.

        This version of Blaze output is probaly obsolete. For the newer version, use method to_blaze_rd() instead.

        Args:
            ntbOut (bool, optional): Should the table be returned by the method? Defaults to True.
            file (str, optional): Where the table should be save (csv file path). Defaults to None.
            output_folder (str, optional): Path to save the file into. Defaults to None.

        Returns:
            pd.DataFrame or None: scorecard table to be imported into Blaze.
        """
        blaze_table =  pd.DataFrame(columns = ['Characteristic','Bin','Label','Score','ScorePoints',
                                'Range','Formula','Number_of_Values','Value1','Value2'])

        scorecard_out = self.scorecard_table_simple()
        for r in scorecard_out.itertuples():  
            if r.Treatment in {'Intercept'}:
                blaze_table=blaze_table.append(pd.DataFrame( data= [['Intercept',1,'K00_All',
                                                                 r.BiXi,0,1,'Intercept=\'integer\'',
                                                                 0,'','']],
                                           columns = ['Characteristic','Bin','Label','Score','ScorePoints',
                                                          'Range','Formula','Number_of_Values','Value1','Value2']))
                blaze_table=blaze_table.append(pd.DataFrame( data= [['Intercept',2,'All Other',
                                                                 0,0,1,'Intercept=\'integer\'',
                                                                 0,'','']],
                                           columns = ['Characteristic','Bin','Label','Score','ScorePoints',
                                                          'Range','Formula','Number_of_Values','Value1','Value2']))

        prv_Variable = ''
        prv_Treatment = ''
        prv_BiXi = ''
        prv_Value1 = ''
        Bin = 1
        k=0
        for r in scorecard_out.itertuples():  
            if (r.Variable==prv_Variable):
                prv_Bin=Bin
                Bin+=1
                if (r.BiXi!=prv_BiXi):
                    k+=1  
            else:  
                prv_Bin=Bin
                Bin=1
                k=0
            Number_of_Values=1   
            Value1 = ''
            Value2 = ''
            Formula = ''
            if pd.notnull(r.Value):
                Value1 = r.Value
                Label = 'K'+str(k)+'_{'+str(Value1)+'}'
                Formula = r.Variable + '= \'character\''
            if r.Value=='null':
                Value1 = ''
                Label = 'NA'
                Formula = ''
            if r.Value=='else':
                prv_Value1 = r.Value
                Value1 = ''
                Label = 'All Other'   
                Formula=''
            if pd.notnull(r.Min) and pd.notnull(r.Max) and (r.Min!=-np.inf) and (r.Max!=np.Inf):
                Number_of_Values=2
                Value1 = r.Min
                Value2 = r.Max
                Label = 'K'+str(k)+'_'+str(Value1)+'_'+str(Value2)
                Formula = r.Variable+ ' \'integer1\' <= .. <\'integer2\''
            if pd.notnull(r.Min) and (r.Max==np.inf):
                Value1 = r.Min
                Label = 'K'+str(k)+'_'+str(Value1)
                Formula = r.Variable+ ' >=\'integer\''  
            if pd.notnull(r.Max) and (r.Min==-np.inf):
                Value1 = r.Max
                Label = 'K'+str(k)+'_'+str(Value1)
                Formula = r.Variable+ ' <\'integer\''
            if (r.Variable!=prv_Variable) and (prv_Treatment not in {'Linear Numerical'}):
                if prv_Value1!='else' and prv_Variable != '':
                    blaze_table=blaze_table.append(pd.DataFrame( data= [[prv_Variable,prv_Bin+1,'All Other',
                                                                     0,'','','',
                                                                     '','','']],
                                               columns = ['Characteristic','Bin','Label','Score','ScorePoints',
                                                              'Range','Formula','Number_of_Values','Value1','Value2']))
                prv_Value1 = ''
                prv_Variable = r.Variable
                prv_Treatment = r.Treatment
                prv_BiXi = r.BiXi
            if r.Treatment not in {'Linear Numerical','Intercept'}:
                blaze_table=blaze_table.append(pd.DataFrame( data= [[r.Variable,Bin,Label,
                                                                     r.BiXi,0,1,Formula,
                                                                     Number_of_Values,Value1,Value2]],
                                               columns = ['Characteristic','Bin','Label','Score','ScorePoints',
                                                              'Range','Formula','Number_of_Values','Value1','Value2']))
                prv_BiXi = r.BiXi
            if r.Treatment in {'Linear Numerical'}:
                blaze_table=blaze_table.append(pd.DataFrame( data= [[r.Variable,1,'K00_All',
                                                                 str(r.BiXi)+'*'+r.Variable,0,1,'',
                                                                 0,'','']],
                                           columns = ['Characteristic','Bin','Label','Score','ScorePoints',
                                                          'Range','Formula','Number_of_Values','Value1','Value2']))
                blaze_table=blaze_table.append(pd.DataFrame( data= [[r.Variable,2,'All Other',
                                                                 0,0,1,'',
                                                                 0,'','']],
                                           columns = ['Characteristic','Bin','Label','Score','ScorePoints',
                                                          'Range','Formula','Number_of_Values','Value1','Value2']))
                prv_Value1 = ''
                prv_Variable = r.Variable
                prv_Treatment = r.Treatment
                prv_BiXi = r.BiXi
           
        if file is not None:
            if output_folder is None:
                output_folder = ''
            blaze_table.to_csv('{}/model/{}'.format(output_folder, file), sep=';', index=False) 
            display(HTML('''<a href={} target="_blank">{}</a>'''.format('{}/model/{}'.format(output_folder, file), 'Link to Blaze scorecard <b>{}/model/{}</b>'.format(output_folder, file))))        
        if ntbOut:
            return blaze_table

    def to_blaze_rd(self, ntbOut=True, file=None, output_folder=None):
        """Creates table with the scorecard which can be later imported in Blaze Advisor.

        Args:
            ntbOut (bool, optional): Should the table be returned by the method? Defaults to True.
            file (str, optional): Where the table should be save (csv file path). Defaults to None.
            output_folder (str, optional): Path to save the file into. Defaults to None.

        Returns:
            pd.DataFrame or None: scorecard table to be imported into Blaze.
        """
        blaze_table =  pd.DataFrame(columns = ['Characteristic','Bin','Label','Score',
                                'Range','Formula','Number_of_Values','Values'])

        scorecard_out = self.scorecard_table_simple()
        for r in scorecard_out.itertuples():  
            if r.Treatment in {'Intercept'}:
                blaze_table=blaze_table.append(pd.DataFrame( data= [['Intercept',1,'All Other',
                                                                 str(r.BiXi).replace(',','.'),1,'',0,'']],
                                           columns = ['Characteristic','Bin','Label','Score',
                                                          'Range','Formula','Number_of_Values','Values']))

        prv_Variable = ''
        prv_Treatment = ''
        prv_BiXi = ''
        prv_Value = ''
        Bin = 1
        k=0
        for r in scorecard_out.itertuples():  
            if (r.Variable==prv_Variable):
                prv_Bin=Bin
                Bin+=1
                if (r.BiXi!=prv_BiXi):
                    k+=1  
            else:  
                prv_Bin=Bin
                Bin=1
                k=0
            Number_of_Values=1   
            Values = ''
            Formula = ''
            if pd.notnull(r.Value):
                Values = r.Value
                Label = 'K'+str(k)+'_{'+str(Values)+'}'
                Formula = '= \'string\''
            if r.Value=='null':
                Values = ''
                Label = 'K'+str(k)+'_NA'
                Formula = 'Is unknown'
                Number_of_Values = 0
            if r.Value=='else':
                prv_Value = r.Value
                Values = ''
                Label = 'All Other'   
                Formula = ''
                Number_of_Values = 0
            if pd.notnull(r.Min) and pd.notnull(r.Max) and (r.Min!=-np.inf) and (r.Max!=np.Inf):
                Number_of_Values=2
                Str1 = str(r.Min).replace(',','.')
                Str2 = str(r.Max).replace(',','.')
                Values = Str1+':'+Str2
                Label = 'K'+str(k)+'_'+Str1+'_'+Str2
                Formula = '\'real1\' <= .. < \'real2\''
            if pd.notnull(r.Min) and (r.Max==np.inf):
                Values = str(r.Min).replace(',','.')
                Label = 'K'+str(k)+'_'+Values
                Formula = '>= \'real\''  
            if pd.notnull(r.Max) and (r.Min==-np.inf):
                Values = str(r.Max).replace(',','.')
                Label = 'K'+str(k)+'_'+Values
                Formula = '< \'real\''
            if (r.Variable!=prv_Variable) and (prv_Treatment not in {'Linear Numerical'}):
                if prv_Value!='else' and prv_Variable != '':
                    blaze_table=blaze_table.append(pd.DataFrame( data= [[prv_Variable,prv_Bin+1,'All Other',
                                                                     0,1,'',0,'']],
                                           columns = ['Characteristic','Bin','Label','Score',
                                                          'Range','Formula','Number_of_Values','Values']))
                prv_Value = ''
                prv_Variable = r.Variable
                prv_Treatment = r.Treatment
                prv_BiXi = r.BiXi
            if r.Treatment not in {'Linear Numerical','Intercept'}:
                blaze_table=blaze_table.append(pd.DataFrame( data= [[r.Variable,Bin,Label,
                                                                     str(r.BiXi).replace(',','.'),1,Formula,Number_of_Values,Values]],
                                           columns = ['Characteristic','Bin','Label','Score',
                                                          'Range','Formula','Number_of_Values','Values']))
                prv_BiXi = r.BiXi
            if r.Treatment in {'Linear Numerical'}:
                blaze_table=blaze_table.append(pd.DataFrame( data= [[r.Variable,1,'K0_All',
                                                                     str(r.BiXi).replace(',','.')+'*char.'+r.Variable,1,'Is known',0,'']],
                                           columns = ['Characteristic','Bin','Label','Score',
                                                          'Range','Formula','Number_of_Values','Values']))
                blaze_table=blaze_table.append(pd.DataFrame( data= [[r.Variable,1,'All Other',
                                                                     str(0.0),1,'Is unknown',0,'']],
                                           columns = ['Characteristic','Bin','Label','Score',
                                                          'Range','Formula','Number_of_Values','Values']))
                prv_Value = ''
                prv_Variable = r.Variable
                prv_Treatment = r.Treatment
                prv_BiXi = r.BiXi

        if file is not None:
            if output_folder is None:
                output_folder = ''
            blaze_table.to_csv('{}/model/{}'.format(output_folder, file), sep=';', index=False) 
            display(HTML('''<a href={} target="_blank">{}</a>'''.format('{}/model/{}'.format(output_folder, file), 'Link to Blaze scorecard <b>{}/model/{}</b>'.format(output_folder, file))))        
        if ntbOut:
            return blaze_table
            
    def to_SQL(self, ntbOut=True, file=None, output_folder=None):
        """Generates SQL code that calculates the score on a Oracle SQL database.

        Args:
            ntbOut (bool, optional): Should the code be returned by the method? Defaults to True.
            file (str, optional): Where the code should be save (file name). Defaults to None.
            output_folder (str, optional): Path to save the file into. Defaults to None.

        Returns:
            str: SQL code
        """
        # OUTER PART TRANSFORMING WOE TO BIXI
        scoring_sql_outer = ['select\n1/(1+exp(-s.LINEAR_SCORE)) as SCORE,\ns.*\nfrom (\n    select\n']
        # INNER PART TRANSFORMING VARIABLE TO WOE
        scoring_sql_inner = ['        select\n']
        nullWOE = None
        elseWOE = None
        lastSuffix = ''
        tmp_variable = ''
        for r in self.scorecard_table_simple().itertuples():
            if r.Treatment in {'WOE Categorical', 'WOE Numerical'}:
                suffix = '_WOE'
                inner_coef = r.WOE
                outer_coef = r.Beta
            elif r.Treatment in {'Dummy Categorical', 'Dummy Numerical'}:
                suffix = '_BTA'
                inner_coef = r.Beta
                outer_coef = 1.0
            elif r.Treatment in {'Linear Numerical'}:
                suffix = '_VAL'
                inner_coef = r.Variable
                outer_coef = r.Beta
            elif r.Treatment in {'Intercept'}:
                suffix = ''
                inner_coef = 1.0
                outer_coef = r.Beta
            if r.Variable != tmp_variable:
                if tmp_variable != '':
                    # OUTER PART TRANSFORMING WOE TO BIXI
                    scoring_sql_outer.append('     + ')
                    # INNER PART TRANSFORMING VARIABLE TO WOE
                    if elseWOE is None:
                        elseWOE = nullWOE
                    if elseWOE is None:
                        elseWOE = 0
                    scoring_sql_inner.append('            else ' + str(elseWOE) + '\n        end as ' + str(tmp_variable) + str(lastSuffix) + ',\n')
                else:
                    # OUTER PART TRANSFORMING WOE TO BIXI
                    scoring_sql_outer.append('    ')
                # OUTER PART TRANSFORMING WOE TO BIXI
                scoring_sql_outer.append('w.' + str(r.Variable) + suffix + ' * ' + str(outer_coef) + '\n')
                # INNER PART TRANSFORMING VARIABLE TO WOE
                scoring_sql_inner.append('        case\n')
                tmp_variable = r.Variable
                nullWOE = None
                elseWOE = None
            if r.Value == 'null':
                scoring_sql_inner.append('            when ' + str(r.Variable) + ' is null then ' + str(inner_coef) + '\n')
                nullWOE = inner_coef
                lastSuffix = suffix
            elif r.Value == 'else':
                elseWOE = inner_coef
                lastSuffix = suffix
            elif pd.notnull(r.Value):
                scoring_sql_inner.append('            when ' + str(r.Variable) + ' = "' + str(r.Value) + '" then ' + str(inner_coef) + '\n')
            elif pd.notnull(r.Min):
                if np.isfinite(r.Max):
                    scoring_sql_inner.append('            when ' + str(r.Variable) + ' < ' + str(r.Max) + ' then ' + str(inner_coef) + '\n')
                else:
                    scoring_sql_inner.append('            when ' + str(r.Variable) + ' >= ' + str(r.Min) + ' then ' + str(inner_coef) + '\n')
            elif r.Treatment in {'Linear Numerical','Intercept'}:
                scoring_sql_inner.append('            when 1=1 then ' + str(inner_coef) + '\n')
                lastSuffix = suffix
        # OUTER PART TRANSFORMING WOE TO BIXI
        scoring_sql_outer.append('    as LINEAR_SCORE,\n    w.*\n    from (\n')
        # INNER PART TRANSFORMING VARIABLE TO WOE
        if elseWOE is None:
            elseWOE = nullWOE
        if elseWOE is None:
            elseWOE = 0
        scoring_sql_inner.append('            else ' + str(elseWOE) + '\n        end as ' + str(tmp_variable) + lastSuffix + '\n')
        scoring_sql_inner.append('        from _SOURCETABLENAME_\n')
        scoring_sql_outer = ''.join(scoring_sql_outer)
        scoring_sql_inner = ''.join(scoring_sql_inner)
        scoring_sql_final = scoring_sql_outer + scoring_sql_inner + '    ) w\n) s'
        scoring_sql_final = scoring_sql_final.replace('"', "'").replace('_Intercept', 'Intercept')

        if file is not None:
            if output_folder is None:
                output_folder = ''
            with open('{}/model/{}'.format(output_folder, file), "w", encoding='utf-8') as output_file:
                output_file.write(scoring_sql_final)
            display(HTML('''<a href={} target="_blank">{}</a>'''.format('{}/model/{}'.format(output_folder, file), 'Link to SQL scorecard <b>{}/model/{}</b>'.format(output_folder, file))))
        if ntbOut:
            return scoring_sql_final

    def to_SQL_with_grouping(self, ntbOut=True, file=None, output_folder=None):
        """Generates SQL code that calculates the score on a Oracle SQL database.

        This version outputs not only the final score but also grouping of each variable as result of the SQL query.

        Args:
            ntbOut (bool, optional): Should the code be returned by the method? Defaults to True.
            file (str, optional): Where the code should be save (file name). Defaults to None.
            output_folder (str, optional): Path to save the file into. Defaults to None.

        Returns:
            str: SQL code
        """

        scorecard_out = self.scorecard_table_simple()
        # OUTER PART TRANSFORMING WOE TO BIXI
        scoring_sql_outer = ['select\n1/(1+exp(s.LINEAR_SCORE)) as SCORE,\ns.*\nfrom (\n    select\n']
        # INNER PART TRANSFORMING VARIABLE TO WOE
        scoring_sql_inner = ['        select\n']
        nullWOE = None
        elseWOE = None
        lastSuffix = ''
        tmp_variable = ''
        for r in scorecard_out.itertuples():
            if r.Treatment in {'WOE Categorical', 'WOE Numerical'}:
                suffix = '_WOE'
                inner_coef = r.WOE
                outer_coef = r.Beta
            elif r.Treatment in {'Dummy Categorical', 'Dummy Numerical'}:
                suffix = '_BTA'
                inner_coef = r.Beta
                outer_coef = 1.0
            elif r.Treatment in {'Linear Numerical'}:
                suffix = '_VAL'
                inner_coef = r.Variable
                outer_coef = r.Beta
            elif r.Treatment in {'Intercept'}:
                suffix = ''
                inner_coef = 1.0
                outer_coef = r.Beta
            if r.Variable != tmp_variable:
                if tmp_variable != '':
                    # OUTER PART TRANSFORMING WOE TO BIXI
                    scoring_sql_outer.append('     + ')
                    # INNER PART TRANSFORMING VARIABLE TO WOE
                    if elseWOE is None:
                        elseWOE = nullWOE
                    if elseWOE is None:
                        elseWOE = 0
                    scoring_sql_inner.append('            else ' + str(elseWOE) + '\n        end as ' + str(tmp_variable) + str(lastSuffix) + ',\n')
                else:
                    # OUTER PART TRANSFORMING WOE TO BIXI
                    scoring_sql_outer.append('    ')
                # OUTER PART TRANSFORMING WOE TO BIXI
                scoring_sql_outer.append('w.' + str(r.Variable) + str(inner_coef) + ' * ' + str(outer_coef) + '\n')
                # INNER PART TRANSFORMING VARIABLE TO WOE
                scoring_sql_inner.append('        case\n')
                tmp_variable = r.Variable
                nullWOE = None
                elseWOE = None
            if r.Value == 'null':
                scoring_sql_inner.append('            when ' + str(r.Variable) + ' is null then ' + str(inner_coef) + '\n')
                nullWOE = inner_coef
                lastSuffix = suffix
            elif r.Value == 'else':
                elseWOE = inner_coef
                lastSuffix = suffix
            elif pd.notnull(r.Value):
                scoring_sql_inner.append('            when ' + str(r.Variable) + ' = "' + str(r.Value) + '" then ' + str(inner_coef) + '\n')
            elif pd.notnull(r.Min):
                if np.isfinite(r.Max):
                    scoring_sql_inner.append('            when ' + str(r.Variable) + ' < ' + str(r.Max) + ' then ' + str(inner_coef) + '\n')
                else:
                    scoring_sql_inner.append('            when ' + str(r.Variable) + ' >= ' + str(r.Min) + ' then ' + str(inner_coef) + '\n')
            elif r.Treatment in {'Linear Numerical','Intercept'}:
                scoring_sql_inner.append('            when 1=1 then ' + str(inner_coef) + '\n')
                lastSuffix = suffix
        # OUTER PART TRANSFORMING WOE TO BIXI
        scoring_sql_outer.append('    as LINEAR_SCORE,\n    w.*\n    from (\n')
        # INNER PART TRANSFORMING VARIABLE TO WOE
        if elseWOE is None:
            elseWOE = nullWOE
        if elseWOE is None:
            elseWOE = 0
        scoring_sql_inner.append('            else ' + str(elseWOE) + '\n        end as ' + str(tmp_variable) + str(inner_coef) + '\n')

        # ----------------------------------------------
        #  for grouping
        # ----------------------------------------------
        # OUTER PART TRANSFORMING WOE TO BIXI
        scoring_sql_outer_group = ['']
        # INNER PART TRANSFORMING VARIABLE TO WOE
        scoring_sql_inner_group = ['          --FOR GROUPING \n']
        nullWOE = None
        elseWOE = None
        lastSuffix = ''
        tmp_variable = ''
        k = 0
        for r in scorecard_out.itertuples():
            if r.Treatment in {'WOE Categorical', 'WOE Numerical'}:
                suffix = '_GROUP'
                inner_coef = r.WOE
                outer_coef = r.Beta
            elif r.Treatment in {'Dummy Categorical', 'Dummy Numerical'}:
                suffix = '_GROUP'
                inner_coef = r.Beta
                outer_coef = 1.0
            elif r.Treatment in {'Linear Numerical'}:
                suffix = '_GROUP'
                inner_coef = r.Variable
                outer_coef = r.Beta
            elif r.Treatment in {'Intercept'}:
                suffix = ''
                inner_coef = 1.0
                outer_coef = r.Beta
            k = k + 1
            if r.Variable != tmp_variable:
                k = 0
                if tmp_variable != '':
                    # OUTER PART TRANSFORMING WOE TO BIXI
                    scoring_sql_outer_group.append('     + ')
                    # INNER PART TRANSFORMING VARIABLE TO WOE
                    if elseWOE is None:
                        elseWOE = nullWOE
                    if elseWOE is None:
                        elseWOE = 0
                    scoring_sql_inner_group.append("            else '" + str(elseWOE) + "'\n        end as " + str(tmp_variable) + lastSuffix + '\n')
                else:
                    # OUTER PART TRANSFORMING WOE TO BIXI
                    scoring_sql_outer_group.append('    ')
                # OUTER PART TRANSFORMING WOE TO BIXI
                scoring_sql_outer_group.append('w.' + str(r.Variable) + str(inner_coef) + ' * ' + str(outer_coef) + '\n')
                # INNER PART TRANSFORMING VARIABLE TO WOE
                scoring_sql_inner_group.append('        case\n')
                tmp_variable = r.Variable
                nullWOE = None
                elseWOE = None
            if r.Value == 'null':
                if r.Treatment in {'WOE Categorical', 'Dummy Categorical'}:
                    scoring_sql_inner_group.append('            when ' + str(r.Variable) + " is null then 'null'" + '\n')
                else:
                    scoring_sql_inner_group.append('            when ' + str(r.Variable) + " is null then '" + str(k) + ". null'" + '\n')
                nullWOE = inner_coef
                lastSuffix = suffix
            elif r.Value == 'else':
                elseWOE = inner_coef
                lastSuffix = suffix
            elif pd.notnull(r.Value):
                scoring_sql_inner_group.append('            when ' + str(r.Variable) + " = '" + str(r.Value) + "' then '" + str(inner_coef) + "'\n")
            elif pd.notnull(r.Min):
                if np.isfinite(r.Max):
                    scoring_sql_inner_group.append('            when ' + str(r.Variable) + ' < ' + str(r.Max) + " then '" + str(k) + ". < " + str(r.Max) + "'\n")
                else:
                    scoring_sql_inner_group.append('            when ' + str(r.Variable) + ' >= ' + str(r.Min) + " then '" + str(k) + ". >= " + str(r.Min) + "'\n")
            elif r.Treatment in {'Linear Numerical','Intercept'}:
                scoring_sql_inner_group.append('            when 1=1 then ' + str(inner_coef) + '\n')
                lastSuffix = suffix
        # OUTER PART TRANSFORMING WOE TO BIXI
        scoring_sql_outer_group.clear()
        # INNER PART TRANSFORMING VARIABLE TO WOE
        if elseWOE is None:
            elseWOE = nullWOE
        if elseWOE is None:
            elseWOE = 0
        scoring_sql_inner_group.append('            else ' + str(elseWOE) + '\n        end as ' + str(tmp_variable) + str(inner_coef) + lastSuffix + '\n')
        scoring_sql_inner_group.append('')
        scoring_sql_outer_group = ''.join(scoring_sql_outer_group)
        scoring_sql_inner_group = ''.join(scoring_sql_inner_group)
        scoring_sql_final_group = scoring_sql_outer_group + scoring_sql_inner_group
        scoring_sql_final_group = scoring_sql_final_group.replace('"', "'").replace('_Intercept', 'Intercept')

        scoring_sql_inner.append(scoring_sql_final_group)
        scoring_sql_inner.append('        from _SOURCETABLENAME_\n')
        scoring_sql_outer = ''.join(scoring_sql_outer)
        scoring_sql_inner = ''.join(scoring_sql_inner)
        scoring_sql_final = scoring_sql_outer + scoring_sql_inner + '    ) w\n) s'
        scoring_sql_final = scoring_sql_final.replace('"', "'").replace('_Intercept', 'Intercept')

        if file is not None:
            if output_folder is None:
                output_folder = ''
            with open('{}/model/{}'.format(output_folder, file), "w", encoding='utf-8') as output_file:
                output_file.write(scoring_sql_final)
            display(HTML('''<a href={} target="_blank">{}</a>'''.format('{}/model/{}'.format(output_folder, file), 'Link to SQL VN scorecard <b>{}/model/{}</b>'.format(output_folder, file))))
        if ntbOut:
            return scoring_sql_final

    def to_python(self, ntbOut=True, file=None, output_folder=None):
        """Produces Python code that can be used for scoring.

        Args:
            ntbOut (bool, optional): Should the code be returned by the method? Defaults to True.
            file (str, optional): Where the code should be save (file name). Defaults to None.
            output_folder (str, optional): Path to save the file into. Defaults to None.

        Returns:
            str: Python code
        """
        # OUTER PART TRANSFORMING WOE TO BIXI
        scoring_python_outer = ['\n    LINEAR_SCORE = \\\n']
                                
        # INNER PART TRANSFORMING VARIABLE TO WOE
        scoring_python_inner = ['def score(row):']
        tmp_variable = ''
        nullWOE = None
        elseWOE = None
        lastSuffix = ''
        scorecard_out = self.scorecard_table_simple()

        for r in scorecard_out.itertuples():
            if r.Treatment in {'WOE Categorical', 'WOE Numerical'}:
                suffix = '_WOE'
                inner_coef = r.WOE
                outer_coef = r.Beta
            elif r.Treatment in {'Dummy Categorical', 'Dummy Numerical'}:
                suffix = '_BTA'
                inner_coef = r.Beta
                outer_coef = 1.0
            elif r.Treatment in {'Linear Numerical'}:
                suffix = '_VAL'
                inner_coef = 'row[\'' + str(r.Variable) + '\']'
                outer_coef = r.Beta
            elif r.Treatment in {'Intercept'}:
                suffix = ''
                inner_coef = 1.0
                outer_coef = r.Beta
            if r.Variable != tmp_variable:
                if tmp_variable != '':
                    # OUTER PART TRANSFORMING WOE TO BIXI
                    scoring_python_outer.append(' + \\\n    ')
                    # INNER PART TRANSFORMING VARIABLE TO WOE
                    if elseWOE is None: elseWOE = nullWOE
                    if elseWOE is None: elseWOE = 0
                    scoring_python_inner.append('    else: ' + str(tmp_variable) + lastSuffix + ' = ' + str(elseWOE) + '\n')
                else:
                    # OUTER PART TRANSFORMING WOE TO BIXI
                    scoring_python_outer.append('    ')

                # OUTER PART TRANSFORMING WOE TO BIXI
                scoring_python_outer.append(str(r.Variable) + suffix + ' * ' + str(outer_coef))
                #INNER PART TRANSFORMING VARIABLE TO WOE
                scoring_python_inner.append('\n')
                tmp_variable = r.Variable
                nullWOE = None
                elseWOE = None
                
                scoring_python_inner.append('    if ')
            else:
                if r.Value != 'else':
                    scoring_python_inner.append('    elif ')
                
            if r.Value == 'null':
                scoring_python_inner.append('row[\'' + str(r.Variable) + '\'] != row[\'' + str(r.Variable) + '\']: ' + str(tmp_variable) + suffix + ' = ' + str(inner_coef) + '\n')
                nullWOE = inner_coef
                lastSuffix = suffix
            elif r.Value == 'else':
                elseWOE = inner_coef
                lastSuffix = suffix
            elif pd.notnull(r.Value):
                scoring_python_inner.append('row[\'' + str(r.Variable) + '\'] == "' + str(r.Value) + '": ' + str(tmp_variable) + suffix + ' = ' + str(inner_coef) + '\n')
            elif pd.notnull(r.Min):
                if np.isfinite(r.Max):
                    scoring_python_inner.append('row[\'' + str(r.Variable) + '\'] < ' + str(r.Max) + ': ' + str(tmp_variable) + suffix + ' = ' + str(inner_coef) + '\n')
                else:
                    scoring_python_inner.append('row[\'' + str(r.Variable) + '\'] >= ' + str(r.Min) + ': ' + str(tmp_variable) + suffix + ' = ' + str(inner_coef) + '\n')
            elif r.Treatment in {'Linear Numerical','Intercept'}:
                scoring_python_inner.append('True: ' + str(r.Variable) + suffix + ' = ' + str(inner_coef) + '\n')
                lastSuffix = suffix

        #INNER PART TRANSFORMING VARIABLE TO WOE
        #INNER PART TRANSFORMING VARIABLE TO WOE
        if elseWOE is None: elseWOE = nullWOE
        if elseWOE is None: elseWOE = 0
        scoring_python_inner.append('    else: ' + str(tmp_variable) + lastSuffix + ' = ' + str(elseWOE) + '\n')



        scoring_python_outer.append('\n\n    SCORE = 1-1/(1+np.exp(LINEAR_SCORE))\n') 
        scoring_python_outer.append('\n    return SCORE\n') 

        scoring_python_outer = ''.join(scoring_python_outer)
        scoring_python_inner = ''.join(scoring_python_inner)
        scoring_python_final = scoring_python_inner + scoring_python_outer

        if file is not None:
            if output_folder is None:
                output_folder = ''
            with open('{}/model/{}'.format(output_folder, file), "w", encoding='utf-8') as output_file:
                output_file.write(scoring_python_final)
            display(HTML('''<a href={} target="_blank">{}</a>'''.format('{}/model/{}'.format(output_folder, file), 'Link to Python scorecard <b>{}/model/{}</b>'.format(output_folder, file))))
        if ntbOut:
            return scoring_python_final
