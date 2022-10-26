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

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import numpy as np
from numpy import inf, mean
import pandas as pd
from sklearn.utils import check_array
from IPython.display import display
import gc
import sys
from .utils import is_number, optimize_dtypes


class FeatureEngineeringFromSlice(BaseEstimator, TransformerMixin):
    """
    Args:
        id_name (string):
            ID column name of the datasets that will be used in fit and transform procedures
        time_name (string ):
            TIME ID column name of the datasets that will be used in fit and transform procedures
        metadata (matrix):
            a matrix of metadata - telling which aggregation families should be performed for which column
            columns:
                1) variable name (variable to be used for FE),
                2) aggregation func (e.g. sum)
        agglist (matrix):
            a matrix of aggregation types - defining the aggregation families
            columns: 
                1) minimal max month,
                2) type of aggregation (basic/parametric/varcomb),
                3) 2nd type of aggregation (simple/ratio),
                4) from (...of basic aggregation or numerator of ratio),
                5) to (...of basic aggregation or numerator of ratio),
                6) func (...of basic aggregation or numerator of ratio),
                7) from (...of denominator of ratio),
                8) to (...of denominator of ratio),
                9) func (...of denominator of ratio),
                10) query (...additional condition for basic aggregation or numerator of ratio)
                11) query (...additional condition for denominator of ratio)
                12) suffix (...suffix for the columns using the queries)
        varcomb (matrix, optional):
            a matrix of varible combinations
            columns:
                1) variable for numerator.
                2) variable for denominator
            segm : matrix, optional 
            a matrix of segmentation variables, for which aggregations of type “segmented” should be created
            columns:
                1)	segmentation variable
        max_time (int, optional):
            number of time units that the aggregations are based on
        uppercase_suffix (boolean, optional):
            boolean if the suffix of the aggregation type should be uppercase
        min fill share (decimal, optional):
            share of filled (non-NaN) rows of feature for this feature to be added to the new dataset (if set to 0, all columns will be added. If set to 1, only fully filled columns will be added)

    Attributes:
        X_in (pd.DataFrame):
            input
        X_out (pd.DataFrame):
            output 
        sql_ (string):
            SQL query which makes the same transformation on Oracle database

    Methods:
        fit(X, y = None):
            go through the aggregation metadata and put all valid aggregations into a special structure
            X (pd.DataFrame):
                the dataframe you want the aggregations to be executed on
        transform(X) :
            execute the aggregations 
            X (pd.DataFrame):
                the dataframe you want the aggregations to be executed on  
    """

    def __init__(self, id_name, time_name, metadata, agglist, varcomb=None, segm=None, max_time=6, uppercase_suffix=True, min_fill_share=0):
        # initiation of the instance

        # metadata: a matrix of metadata - telling which aggregation families should be performed for which column:
        # in first column there should be a column name
        # in second column there should be the aggregation
        self.metadata = check_array(metadata, dtype=object, force_all_finite=False)
        self.metadata = pd.DataFrame(self.metadata)
        self.metadata.columns = ['name', 'agg']

        # agglist: a matrix of aggregation types - defining the aggregation families
        # columns: minimal max month, type of aggregation (basic/parametric), 2nd type of aggregation (simple/ratio),
        # from, to, func (...of basic aggregation or numerator of ratio), from, to, func (...of denominator of ratio)
        self.agglist = check_array(agglist, dtype=object, force_all_finite=False)
        self.agglist = pd.DataFrame(self.agglist)
        self.agglist.columns = ['time', 'type', 'type2', 'from1', 'to1',
                                'func1', 'from2', 'to2', 'func2', 'query1', 'query2', 'suffix']

        # varcomb: a matrix for special aggregations combining two varibles
        if varcomb is not None:
            self.varcomb = check_array(varcomb, dtype=object, force_all_finite=False)
            self.varcomb = pd.DataFrame(self.varcomb)
            self.varcomb.columns = ['var1', 'var2']
        else:
            self.varcomb = None
        # segm: a matrix for secial aggregations which are segmented by other variable
        if segm is not None:
            self.segm = check_array(segm, dtype=object, force_all_finite=False)
            self.segm = pd.DataFrame(self.segm)
            self.segm.columns = ['segmentation']
        else:
            self.segm = None

        # max_time: number of time units that the aggregations are based on
        self.max_time = max_time

        # ID and TIME ID column names of the datasets that will be used in fit and transform procedures
        self.id_name = id_name
        self.time_name = time_name

        # boolean if the suffix of the aggregation type should be uppercase
        self.uppercase_suffix = uppercase_suffix

        # share of filled (non-NaN) rows of feature to be added to the new dataset
        self.min_fill_share = min_fill_share

    def __aggregateRatio(self, recentColName, recentFrom, recentTo, recentFunc, oldColName, oldFrom, oldTo, oldFunc, recentQuery=None, oldQuery=None, suffix='', segm_var=None):

        # segmentation parameters if segm_var exists
        if segm_var is not None:
            segm_values = self.X_in[segm_var].astype(str).unique()
            segm_str = []  # segmenting condition in python code
            segm_sql = []  # segmenting condition in sql code
            segm_col = []  # segmenting condition in column name
            for v in segm_values:
                segm_str.append(' & (self.X_in[segm_var].astype(str) == "'+str(v)+'")')
                segm_sql.append(' and '+segm_var+" = '"+str(v)+"'")
                segm_col.append('_'+segm_var+'_'+str(v))
        else:
            segm_str = segm_sql = segm_col = segm_values = ['']

        for i in range(len(segm_values)):

            # if the time period is trivial, add only the number to new variable name, else add there the aggregation and
            # the period from - to
            if recentTo == recentFrom:
                column_name = recentFunc + str(int(recentFrom))
            elif recentTo == inf:
                column_name = recentFunc + str(int(recentFrom)) + '_INF'
            else:
                column_name = recentFunc + str(int(recentFrom)) + '_' + str(int(recentTo))

            if self.uppercase_suffix:
                column_name = column_name.upper()

            # aggregation of the recent time period, group by ID
            if (recentQuery is not None) and (recentQuery != 'XNA'):
                recentData = eval('self.X_in[(self.X_in[self.time_name] >= recentFrom) & (self.X_in[self.time_name] <= recentTo)' + segm_str[i] + ']'
                                  + '.query("' + recentQuery + '")'
                                  + '.groupby([self.id_name])' + '["' + recentColName + '"].' + recentFunc + '()')
            else:
                recentData = eval('self.X_in[(self.X_in[self.time_name] >= recentFrom) & (self.X_in[self.time_name] <= recentTo)' + segm_str[i] + ']'
                                  + '.groupby([self.id_name])' + '["' + recentColName + '"].' + recentFunc + '()')

            # if the time period is trivial, add only the number to new variable name, else add there the aggregation and
            # the period from - to
            if oldTo == oldFrom:
                str2 = oldFunc + str(int(oldFrom))
            elif oldTo == inf:
                str2 = oldFunc + str(int(oldFrom)) + '_INF'
            else:
                str2 = oldFunc + str(int(oldFrom)) + '_' + str(int(oldTo))

            if self.uppercase_suffix:
                str2 = str2.upper()

            # aggregation of the old time period, group by ID
            if (oldQuery is not None) and (oldQuery != 'XNA'):
                oldData = eval('self.X_in[(self.X_in[self.time_name] >= oldFrom) & (self.X_in[self.time_name] <= oldTo)' + segm_str[i] + ']'
                               + '.query("' + oldQuery + '")'
                               + '.groupby([self.id_name])' + '["' + oldColName + '"].' + oldFunc + '()')
            else:
                oldData = eval('self.X_in[(self.X_in[self.time_name] >= oldFrom) & (self.X_in[self.time_name] <= oldTo)' + segm_str[i] + ']'
                               + '.groupby([self.id_name])' + '["' + oldColName + '"].' + oldFunc + '()')

            # join recent and old time period, calculate the ratio
            resultData = pd.concat([recentData, oldData], axis=1)
            resultData.columns = ['recent', 'old']
            if recentColName == oldColName:
                newColName = recentColName + '_' + column_name + '_' + str2 + segm_col[i]
            else:
                newColName = recentColName + '_' + column_name + '_' + oldColName + '_' + str2 + segm_col[i]
            if len(suffix) > 0:
                newColName = newColName + '_' + suffix
            resultData[newColName] = resultData['recent']/resultData['old']

            # SQL string
            if recentFunc == 'mean':
                func_out1 = 'avg'
            else:
                func_out1 = recentFunc
            if oldFunc == 'mean':
                func_out2 = 'avg'
            else:
                func_out2 = oldFunc

            if (oldTo == inf) and (oldQuery is not None) and (oldQuery != 'XNA'):
                sql_tmp_denom = func_out2 + '(case when ' + oldQuery.replace('==', '=') + ' and ' + \
                    self.time_name + '>=' + str(int(oldFrom)) + segm_sql[i] + ' then ' + oldColName + ' end)'
            elif (oldTo == inf):
                sql_tmp_denom = func_out2 + '(case when ' + self.time_name + '>=' + \
                    str(int(oldFrom)) + segm_sql[i] + ' then ' + oldColName + ' end)'
            elif (oldQuery is not None) and (oldQuery != 'XNA'):
                sql_tmp_denom = func_out2 + '(case when ' + oldQuery.replace('==', '=') + ' and ' + self.time_name + '>=' + str(
                    int(oldFrom)) + ' and ' + self.time_name + '<=' + str(int(oldTo)) + segm_sql[i] + ' then ' + oldColName + ' end)'
            else:
                sql_tmp_denom = func_out2 + '(case when ' + self.time_name + '>=' + str(
                    int(oldFrom)) + ' and ' + self.time_name + '<=' + str(int(oldTo)) + segm_sql[i] + ' then ' + oldColName + ' end)'
            if (recentTo == inf) and (recentQuery is not None) and (recentQuery != 'XNA'):
                sql_tmp_numer = func_out1 + '(case when ' + recentQuery.replace('==', '=') + ' and ' + \
                    self.time_name + '>=' + str(int(recentFrom)) + \
                    segm_sql[i] + ' then ' + recentColName + ' end)'
            elif (recentTo == inf):
                sql_tmp_numer = func_out1 + '(case when ' + self.time_name + '>=' + \
                    str(int(recentFrom)) + segm_sql[i] + ' then ' + recentColName + ' end)'
            elif (recentQuery is not None) and (recentQuery != 'XNA'):
                sql_tmp_numer = func_out1 + '(case when ' + recentQuery.replace('==', '=') + ' and ' + self.time_name + '>=' + str(
                    int(recentFrom)) + ' and ' + self.time_name + '<=' + str(int(recentTo)) + segm_sql[i] + ' then ' + recentColName + ' end)'
            else:
                sql_tmp_numer = func_out1 + '(case when ' + self.time_name + '>=' + str(int(recentFrom)) + ' and ' + \
                    self.time_name + '<=' + str(int(recentTo)) + segm_sql[i] + ' then ' + recentColName + ' end)'

            sql_tmp = 'case when ' + sql_tmp_denom + ' <> 0 then ' + sql_tmp_numer + ' / ' + sql_tmp_denom + ' end'

            sql_full = ',' + sql_tmp + ' as ' + newColName + '\n'

            # calculate fill share to compare it with min_fill_share param
            fill_share = resultData.shape[0]/self.X_out.shape[0]

            # add the new column to X_out data
            if fill_share >= self.min_fill_share:
                # add it only if it is not there yet
                if (newColName not in self.X_buff) and (newColName not in self.X_out):
                    self.X_buff = pd.concat([self.X_buff, optimize_dtypes(resultData[newColName].to_frame())], axis=1)
                    self.aggnames_.append(newColName)
                    if i == len(segm_values) - 1:
                        print('Variable', newColName, 'created. (', self.nr_columns_done_print, '/', self.nr_columns_total, ')')
                    else:
                        print('Variable', newColName, 'created.')
                    self.strsql_.append(sql_full)
                else:
                    print('Variable with name', newColName, 'already in the dataset. (', self.nr_columns_done_print, '/', self.nr_columns_total, ')')
            else:
                print('Variable', newColName, 'fill share <', str(self.min_fill_share), '(', self.nr_columns_done_print, '/', self.nr_columns_total, ')')

    def __aggregateSimple(self, colName, aggFrom, aggTo, aggFunc, query=None, suffix='', segm_var=None):

        # segmentation parameters if segm_var exists
        if segm_var is not None:
            segm_values = self.X_in[segm_var].astype(str).unique()
            segm_str = []  # segmenting condition in python code
            segm_sql = []  # segmenting condition in sql code
            segm_col = []  # segmenting condition in column name
            for v in segm_values:
                segm_str.append(' & (self.X_in[segm_var].astype(str) == "'+str(v)+'")')
                segm_sql.append(' and '+segm_var+" = '"+str(v)+"'")
                segm_col.append('_'+segm_var+'_'+str(v))
        else:
            segm_str = segm_sql = segm_col = segm_values = ['']

        for i in range(len(segm_values)):

            # if the time period is trivial, add only the number to new variable name, else add there the aggregation and
            # the period from - to
            if aggTo == aggFrom:
                column_name = aggFunc + str(int(aggFrom))
            elif aggTo == inf:
                column_name = aggFunc + str(int(aggFrom)) + '_INF'
            else:
                column_name = aggFunc + str(int(aggFrom)) + '_' + str(int(aggTo))

            if self.uppercase_suffix:
                column_name = column_name.upper()

            # aggregation of the time period, group by ID
            if (query is not None) and (query != 'XNA'):
                aggData = eval('self.X_in[(self.X_in[self.time_name] >= aggFrom) & (self.X_in[self.time_name] <= aggTo)' + segm_str[i] + ']'
                               + '.query("' + query + '")'
                               + '.groupby([self.id_name])' + '["' + colName + '"].' + aggFunc + '()')
            else:
                aggData = eval('self.X_in[(self.X_in[self.time_name] >= aggFrom) & (self.X_in[self.time_name] <= aggTo)' + segm_str[i] + ']'
                               + '.groupby([self.id_name])' + '["' + colName + '"].' + aggFunc + '()')

            # transform the new column to data frame
            newColName = colName + '_' + column_name + segm_col[i]
            if len(suffix) > 0:
                newColName = newColName + '_' + suffix
            aggData = aggData.to_frame()
            aggData.columns = [newColName]

            # SQL string
            if aggFunc == 'mean':
                func_out = 'avg'
            else:
                func_out = aggFunc

            if (aggTo == inf) and (query is not None) and (query != 'XNA'):
                sql_tmp = func_out + '(case when ' + query.replace('==', '=') + ' and ' + self.time_name + \
                    '>=' + str(int(aggFrom)) + segm_sql[i] + ' then ' + colName + ' end)'
            elif (aggTo == inf):
                sql_tmp = func_out + '(case when ' + self.time_name + '>=' + \
                    str(int(aggFrom)) + segm_sql[i] + ' then ' + colName + ' end)'
            elif (query is not None) and (query != 'XNA'):
                sql_tmp = func_out + '(case when ' + query.replace('==', '=') + ' and ' + self.time_name + '>=' + str(
                    int(aggFrom)) + ' and ' + self.time_name + '<=' + str(int(aggTo)) + segm_sql[i] + ' then ' + colName + ' end)'
            else:
                sql_tmp = func_out + '(case when ' + self.time_name + '>=' + str(int(aggFrom)) + ' and ' + \
                    self.time_name + '<=' + str(int(aggTo)) + segm_sql[i] + ' then ' + colName + ' end)'

            sql_full = ',' + sql_tmp + ' as ' + newColName + '\n'

            # calculate fill share to compare it with min_fill_share param
            fill_share = aggData.shape[0]/self.X_out.shape[0]

            # add the new column to X_out data
            if fill_share >= self.min_fill_share:
                # add it only if it is not there yet
                if (newColName not in self.X_buff) and (newColName not in self.X_out):
                    self.X_buff = pd.concat([self.X_buff, optimize_dtypes(aggData)], axis=1)
                    self.aggnames_.append(newColName)
                    if i == len(segm_values) - 1:
                        print('Variable', newColName, 'created. (', self.nr_columns_done_print, '/', self.nr_columns_total, ')')
                    else:
                        print('Variable', newColName, 'created.')
                    self.strsql_.append(sql_full)
                else:
                    print('Variable with name', newColName, 'already in the dataset. (', self.nr_columns_done_print, '/', self.nr_columns_total, ')')
            else:
                print('Variable', newColName, 'fill share <', str(self.min_fill_share), '(', self.nr_columns_done_print, '/', self.nr_columns_total, ')')

    def __create_omnimeta_table(self):
        # this functions takes generated metadata tables and tries to figure out how many new features will be created based on those tables

        self.omnimeta_ = list()

        # 1a. basic simple features
        for _, a in self.agglist_[(self.agglist_['type'] == 'basic') &
                                  (self.agglist_['type2'] == 'simple') &
                                  ((self.agglist_['time'] <= self.max_time) |
                                   (self.agglist_['time'] == inf))].iterrows():
            for _, m in self.metadata_[(self.metadata_['agg'] == 'basic')].iterrows():
                self.omnimeta_.append({'call': 'aggregateSimple', 'colName': m['name'], 'recentFrom': a['from1'], 'recentTo': a['to1'], 'recentFunc': a['func1'], 'recentQuery': a['query1'], 'suffix': a['suffix'], 'segm_var': None, 'cnt_features': 1, 'origin': '1a', 'oldColName': None, 'oldFrom': None, 'oldTo': None, 'oldFunc': None, 'oldQuery': None
                                       })

        # 1b. basic ratios
        for _, a in self.agglist_[(self.agglist_['type'] == 'basic') &
                                   (self.agglist_['type2'] == 'ratio') &
                                   ((self.agglist_['time'] <= self.max_time) |
                                    (self.agglist_['time'] == inf))].iterrows():
            for _, m in self.metadata_[(self.metadata_['agg'] == 'basic')].iterrows():
                self.omnimeta_.append({'call': 'aggregateRatio', 'colName': m['name'], 'recentFrom': a['from1'], 'recentTo': a['to1'], 'recentFunc': a['func1'], 'oldColName': m['name'], 'oldFrom': a['from2'], 'oldTo': a['to2'], 'oldFunc': a['func2'], 'recentQuery': a['query1'], 'oldQuery': a['query2'], 'suffix': a['suffix'], 'segm_var': None, 'cnt_features': 1, 'origin': '1b'
                                       })

        # 2a. parametric simple aggregations
        for _, a in self.agglist_[(self.agglist_['type'] == 'parametric') &
                                   (self.agglist_['type2'] == 'simple') &
                                   ((self.agglist_['time'] <= self.max_time) |
                                    (self.agglist_['time'] == inf))].iterrows():
            for _, m in self.metadata_[(self.metadata_['agg'] != 'basic')].iterrows():
                if a['func1'] == 'parametric':
                    func1 = m['agg']
                else:
                    func1 = a['func1']
                self.omnimeta_.append({'call': 'aggregateSimple', 'colName': m['name'], 'recentFrom': a['from1'], 'recentTo': a['to1'], 'recentFunc': func1, 'recentQuery': a['query1'], 'suffix': a['suffix'], 'segm_var': None, 'cnt_features': 1, 'origin': '2a', 'oldColName': None, 'oldFrom': None, 'oldTo': None, 'oldFunc': None, 'oldQuery': None
                                       })

        # 2b. parametric ratios
        for _, a in self.agglist_[(self.agglist_['type'] == 'parametric') &
                                   (self.agglist_['type2'] == 'ratio') &
                                   ((self.agglist_['time'] <= self.max_time) |
                                    (self.agglist_['time'] == inf))].iterrows():
            for _, m in self.metadata_[(self.metadata_['agg'] != 'basic')].iterrows():
                if a['func1'] == 'parametric':
                    func1 = m['agg']
                else:
                    func1 = a['func1']
                if a['func2'] == 'parametric':
                    func2 = m['agg']
                else:
                    func2 = a['func2']
                self.omnimeta_.append({'call': 'aggregateRatio', 'colName': m['name'], 'recentFrom': a['from1'], 'recentTo': a['to1'], 'recentFunc': func1, 'oldColName': m['name'], 'oldFrom': a['from2'], 'oldTo': a['to2'], 'oldFunc': func2, 'recentQuery': a['query1'], 'oldQuery': a['query2'], 'suffix': a['suffix'], 'segm_var': None, 'cnt_features': 1, 'origin': '2b'
                                       })

        # 3. variable combination ratios
        if self.varcomb_ is not None:
            for _, a in self.agglist_[(self.agglist_['type'] == 'varcomb') &
                                       (self.agglist_['type2'] == 'ratio') &
                                       ((self.agglist_['time'] <= self.max_time) |
                                        (self.agglist_['time'] == inf))].iterrows():
                for _, v in self.varcomb_.iterrows():
                    self.omnimeta_.append({'call': 'aggregateRatio', 'colName': v['var1'], 'recentFrom': a['from1'], 'recentTo': a['to1'], 'recentFunc': a['func1'], 'oldColName': v['var2'], 'oldFrom': a['from2'], 'oldTo': a['to2'], 'oldFunc': a['func2'], 'recentQuery': a['query1'], 'oldQuery': a['query2'], 'suffix': a['suffix'], 'segm_var': None, 'cnt_features': 1, 'origin': '3'
                                           })

        # 4a. segmented basic simple aggregations
        if self.segm_ is not None:
            for _, a in self.agglist_[(self.agglist_['type'] == 'segmented') &
                                       (self.agglist_['type2'] == 'simple') &
                                       ((self.agglist_['time'] <= self.max_time) |
                                        (self.agglist_['time'] == inf))].iterrows():
                for _, m in self.metadata_[(self.metadata_['agg'] == 'basic')].iterrows():
                    for _, s in self.segm_.iterrows():
                        self.omnimeta_.append({'call': 'aggregateSimple', 'colName': m['name'], 'recentFrom': a['from1'], 'recentTo': a['to1'], 'recentFunc': a['func1'], 'recentQuery': a['query1'], 'suffix': a['suffix'], 'segm_var': s['segmentation'], 'cnt_features': s['cnt_segments'], 'origin': '4a', 'oldColName': None, 'oldFrom': None, 'oldTo': None, 'oldFunc': None, 'oldQuery': None
                                               })

        # 4b. segmented basic ratios
        if self.segm_ is not None:
            for _, a in self.agglist_[(self.agglist_['type'] == 'segmented') &
                                       (self.agglist_['type2'] == 'ratio') &
                                       ((self.agglist_['time'] <= self.max_time) |
                                        (self.agglist_['time'] == inf))].iterrows():
                for _, m in self.metadata_[(self.metadata_['agg'] == 'basic')].iterrows():
                    for _, s in self.segm_.iterrows():
                        self.omnimeta_.append({'call': 'aggregateRatio', 'colName': m['name'], 'recentFrom': a['from1'], 'recentTo': a['to1'], 'recentFunc': a['func1'], 'oldColName': m['name'], 'oldFrom': a['from2'], 'oldTo': a['to2'], 'oldFunc': a['func2'], 'recentQuery': a['query1'], 'oldQuery': a['query2'], 'suffix': a['suffix'], 'segm_var': s['segmentation'], 'cnt_features': s['cnt_segments'], 'origin': '4b'
                                               })

        # convert omnimeta list to pandas dataframe
        self.omnimeta_ = pd.DataFrame(self.omnimeta_)[['call', 'colName', 'recentFrom', 'recentTo', 'recentFunc',
                                                       'oldColName', 'oldFrom', 'oldTo', 'oldFunc', 'recentQuery', 'oldQuery', 'suffix', 'segm_var', 'cnt_features']]

        # deduplication
        self.omnimeta_ = self.omnimeta_.drop_duplicates(inplace=False)

        # sum of feature numbers
        self.nr_columns_total = self.omnimeta_['cnt_features'].sum()

        return

    def fit(self, X, y=None):
        # go through the aggregation metadata and put all valid aggregations into a special structure

        # boolean telling that the dataset is not valid
        cantfit = False

        # check whether the history is long enough and ID and TIME ID columns are present in the X data set
        if self.max_time < self.agglist['time'].min():
            print('Too short history (max_time needs to be >=', self.agglist['time'].min(),
                  'for at least 1 aggrgegation to be valid)!')
            cantfit = True
        if self.id_name not in X:
            print('ID column', self.id_name, 'not present in the dataset!')
            cantfit = True
        if self.time_name not in X:
            print('Time column', self.time_name, 'not present in the dataset!')
            cantfit = True

        # go through each row of metadata and check whether the column name from that row is in the dataset
        if not cantfit:

            self.metadata_ = self.metadata.copy()

            # validity checks of metadata table
            # new column telling us whether the row is valid for the data provided
            self.metadata_['valid'] = 'OK'

            self.metadata_.loc[~self.metadata_['name'].isin(X.columns), 'valid'] = 'name missing in provided dataset'

            # each column (name) mentioned in metadata can have either a special aggregation type (agg) or no no special aggregation (empty field agg)
            # each of the should be base for basic aggregation (defined in agglist) and those with special aggregation also for that
            # for each column name mentioned in meta data, we will duplicate it with aggregation type (agg) = 'basic'
            # then we will keep only the rows with special aggregations and the rows with these 'basic'. the other rows will be deleted
            metadata_basics = self.metadata_.copy()
            metadata_basics['agg'] = 'basic'
            metadata_basics.drop_duplicates(inplace=True)
            self.metadata_ = self.metadata_[pd.notnull(self.metadata_['agg'])].append(
                metadata_basics).sort_values(['name', 'agg']).reset_index().drop(['index'], axis=1)

            # keep only valid rows and deduplicate
            metadata_invalid = self.metadata_[self.metadata_['valid'] != 'OK'].copy()
            self.metadata_ = self.metadata_[self.metadata_['valid'] == 'OK'].drop_duplicates(inplace=False)

            # print invalid columns so user can review them
            if len(metadata_invalid) > 0:
                print('The following rows of column meta data were ommited as they are invalid (reason given in last column):')
                display(metadata_invalid)

            if len(self.metadata_) == 0:
                self.metadata_ = None

            self.agglist_ = self.agglist.copy()

            # validity checks of agglist table
            # new column telling us whether the row is valid for the data provided
            self.agglist_['valid'] = 'OK'

            self.agglist_.loc[self.agglist_['time'] < 1, 'valid'] = 'time < 1'
            self.agglist_.loc[~self.agglist_['type'].isin(
                {'basic', 'parametric', 'varcomb', 'segmented'}), 'valid'] = 'type not in {basic, parametric, varcomb, segmented}'
            self.agglist_.loc[~self.agglist_['type2'].isin(
                {'ratio', 'simple'}), 'valid'] = 'type2 not in {ratio, simple}'
            self.agglist_.loc[self.agglist_['from1'] > self.agglist_['to1'], 'valid'] = 'from1 > to1'
            self.agglist_.loc[(self.agglist_['type2'] == 'ratio') & (
                self.agglist_['from2'] > self.agglist_['to2']), 'valid'] = 'from2 > to2'
            self.agglist_.loc[(self.agglist_['type2'] == 'simple'), 'from2'] = 0
            self.agglist_.loc[(self.agglist_['type2'] == 'simple'), 'to2'] = 0
            self.agglist_.loc[(self.agglist_['type2'] == 'simple'), 'func2'] = 'XNA'
            self.agglist_.loc[(self.agglist_['type2'] == 'simple'), 'query2'] = 'XNA'
            self.agglist_.loc[pd.isnull(self.agglist_['query1']), 'query1'] = 'XNA'
            self.agglist_.loc[pd.isnull(self.agglist_['query2']), 'query2'] = 'XNA'
            self.agglist_.loc[pd.isnull(self.agglist_['suffix']), 'suffix'] = ''

            # keep only valid rows and deduplicate
            agglist_invalid = self.agglist_[self.agglist_['valid'] != 'OK'].copy()
            self.agglist_ = self.agglist_[self.agglist_['valid'] == 'OK'].drop_duplicates(inplace=False)

            # print invalid columns so user can review them
            if len(agglist_invalid) > 0:
                print('The following rows of aggregations meta data were ommited as they are invalid (reason given in last column):')
                display(agglist_invalid)

            if len(self.agglist_) == 0:
                self.agglist_ = None

            if self.varcomb is not None:

                self.varcomb_ = self.varcomb.copy()

                # validity checks of varcomb table
                # new column telling us whether the row is valid for the data provided
                self.varcomb_['valid'] = 'OK'

                self.varcomb_.loc[~self.varcomb_['var1'].isin(X.columns), 'valid'] = 'var1 missing in provided dataset'
                self.varcomb_.loc[~self.varcomb_['var2'].isin(X.columns), 'valid'] = 'var2 missing in provided dataset'

                # keep only valid rows and deduplicate
                varcomb_invalid = self.varcomb_[self.varcomb_['valid'] != 'OK'].copy()
                self.varcomb_ = self.varcomb_[self.varcomb_['valid'] == 'OK'].drop_duplicates(inplace=False)

                # print invalid columns so user can review them
                if len(varcomb_invalid) > 0:
                    print('The following rows of var combination meta data were ommited as they are invalid (reason given in last column):')
                    display(varcomb_invalid)

                if len(self.varcomb_) == 0:
                    self.varcomb_ = None

            else:
                self.varcomb_ = None

            if self.segm is not None:

                self.segm_ = self.segm.copy()

                # validity checks of segm table
                # new column telling us whether the row is valid for the data provided
                self.segm_['valid'] = 'OK'
                self.segm_['cnt_segments'] = 1

                self.segm_.loc[~self.segm_['segmentation'].isin(
                    X.columns), 'valid'] = 'segmentation missing in provided dataset'

                # count segments per segmentation variable
                segment_variables = self.segm_[(self.segm_['segmentation'] != '') & (
                    self.segm_['valid'] == 'OK')]['segmentation'].unique()
                for sv in segment_variables:
                    sc = len(set(X[sv].unique()) - {np.nan})
                    self.segm_.loc[self.segm_['segmentation'] == sv, 'cnt_segments'] = sc
                    if sc < 1:
                        self.segm_.loc[self.segm_['segmentation'] == sv,
                                       'valid'] = 'segmentation empty in provided dataset'

                # keep only valid rows and deduplicate
                segm_invalid = self.segm_[self.segm_['valid'] != 'OK'].copy()
                self.segm_ = self.segm_[self.segm_['valid'] == 'OK'].drop_duplicates(inplace=False)

                # print invalid columns so user can review them
                if len(segm_invalid) > 0:
                    print('The following rows of segmentation meta data were ommited as they are invalid (reason given in last column):')
                    display(segm_invalid)

                if len(self.segm_) == 0:
                    self.segm_ = None

            else:
                self.segm_ = None

            self.__create_omnimeta_table()
            self.omnimeta_['cumsum_features'] = self.omnimeta_['cnt_features'].cumsum()

            if self.nr_columns_total == 0:
                self.omnimeta_ = None
            else:
                print('Expected number of new columns:', self.nr_columns_total)

            #necessary columns to copy in transform() into self.X_in
            self.cols_needed_in_copy = list(set([self.id_name, self.time_name] + \
                [col for col in self.omnimeta_['colName'].unique()] + [col for col in self.omnimeta_['oldColName'].unique()] + \
                [col for col in self.omnimeta_['segm_var'].unique()]) - {'',None,np.nan})

            #RAM usage estimation
            mem_X = X.memory_usage(index = True, deep = True)
            rows_X = X.shape[0]
            entities_X = X[self.id_name].nunique()
            mem_estimate = 0
            for _, feature in self.omnimeta_.iterrows():
                mem_feature = (mem_X[feature['colName']] / rows_X) * entities_X * feature['cnt_features']
                mem_estimate += mem_feature
            mem_estimate += 2 * sys.getsizeof(X[self.cols_needed_in_copy])
            print('Rough upper estimate of memory usage:',round(mem_estimate/1024/1024/1024,3),'GB')

        if not hasattr(self, 'omnimeta_'):
            print('There are no valid aggregations fulfilling given criteria!')

        return

    def transform(self, X, resume = False, resume_from = None):
        # execute the aggregations

        check_is_fitted(self, 'omnimeta_')

        if not resume:

            self.X_in = X
            self.aggnames_ = list()

            # self.X_out = output structure, granularity of id_name
            self.X_out = self.X_in.groupby([self.id_name]).size().to_frame()
            # self.X_buff = buffer for output structure
            self.X_buff = self.X_out.drop(self.X_out.columns, axis=1, inplace=False)

            # SQL string
            self.strsql_ = ['select ' + self.id_name + '\n'
                        ',' + 'count(*) as ORIG_ROWS_COUNT \n']

            if self.uppercase_suffix:
                newColName = 'ORIG_ROWS_COUNT'
            else:
                newColName = 'orig_rows_count'
            self.X_out.columns = [newColName]
            self.aggnames_.append(newColName)
            print('Variable', newColName, 'created.')

            # self.nr_columns_done = number of features already processed
            self.nr_columns_done = 0
            self.nr_columns_done_print = 0

        if (resume) and (not resume_from):
            self.resume_from = self.nr_columns_done
        elif resume:
            self.resume_from = resume_from-1

        # for cycle through all aggregations (omnimeta_ rows)
        for _, o in self.omnimeta_.iterrows():

            if (not resume) or (resume and o['cumsum_features'] > self.resume_from):

                self.nr_columns_done_print = o['cumsum_features']

                try:

                    # a. simple aggregation
                    if o['call'] == 'aggregateSimple':
                        self.__aggregateSimple(o['colName'], o['recentFrom'], o['recentTo'],
                                               o['recentFunc'], o['recentQuery'], o['suffix'], o['segm_var'])

                    # b. ratio aggregation
                    elif o['call'] == 'aggregateRatio':
                        self.__aggregateRatio(o['colName'], o['recentFrom'], o['recentTo'], o['recentFunc'], o['oldColName'],
                                              o['oldFrom'], o['oldTo'], o['oldFunc'], o['recentQuery'], o['oldQuery'], o['suffix'], o['segm_var'])

                except:

                    print('Problem occurred creating column given by following parameters:', {'name/var1': o['colName'], 'from1': o['recentFrom'], 'to1': o['recentTo'], 'query1': o['recentQuery'],
                                                                                              'name/var2': o['oldColName'], 'from2': o['oldFrom'], 'to2': o['oldTo'], 'query2': o['oldQuery'],
                                                                                              'segmentation': o['segm_var']})

                # if at least 50 columns in buffer, move data from buffer to output
                if len(self.X_buff.columns) >= 50:
                    self.X_out = pd.concat([self.X_out, self.X_buff], axis=1)
                    self.X_buff.drop(self.X_buff.columns, axis=1, inplace=True)

                    gc.collect()

                self.nr_columns_done = o['cumsum_features']

        # move the rest of the data from buffer to the output
        if len(self.X_buff.columns) > 0:
            self.X_out = pd.concat([self.X_out, self.X_buff], axis=1)
            self.X_buff.drop(self.X_buff.columns, axis=1, inplace=True)

        gc.collect()

        # SQL string
        if self.max_time == inf:
            self.strsql_.append('from _TABLENAME_\n' +
                             'group by ' + self.id_name + '\n')
        else:
            self.strsql_.append('from _TABLENAME_\n' +
                             'where ' + self.time_name + ' <= ' + str(int(self.max_time)) + '\n' +
                             'group by ' + self.id_name + '\n')
        self.strsql_ = ''.join(self.strsql_)

        return self.X_out
