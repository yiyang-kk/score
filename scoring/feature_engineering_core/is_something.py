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
from numpy import inf, mean
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from sklearn.utils import check_array
import gc
import sys
from IPython.display import display
import numbers
from .utils import is_number, optimize_dtypes

class IsSomething(BaseEstimator, TransformerMixin):
    """
    Args:
        id_name (string):
            ID column name of the datasets that will be used in fit and transform procedures
        time_name (string):
            TIME ID column name of the datasets that will be used in fit and transform procedures
        issomemeta (matrix):
            a matrix of aggregation types - defining the aggregation families
            columns: 
                1) column - name of column the condition is based on
                2) min order - minimal time order index which the condition is calculated for
                3) max order - maximal time order index which the condition is calculated for
                4) mid order - if filled, ratios of the occurencies condition fulfilled in intervals [min order,mid order] and (mid order,max order] are calculated
                5) threshold - if filled, counts of how many time the condition was fulfilled at least threshold-times in a row are calculated
                6) condition type - can be: 
                    a. asc (column values were ascending in time)
                    b. desc (column values were descending in time)
                    c. notasc (column values were not ascending in time)
                    d. notdesc (column values were not descending in time)
                    e. > (column values were greater than condition value)
                    f. >= (column values were greater than or equal to condition value)
                    g. < (column values were lower than condition value)
                    h. <= (column values were lower than or equal to condition value)
                    i. = (column values were equal to condition value)
                7) condition value - value for conditin types >,>=,<,<=,=
        max_time (int, optional):
            number of time units that the aggregations are based on. if max_order in issomemeta is greater than this, this overrides it. infinity by default
        uppercase_suffix (boolean, optional):
            boolean if the suffix of the aggregation type should be uppercase
            
    Attributes:
        X_in (pd.DataFrame)
            input
        X_out (pd.DataFrame)
            output 
            
    Methods:
        fit(X, y = None) :
            go through the issomething metadata and put all valid aggregations into a special structure
            X (pd.DataFrame):
                the dataframe you want the issomething aggregations to be executed on
        transform(X) :
            execute the issomething aggregations 
            X (pd.DataFrame):
                the dataframe you want the issomething aggregations to be executed on  
    """ 
    
    def __init__(self, id_name, time_name, issomemeta, max_time = np.inf, uppercase_suffix = True):
    # initiation of the instance
    
        # a matix of metadata telling what will be the columns and conditions the aggregations will be created from
        self.issomemeta = check_array(issomemeta, dtype=object, force_all_finite=False)
        self.issomemeta = pd.DataFrame(self.issomemeta)
        self.issomemeta.columns = ['column','min order','max order','mid order','threshold','condition type','condition value']
        
        # name of column with application ID
        self.id_name = id_name
        
        # name of column with time (aggregated time, i.e. slice index)
        self.time_name = time_name
        
        # number of time units that the aggregations are based on
        self.max_time = max_time
        
        # boolean if the suffix of the aggregation type should be uppercase
        self.uppercase_suffix = uppercase_suffix
    
    def fit(self, X, y=None):
    # go through the IsSomething metadata and put all valid aggregations into a special structure
          
        # boolean telling that the dataset is not valid
        cantfit = False
        
        # check whether the id_name and time_name columns are present in the X dataset
        if self.id_name not in X:
            print('ID column',self.id_name,'not present in the dataset!')
            cantfit = True
        if self.time_name not in X:
            print('Time column',self.time_name,'not present in the dataset!')
            cantfit = True
        
        # go through each row of metadata and check whether the column name from that row is in the dataset    
        if not cantfit:
            self.issomemeta_ = self.issomemeta.copy()
            
            # do some basic validity checks of metadata tables
            # new column telling us whether the row is valid for the data provided
            self.issomemeta_['valid'] = 'OK'
            # new column telling us how many new features will be created thanks to this one metadata row
            self.issomemeta_['cnt_features'] = 9
            
            self.issomemeta_.loc[~self.issomemeta_['column'].isin(X.columns), 'valid'] = 'column missing in provided dataset'
            self.issomemeta_.loc[self.issomemeta_['min order'] > self.issomemeta_['max order'], 'valid'] = 'min order > max order'
            self.issomemeta_.loc[pd.isnull(self.issomemeta_['min order']), 'min order'] = 1
            self.issomemeta_.loc[pd.isnull(self.issomemeta_['max order']), 'max order'] = np.inf
            self.issomemeta_.loc[(pd.notnull(self.issomemeta_['mid order'])) & (self.issomemeta_['min order'] > self.issomemeta_['mid order']), 'valid'] = 'min order > mid order'
            self.issomemeta_.loc[(pd.notnull(self.issomemeta_['mid order'])) & (self.issomemeta_['mid order'] > self.issomemeta_['max order']), 'valid'] = 'mid order > max order'
            self.issomemeta_.loc[~self.issomemeta_['condition type'].isin({'>=','>','<=','<','=','==','asc','desc','notasc','notdesc'}), 'valid'] = 'invalid condition type'
            self.issomemeta_.loc[(self.issomemeta_['condition type'].isin({'>=','>','<=','<','=','=='})) & (pd.isnull(self.issomemeta_['condition value'])), 'valid'] = 'missing condition value'
            self.issomemeta_.loc[pd.notnull(self.issomemeta_['mid order']), 'cnt_features'] = self.issomemeta_.loc[pd.notnull(self.issomemeta_['mid order']), 'cnt_features'] + 2
            self.issomemeta_.loc[pd.notnull(self.issomemeta_['threshold']), 'cnt_features'] = self.issomemeta_.loc[pd.notnull(self.issomemeta_['threshold']), 'cnt_features'] + 4
                                 
            # keep only valid rows and deduplicate
            issomemeta_invalid = self.issomemeta_[self.issomemeta_['valid'] != 'OK'].drop(['cnt_features'],axis=1,inplace=False)
            self.issomemeta_ = self.issomemeta_[self.issomemeta_['valid'] == 'OK'].drop_duplicates(inplace=False)             
            
            # expected number of new features is sum of expected number of new features of each row
            self.nr_columns_total = self.issomemeta_['cnt_features'].sum()
            self.issomemeta_['cumsum_features'] = self.issomemeta_['cnt_features'].cumsum()
                    
            if self.nr_columns_total == 0:       
                self.issomemeta_ = None
            else:
                print('Expected number of new columns:',self.nr_columns_total)
                
            # print invalid columns so user can review them
            if len(issomemeta_invalid) > 0:
                print('The following rows of issomething meta data were ommited as they are invalid (reason given in last column):')
                display(issomemeta_invalid)  

            #necessary columns to copy in transform() into self.X_in
            self.cols_needed_in_copy = list(set([self.id_name, self.time_name] + [col for col in self.issomemeta_['column'].unique()]) - {'',None,np.nan})

            #RAM usage estimation
            entities_X = X[self.id_name].nunique()
            mem_estimate = 0
            for _, feature in self.issomemeta_.iterrows():
                mem_feature = 8 * entities_X * feature['cnt_features']
                mem_estimate += mem_feature
            mem_estimate += 2 * sys.getsizeof(X[(X[self.time_name] >= 1) & (X[self.time_name] <= self.max_time)][self.cols_needed_in_copy])
            print('Rough upper estimate of memory usage:',round(mem_estimate/1024/1024/1024,3),'GB')
                
        if not (hasattr(self, 'issomemeta_')):
            print('There are no valid aggregations fulfilling given criteria!')
        
        return
    
    def __isSomethingCols(self,columnName,conditionType,conditionValue,timeFrom,timeTo,threshold,timeMiddle):
    # creates boolean columns indicating for which row the condition (given by conditionType and conditionValue)
    # is fulfilled for column columnName
    
        newData = self.X_in[[self.id_name,self.time_name,columnName]].copy() \
                            .sort_values([self.id_name,self.time_name],ascending=[True,False])
                            
        X_tmp = self.X_out.drop(self.X_out.columns,axis=1,inplace=False)
        
        # fix timeTo and timeMiddle for low max_time
        if timeTo > self.max_time:
            timeTo = self.max_time
        if (timeMiddle is not None) and (pd.notnull(timeMiddle)) and (timeMiddle >= timeTo):
            timeMiddle = None
        
        if conditionType in set(['asc','desc','notasc','notdesc']):
    
            # the original columnName is transformed to difference from previous time
            # so ascending/descending condition is easily calculated as signature of such difference
            newData['diff'] = newData.groupby([self.id_name])[columnName].diff()
            
            if conditionType == 'asc': 
                conditionType2 = '>'
            elif conditionType == 'desc': 
                conditionType2 = '<'
            elif conditionType == 'notasc': 
                conditionType2 = '<='
            elif conditionType == 'notdesc': 
                conditionType2 = '>='
            conditionValue = '0'
            condition = "newData['diff']" + str(conditionType2) + str(conditionValue)
            
            conditionName = conditionType
        else:
            if conditionType == '=':
                conditionType2 = '=='
            else:
                conditionType2 = conditionType
            condition = "newData['" + str(columnName) + "']" + str(conditionType2) + str(conditionValue)
            
            mapping = {'==':'e', '<=':'le', '<':'l', '>=':'ge', '>':'g'}
            conditionName = conditionType2
            for key in mapping:
                conditionName = conditionName.replace(key, mapping[key])
            if conditionValue.is_integer(): 
                conditionName = conditionName + str(int(conditionValue))
            else:
                conditionName = conditionName + str(conditionValue)
       
        # evaluate condition - transform vector to boolean values, and transform them to 0/1 values
        newData['is_something'] = eval(condition)*1
        newData['not_something'] = 1-newData['is_something']
        
        # and get only period we are interested in
        newData = newData[(newData[self.time_name] >= timeFrom) & (newData[self.time_name] <= timeTo)].copy()
        
        # vector of previous values: first value is np.nan (as there is no previous value)
        newData['lag_is_something'] = newData.groupby([self.id_name])['is_something'].shift(+1)
        
        # did something change compared to previous element: first element is always changed
        newData['is_something_changed'] = (newData['is_something'] != newData['lag_is_something'])*1
        
        # sequence number of change
        newData['flip_sequence'] = newData.groupby([self.id_name])['is_something_changed'].cumsum()
        
        # aggregate the sequence change
        # for each sequence number, we would like to get whether it is True/False and what is its length
        flip_agg = newData.groupby([self.id_name,'flip_sequence']) \
            .aggregate({'flip_sequence':['count'],'is_something':['max']})
        flip_agg.columns = ('length','val')
        flip_agg['cum_dist'] = flip_agg.groupby([self.id_name])['length'].cumsum()
        flip_agg['cum_dist'] = - flip_agg['cum_dist'] + flip_agg.groupby([self.id_name])['length'].sum()    
        
        # name of new columns - this part will be common for all of them
        if np.isinf(timeTo): timeTo = 'inf'
        else: timeTo = int(timeTo)
        baseName = str(columnName) + '_' + str(int(timeFrom)) + '_' + str(timeTo) + '_' + str(conditionName) + '_'
        
        # Sum Is Something {x, y} (Number of months with balance > $0 in last 12 months)
        colName = baseName + 'sumis'
        if self.uppercase_suffix: colName = colName.upper()
        if (colName not in X_tmp) and (colName not in self.X_buff) and (colName not in self.X_out):
            X_tmp[colName] = flip_agg[flip_agg['val'] == 1].groupby([self.id_name])['length'].sum()
            X_tmp.loc[pd.isnull(X_tmp[colName]),colName] = 0
            self.aggnames_.append(colName)
            print('Variable',colName,'created.')
        colName = baseName + 'sumnot'
        if self.uppercase_suffix: colName = colName.upper()
        if (colName not in X_tmp) and (colName not in self.X_buff) and (colName not in self.X_out):
            X_tmp[colName] = flip_agg[flip_agg['val'] == 0].groupby([self.id_name])['length'].sum()
            X_tmp.loc[pd.isnull(X_tmp[colName]),colName] = 0
            self.aggnames_.append(colName)
            print('Variable',colName,'created.')
        
        # Max Consecutive Is Something {x, y} (Maximum number of consecutive months with balance > $0 in last 12 months)
        colName = baseName + 'maxis'
        if self.uppercase_suffix: colName = colName.upper()
        if (colName not in X_tmp) and (colName not in self.X_buff) and (colName not in self.X_out):
            X_tmp[colName] = flip_agg[flip_agg['val'] == 1].groupby([self.id_name])['length'].max()
            X_tmp.loc[pd.isnull(X_tmp[colName]),colName] = 0
            self.aggnames_.append(colName)
            print('Variable',colName,'created.')
        colName = baseName + 'maxnot'
        if self.uppercase_suffix: colName = colName.upper()
        if (colName not in X_tmp) and (colName not in self.X_buff) and (colName not in self.X_out):
            X_tmp[colName] = flip_agg[flip_agg['val'] == 0].groupby([self.id_name])['length'].max()
            X_tmp.loc[pd.isnull(X_tmp[colName]),colName] = 0
            self.aggnames_.append(colName)
            print('Variable',colName,'created.')
        
        # Avg Length of Consecutive Is Something {x, y} (Average length of cosecutive months 
        # with balance >$0 in last 12 months)
        colName = baseName + 'meanis'
        if self.uppercase_suffix: colName = colName.upper()
        if (colName not in X_tmp) and (colName not in self.X_buff) and (colName not in self.X_out):
            X_tmp[colName] = flip_agg[flip_agg['val'] == 1].groupby([self.id_name])['length'].mean()
            X_tmp.loc[pd.isnull(X_tmp[colName]),colName] = 0
            self.aggnames_.append(colName)
            print('Variable',colName,'created.')
        colName = baseName + 'meannot'
        if self.uppercase_suffix: colName = colName.upper()
        if (colName not in X_tmp) and (colName not in self.X_buff) and (colName not in self.X_out):
            X_tmp[colName] = flip_agg[flip_agg['val'] == 0].groupby([self.id_name])['length'].mean()
            X_tmp.loc[pd.isnull(X_tmp[colName]),colName] = 0
            self.aggnames_.append(colName)
            print('Variable',colName,'created.')
        
        # Consecutive Is Something from Current {x, y}   (Number of consecutive months
        # with balance $>0 in last 12 months, starting with current month)
        maxconfixval = flip_agg.loc[(flip_agg['cum_dist']==0)].groupby([self.id_name])['val'].min()
        maxconfix = flip_agg.loc[(flip_agg['cum_dist']==0)].groupby([self.id_name])['length'].min()
        colName = baseName + 'maxconisfix'
        if self.uppercase_suffix: colName = colName.upper()
        if (colName not in X_tmp) and (colName not in self.X_buff) and (colName not in self.X_out):
            X_tmp[colName] = maxconfixval * maxconfix
            X_tmp.loc[pd.isnull(X_tmp[colName]),colName] = 0
            self.aggnames_.append(colName)
            print('Variable',colName,'created.')
        colName = baseName + 'maxconnotfix'
        if self.uppercase_suffix: colName = colName.upper()
        if (colName not in X_tmp) and (colName not in self.X_buff) and (colName not in self.X_out):
            X_tmp[colName] = (1-maxconfixval) * maxconfix
            X_tmp.loc[pd.isnull(X_tmp[colName]),colName] = 0
            self.aggnames_.append(colName)
            print('Variable',colName,'created.')
        
        if threshold is not None and pd.notnull(threshold):
            # Number of Instances of Consecutive Not Something {x, y} is greater than z (Number of cases where
            # Balance =$0 for more than or equal 3 consecutive months in last 12 months)
            colName = baseName + 'conis' + str(int(threshold))
            if self.uppercase_suffix: colName = colName.upper()
            if (colName not in X_tmp) and (colName not in self.X_buff) and (colName not in self.X_out):
                X_tmp[colName] = flip_agg[(flip_agg['val'] == 1) & (flip_agg['length'] > threshold)].\
                    groupby([self.id_name])['length'].count()
                X_tmp.loc[pd.isnull(X_tmp[colName]),colName] = 0
                self.aggnames_.append(colName)
                print('Variable',colName,'created.')
            colName = baseName + 'connot' + str(int(threshold))
            if self.uppercase_suffix: colName = colName.upper()
            if (colName not in X_tmp) and (colName not in self.X_buff) and (colName not in self.X_out):
                X_tmp[colName] = flip_agg[(flip_agg['val'] == 0) & (flip_agg['length'] > threshold)].\
                    groupby([self.id_name])['length'].count()
                X_tmp.loc[pd.isnull(X_tmp[colName]),colName] = 0
                self.aggnames_.append(colName)
                print('Variable',colName,'created.')
                
            #Distance from last Consecutive Is Something {x, y} greater than z (Distance in months from the last
            # occurence of Balance >$0 for more than 3 consecutive months in last 12 months)
            colName = baseName + 'distconis' + str(int(threshold))
            if self.uppercase_suffix: colName = colName.upper()
            if (colName not in X_tmp) and (colName not in self.X_buff) and (colName not in self.X_out):
                X_tmp[colName] = flip_agg[(flip_agg['val'] == 1) & (flip_agg['length'] > threshold)].\
                    groupby([self.id_name])['cum_dist'].min()
                X_tmp.loc[pd.isnull(X_tmp[colName]),colName] = 0
                self.aggnames_.append(colName)
                print('Variable',colName,'created.')
            colName = baseName + 'distconnot' + str(int(threshold))
            if self.uppercase_suffix: colName = colName.upper()
            if (colName not in X_tmp) and (colName not in self.X_buff) and (colName not in self.X_out):
                X_tmp[colName] = flip_agg[(flip_agg['val'] == 0) & (flip_agg['length'] > threshold)].\
                    groupby([self.id_name])['cum_dist'].min()
                X_tmp.loc[pd.isnull(X_tmp[colName]),colName] = 0
                self.aggnames_.append(colName)
                print('Variable',colName,'created.')
        
        if timeMiddle is not None and pd.notnull(timeMiddle):
            # Ratio Sum Is Something {x, s} and Sum Is Something {s, y} (Percentage of months where Balance >$0 
            # in last 12 months)
            colName = baseName + 'r' + str(int(timeMiddle)) + '_is'
            if self.uppercase_suffix: colName = colName.upper() 
            if (colName not in X_tmp) and (colName not in self.X_buff) and (colName not in self.X_out):
                X_tmp[colName] = newData[(newData[self.time_name] >= timeFrom) & (newData[self.time_name] <= timeMiddle)].\
                    groupby([self.id_name])['is_something'].sum() / \
                    newData[(newData[self.time_name] > timeMiddle) & (newData[self.time_name] <= timeTo)].\
                    groupby([self.id_name])['is_something'].sum()
                self.aggnames_.append(colName)
                print('Variable',colName,'created.')
            
            colName = baseName + 'r' + str(int(timeMiddle)) + '_not' 
            if self.uppercase_suffix: colName = colName.upper()
            if (colName not in X_tmp) and (colName not in self.X_buff) and (colName not in self.X_out):
                X_tmp[colName] = newData[(newData[self.time_name] >= timeFrom) & (newData[self.time_name] <= timeMiddle)].\
                    groupby([self.id_name])['not_something'].sum() / \
                    newData[(newData[self.time_name] > timeMiddle) & (newData[self.time_name] <= timeTo)].\
                    groupby([self.id_name])['not_something'].sum()
                self.aggnames_.append(colName)
                print('Variable',colName,'created.')
            
        # Ratio of Is Something {x,y} and Number of Records
        colName = baseName + 'r' + '_sumis'
        if self.uppercase_suffix: colName = colName.upper()
        if (colName not in X_tmp) and (colName not in self.X_buff) and (colName not in self.X_out):
            X_tmp[colName] = flip_agg[flip_agg['val'] == 1].groupby([self.id_name])['length'].sum() / \
                newData.groupby([self.id_name])['is_something'].count()
            self.aggnames_.append(colName)
            print('Variable',colName,'created. (',self.nr_columns_done_print,'/',self.nr_columns_total,')')
            
        # data types optimization to save RAM
        self.X_buff = pd.concat([self.X_buff,optimize_dtypes(X_tmp)], axis = 1)
            
        return
                    
    def transform(self, X, resume = False, resume_from = None):
    # execute the IsSomething aggregations
    
        check_is_fitted(self, ['issomemeta_'])

        if not resume:

            self.X_in = X[(X[self.time_name] >= 1) & (X[self.time_name] <= self.max_time)][self.cols_needed_in_copy].copy()
            self.aggnames_ = list()
        
            self.X_out = self.X_in.groupby([self.id_name]).size().to_frame()
            if self.uppercase_suffix: newColName = 'ORIG_ROWS_COUNT'
            else: newColName = 'orig_rows_count'
            self.X_out.columns = [newColName]
            self.aggnames_.append(newColName)
        
            # self.X_buff = buffer which will be used to temporarily add new columns before they are all joined to X_out
            self.X_buff = self.X_out.drop(self.X_out.columns,axis=1,inplace=False)
        
            # self.nr_columns_done = count of new columns that have been already created
            self.nr_columns_done = 0
            self.nr_columns_done_print = 0

        if (resume) and (not resume_from):
            self.resume_from = self.nr_columns_done
        elif resume:
            self.resume_from = resume_from-1
        
        for _, i in self.issomemeta_.iterrows():

            if (not resume) or (resume and i['cumsum_features'] > self.resume_from):

                self.nr_columns_done_print = i['cumsum_features']

                try:

                    self.__isSomethingCols(i['column'],i['condition type'],i['condition value'],i['min order'],i['max order'],i['threshold'],i['mid order'])

                except:

                    print('Problem occurred creating column given by following parameters:', {'column': i['column'], 'min order': i['min order'], 'max order': i['max order'], 'mid order': i['mid order'],
                                                                                              'threshold': i['threshold'], 'condition type': i['condition type'], 'condition value': i['condition value']})

                # if at least 50 columns in buffer, move data from buffer to output and opitmize data types
                if len(self.X_buff.columns) >= 50:
                    self.X_out = pd.concat([self.X_out, self.X_buff], axis=1)
                    self.X_buff.drop(self.X_buff.columns, axis=1, inplace=True)
                
                gc.collect()

                self.nr_columns_done = i['cumsum_features']
            
        if len(self.X_buff.columns) > 0:
            self.X_out = pd.concat([self.X_out, self.X_buff], axis=1)
            self.X_buff.drop(self.X_buff.columns, axis=1, inplace=True)

        del self.X_in
        
        gc.collect()
        
        self.strsql_ = 'This class does not support SQL string creation.'
        
        return self.X_out
        
   