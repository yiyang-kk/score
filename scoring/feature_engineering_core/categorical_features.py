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



class CategoricalFeatures(BaseEstimator, TransformerMixin):
    """
    Args:
        id_name (string): 
            ID column name of the datasets that will be used in fit and transform procedures
        from type (string):
            can be either 'slice' or 'raw' determining whether the features are created from pre-aggregated slices (i.e. after Slicer) or from raw-granularity data (i.e. after OrderAssigner)
        catmeta (matrix):
            matrix of slice metadata - defining how the raw data set should be sliced, columns: 
            1)	variable: name of categorical variable the features will be based on
            2)	aggregation type: mode, nunique, last, first, argmax, argmaxsum, argmaxmean, argmin, argminsum, argminmean, nchanges, tschange
            3)	from: minimal time index that will be taken into account for the aggragation calculation
            4)	to: maxinal time index that will be taken into account for the aggragation calculation
            5)	nancategorical: whether NaN is a separate category or whether such rows should not be used
            6)	metric: for argmax/argmin/argmaxsum/argminsum/argmaxmean/argminmean type of aggregations, this is the metric the max/sum/mean is calculated from
            7)	granularity: for tschange aggregation (when calculated from raw data), this is the time unit the time since is calculated in
        slice_name (string):
            time index (from Slicer or Order Assigner)
        time_name (string, optional ):
            from_type = 'raw', this should be time of the transaction
            defaults to None.
        time_max_name (string, optional):
            for from_type = 'raw', time dimension of the ID (e.g. in the final aggregations granularity)
            defaults to None.
        time_format (string, optional):
            the format of time columns
            defaults to '%Y-%m-%d'.
        uppercase_suffix (boolean, optional):
            boolean if the suffix of the aggregation type should be uppercase
            defaults to True. 

    Attributes:
        X_in (pd.DataFrame):
            input
        X_out (pd.DataFrame):
            output 
        sql_ (string):
            SQL query which makes the same transformation on Oracle database
            
    Methods:
        fit(X, y = None):
            go through the aggregation metadata and put all valid aggregations into a special structure.
            X (pd.DataFrame):
                the dataframe you want the aggregations to be executed on
        transform(X):
            execute the time since aggregations   
            X (pd.DataFrame):
                the dataframe you want the aggregations to be executed on
    """
    
    def __init__(self, id_name, from_type, catmeta, slice_name, time_name=None, time_max_name=None, time_format = '%Y-%m-%d', uppercase_suffix = True):
    # initiation of the instance
        
        # id_name: ID column name of the dataset that will be used in fit and transform procedures
        self.id_name = id_name
        
        # from_type: can be either 'slice' or 'raw' determining whether the features are created from pre-aggregated slices (i.e. after Slicer) or from raw-granularity data (i.e. after OrderAssigner)
        self.from_type = from_type
        
        # catmeta: matrix of categorical features metadata
        self.catmeta = check_array(catmeta, dtype=object, force_all_finite=False)
        self.catmeta = pd.DataFrame(self.catmeta)
        self.catmeta.columns = ['variable','aggregation','from','to','nancategorical','metric','granularity']
        
        # slice_name: time index (from Slicer or Order Assigner)
        self.slice_name = slice_name
            
        # time_name: for from_type = 'raw', this should be time of the transaction, for from_type = 'slice'
        if self.from_type == 'slice':
            self.time_name = 'slices'
        else:
            self.time_name = time_name
            
        # time_max_name: for from_type = 'raw', time dimension of the ID (e.g. in the final aggregations granularity)
        self.time_max_name = time_max_name
        
        # time_format: the format of time columns
        self.time_format = time_format
        
        # uppercase_suffix: boolean if the suffix of the aggregation type should be uppercase
        self.uppercase_suffix = uppercase_suffix
        
    def fit(self, X, y=None):
    # go through the categorical metadata and put all valid aggregations into a special structure
          
        # boolean telling that the dataset is not valid
        cantfit = False
        
        if (self.from_type != 'slice') and (self.from_type != 'raw'):
            print('from_type must be "slice" or "raw"!')
            cantfit = True
        if (self.from_type == 'raw') and (self.time_name is None):
            print('For aggregation from raw data, time_name parameter must be filled')
            cantfit = True
        if (self.from_type == 'raw') and (self.time_max_name is None):
            print('For aggregation from raw data, time_max_name parameter must be filled')
            cantfit = True
        
        # check if the neccessary columns are present in the X dataset
        if self.id_name not in X:
            print('ID column',self.id_name,'not present in the dataset!')
            cantfit = True
        if self.slice_name not in X:
            print('Time index column',self.slice_name,'not present in the dataset!')
            cantfit = True
        if (self.from_type == 'raw') and (self.time_name not in X):
            print('Time index column',self.time_name,'not present in the dataset!')
            cantfit = True
        if (self.from_type == 'raw') and (self.time_max_name not in X):
            print('Time index column',self.time_max_name,'not present in the dataset!')
            cantfit = True
    
        # go through each row of cat metadata and check whether the column name from that row is in the dataset
        if not cantfit:
                
            self.catmeta_ = self.catmeta.copy()
            
            # do some basic validity checks of metadata tables
            # new column telling us whether the row is valid for the data provided
            self.catmeta_['valid'] = 'OK'
            
            self.catmeta_.loc[~self.catmeta_['variable'].isin(X.columns), 'valid'] = 'variable missing in provided dataset'
            self.catmeta_.loc[(self.catmeta_['aggregation'].isin({'argmax','argmaxsum','argmaxmean','argmin','argminsum','argminmean'}))
                                                & (pd.isnull(self.catmeta_['metric'])),'valid'] = 'aggregation missing metric'
            # if we calculate the features from raw data, we have to specify the time granularity for "raw" and "tschange" features            
            if (self.from_type == 'raw'):
                self.catmeta_.loc[(self.catmeta_['aggregation'].isin({'tschange'}))
                                                & (pd.isnull(self.catmeta_['granularity'])),'valid'] = 'aggregation missing granularity'
            # if we calculate the features from slices, we assign the time granularity to be "slices" as standard time units dont make sense in this case
            elif (self.from_type == 'slice'):
                self.catmeta_.loc[self.catmeta_['aggregation'].isin({'tschange'}),'granularity'] = 'slices'
            self.catmeta_.loc[pd.isnull(self.catmeta_['nancategorical']),'nancategorical'] = 0
            self.catmeta_.loc[pd.isnull(self.catmeta_['metric']),'metric'] = ''
            self.catmeta_.loc[pd.isnull(self.catmeta_['granularity']),'granularity'] = ''
            self.catmeta_.loc[~self.catmeta_['aggregation'].isin({'tschange'}),'granularity'] = ''
                                 
            # keep only valid rows and deduplicate
            catmeta_invalid = self.catmeta_[self.catmeta_['valid'] != 'OK'].copy()
            self.catmeta_ = self.catmeta_[self.catmeta_['valid'] == 'OK'].drop_duplicates(inplace=False)   
            
            # expected number of new features is sum of expected number of new features of each row
            self.nr_columns_total = self.catmeta_.shape[0]
                    
            if self.nr_columns_total == 0:       
                self.catmeta_ = None
            else:
                print('Expected number of new columns:',self.nr_columns_total)
                
            # print invalid columns so user can review them
            if len(catmeta_invalid) > 0:
                print('The following rows of slice meta data were ommited as they are invalid (reason given in last column):')
                display(catmeta_invalid)        

            #necessary columns to copy in transform() into self.X_in
            self.cols_needed_in_copy = list(set([self.id_name, self.time_name, self.time_max_name, self.slice_name] + \
                [col for col in self.catmeta_['variable'].unique()] + [col for col in self.catmeta_['metric'].unique()]) - {'',None,np.nan,'slices'})

            #RAM usage estimation
            mem_X = X.memory_usage(index = True, deep = True)
            rows_X = X.shape[0]
            entities_X = X[self.id_name].nunique()
            mem_estimate = 0
            for _, feature in self.catmeta_.iterrows():
                if len(feature['metric']) > 0:
                    mem_feature = (mem_X[feature['metric']] / rows_X) * entities_X
                else:
                    mem_feature = (mem_X[feature['variable']] / rows_X) * entities_X
                mem_estimate += mem_feature
            mem_estimate += 2 * sys.getsizeof(X[self.cols_needed_in_copy])
            mem_estimate += 128 * rows_X #for the new columns with time diffs
            print('Rough upper estimate of memory usage:',round(mem_estimate/1024/1024/1024,3),'GB')
                
        if not (hasattr(self, 'catmeta_')):
            print('There are no valid categorical aggregations fulfilling given criteria!')
        
        return
    
    def __aggregateCategorical(self,variable,aggFrom,aggTo,func,nancat,timeVar,metric,gran):
    # aggregation based on a categorical variable
        
        # name of the column with time difference for this specific feature consist
            
        # create temporary data set with the aggregation
        #  1. filter only TIME_NAME between FROM_NAME and TO_NAME
        #  2. copy important columns into a new data structure
        colsubset = list({self.id_name,self.slice_name,timeVar,variable}) #using set here because timeVar can equal to slice_name
        if gran != '':
            diff_name = 'diff_' + gran
            colsubset.append(diff_name)
        if len(metric) > 0: colsubset.append(metric)
        if len(gran) > 0 and gran != 'slices': colsubset.extend(['diffR','diffD'])
        X_tmp = self.X_in.loc[(self.X_in[self.slice_name] >= aggFrom) & (self.X_in[self.slice_name] <= aggTo),colsubset].copy()
        
        # if slice index is time variable, we need it to invert so ascending means from oldest to newest
        if self.slice_name == timeVar:
            X_tmp['_slice'] = -X_tmp[self.slice_name]
            timeVar = '_slice'
            sqlTimeVarSort = self.slice_name + ' desc'
        else:
            sqlTimeVarSort = timeVar + ' asc'
                
        # strings for given time granularity (makes sense only for func=='tschange', but for other func values, gran=='', so this part of code does not run)
        if gran == '':
            pass
        elif gran == 'slices':
            str2 = ''
            tsql = self.slice_name
        elif gran == 'years':
            str2 = 'y'
            tsql = 'floor(months_between(' + self.time_max_name + ',' + self.time_name + ')/12)'
        elif gran == 'months':
            str2 = 'm'
            tsql = 'floor(months_between(' + self.time_max_name + ',' + self.time_name + '))'
        elif gran == 'weeks':
            str2 = 'w'
            tsql = 'floor((' + self.time_max_name + ' - ' + self.time_name + ')/7)'
        elif gran == 'days':
            str2 = 'd'
            tsql = 'floor(' + self.time_max_name + ' - ' + self.time_name + ')'
        elif gran == 'hours':
            str2 = 'h'
            tsql = 'floor((' + self.time_max_name + ' - ' + self.time_name + ')*24)'
        elif gran == 'minutes':
            str2 = 'min'
            tsql = 'floor((' + self.time_max_name + ' - ' + self.time_name + ')*1440)'
        elif gran == 'seconds':
            str2 = 's'
            tsql = 'floor((' + self.time_max_name + ' - ' + self.time_name + ')*86400)'
                
        X_tmp = X_tmp.sort_values([self.id_name, self.slice_name, timeVar], ascending=[True,False,True])
        
        #  3. fill NA values
        if nancat == 1:
            #  3a. if NA is a separate category
            X_tmp[variable] = X_tmp[variable].astype(str)
            X_tmp.loc[X_tmp[variable].str.lower() == 'nan',variable] = 'NaN'
            sql_varname = 'nvl(' + variable + ",'NaN')"
            sql_tablename = '_TABLENAME_'
        else:
            #  3b. if NA is skipped
            #X_tmp[variable] = X_tmp[variable].astype(str)
            #X_tmp = X_tmp[X_tmp[variable].str.lower() != 'nan']
            X_tmp = X_tmp[pd.notnull(X_tmp[variable])]
            X_tmp[variable] = X_tmp[variable].astype(str)
            sql_varname = variable
            sql_tablename = '(select * from _TABLENAME_ where ' + variable + ' is not null)'
            
        # creating sql query is complicated here, so we need to determine whether we need to use subquery
        sql_subquery = False
        
        #  4. apply function - each function is applied in an special way as they are not simple pandas agg funtions
        if (func == 'mode'):
            #  4a. MODE: the most common category. mode returns multiple values if there are more of them. in such case
            #      we select just one. if there is no mode (all NaNs) we put NaN there
            aggstr = 'X_tmp.groupby(["' + self.id_name + '"])' + \
            '["' + variable + '"].apply(lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan)'
            column_name = variable + '_' + func + str(int(aggFrom)) + '_' + str(int(aggTo))
            if self.uppercase_suffix: column_name = column_name.upper()
            sql_full = ',' + 'stats_mode(' + sql_varname + ') as ' + column_name + '\n'
        elif (func == 'nunique'):
            #  4b. NUNIQUE: number of unique categories (a.k.a. count distinct)
            aggstr = 'X_tmp.groupby(["' + self.id_name + '"])' + \
            '["' + variable + '"].apply(lambda x: x.nunique())'
            column_name = variable + '_' + func + str(int(aggFrom)) + '_' + str(int(aggTo))
            if self.uppercase_suffix: column_name = column_name.upper()
            sql_full = ',' + 'count(distinct ' + sql_varname + ') as ' + column_name + '\n'
        elif (func == 'last'):
            #  4c. LAST: the most recent observation in the given time (argmax by time variable)
            aggstr = 'X_tmp.groupby(["' + self.id_name + '"])' + \
            '["' + self.time_name + '","' + variable + '"].' + \
            'apply(lambda x: x.loc[x["' + self.time_name + '"].idxmax(),"' + variable + '"])'
            column_name = variable + '_' + func + str(int(aggFrom)) + '_' + str(int(aggTo))
            if self.uppercase_suffix: column_name = column_name.upper()
            sql_full = ',max(TAB_' + column_name + '.val) as ' + column_name + '\n'
            sql_subquery = " (select distinct * from\n" + \
                          "  (select " + self.id_name + "\n" + \
                          "   ,last_value(" + sql_varname + ") over (partition by " + self.id_name + " order by " + sqlTimeVarSort + ") as val\n" + \
                          "   from " + sql_tablename + "\n   where " + self.slice_name + " <= " + str(int(aggTo)) + " and " + self.slice_name + " >= " + str(int(aggFrom)) + ")) TAB_" + column_name + "\n"
        elif (func == 'first'):
            #  4d. FIRST: the most recent observation in the given time (argmin by time variable)
            aggstr = 'X_tmp.groupby(["' + self.id_name + '"])' + \
            '["' + self.time_name + '","' + variable + '"].' + \
            'apply(lambda x: x.loc[x["' + self.time_name + '"].idxmin(),"' + variable + '"])'
            column_name = variable + '_' + func + str(int(aggFrom)) + '_' + str(int(aggTo))
            if self.uppercase_suffix: column_name = column_name.upper()
            sql_full = ',max(TAB_' + column_name + '.val) as ' + column_name + '\n'
            sql_subquery = " (select distinct * from\n" + \
                          "  (select " + self.id_name + "\n" + \
                          "   ,first_value(" + sql_varname + ") over (partition by " + self.id_name + " order by " + sqlTimeVarSort + ") as val\n" + \
                          "   from " + sql_tablename + "\n   where " + self.slice_name + " <= " + str(int(aggTo)) + " and " + self.slice_name + " >= " + str(int(aggFrom)) + ")) TAB_" + column_name + "\n"
        elif (func == 'argmax'):
            #  4e. ARGMAX: observation with the highest value of metric
            aggstr = 'X_tmp.groupby(["' + self.id_name + '"])' + \
            '["' + metric + '","' + variable + '"].' + \
            'apply(lambda x: x.loc[x["' + metric + '"].idxmax(),"' + variable + '"])'
            column_name = variable + '_' + func + '_' + metric + str(int(aggFrom)) + '_' + str(int(aggTo))
            if self.uppercase_suffix: column_name = column_name.upper()
            sql_full = ',max(TAB_' + column_name + '.val) as ' + column_name + '\n'
            sql_subquery = " (select distinct * from\n" + \
                          "  (select " + self.id_name + "\n" + \
                          "   ,first_value(" + sql_varname + ") over (partition by " + self.id_name + " order by " + metric + " desc) as val\n" + \
                          "   from " + sql_tablename + "\n   where " + self.slice_name + " <= " + str(int(aggTo)) + " and " + self.slice_name + " >= " + str(int(aggFrom)) + ")) TAB_" + column_name + "\n"
        elif (func == 'argmin'):
            #  4e2. ARGMIN: observation with the highest value of metric
            aggstr = 'X_tmp.groupby(["' + self.id_name + '"])' + \
            '["' + metric + '","' + variable + '"].' + \
            'apply(lambda x: x.loc[x["' + metric + '"].idxmin(),"' + variable + '"])'
            column_name = variable + '_' + func + '_' + metric + str(int(aggFrom)) + '_' + str(int(aggTo))
            if self.uppercase_suffix: column_name = column_name.upper()
            sql_full = ',max(TAB_' + column_name + '.val) as ' + column_name + '\n'
            sql_subquery = " (select distinct * from\n" + \
                          "  (select " + self.id_name + "\n" + \
                          "   ,first_value(" + sql_varname + ") over (partition by " + self.id_name + " order by " + metric + " asc) as val\n" + \
                          "   from " + sql_tablename + "\n   where " + self.slice_name + " <= " + str(int(aggTo)) + " and " + self.slice_name + " >= " + str(int(aggFrom)) + ")) TAB_" + column_name + "\n"
        elif (func == 'argmaxsum'):
            #  4f. ARGMAXSUM: observation with the highest sum of metric
            aggstr = 'X_tmp.groupby(["' + self.id_name + '","' + variable +'"])' + \
            '["' + metric + '"].sum().reset_index().groupby(["' + self.id_name + '"])' + \
            '.apply(lambda x: x.loc[x["' + metric + '"].idxmax(),"' + variable + '"])'
            column_name = variable + '_' + func + '_' + metric + str(int(aggFrom)) + '_' + str(int(aggTo))
            if self.uppercase_suffix: column_name = column_name.upper()
            sql_full = ',max(TAB_' + column_name + '.val) as ' + column_name + '\n'
            sql_subquery = " (select distinct * from\n" + \
                          "  (select " + self.id_name + "\n" + \
                          "   ,first_value(val) over (partition by " + self.id_name + " order by val desc) as val from\n" + \
                          "   (select " + self.id_name + "\n    ," + sql_varname + " as val\n" + \
                          "    ,sum(" + metric + ") as val\n" + \
                          "    from " + sql_tablename + "\n    where " + self.slice_name + " <= " + str(int(aggTo)) + " and " + self.slice_name + " >= " + str(int(aggFrom)) + "\n" + \
                          "    group by " + self.id_name + ", " + sql_varname + "))) TAB_" + column_name + "\n"
        elif (func == 'argminsum'):
            #  4f2. ARGMINSUM: observation with the highest sum of metric
            aggstr = 'X_tmp.groupby(["' + self.id_name + '","' + variable +'"])' + \
            '["' + metric + '"].sum().reset_index().groupby(["' + self.id_name + '"])' + \
            '.apply(lambda x: x.loc[x["' + metric + '"].idxmin(),"' + variable + '"])'
            column_name = variable + '_' + func + '_' + metric + str(int(aggFrom)) + '_' + str(int(aggTo))
            if self.uppercase_suffix: column_name = column_name.upper()
            sql_full = ',max(TAB_' + column_name + '.val) as ' + column_name + '\n'
            sql_subquery = " (select distinct * from\n" + \
                          "  (select " + self.id_name + "\n" + \
                          "   ,first_value(val) over (partition by " + self.id_name + " order by val asc) as val from\n" + \
                          "   (select " + self.id_name + "\n    ," + sql_varname + " as val\n" + \
                          "    ,sum(" + metric + ") as val\n" + \
                          "    from " + sql_tablename + "\n    where " + self.slice_name + " <= " + str(int(aggTo)) + " and " + self.slice_name + " >= " + str(int(aggFrom)) + "\n" + \
                          "    group by " + self.id_name + ", " + sql_varname + "))) TAB_" + column_name + "\n"
        elif (func == 'argmaxmean'):
            #  4g. ARGMAXMEAN: observation with the highest mean of metric
            aggstr = 'X_tmp.groupby(["' + self.id_name + '","' + variable +'"])' + \
            '["' + metric + '"].mean().reset_index().groupby(["' + self.id_name + '"])' + \
            '.apply(lambda x: x.loc[x["' + metric + '"].idxmax(),"' + variable + '"])'
            column_name = variable + '_' + func + '_' + metric + str(int(aggFrom)) + '_' + str(int(aggTo))
            if self.uppercase_suffix: column_name = column_name.upper()
            sql_full = ',max(TAB_' + column_name + '.val) as ' + column_name + '\n'
            sql_subquery = " (select distinct * from\n" + \
                          "  (select " + self.id_name + "\n" + \
                          "   ,first_value(val) over (partition by " + self.id_name + " order by val desc) as val from\n" + \
                          "   (select " + self.id_name + "\n    ," + sql_varname + " as val\n" + \
                          "    ,avg(" + metric + ") as val\n" + \
                          "    from " + sql_tablename + "\n    where " + self.slice_name + " <= " + str(int(aggTo)) + " and " + self.slice_name + " >= " + str(int(aggFrom)) + "\n" + \
                          "    group by " + self.id_name + ", " + sql_varname + "))) TAB_" + column_name + "\n"
        elif (func == 'argminmean'):
            #  4g2. ARGMINMEAN: observation with the highest mean of metric
            aggstr = 'X_tmp.groupby(["' + self.id_name + '","' + variable +'"])' + \
            '["' + metric + '"].mean().reset_index().groupby(["' + self.id_name + '"])' + \
            '.apply(lambda x: x.loc[x["' + metric + '"].idxmin(),"' + variable + '"])'
            column_name = variable + '_' + func + '_' + metric + str(int(aggFrom)) + '_' + str(int(aggTo))
            if self.uppercase_suffix: column_name = column_name.upper()
            sql_full = ',max(TAB_' + column_name + '.val) as ' + column_name + '\n'
            sql_subquery = " (select distinct * from\n" + \
                          "  (select " + self.id_name + "\n" + \
                          "   ,first_value(val) over (partition by " + self.id_name + " order by val asc) as val from\n" + \
                          "   (select " + self.id_name + "\n    ," + sql_varname + " as val\n" + \
                          "    ,avg(" + metric + ") as val\n" + \
                          "    from " + sql_tablename + "\n    where " + self.slice_name + " <= " + str(int(aggTo)) + " and " + self.slice_name + " >= " + str(int(aggFrom)) + "\n" + \
                          "    group by " + self.id_name + ", " + sql_varname + "))) TAB_" + column_name + "\n"
        elif (func == 'nchanges'):
            #  4h. NCHANGES: number of changes in the given time
            X_tmp['prev_cat'] = X_tmp[variable].shift(1)
            X_tmp['prev_id'] = X_tmp[self.id_name].shift(1)
            X_tmp.loc[X_tmp[self.id_name] != X_tmp['prev_id'], 'prev_cat'] = X_tmp.loc[X_tmp[self.id_name] != X_tmp['prev_id'], variable]        
            aggstr = 'X_tmp.groupby(["' + self.id_name + '"])' + \
            '["' + variable + '", "prev_cat"].apply(lambda x: sum(x["' + variable + '"] != x["prev_cat"]))'
            column_name = variable + '_' + func + str(int(aggFrom)) + '_' + str(int(aggTo))
            if self.uppercase_suffix: column_name = column_name.upper()
            sql_full = ',max(TAB_' + column_name + '.val) as ' + column_name + '\n'
            sql_subquery = " (select " + self.id_name + "\n" + \
                          "  ,sum(case when val <> lagged then 1 else 0 end) as val from\n" + \
                          "  (select " + self.id_name + "\n   ," + sql_varname + " as val\n" + \
                          "   ,lag(" + sql_varname + ") over (partition by " + self.id_name + " order by " + sqlTimeVarSort + ") as lagged\n" + \
                          "   from " + sql_tablename + "\n   where " + self.slice_name + " <= " + str(int(aggTo)) + " and " + self.slice_name + " >= " + str(int(aggFrom)) + ")\n" + \
                          "  group by " + self.id_name + ") TAB_" + column_name + "\n"
        elif (func == 'tschange') & (gran != ''):
            #  4i. TSCHANGE: time since last change
            X_tmp['prev_cat'] = X_tmp[variable].shift(1)
            X_tmp['prev_id'] = X_tmp[self.id_name].shift(1)
            X_tmp.loc[X_tmp[self.id_name] != X_tmp['prev_id'], 'prev_cat'] = X_tmp.loc[X_tmp[self.id_name] != X_tmp['prev_id'], variable]
            X_tmp['change'] = X_tmp['prev_cat'] != X_tmp[variable]      
            X_tmp['changeDiff'] = X_tmp[diff_name]
            X_tmp.loc[X_tmp['change']==False,'changeDiff'] = np.nan
            aggstr = 'X_tmp.groupby(["' + self.id_name + '"])["changeDiff"].min()'
            column_name = variable + '_' + func + '_' + str2 + '_' + str(int(aggFrom)) + '_' + str(int(aggTo))
            if self.uppercase_suffix: column_name = column_name.upper()
            sql_full = ',max(TAB_' + column_name + '.val) as ' + column_name + '\n'
            sql_subquery = " (select " + self.id_name + "\n" + \
                          "  ,min(timediff) as val from\n" + \
                          "  (select " + self.id_name + "\n   ," + sql_varname + " as val\n" + \
                          "   ,lag(" + sql_varname + ") over (partition by " + self.id_name + " order by " + sqlTimeVarSort + ") as lagged\n" + \
                          "   ," + tsql + " as timediff" + \
                          "   from " + sql_tablename + "\n   where " + self.slice_name + " <= " + str(int(aggTo)) + " and " + self.slice_name + " >= " + str(int(aggFrom)) + ")\n" + \
                          "   where val <> lagged\n" + \
                          "  group by " + self.id_name + ") TAB_" + column_name + "\n"
        
        if self.uppercase_suffix: column_name = column_name.upper()
        X_tmp = eval(aggstr + '.to_frame()')
        X_tmp.columns = [column_name]
        
        # add the new column to X_out data     
        if (column_name not in self.X_buff) and (column_name not in self.X_out):
            self.X_buff = pd.concat([self.X_buff,optimize_dtypes(X_tmp)], axis = 1)
            self.catnames_.append(column_name)
            self.strsql_.append(sql_full)
            if sql_subquery:
                self.strsql_end.append("left join" + sql_subquery + " on t."+self.id_name+"=TAB_"+column_name+"."+self.id_name + "\n")
            print('Variable',column_name,'created. (',self.nr_columns_done_print,'/',self.nr_columns_total,')')
        
        return 
            
    def __precalculate_time_differences(self):
    # this function calculates time diffs in the original data (X) in time granularities required by metadata (self.catmeta_) and adds the to self.X_in
    
        print('Precalculating time differences for input data...')
    
        # prepare the columns with time differences  
        print(' calculating time difference...')
        self.X_in['diffD'] = self.X_in[self.time_max_name] - self.X_in[self.time_name]
        print(' calculating relative delta...')
        self.X_in['diffR'] = self.X_in[[self.time_name,self.time_max_name]].apply(lambda x: relativedelta(dt1=x[1],dt2=x[0]), axis=1)
        
        # calculate the difference in the target time units
        for gran in self.catmeta_['granularity'].unique():
            if gran == 'years':
                print(' calculating diff in years...')
                self.X_in['diff_' + gran] = self.X_in['diffR'].apply(lambda x: x.years)
            elif gran == 'months':
                print(' calculating diff in months...')
                self.X_in['diff_' + gran] = self.X_in['diffR'].apply(lambda x: 12*x.years+x.months)
            elif gran == 'weeks':
                print(' calculating diff in weeks...')
                self.X_in['diff_' + gran] = self.X_in['diffD'].apply(lambda x: x.days//7)
            elif gran == 'days':
                print(' calculating diff in days...')
                self.X_in['diff_' + gran] = self.X_in['diffD'].apply(lambda x: x.days)
            elif gran == 'hours':
                print(' calculating diff in hours...')
                self.X_in['diff_' + gran] = self.X_in['diffD'].apply(lambda x: x.days*24 + x.seconds//3600)
            elif gran == 'minutes':
                print(' calculating diff in minutes...')
                self.X_in['diff_' + gran] = self.X_in['diffD'].apply(lambda x: x.days*1440 + x.seconds//60)
            elif gran == 'seconds':
                print(' calculating diff in seconds...')
                self.X_in['diff_' + gran] = self.X_in['diffD'].apply(lambda x: x.days*86400 + x.seconds)
    
        return                
                
    def transform(self, X, resume = False, resume_from = None):
    # execute the categorical aggregations
    
        check_is_fitted(self, ['catmeta_'])

        if not resume:
        
            self.X_in = X[self.cols_needed_in_copy].copy()
            self.catnames_ = list()
            self.strsql_ = []
            self.strsql_end = []
                    
            if self.from_type == 'raw':
                # if we have raw data (transactions), we calculate all time differences that we need for the aggregations
                self.__precalculate_time_differences()
            elif self.from_type == 'slice':
                # if we have sliced data, we just calculate time difference in "number of slices"
                self.X_in['diff_slices'] = self.X_in[self.slice_name]-1
            
            self.X_out = self.X_in.groupby([self.id_name]).size().to_frame()
            
            # self.X_buff = buffer which will be used to temporarily add new columns before they are all joined to X_out
            self.X_buff = self.X_out.drop(self.X_out.columns,axis=1,inplace=False)
            
            #SQL string
            self.strsql_ = ['select t.' + self.id_name + '\n' \
                            ',' + 'count(t.' + self.id_name + ') as ORIG_ROWS_COUNT \n']
            
            if self.uppercase_suffix: newColName = 'ORIG_ROWS_COUNT'
            else: newColName = 'orig_rows_count'
            self.X_out.columns = [newColName]
            self.catnames_.append(newColName)
            print('Variable',newColName,'created.')

            # self.nr_columns_done = count of new columns that have been already created
            self.nr_columns_done = 0
            self.nr_columns_done_print = 0

        if (resume) and (not resume_from):
            self.resume_from = self.nr_columns_done
            self.nr_columns_done_print = self.nr_columns_done
        elif resume:
            self.resume_from = resume_from-1
            self.nr_columns_done_print = self.nr_columns_done
                
        # for each cat metadata in list, make the aggregation
        for ci, c in self.catmeta_.iterrows():

            if (not resume) or (resume and ci+1 > self.resume_from):

                self.nr_columns_done_print += 1

                try:

                    # note that for self.from_type=='slice', c['gran']='slices' as it is assigned in fit method
                    if self.from_type == 'slice':
                        self.__aggregateCategorical(c['variable'],c['from'],c['to'],c['aggregation'],c['nancategorical'],self.slice_name,c['metric'],c['granularity'])
                    else:
                        self.__aggregateCategorical(c['variable'],c['from'],c['to'],c['aggregation'],c['nancategorical'],self.time_name,c['metric'],c['granularity'])

                except:

                    print('Problem occurred creating column given by following parameters:', {'variable':c['variable'], 'from':c['from'], 'to':c['to'], 'metric':c['metric'], 'granularity':c['granularity']})
                
                # if at least 20 columns in buffer, move data from buffer to output
                if len(self.X_buff.columns) >= 20:
                    self.X_out = pd.concat([self.X_out, self.X_buff], axis=1)
                    self.X_buff.drop(self.X_buff.columns, axis=1, inplace=True)
                    
                gc.collect()

                self.nr_columns_done += 1
        
        # move the rest of the data from buffer to the output
        if len(self.X_buff.columns) > 0:
            self.X_out = pd.concat([self.X_out, self.X_buff], axis=1)
            self.X_buff.drop(self.X_buff.columns, axis=1, inplace=True)
        
        gc.collect()
        
        #SQL string
        self.strsql_.append('from _TABLENAME_ t\n')
        self.strsql_.extend(self.strsql_end)
        self.strsql_.append('group by t.' + self.id_name + '\n')
        self.strsql_ = ''.join(self.strsql_)
        #print(self.strsql_)

        gc.collect()
        
        return self.X_out   