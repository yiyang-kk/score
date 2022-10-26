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


class TimeSinceCalc(BaseEstimator, TransformerMixin):
    """
    Args:
        id_name (string):
            ID column name of the datasets that will be used in fit and transform procedures
        time_name (string):
            TIME ID column name of the datasets that will be used in fit and transform procedures
        time_max_name (string):
            TIME MAX column is the column of "current time" for each ID, i.e. time from which the history is calculated back from
            Example: time_max_name = application date, time_name = transaction date, only rows with time_name <= time_max_name are taken into account
        timesincemeta (matrix):
            a matrix of slice metadata - defining how the raw data set should be sliced
            columns: 
                1) granularity: time units in which the time since should be calculated in, can be 'years', 'months', 'weeks', 'days', 'hours', 'minutes', 'seconds'
                2) type: "last" or "first" - whether to calculated time since first or last transaction which happened in the history and is fulfilling the given condition
            The other columns can specify up to two conditions which transaction must fulfill to enter the aggregation:
                3) condition1: name of the column which the condition is based on. if empty, the condition is considered to be always true.
                4) from1: if condition1 column is numeric, specify the left boundary of interval of values where the condition is considered fulfilled. If empty, the left boundary is considered to be -infinity.
                5) from1eq: 0 or 1, specifying whether the inequality is sharp (1 for sharp)
                6) to1: if condition1 column is numeric, specify the right boundary of interval of values where the condition is considered fulfilled. If empty, the left boundary is considered to be +infinity.
                7) to1eq: 0 or 1, specifying whether the inequality is sharp (1 for sharp)
                8) category1: if condition1 column is categorical, specify the category which the column should be equal to for the condition to be true. If empty, the algorithm scans the dataset for all the possible categories and uses each one of them as a separate condition. NaN is not considered a category.
                9) - 14) the same for second condition. if empty, the condition is considered to be always true.
        time_format (string, optional):
            the format of time columns
        uppercase_suffix (boolean, optional):
            boolean if the suffix of the aggregation type should be uppercase
        keyword (string, optional):
            string that will be used as prefix for the new columns

    Attributes:
        X_in (pd.DataFrame):
            input
        X_out (pd.DataFrame):
            output 
        sql_ (string):
            SQL query which makes the same transformation on Oracle database

    Methods:
        fit(X, y = None) :
            go through the aggregation metadata and put all valid aggregations into a special structure.
            X (pd.DataFrame):
                the dataframe you want the aggregations to be executed on
        transform(X) :
            execute the time since aggregations   
            X (pd.DataFrame):
                the dataframe you want the aggregations to be executed on
    """

    def __init__(self, id_name, time_name, time_max_name, timesincemeta, time_format='%Y-%m-%d', uppercase_suffix=True, keyword='entry'):
        self.id_name = id_name
        self.time_name = time_name
        self.time_max_name = time_max_name
        self.timesincemeta = check_array(timesincemeta, dtype=object, force_all_finite=False)
        self.timesincemeta = pd.DataFrame(self.timesincemeta)
        self.timesincemeta.columns = ['granularity', 'type', 'condition1', 'from1', 'from1eq',
                                      'to1', 'to1eq', 'category1', 'condition2', 'from2', 'from2eq', 'to2', 'to2eq', 'category2']
        self.time_format = time_format
        self.uppercase_suffix = uppercase_suffix
        # keyword: new variable names will start with this keyword
        self.keyword = keyword

    def __assign_ineq_types(self, condition_number_str):
        # this method process all rows of timesincemeta_ and properly assigns ineqality types
        # as there might be multiple conditions in each row, this method takes number of the condition as argument, so it can be called multiple times and the chunks of codes are not repeated
        # the argument must be formatted as string

        condition_name = 'condition' + condition_number_str
        from_name = 'from' + condition_number_str
        fromeq_name = 'from' + condition_number_str + 'eq'
        to_name = 'to' + condition_number_str
        toeq_name = 'to' + condition_number_str + 'eq'
        category_name = 'category' + condition_number_str
        ineq_name = 'ineq' + condition_number_str

        self.timesincemeta_.loc[(pd.notnull(self.timesincemeta_[condition_name]))
                                & (self.timesincemeta_['valid'] == 'OK')
                                & (pd.notnull(self.timesincemeta_[from_name]))
                                & ((self.timesincemeta_[fromeq_name] == 0) | (pd.isnull(self.timesincemeta_[fromeq_name])))
                                & (pd.isnull(self.timesincemeta_[to_name])), ineq_name] = 'g'
        self.timesincemeta_.loc[(pd.notnull(self.timesincemeta_[condition_name]))
                                & (self.timesincemeta_['valid'] == 'OK')
                                & (pd.notnull(self.timesincemeta_[from_name]))
                                & (self.timesincemeta_[fromeq_name] == 1)
                                & (pd.isnull(self.timesincemeta_[to_name])), ineq_name] = 'ge'
        self.timesincemeta_.loc[(pd.notnull(self.timesincemeta_[condition_name]))
                                & (self.timesincemeta_['valid'] == 'OK')
                                & (pd.isnull(self.timesincemeta_[from_name]))
                                & (pd.notnull(self.timesincemeta_[to_name]))
                                & ((self.timesincemeta_[toeq_name] == 0) | (pd.isnull(self.timesincemeta_[toeq_name]))), ineq_name] = 'l'
        self.timesincemeta_.loc[(pd.notnull(self.timesincemeta_[condition_name]))
                                & (self.timesincemeta_['valid'] == 'OK')
                                & (pd.isnull(self.timesincemeta_[from_name]))
                                & (pd.notnull(self.timesincemeta_[to_name]))
                                & (self.timesincemeta_[toeq_name] == 1), ineq_name] = 'le'
        self.timesincemeta_.loc[(pd.notnull(self.timesincemeta_[condition_name]))
                                & (self.timesincemeta_['valid'] == 'OK')
                                & (pd.notnull(self.timesincemeta_[from_name]))
                                & ((self.timesincemeta_[fromeq_name] == 0) | (pd.isnull(self.timesincemeta_[fromeq_name])))
                                & (pd.notnull(self.timesincemeta_[to_name]))
                                & ((self.timesincemeta_[toeq_name] == 0) | (pd.isnull(self.timesincemeta_[toeq_name]))), ineq_name] = 'gl'
        self.timesincemeta_.loc[(pd.notnull(self.timesincemeta_[condition_name]))
                                & (self.timesincemeta_['valid'] == 'OK')
                                & (pd.notnull(self.timesincemeta_[from_name]))
                                & (self.timesincemeta_[fromeq_name] == 1)
                                & (pd.notnull(self.timesincemeta_[to_name]))
                                & ((self.timesincemeta_[toeq_name] == 0) | (pd.isnull(self.timesincemeta_[toeq_name]))), ineq_name] = 'gel'
        self.timesincemeta_.loc[(pd.notnull(self.timesincemeta_[condition_name]))
                                & (self.timesincemeta_['valid'] == 'OK')
                                & (pd.notnull(self.timesincemeta_[from_name]))
                                & ((self.timesincemeta_[fromeq_name] == 0) | (pd.isnull(self.timesincemeta_[fromeq_name])))
                                & (pd.notnull(self.timesincemeta_[to_name]))
                                & (self.timesincemeta_[toeq_name] == 1), ineq_name] = 'gle'
        self.timesincemeta_.loc[(pd.notnull(self.timesincemeta_[condition_name]))
                                & (self.timesincemeta_['valid'] == 'OK')
                                & (pd.notnull(self.timesincemeta_[from_name]))
                                & (self.timesincemeta_[fromeq_name] == 1)
                                & (pd.notnull(self.timesincemeta_[to_name]))
                                & (self.timesincemeta_[toeq_name] == 1), ineq_name] = 'gele'
        self.timesincemeta_.loc[(pd.notnull(self.timesincemeta_[condition_name]))
                                & (self.timesincemeta_['valid'] == 'OK')
                                & (pd.isnull(self.timesincemeta_[from_name]))
                                & (pd.isnull(self.timesincemeta_[to_name]))
                                & (self.timesincemeta_[category_name] != 'nan'), ineq_name] = 'c'
        self.timesincemeta_.loc[(pd.notnull(self.timesincemeta_[condition_name]))
                                & (self.timesincemeta_['valid'] == 'OK')
                                & (pd.isnull(self.timesincemeta_[from_name]))
                                & (pd.isnull(self.timesincemeta_[to_name]))
                                & (self.timesincemeta_[category_name] == 'nan'), ineq_name] = 'a'

    def __create_categorical_rows(self, condition_number_str, metadata_row, data_column):
        # from one row in metadata of type 'a' (feature for all categories from given data_column) create multiple rows of type 'c' (category of given data_column equal to a certain value)
        # deletion of the original row is not done in this method!

        category_name = 'category' + condition_number_str
        ineq_name = 'ineq' + condition_number_str

        unique_values = list(set(data_column.unique()) - {np.nan})

        for val in unique_values:
            newrow = metadata_row.copy()
            newrow[ineq_name] = 'c'
            newrow[category_name] = str(val)
            self.timesincemeta_ = self.timesincemeta_.append(newrow, ignore_index=True)

        return

    def fit(self, X, y=None):
        # creates internal structures which are then used to calculate the features

        # boolean telling that the dataset is not valid
        cantfit = False

        # check whether the history is long enough and ID and TIME ID columns are present in the X data set
        if self.id_name not in X:
            print('ID column', self.id_name, 'not present in the dataset!')
            cantfit = True
        if self.time_name not in X:
            print('Time column', self.time_name, 'not present in the dataset!')
            cantfit = True
        if self.time_max_name not in X:
            print('Time max column', self.time_max_name, 'not present in the dataset!')
            cantfit = True

        # go through each row of metadata and check whether the column name from that row is in the dataset
        if not cantfit:
            self.timesincemeta_ = self.timesincemeta.copy()
            self.timesincemeta_['valid'] = 'OK'
            self.timesincemeta_['ineq1'] = 'n'
            self.timesincemeta_['ineq2'] = 'n'

            self.timesincemeta_['category1'] = self.timesincemeta_['category1'].astype(str)
            self.timesincemeta_['category2'] = self.timesincemeta_['category2'].astype(str)

            # basic validity checks of the rows
            self.timesincemeta_.loc[pd.isnull(self.timesincemeta_['granularity']),
                                    'valid'] = 'granularity is not filled'
            self.timesincemeta_.loc[~self.timesincemeta_['type'].isin(
                {'first', 'last'}), 'valid'] = 'col not in {first,last}'
            self.timesincemeta_.loc[(pd.notnull(self.timesincemeta_['condition1']))
                                    & (~self.timesincemeta_['condition1'].isin(X.columns)), 'valid'] = 'condition1 missing in provided dataset'
            self.timesincemeta_.loc[(pd.notnull(self.timesincemeta_['condition2']))
                                    & (~self.timesincemeta_['condition2'].isin(X.columns)), 'valid'] = 'condition2 missing in provided dataset'

            # set proper inequality types for easy processing in transform method
            self.__assign_ineq_types('1')
            self.__assign_ineq_types('2')

            # process conditions for categorical variables where user wants to create a feature for each category
            # these conditions are originally as one row in the metadata. we must change this one row to multiple rows: one per each distinct category
            for _, rw in self.timesincemeta_.iterrows():
                if rw['ineq1'] == 'a':
                    self.__create_categorical_rows('1', rw, X[rw['condition1']])
            self.timesincemeta_ = self.timesincemeta_[self.timesincemeta_['ineq1'] != 'a']
            for _, rw in self.timesincemeta_.iterrows():
                if rw['ineq2'] == 'a':
                    self.__create_categorical_rows('2', rw, X[rw['condition2']])
            self.timesincemeta_ = self.timesincemeta_[self.timesincemeta_['ineq2'] != 'a']

            # count the new features

            # keep only valid rows and deduplicate
            timesincemeta_invalid = self.timesincemeta_[self.timesincemeta_['valid'] != 'OK'].copy()
            self.timesincemeta_ = self.timesincemeta_[
                self.timesincemeta_['valid'] == 'OK'].drop_duplicates(inplace=False)

            # expected number of new features is number of rows
            self.nr_columns_total = self.timesincemeta_.shape[0]

            if self.nr_columns_total == 0:
                self.timesincemeta_ = None
            else:
                print('Expected number of new columns:', self.nr_columns_total)

            # print invalid columns so user can review them
            if len(timesincemeta_invalid) > 0:
                print('The following rows of time since meta data were ommited as they are invalid (reason given in last column):')
                display(timesincemeta_invalid)

            #necessary columns to copy in transform() into self.X_in
            self.cols_needed_in_copy = list(set([self.id_name, self.time_name, self.time_max_name] + \
                [col for col in self.timesincemeta_['condition1'].unique()] + [col for col in self.timesincemeta_['condition2'].unique()]) - {'',None,np.nan})

            #RAM usage estimation
            rows_X = X.shape[0]
            entities_X = X[self.id_name].nunique()
            mem_estimate = self.timesincemeta_.shape[0] * 8 * entities_X
            mem_estimate += sys.getsizeof(X[self.cols_needed_in_copy])
            mem_estimate += 128 * rows_X #for the new columns with time diffs
            print('Rough upper estimate of memory usage:',round(mem_estimate/1024/1024/1024,3),'GB')

        if not hasattr(self, 'timesincemeta_'):
            print('There are no valid time since aggregations fulfilling given criteria!')

        return

    def __applycondition(self, dt, cond, ctp, cfrom, cto, ccat):
        # restrains dataset dt using a condition

        csql = ''
        cstr = ''

        # condition on categorical variable
        if ctp == 'c':
            dt = dt[dt[cond].astype(str) == ccat]
            cstr = str(cond) + '_' + str(ccat)
            csql = str(cond) + "=" + "'" + str(ccat) + "'"
        # condition - interval [) for numeric variable
        elif ctp == 'gel':
            dt = dt[(dt[cond] >= cfrom) & (dt[cond] < cto)]
            cstr = str(cond) + '_' + str(int(cfrom)) + '_' + str(int(cto))
            csql = str(cond) + ">=" + str(cfrom) + " and " + str(cond) + "<" + str(cto)
        # condition - interval (] for numeric variable
        elif ctp == 'gle':
            dt = dt[(dt[cond] > cfrom) & (dt[cond] <= cto)]
            cstr = str(cond) + '_' + str(int(cfrom)) + '_' + str(int(cto))
            csql = str(cond) + ">" + str(cfrom) + " and " + str(cond) + "<=" + str(cto)
        # condition - interval [] for numeric variable
        elif ctp == 'gele':
            dt = dt[(dt[cond] >= cfrom) & (dt[cond] <= cto)]
            cstr = str(cond) + '_' + str(int(cfrom)) + '_' + str(int(cto))
            csql = str(cond) + ">=" + str(cfrom) + " and " + str(cond) + "<=" + str(cto)
        # condition - interval () for numeric variable
        elif ctp == 'gl':
            dt = dt[(dt[cond] > cfrom) & (dt[cond] < cto)]
            cstr = str(cond) + '_' + str(int(cfrom)) + '_' + str(int(cto))
            csql = str(cond) + ">" + str(cfrom) + " and " + str(cond) + "<" + str(cto)
        # condition - greater than or equal to
        elif ctp == 'ge':
            dt = dt[dt[cond] >= cfrom]
            cstr = str(cond) + '_' + str(ctp) + str(int(cfrom))
            csql = str(cond) + ">=" + str(cfrom)
        # condition - greater than
        elif ctp == 'g':
            dt = dt[dt[cond] > cfrom]
            cstr = str(cond) + '_' + str(ctp) + str(int(cfrom))
            csql = str(cond) + ">" + str(cfrom)
        # condition - lower than or equal to
        elif ctp == 'le':
            dt = dt[dt[cond] <= cto]
            cstr = str(cond) + '_' + str(ctp) + str(int(cto))
            csql = str(cond) + "<=" + str(cto)
        # condition - lower than
        elif ctp == 'l':
            dt = dt[dt[cond] < cto]
            cstr = str(cond) + '_' + str(ctp) + str(int(cto))
            csql = str(cond) + "<" + str(cto)
        # condition - equal to
        elif ctp == 'g':
            dt = dt[dt[cond] == cfrom]
            cstr = str(cond) + str(int(cfrom))
            csql = str(cond) + "=" + str(cfrom)

        return csql, cstr, dt

    def __timesince(self, gran, firstlast, cond1, cond1tp, from1, to1, cat1, cond2, cond2tp, from2, to2, cat2):

        csql1 = ccolumn_name = csql2 = cstr2 = cstr = ''

        # name of the column with time difference for this specific feature consist
        diff_name = 'diff_' + gran

        needed_columns = [self.id_name, 'diff_' + gran]
        if cond1tp != 'n':
            needed_columns.append(cond1)
        if cond2tp != 'n':
            needed_columns.append(cond2)
        X_cond = self.X_in[needed_columns].copy()

        # apply conditions to the dataset
        if cond1tp != 'n':
            csql1, ccolumn_name, X_cond = self.__applycondition(X_cond, cond1, cond1tp, from1, to1, cat1)

        if cond2tp != 'n':
            csql2, cstr2, X_cond = self.__applycondition(X_cond, cond2, cond2tp, from2, to2, cat2)

        # strings for the right granularity
        if gran == 'years':
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

        # calculate the right aggregation (first or last entry)
        if firstlast == 'first':
            X_grp = X_cond.groupby(self.id_name)[diff_name].max().to_frame()
            column_name = 'f'
            flsql = 'max('
        elif firstlast == 'last':
            X_grp = X_cond.groupby(self.id_name)[diff_name].min().to_frame()
            column_name = 'l'
            flsql = 'min('

        # put together name of the variable
        if len(ccolumn_name) + len(cstr2) > 0:
            if len(ccolumn_name) > 0:
                if len(cstr2) > 0:
                    cstr = ccolumn_name + '_' + cstr2
                else:
                    cstr = ccolumn_name
            else:
                cstr = cstr2
            cstr = '_' + cstr
        varname = self.keyword + '_' + column_name + cstr + '_' + str2
        if self.uppercase_suffix:
            varname = varname.upper()
        X_grp.columns = [varname]

        # put together sql code of the variable
        if len(csql1) + len(csql2) > 0:
            if len(csql1) > 0 and len(csql2) > 0:
                csql = csql1 + ' and ' + csql2
            elif len(csql1) > 0:
                csql = csql1
            else:
                csql = csql2
        else:
            csql = '1=1'
        sql = ',' + flsql + 'case when ' + csql + ' then ' + tsql + ' end) as ' + varname + '\n'

        if (varname not in self.X_buff) and (varname not in self.X_out):
            self.X_buff = pd.concat([self.X_buff, optimize_dtypes(X_grp)], axis=1)
            self.tsnames_.append(varname)
            print('Variable', varname, 'created. (', self.nr_columns_done_print, '/', self.nr_columns_total, ')')
            self.strsql_.append(sql)

        return

    def __precalculate_time_differences(self, resume = False):
        # this function calculates time diffs in the original data (X) in time granularities required by metadata (self.timesincemeta_) and adds the to self.X_in

        print('Precalculating time differences for input data...')

            # prepare the columns with time differences
        if (not resume) or ('diffD' not in self.X_in.columns):
            print(' calculating time difference...')
            self.X_in['diffD'] = self.X_in[self.time_max_name] - self.X_in[self.time_name]
        if (not resume) or ('diffR' not in self.X_in.columns):
            print(' calculating relative delta...')
            self.X_in['diffR'] = self.X_in[[self.time_name, self.time_max_name]].apply(
                lambda x: relativedelta(dt1=x[1], dt2=x[0]), axis=1)

            # calculate the difference in the target time units
        for gran in self.timesincemeta_['granularity'].unique():
            if (not resume) or ('diff_' + gran not in self.X_in.columns):
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
        # execute the transformations

        check_is_fitted(self, ['timesincemeta_'])

        if not resume:

            self.X_out_prepared = False

            # remove the information from the future
            self.X_in = X[self.cols_needed_in_copy].copy()
            self.tsnames_ = list()

            # apply correct format to the ditetime columns
            self.X_in[self.time_name] = pd.to_datetime(self.X_in[self.time_name], format=self.time_format)
            self.X_in[self.time_max_name] = pd.to_datetime(
                self.X_in[self.time_max_name], format=self.time_format)
            self.X_in = self.X_in[self.X_in[self.time_max_name] >= self.X_in[self.time_name]].copy()

        # add new rows with time differences between time_name and time_max_name in all needed granularities (time units)
        self.__precalculate_time_differences(resume)

        if (not resume) or (self.X_out_prepared == False):

            # first aggregation is just the count
            self.X_out = pd.DataFrame(self.X_in.groupby(self.id_name)[self.id_name].count())
            varname = 'rows_count'
            if self.uppercase_suffix:
                varname = varname.upper()
            self.tsnames_.append(varname)
            self.X_out.columns = [varname]

            # self.X_buff = buffer which will be used to temporarily add new columns before they are all joined to X_out
            self.X_buff = self.X_out.drop(self.X_out.columns, axis=1, inplace=False)

            # SQL string
            self.strsql_ = ['select ' + self.id_name + '\n'
                         ',' + 'count(*) as ' + varname + '\n']

            # self.nr_columns_done = count of new columns that have been already created
            self.nr_columns_done = 0
            self.nr_columns_done_print = 0
            self.X_out_prepared = True

        if (resume) and (not resume_from):
            self.resume_from = self.nr_columns_done
            self.nr_columns_done_print = self.nr_columns_done
        elif resume:
            self.resume_from = resume_from-1
            self.nr_columns_done_print = self.nr_columns_done
        
        print('Creating new columns...')

        # for each row in time since metadata apply the transformation
        for ti, t in self.timesincemeta_.iterrows():

            if (not resume) or (resume and ti+1 > self.resume_from):

                self.nr_columns_done_print += 1

                try:

                    self.__timesince(t['granularity'], t['type'], t['condition1'], t['ineq1'], t['from1'], t['to1'], t['category1'],
                                    t['condition2'], t['ineq2'], t['from2'], t['to2'], t['category2'])

                except:

                    print('Problem occurred creating column given by following parameters:', {'granularity': t['granularity'], 'type': t['type'],
                                                                                              'condition1': t['condition1'], 'from1': t['from1'], 'to1': t['to1'], 'category1': t['category1'],
                                                                                              'condition2': t['condition2'], 'from2': t['from2'], 'to2': t['to2'], 'category2': t['category2']})

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

        del self.X_in

        gc.collect()

        # SQL string
        self.strsql_.append('from _TABLENAME_\n' +
                         'where ' + self.time_name + ' <= ' + self.time_max_name + '\n' +
                         'group by ' + self.id_name + '\n')
        self.strsql_ = ''.join(self.strsql_)

        return self.X_out
