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
from IPython.display import display
import numbers
from .utils import is_number, optimize_dtypes

class Interactions(BaseEstimator, TransformerMixin):
    '''
    Creates combination from the columns of a dataset. The available types of combinations are:
    -	sum: sum of two variables, i.e. var1 + var2, makes sense for numbers only
    -	product: product of two variables, i.e. var1 * var2, makes sense for numbers only
    -	difference: difference of two variables, i.e. var1 - var2, makes sense for numbers only
    -	ratio: ratio of two variables, i.e. var1/var2, makes sense for numbers only
    -	cartesian: Cartesian product of two variables, makes sense for strings only
    -	equality: comparison of two variables, the result is one of following: “<”, “=”, “>”, can be applied either to strings or to numbers
    -	length: length of string variable – in this case, variable2 is not used
    -	clean: applies “cleaning”, i.e. filling of NaNs and infinites, to one variable – in this case, variable2 is not used    

    Args:
        id_name (string ): ID column name of the datasets
        intermeta (matrix): matrix of slice metadata - defining how the raw data set should be sliced, columns: 
            1)	variable1: name of variable that should be combined with variable2
            2)	variable2: name of variable that should be combined with variable1
            3)	type: type of combination (the available types are mentioned in the description above)
            4)	in_nan: how NaNs should be treated BEFORE the transformation (can be left empty)
            5)	in_inf: how Infinities should be treated BEFORE the transformation (can be left empty)
            6)	out_nan: how NaNs should be treated AFTER the transformation (can be left empty)
            7)	out_inf: how Infinities should be treated AFTER the transformation (can be left empty)
            8)	bins: for “quantiles” type of transformation, how many quantiles should be created
        uppercase_suffix (boolean, optional): boolean if the suffix of the aggregation type should be uppercase 

    Attributes:
        X_in (pd.DataFrame): input
        X_out (pd.DataFrame): output 
        sql_ (string): SQL query which makes the same transformation on Oracle database

    Methods:
        fit(X, y = None) : go through the interaction metadata and put all valid aggregations into a special structure
            X (pd.DataFrame): the dataframe you want the aggregations to be executed on
        transform(X) : execute the interactions 
            X (pd.DataFrame): the dataframe you want the aggregations to be executed on 
    '''

    def __init__(self, id_name, intermeta, uppercase_suffix=True):
        self.id_name = id_name
        self.intermeta = check_array(intermeta, dtype=object, force_all_finite=False)
        self.intermeta = pd.DataFrame(self.intermeta)
        self.intermeta.columns = ['variable1', 'variable2', 'type', 'in_nan', 'in_inf', 'out_nan', 'out_inf', 'bins']
        self.uppercase_suffix = uppercase_suffix

    def fit(self, X, y=None):
        # creates internal structures which are then used to calculate the features

        # boolean telling that the dataset is not valid
        cantfit = False

        # check whether the history is long enough and ID and TIME ID columns are present in the X data set
        if self.id_name not in X.columns:
            print('ID column', self.id_name, 'not present in the dataset!')
            cantfit = True

        if not cantfit:
            self.intermeta_ = self.intermeta.copy()

            # do some basic validity checks of metadata tables
            # new column telling us whether the row is valid for the data provided
            self.intermeta_['valid'] = 'OK'

            self.intermeta_.loc[~self.intermeta_['variable1'].isin(
                X.columns), 'valid'] = 'variable1 missing in provided dataset'
            self.intermeta_.loc[(pd.notnull(self.intermeta_['variable2'])) & (
                ~self.intermeta_['variable2'].isin(X.columns)), 'valid'] = 'variable2 missing in provided dataset'
            self.intermeta_.loc[((pd.isnull(self.intermeta_['variable1'])) | (pd.isnull(self.intermeta_['variable2']))) & (self.intermeta_['type'].isin(
                {'sum', 'product', 'cartesian', 'difference', 'ratio', 'equality'})), 'valid'] = 'variable1 or variable2 not filled, both needed for this type'

            self.intermeta_.loc[self.intermeta_['type'].isin({'length', 'clean', 'quantiles'}), 'variable2'] = ''
            self.intermeta_.loc[((pd.isnull(self.intermeta_['bins'])) | (self.intermeta_['bins'] < 2))
                                & (self.intermeta_['type'].isin({'cartesian', 'quantiles'})), 'bins'] = 10

            # keep only valid rows and deduplicate
            intermeta_invalid = self.intermeta_[self.intermeta_['valid'] != 'OK'].copy()
            self.intermeta_ = self.intermeta_[self.intermeta_['valid'] == 'OK'].drop_duplicates(inplace=False)

            # expected number of new features is sum of expected number of new features of each row
            self.nr_columns_total = self.intermeta_.shape[0]

            if self.nr_columns_total == 0:
                self.intermeta_ = None
            else:
                print('Expected number of new columns:', self.nr_columns_total)

            # print invalid columns so user can review them
            if len(intermeta_invalid) > 0:
                print('The following rows of interactions meta data were ommited as they are invalid (reason given in last column):')
                display(intermeta_invalid)

            #necessary columns to copy in transform() into self.X_in
            self.cols_needed_in_copy = list(set([self.id_name] + \
                [col for col in self.intermeta_['variable1'].unique()] + [col for col in self.intermeta_['variable2'].unique()]) - {'',None,np.nan})

            #RAM usage estimation
            mem_X = X.memory_usage(index = True, deep = True)
            rows_X = X.shape[0]
            mem_estimate = 0
            for _, feature in self.intermeta_.iterrows():
                if feature['type'] == 'cartesian': bytes_feature = 128
                if feature['type'] == 'quantiles': bytes_feature = 16
                if feature['type'] == 'equality': bytes_feature = 64
                if feature['type'] == 'length': bytes_feature = 8
                else: 
                    bytes_feature = mem_X[feature['variable1']] / rows_X
                    if len(feature['variable2']) > 0:
                        if mem_X[feature['variable2']] > bytes_feature:
                            bytes_feature = mem_X[feature['variable2']] / rows_X
                mem_feature = bytes_feature * rows_X
                mem_estimate += mem_feature
            print('Rough upper estimate of memory usage:',round(mem_estimate/1024/1024/1024,3),'GB')

        if not hasattr(self, 'intermeta_'):
            print('There are no valid interactions meta data fulfilling given criteria!')
        return

    def __treat(self, col, colsql, nans, infs):
        # function __treat replaces NaNs and infinities by values given by parameters
        if col.dtype.name == 'category':
            ncol = col.astype(str)
        else:
            ncol = col
        ncolsql = colsql
        if infs == 'nan':
            infs = np.nan
        if (nans is not None) and (pd.notnull(nans)):
            ncol = ncol.fillna(nans)
            if is_number(nans):
                ncolsql = 'nvl(' + ncolsql + ',' + str(nans) + ')'
            else:
                ncolsql = 'nvl(' + ncolsql + ",'" + str(nans) + "')"
        if (infs is not None):
            if (is_number(infs)):
                ncol = ncol.replace([np.inf, -np.inf], [infs, -infs])
            else:
                ncol = ncol.replace([np.inf, -np.inf], [str(infs), '-' + str(infs)])
        return ncol, ncolsql

    def __sum2(self, col1, col2, in_nan, in_inf, out_nan, out_inf):
        # function __sum2 sums two numerical columns and treats NaNs and infs on input and output
        c1 = self.X_in[col1]
        c1sql = col1
        c2 = self.X_in[col2]
        c2sql = col2
        c1, c1sql = self.__treat(c1, c1sql, in_nan, in_inf)
        c2, c2sql = self.__treat(c2, c2sql, in_nan, in_inf)
        ncol, ncolsql = self.__treat(c1 + c2, c1sql + '+' + c2sql, out_nan, out_inf)
        ncolname = 's_' + col1 + '_' + col2
        if self.uppercase_suffix:
            ncolname = ncolname.upper()
        sql = ' ,' + ncolsql + ' as ' + ncolname + '\n'
        self.strsql_.append(sql)
        self.X_buff[ncolname] = ncol
        print('Variable', ncolname, 'created. (', self.nr_columns_done_print, '/', self.nr_columns_total, ')')
        return

    def __product(self, col1, col2, in_nan, in_inf, out_nan, out_inf):
        # function __product multiplies two numerical columns and treats NaNs and infs on input and output
        c1 = self.X_in[col1]
        c1sql = col1
        c2 = self.X_in[col2]
        c2sql = col2
        c1, c1sql = self.__treat(c1, c1sql, in_nan, in_inf)
        c2, c2sql = self.__treat(c2, c2sql, in_nan, in_inf)
        ncol, ncolsql = self.__treat(c1 * c2, c1sql + '*' + c2sql, out_nan, out_inf)
        ncolname = 'p_' + col1 + '_' + col2
        if self.uppercase_suffix:
            ncolname = ncolname.upper()
        sql = ' ,' + ncolsql + ' as ' + ncolname + '\n'
        self.strsql_.append(sql)
        self.X_buff[ncolname] = ncol
        print('Variable', ncolname, 'created. (', self.nr_columns_done_print, '/', self.nr_columns_total, ')')
        return

    def __difference(self, col1, col2, in_nan, in_inf, out_nan, out_inf):
        # function __difference subtracts two numerical columns and treats NaNs and infs on input and output
        c1 = self.X_in[col1]
        c1sql = col1
        c2 = self.X_in[col2]
        c2sql = col2
        c1, c1sql = self.__treat(c1, c1sql, in_nan, in_inf)
        c2, c2sql = self.__treat(c2, c2sql, in_nan, in_inf)
        ncol, ncolsql = self.__treat(c1 - c2, c1sql + '-' + c2sql, out_nan, out_inf)
        ncolname = 'd_' + col1 + '_' + col2
        if self.uppercase_suffix:
            ncolname = ncolname.upper()
        sql = ' ,' + ncolsql + ' as ' + ncolname + '\n'
        self.strsql_.append(sql)
        self.X_buff[ncolname] = ncol
        print('Variable', ncolname, 'created. (', self.nr_columns_done_print, '/', self.nr_columns_total, ')')
        return

    def __ratio(self, col1, col2, in_nan, in_inf, out_nan, out_inf):
        # function __ratio divides two numerical columns and treats NaNs and infs on input and output
        c1 = self.X_in[col1]
        c1sql = col1
        c2 = self.X_in[col2]
        c2sql = col2
        c1, c1sql = self.__treat(c1, c2sql, in_nan, in_inf)
        c2, c2sql = self.__treat(c2, c2sql, in_nan, in_inf)
        ncol, ncolsql = self.__treat(c1 / c2, c1sql + '/' + c2sql, out_nan, out_inf)
        ncolname = 'r_' + col1 + '_' + col2
        if self.uppercase_suffix:
            ncolname = ncolname.upper()
        sql = ' ,case when ' + c2sql + '=0 then ' + 'sign(' + c1sql + ')*' + str(out_inf) + ' else ' + ncolsql +\
              ' end as ' + ncolname + '\n'
        self.strsql_.append(sql)
        self.X_buff[ncolname] = ncol
        print('Variable', ncolname, 'created. (', self.nr_columns_done_print, '/', self.nr_columns_total, ')')
        return

    def __cartesian(self, col1, col2, in_nan, in_inf, out_nan, out_inf, bins):
        # function __cartesian outputs cartesian product of two columns. If any of them is numerical, it first calculates quantiles (so the variable becomes categorical).
        c1 = self.X_in[col1]
        c1sql = col1
        c2 = self.X_in[col2]
        c2sql = col2
        c1, c1sql = self.__treat(c1, c1sql, in_nan, in_inf)
        c2, c2sql = self.__treat(c2, c2sql, in_nan, in_inf)
        if c1.dtype != object:
            c1 = pd.qcut(c1.astype(float), q=int(bins), duplicates='drop')
            c1sql = '/*quantile not supported by SQL*/'
        if c2.dtype != object:
            c2 = pd.qcut(c2.astype(float), q=int(bins), duplicates='drop')
            c2sql = '/*quantile not supported by SQL*/'
        ncol, ncolsql = self.__treat(c1.astype(str) + '_' + c2.astype(str), c1sql +
                                     " || '_' || " + c2sql, out_nan, out_inf)
        ncolname = 'x_' + col1 + '_' + col2
        if self.uppercase_suffix:
            ncolname = ncolname.upper()
        sql = ' ,' + ncolsql + ' as ' + ncolname + '\n'
        self.strsql_.append(sql)
        self.X_buff[ncolname] = ncol
        print('Variable', ncolname, 'created. (', self.nr_columns_done_print, '/', self.nr_columns_total, ')')
        return

    def __equality(self, col1, col2, in_nan, in_inf, out_nan, out_inf):
        # function __equality compares values of two variables and returns (in)equality sign
        c1 = self.X_in[col1]
        c1sql = col1
        c2 = self.X_in[col2]
        c2sql = col2
        c1, c1sql = self.__treat(c1, c1sql, in_nan, in_inf)
        c2, c2sql = self.__treat(c2, c2sql, in_nan, in_inf)
        flag_l = (c1 < c2)
        flag_g = (c1 > c2)
        flag_e = (c1 == c2)
        ncol = c1.astype(str).copy()
        ncol[flag_l] = '<'
        ncol[flag_g] = '>'
        ncol[flag_e] = '='
        ncolsql = 'case when ' + c1sql + '<' + c2sql + " then '<' when " + c1sql + '<' + c2sql + " then '>' when " +\
            c1sql + '=' + c2sql + " then '=' end"
        ncol, ncolsql = self.__treat(ncol, ncolsql, out_nan, out_inf)
        ncolname = 'e_' + col1 + '_' + col2
        if self.uppercase_suffix:
            ncolname = ncolname.upper()
        sql = ' ,' + ncolsql + ' as ' + ncolname + '\n'
        self.strsql_.append(sql)
        self.X_buff[ncolname] = ncol
        print('Variable', ncolname, 'created. (', self.nr_columns_done_print, '/', self.nr_columns_total, ')')
        return

    def __quantiles(self, col, in_nan, in_inf, out_nan, out_inf, bins):
        # function __quantiles transforms numerical variable into categorical. it creates splits so the variable transforms into given number of categories (this number is given by parameter) of the same size
        c1 = self.X_in[col]
        c1sql = col
        c1, c1sql = self.__treat(c1, c1sql, in_nan, in_inf)
        ncol, ncolsql = self.__treat(pd.qcut(c1.astype(float), q=int(bins), duplicates='drop'),
                                     '/*quantile not supported by SQL*/', out_nan, out_inf)
        ncolname = 'q_' + col
        if self.uppercase_suffix:
            ncolname = ncolname.upper()
        sql = ' ,' + ncolsql + ' as ' + ncolname + '\n'
        self.strsql_.append(sql)
        self.X_buff[ncolname] = ncol
        print('Variable', ncolname, 'created. (', self.nr_columns_done_print, '/', self.nr_columns_total, ')')
        return

    def __str_len(self, col, in_nan, in_inf, out_nan, out_inf):
        # function __str_len calculates lenght of string variable
        c1 = self.X_in[col]
        c1sql = col
        c1, c1sql = self.__treat(c1, c1sql, in_nan, in_inf)
        c1[pd.isnull(c1)] = ''
        ncol, ncolsql = self.__treat(c1.astype(str).str.len(), 'length('+c1sql+')', out_nan, out_inf)
        ncolname = 'l_' + col
        if self.uppercase_suffix:
            ncolname = ncolname.upper()
        sql = ' ,' + ncolsql + ' as ' + ncolname + '\n'
        self.strsql_.append(sql)
        self.X_buff[ncolname] = ncol
        print('Variable', ncolname, 'created. (', self.nr_columns_done_print, '/', self.nr_columns_total, ')')
        return

    def __clean(self, col, in_nan, in_inf, out_nan, out_inf):
        # function __clean replaces NaNs and infinities on input and input (actually it makes sense only once, but it is done twice to be consistent with the other functions)
        c1 = self.X_in[col]
        c1sql = col
        c1, c1sql = self.__treat(c1, c1sql, in_nan, in_inf)
        ncol, ncolsql = self.__treat(c1, c1sql, out_nan, out_inf)
        ncolname = 'c_' + col
        if self.uppercase_suffix:
            ncolname = ncolname.upper()
        sql = ' ,' + ncolsql + ' as ' + ncolname + '\n'
        self.strsql_.append(sql)
        self.X_buff[ncolname] = ncol
        print('Variable', ncolname, 'created. (', self.nr_columns_done_print, '/', self.nr_columns_total, ')')
        return

    def transform(self, X, resume = False, resume_from = None):
        
        check_is_fitted(self, ['intermeta_'])

        if not resume:

            self.X_in = X
            self.X_out = X[[self.id_name]].copy()
            self.X_buff = self.X_out.drop(self.X_out.columns, axis=1, inplace=False)

            self.aggnames_ = list()
            self.strsql_ = ['select t.*\n']

            # self.nr_columns_done = count of new columns that have been already created
            self.nr_columns_done = 0
            self.nr_columns_done_print = 0

        if (resume) and (not resume_from):
            self.resume_from = self.nr_columns_done
            self.nr_columns_done_print = self.nr_columns_done
        elif resume:
            self.resume_from = resume_from-1
            self.nr_columns_done_print = self.nr_columns_done

        for ii, i in self.intermeta_.iterrows():

            if (not resume) or (resume and ii+1 > self.resume_from):

                self.nr_columns_done_print += 1

                try:

                    if i['type'] == 'sum':
                        self.__sum2(i['variable1'], i['variable2'], i['in_nan'], i['in_inf'], i['out_nan'], i['out_inf'])
                    elif i['type'] == 'product':
                        self.__product(i['variable1'], i['variable2'], i['in_nan'], i['in_inf'], i['out_nan'], i['out_inf'])
                    elif i['type'] == 'difference':
                        self.__difference(i['variable1'], i['variable2'], i['in_nan'], i['in_inf'], i['out_nan'], i['out_inf'])
                    elif i['type'] == 'ratio':
                        self.__ratio(i['variable1'], i['variable2'], i['in_nan'], i['in_inf'], i['out_nan'], i['out_inf'])
                    elif i['type'] == 'cartesian':
                        self.__cartesian(i['variable1'], i['variable2'], i['in_nan'],
                                        i['in_inf'], i['out_nan'], i['out_inf'], i['bins'])
                    elif i['type'] == 'equality':
                        self.__equality(i['variable1'], i['variable2'], i['in_nan'], i['in_inf'], i['out_nan'], i['out_inf'])
                    elif i['type'] == 'quantiles':
                        self.__quantiles(i['variable1'], i['in_nan'], i['in_inf'], i['out_nan'], i['out_inf'], i['bins'])
                    elif i['type'] == 'length':
                        self.__str_len(i['variable1'], i['in_nan'], i['in_inf'], i['out_nan'], i['out_inf'])
                    elif i['type'] == 'clean':
                        self.__clean(i['variable1'], i['in_nan'], i['in_inf'], i['out_nan'], i['out_inf'])

                except:

                    print('Problem occurred creating column given by following parameters:', {'var1': i['variable1'], 'var2': i['variable2'], 'type': i['type']})

                # if at least 20 columns in buffer, move data from buffer to output
                # optimization of data types is done here and not in the particular methods as it is easier here and if the buffer is emptied often enough, it should not affect RAM consumption too much
                if len(self.X_buff.columns) >= 20:
                    self.X_out = pd.concat([self.X_out, optimize_dtypes(self.X_buff)], axis=1)
                    self.X_buff.drop(self.X_buff.columns, axis=1, inplace=True)

                gc.collect()

                self.nr_columns_done += 1

        # move the rest of the data from buffer to the output
        if len(self.X_buff.columns) > 0:
            self.X_out = pd.concat([self.X_out, optimize_dtypes(self.X_buff)], axis=1)
            self.X_buff.drop(self.X_buff.columns, axis=1, inplace=True)

        gc.collect()

        self.strsql_.append('from _TABLENAME_ t\n')
        self.strsql_ = ''.join(self.strsql_)
        gc.collect()
        return self.X_out
