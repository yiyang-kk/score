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
import pandas as pd
import numpy as np
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from sklearn.utils import check_array
from IPython.display import display
import gc
import sys
from .utils import is_number, optimize_dtypes


class Slicer(BaseEstimator, TransformerMixin):
    """
    Args:
        id_name (string):
            ID column name of the datasets
            that will be used in fit and transform procedures
        time_name (string):
            TIME ID column name of the datasets
            that will be used in fit and transform procedures
        time_max_name (string):
            TIME MAX column is the column of "current time" for each ID,
            i.e. time from which the history is calculated back from
            Example:
                time_max_name = application date, 
                time_name = transaction date, 
                only rows with time_name <= time_max_name are taken into account
        slicemeta (matrix):
            a matrix of slice metadata
                - defining how the raw data set should be sliced
            columns:
                1) variable name (variable to be aggregated),
                2) aggregation func (e.g. sum),
                3) segmentation variable (meaning the aggregation
                should be segmented by this variable) which can be also empty,
                4) relative segmentation flag (int 0/1) (if the segmented aggr. 
                should be calculated as ratio to the non-segm. aggr.)    
                5) categorical variable flag (int 0/1) flag
                whether the variable is categorical
                (for such variables columns 4 and 5 are ignored)
                6) 1 if np.nan is considered a category
                    (applies only if column 5 indicates the variable is categorical) or 0 if removed
                7) metric for some categorical variable aggregation:
                    for argmax/argmin/argmaxsum/argminsum/argmaxmean/argminmean type of aggregations, 
                    this is the metric the max/sum/mean is calculated from
        time_granularity (string, optional):
            granularity of the time slices, can be 'months', 'weeks', 'days', 'hours'
        history_length (int, optional):
            how long the history should be (examplle values: 12 or 24 for months, 30 for days etc.)
        time_format (string, optional):
            the format of time columns
        uppercase_suffix (boolean, optional):
            boolean if the suffix of the aggregation type should be uppercase

    Attributes:
        X_in (pd.DataFrame):
            input
        X_out (pd.DataFrame):
            output
        sql_ (string):
            SQL query which makes the same transformation on Oracle database

    Methods:
        fit(X, y = None) :
            go through the aggregation metadata and put all valid aggregations into a special structure
            X (pd.DataFrame):
                the dataframe you want the aggregations to be executed on
        transform(X) :
            execute the slice aggregations
            X (pd.DataFrame):
                the dataframe you want the aggregations to be executed on
    """

    def __init__(self, id_name, time_name, time_max_name, slicemeta,
                 time_granularity='months', history_length=12, time_format='%Y-%m-%d', uppercase_suffix=True):
        # initiation of the instance

        # a matrix of slice metadata - defining how the raw data set should be sliced
        # columns: Variable name (variable to be aggregated), aggregation func (e.g. sum),
        #          segmentation variable (meaning the aggregation should be segmented by this variable)
        #               which can be also empty,
        #          relative segmentation flag (if the segmented aggr.
        #               should be calculated as ratio to the non-segm. aggr.)
        self.slicemeta = check_array(slicemeta, dtype=object, force_all_finite=False)
        if self.slicemeta.shape[1] < 7:
            zeros = np.zeros((self.slicemeta.shape[0], 7-self.slicemeta.shape[1]))
            self.slicemeta = np.append(self.slicemeta, zeros, axis=1)
        self.slicemeta = pd.DataFrame(self.slicemeta)
        self.slicemeta.columns = ['col', 'func', 'segm_var', 'relative', 'cat', 'nancat', 'metric']

        # ID and TIME ID column names of the datasets that will be used in fit and transform procedures
        # TIME MAX column is the column of "current time" for each ID,
        #       i.e. time from which the history is calculated back from
        # Example: time_max_name = application date,
        #          time_name = transaction date, only rows with time_name <= time_max_name
        #          are taken into account
        self.id_name = id_name
        self.time_name = time_name
        self.time_max_name = time_max_name

        # the format of time columns
        self.time_format = time_format
        # granularity of the time slices, can be 'months', 'weeks', 'days', 'hours'
        self.time_granularity = time_granularity
        # how long the history should be (examplle values: 12 or 24 for months, 30 for days etc.)
        self.history_length = history_length

        # boolean if the suffix of the aggregation type should be uppercase
        self.uppercase_suffix = uppercase_suffix

    def __relative_time(self, col, newcol, deltacol, adddelta):
        # this function creates newcol from col adding delta of time: (deltacol+addelta)*self.time_granularity

        self.X_out[newcol] = self.X_out.apply(
            lambda row: row[col] +
            relativedelta(**{self.time_granularity: -row[deltacol] + adddelta}), axis=1)  # using key-word arguments

    def __slice_general(self, variable, func, add_col):
        """Overloaded - creates both the python and sql transformation

        general slice: func(variable) group by self.id_name, self.order_name

        create temporary data set with the aggregation
         1. merge X_in and X_out on ID_NAME
         2. filter only TIME_NAME between FROM_NAME and TO_NAME
         3. calculate func ... group by ...

        Arguments:
            variable {string} -- variable on which action will be performed
            func {string} -- function which will be applied
            add_col {str?} -- should the column be added

        Returns:
            Temporary part of the dataset with given aggregation
            Also appends to final sql
        """
        X_tmp = pd.merge(self.X_in[[self.id_name, self.time_name, variable]],
                         self.X_base,
                         on=self.id_name)
        X_tmp = X_tmp[(X_tmp[self.time_name] > X_tmp[self.from_name]) &
                      (X_tmp[self.time_name] <= X_tmp[self.to_name])]

        X_tmp = getattr(X_tmp.groupby([self.id_name, self.order_name])[variable], func)().to_frame()

        # set the column name of the aggregation
        column_name = f"{variable}_{(func.upper() if self.uppercase_suffix else func)}"
        X_tmp.columns = [column_name]

        # SQL string
        func_out = func if func != 'mean' else 'avg'
        sql_tmp = f"{func_out}({variable})"
        sql_full = f",{sql_tmp} as {column_name}\n"

        # add the aggregation to self.X_buff, if we want to
        if add_col == 1:
            X_tmp = optimize_dtypes(X_tmp)
            self.X_buff = pd.concat([self.X_buff, X_tmp], axis=1)
            print(f"Variable {column_name} created. ({self.nr_columns_done_print} / {self.nr_columns_total})")
            self.strsql_.append(sql_full)

        return X_tmp, sql_tmp

    def __slice_segmented(self, variable, func, segm_var, add_col):
        # segmented slice: func(variable) group by self.id_name, self.order_name, segm_var

        # create temporary data set with the aggregations
        #  1. merge X_in and X_out on ID_NAME
        #  2. filter only TIME_NAME between FROM_NAME and TO_NAME
        #  3. calculate func ... group by ...
        #  4. make simple columns from column multiindex
        X_tmp = pd.merge(self.X_in[[self.id_name, self.time_name, variable, segm_var]],
                         self.X_base,
                         on=self.id_name)
        X_tmp = X_tmp[(X_tmp[self.time_name] > X_tmp[self.from_name]) &
                      (X_tmp[self.time_name] <= X_tmp[self.to_name])]
        X_tmp[segm_var] = X_tmp[segm_var].astype(str)
        X_tmp = getattr(X_tmp.groupby([self.id_name, self.order_name, segm_var])[variable], func)() \
            .to_frame() \
            .unstack(level=-1)

        # set the column names of the aggregations
        segments = X_tmp.columns.get_level_values(1).astype(str)
        column_names = (X_tmp.columns.get_level_values(0).astype(str) + '_' +
                        (segm_var.upper() if self.uppercase_suffix else segm_var) + '_' +
                        segments + '_' +
                        (func.upper() if self.uppercase_suffix else func))

        X_tmp.columns = column_names

        # SQL string
        func_out = func if func != 'mean' else 'avg'
        sql_list = []
        sql_full = ''
        for i, segment in enumerate(segments):
            sql_tmp = f"{func_out}(case when {segm_var} = '{segment}' then {variable} end)"
            sql_list.append(sql_tmp)
            sql_full = sql_full + f", {sql_tmp} as {column_names[i]}\n"

        # add the aggregation to self.X_buff, if we want to
        if add_col == 1:
            X_tmp = optimize_dtypes(X_tmp)
            self.X_buff = pd.concat([self.X_buff, X_tmp], axis=1)
            print(f'Variables {list(column_names)} created. ({self.nr_columns_done_print}/{self.nr_columns_total})')
            self.strsql_.append(sql_full)

        return X_tmp, sql_list

    def __slice_relative(self, variable, func, segm_var, add_col):
        # relative segmented slice: general slice / segmented slice

        # call general and segmented slice
        gen, sql_gen = self.__slice_general(variable, func, 0)
        segm, sql_segm = self.__slice_segmented(variable, func, segm_var, 0)

        # merge both slices and calculate ratio
        segmrel = pd.merge(gen, segm, left_index=True, right_index=True)
        X_tmp = segmrel[list(segm.columns)].div(segmrel[gen.columns[0]], axis='index')

        # set the column names of the aggregations
        r_suff = '_R' if self.uppercase_suffix else '_r'
        X_tmp.columns = X_tmp.columns + r_suff

        # SQL string
        for i, single_sql in enumerate(sql_segm):
            sql_full = f',case when {sql_gen} + <> 0 then {single_sql}/{sql_gen} end as {X_tmp.columns[i]}\n'
        # add the aggregation to self.X_buff, if we want to
        if add_col == 1:
            X_tmp = optimize_dtypes(X_tmp)
            self.X_buff = pd.concat([self.X_buff, X_tmp], axis=1)
            print(f'Variables {list(X_tmp.columns)} created. ({self.nr_columns_done_print}/{self.nr_columns_total})')
            self.strsql_.append(sql_full)

    def __slice_categorical(self, variable, func, nancat, metric, add_col):

        # slice based on a categorical variable

        # create temporary data set with the aggregation
        #  1. merge X_in and X_out on ID_NAME
        #  2. filter only TIME_NAME between FROM_NAME and TO_NAME

        # if there is metric defined for given slice, X takes the columns with given metric (?)
        if len(metric) > 0:
            X_tmp = pd.merge(self.X_in[[self.id_name, self.time_name, variable, metric]],
                             self.X_base,
                             on=self.id_name)

        # otherwise we take only subset
        else:
            X_tmp = pd.merge(self.X_in[[self.id_name, self.time_name, variable]],
                             self.X_out[[self.order_name, self.from_name, self.to_name]],
                             on=self.id_name)
        X_tmp.reset_index(inplace=True)
        X_tmp = X_tmp[(X_tmp[self.time_name] > X_tmp[self.from_name]) &
                      (X_tmp[self.time_name] <= X_tmp[self.to_name])] \
            .sort_values([self.id_name, self.order_name, self.time_name, variable],
                         ascending=[True, False, True, False])

        #  3. fill NA values
        if nancat == 1:
            #  3a. if NA is a separate category
            X_tmp[variable] = X_tmp[variable].astype(str)
            X_tmp.loc[X_tmp[variable].str.lower() == 'nan', variable] = 'NaN'
            sql_varname = 'nvl(' + variable + ",'NaN')"
            sql_tablename = '_TABLENAME_'
        else:
            #  3b. if NA is skipped
            X_tmp = X_tmp[pd.notnull(X_tmp[variable])]
            X_tmp[variable] = X_tmp[variable].astype(str)
            sql_varname = variable
            sql_tablename = '(select * from _TABLENAME_ where ' + variable + ' is not null)'

        # creating sql query is complicated here, so we need to determine whether we need to use subquery
        sql_subquery = ''
        column_name = variable + '_' + func
        sql_done = False
        #  4. apply function - each function is applied in an special way as they are not simple pandas agg funtions

        if (func in ['last', 'first']):
            #  4c. LAST: the most recent observation in the given time (argmax by time variable)
            #  4d. FIRST: the most recent observation in the given time (argmin by time variable)
            X_tmp = X_tmp.groupby([self.id_name, self.order_name])[self.time_name, variable]

            sql_subquery = f"""
                (select distinct *
                   from (select {self.id_name}
                                ,{self.strsql_timecondition} as {self.order_name}
                                ,{func}_value({sql_varname}) over (partition by {self.id_name},
                                                                                {self.strsql_timecondition0}
                                                                       order by {self.time_name} asc) as val
                           from {sql_tablename} 
                          where {self.strsql_timecondition} <= {self.history_length}
                            and {self.strsql_timecondition} >= 1)) TAB_{column_name}

                """

        elif (func in ['argmax', 'argmin']):
            #  4e. ARGMAX: observation with the highest value of metric
            #  4e2. ARGMIN: observation with the highest value of metric
            X_tmp = X_tmp.groupby([self.id_name, self.order_name])[metric, variable]
            column_name = column_name + '_' + metric
            direction = 'asc' if func == 'argmin' else 'desc'
            sql_subquery = f"""
                (select distinct *
                   from (select {self.id_name}
                                ,{self.strsql_timecondition} as {self.order_name}
                                ,first_value({sql_varname}) over (partition by {self.id_name},
                                                                               {self.strsql_timecondition0}
                                                                      order by {metric} {direction}) as val
                           from {sql_tablename}
                          where {self.strsql_timecondition} <= {self.history_length}
                            and {self.strsql_timecondition} >= 1)) TAB_{column_name}

                """
        elif (func in ['argmaxsum', 'argminsum', 'argmaxmean', 'argminmean']):
            #  4f. ARGMAXSUM: observation with the highest sum of metric
            #  4f2. ARGMINSUM: observation with the lowest sum of metric
            #  4g. ARGMAXMEAN: observation with the highest mean of metric
            #  4g2. ARGMINMEAN: observation with the lowest mean of metric

            # bit hacky:
            # when the chosen function contains 'sum',
            # we use 'sum' method of the pandas dataframe, which is called using getattr (get attribute).
            # Otherwise we get mean.
            # Parentheses after getattr are needed as we only got the function. It needs to be called.

            X_tmp = getattr(
                X_tmp.groupby([self.id_name, self.order_name, variable])[metric],
                'sum' if 'sum' in func else 'mean')() \
                .reset_index() \
                .groupby([self.id_name, self.order_name])
            direction = 'asc' if 'min' in func else 'desc'
            sql_func = 'sum' if 'sum' in func else 'avg'
            column_name = column_name + '_' + metric
            sql_subquery = f"""
                (select distinct *
                   from (select {self.id_name}
                                ,{self.order_name}
                                ,first_value(val) over (partition by {self.id_name},
                                                                     {self.order_name}
                                                            order by val {direction}) as val
                           from (select {self.id_name}
                                        ,{self.strsql_timecondition} as {self.order_name},
                                        ,{sql_varname} as val
                                        ,{sql_func}({metric}) as val
                                   from {sql_tablename}
                                  where {self.strsql_timecondition} <= {self.history_length}
                                    and {self.strsql_timecondition}
                               group by {self.id_name}, {self.strsql_timecondition}, {sql_varname}))) TAB_{column_name}

                """

        elif (func == 'nchanges'):
            #  4h. NCHANGES: number of changes in the given time
            #  TODO: there is a problem with the result
            X_tmp['prev_cat'] = X_tmp[variable].shift(1)
            X_tmp['prev_id'] = X_tmp[self.id_name].shift(1)

            selector = X_tmp[self.id_name] != X_tmp['prev_id']
            X_tmp.loc[selector, 'prev_cat'] = X_tmp.loc[selector, variable]

            X_tmp = X_tmp.groupby([self.id_name, self.order_name])[variable, "prev_cat"]

            sql_subquery = f"""
                (select {self.id_name}
                        ,{self.order_name}
                        ,sum(case when val <> lagged then 1 else 0 end) as val
                   from (select {self.id_name}
                                ,{self.strsql_timecondition} as {self.order_name}
                                ,{sql_varname} as val
                                ,lag({sql_varname}) over (partition by {self.id_name}, {self.strsql_timecondition0}
                                                              order by {self.time_name} asc) as lagged
                           from {sql_tablename}
                          where {self.strsql_timecondition} <= {self.history_length}
                            and {self.strsql_timecondition} >= 1)
               group by {self.id_name}, {self.order_name}) TAB_{column_name}

                """
        if (func == 'mode'):
            #  4a. MODE: the most common category. mode returns multiple values if there are more of them. in such case
            #      we select just one. if there is no mode (all NaNs) we put NaN there

            X_tmp = X_tmp.groupby([self.id_name, self.order_name])[variable]
            sql_full = f',stats_mode({sql_varname}) as {column_name}\n'
            sql_done = True
        elif (func == 'nunique'):
            #  4b. NUNIQUE: number of unique categories (a.k.a. count distinct)
            X_tmp = X_tmp.groupby([self.id_name, self.order_name])[variable]
            sql_full = f',count(distinct {sql_varname}) as column_name\n'
            sql_done = True

        function_mapping = dict(
            mode=lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan,
            nunique=lambda x: x.nunique(),
            first=lambda x: x.loc[x[self.time_name].idxmin(), variable],
            last=lambda x: x.loc[x[self.time_name].idxmax(), variable],
            argmax=lambda x: x.loc[x[metric].idxmax(), variable],
            argmin=lambda x: x.loc[x[metric].idxmin(), variable],
            argmaxsum=lambda x: x.loc[x[metric].idxmax(), variable],
            argminsum=lambda x: x.loc[x[metric].idxmin(), variable],
            argmaxmean=lambda x: x.loc[x[metric].idxmax(), variable],
            argminmean=lambda x: x.loc[x[metric].idxmin(), variable],
            nchanges=lambda x: sum(x[variable] != x["prev_cat"])
        )
        X_tmp = X_tmp.apply(function_mapping[func]).to_frame()
        if self.uppercase_suffix:
            column_name = column_name.upper()
        if not sql_done:
            sql_full = f",max(TAB_{column_name}.val) as {column_name}\n"

        X_tmp.columns = [column_name]

        # add the aggregation to self.X_buff, if we want to
        if add_col == 1:
            X_tmp = optimize_dtypes(X_tmp)
            self.X_buff = pd.concat([self.X_buff, X_tmp], axis=1)
            print(f'Variable {column_name} created. ({self.nr_columns_done_print}/{self.nr_columns_total})')
            self.strsql_.append(sql_full)
            if sql_subquery:
                self.strsql_end.append(f"""
                                        left join {sql_subquery}
                                               on t.{self.id_name}=TAB_{column_name}.{self.id_name}
                                              and t.{self.order_name}=TAB_{column_name}.{self.order_name}
                                    """)

    def fit(self, X, y=None):
        # go through the slice metadata and put all valid aggregations into a special structure

        # boolean telling that the dataset is not valid
        cantfit = False

        # check whether the history is long enough and ID and TIME ID columns are present in the X data set
        if self.id_name not in X.columns:
            print('ID column', self.id_name, 'not present in the dataset!')
            cantfit = True
        if self.time_name not in X.columns:
            print('Time column', self.time_name, 'not present in the dataset!')
            cantfit = True
        if self.time_max_name not in X.columns:
            print('Time max column', self.time_max_name, 'not present in the dataset!')
            cantfit = True

        if not cantfit:
            self.slicemeta_ = self.slicemeta.copy()

            # do some basic validity checks of metadata tables
            # new column telling us whether the row is valid for the data provided
            self.slicemeta_['valid'] = 'OK'
            # new column telling us how many new features will be created thanks to this one metadata row
            self.slicemeta_['cnt_features'] = 1

            self.slicemeta_.loc[~self.slicemeta_['col'].isin(X.columns), 'valid'] = 'col missing in provided dataset'
            self.slicemeta_.loc[(pd.notnull(self.slicemeta_['segm_var'])) & (
                ~self.slicemeta_['segm_var'].isin(X.columns)), 'valid'] = 'segm_var missing in provided dataset'
            self.slicemeta_.loc[(pd.isnull(self.slicemeta_['segm_var'])), 'segm_var'] = ''
            self.slicemeta_.loc[(pd.isnull(self.slicemeta_['metric'])), 'metric'] = ''
            self.slicemeta_.loc[(self.slicemeta_['cat'] == 1) & (pd.isnull(self.slicemeta_['nancat'])), 'nancat'] = 0
            self.slicemeta_.loc[(self.slicemeta_['cat'] == 1) &
                                (((self.slicemeta_['func'] == 'argmax') |
                                  (self.slicemeta_['func'] == 'argmaxsum') |
                                    (self.slicemeta_['func'] == 'argmaxmean') |
                                    (self.slicemeta_['func'] == 'argmin') |
                                    (self.slicemeta_['func'] == 'argminsum') |
                                    (self.slicemeta_['func'] == 'argminmean')) &
                                 (self.slicemeta_['metric'] == '')), 'valid'] = 'missing metric'
            self.slicemeta_['relative'] = self.slicemeta_['relative'].fillna(0).astype('int')

            # count the new features
            segment_variables = self.slicemeta_[(self.slicemeta_['segm_var'] != '') & (
                self.slicemeta_['valid'] == 'OK')]['segm_var'].unique()
            for sv in segment_variables:
                sc = len(set(X[sv].unique()) - {np.nan})
                self.slicemeta_.loc[self.slicemeta_['segm_var'] == sv, 'cnt_features'] = sc
                if sc < 1:
                    self.slicemeta_.loc[self.slicemeta_['segm_var'] == sv,
                                        'valid'] = 'segm_var empty in provided dataset'

            # keep only valid rows and deduplicate
            slicemeta_invalid = self.slicemeta_[self.slicemeta_['valid']
                                                != 'OK'].drop(['cnt_features'], axis=1, inplace=False)
            self.slicemeta_ = self.slicemeta_[self.slicemeta_['valid'] == 'OK'].drop_duplicates(inplace=False)

            # expected number of new features is sum of expected number of new features of each row
            self.nr_columns_total = self.slicemeta_['cnt_features'].sum()
            self.slicemeta_['cumsum_features'] = self.slicemeta_['cnt_features'].cumsum()

            if self.nr_columns_total == 0:
                self.slicemeta_ = None
            else:
                print('Expected number of new columns:', self.nr_columns_total)

            # print invalid columns so user can review them
            if len(slicemeta_invalid) > 0:
                print(('The following rows of slice meta data were ommited as they are invalid '
                       '(reason given in last column):'))
                display(slicemeta_invalid)

            #necessary columns to copy in transform() into self.X_in
            self.cols_needed_in_copy = list(set([self.id_name, self.time_name, self.time_max_name] + \
                [col for col in self.slicemeta_['col'].unique()] + [col for col in self.slicemeta_['segm_var'].unique()] + \
                [col for col in self.slicemeta_['metric'].unique()]) - {'',None,np.nan})

            #RAM usage estimation
            mem_X = X.memory_usage(index = True, deep = True)
            rows_X = X.shape[0]
            entities_X = X[self.id_name].nunique()
            mem_estimate = 0
            for _, feature in self.slicemeta_.iterrows():
                if len(feature['metric']) > 0:
                    mem_feature = (mem_X[feature['metric']] / rows_X) * entities_X * self.history_length * feature['cnt_features']
                else:
                    mem_feature = (mem_X[feature['col']] / rows_X) * entities_X * self.history_length * feature['cnt_features']
                mem_estimate += mem_feature
            mem_estimate += 2 * sys.getsizeof(X[self.cols_needed_in_copy])
            print('Rough upper estimate of memory usage:',round(mem_estimate/1024/1024/1024,3),'GB')

        if not hasattr(self, 'slicemeta_'):
            print('There are no valid slice aggregations fulfilling given criteria!')

        return

    def transform(self, X, resume = False, resume_from = None):
        # execute the slice aggregations

        check_is_fitted(self, ['slicemeta_'])

        if not resume:

            self.X_in = X[self.cols_needed_in_copy].copy()
            self.X_in[self.time_name] = pd.to_datetime(self.X_in[self.time_name], format=self.time_format)
            self.X_in[self.time_max_name] = pd.to_datetime(
                self.X_in[self.time_max_name], format=self.time_format)
            self.X_in = self.X_in[self.X_in[self.time_max_name] >= self.X_in[self.time_name]]
            self.aggnames_ = list()

            # names of new columns
            if self.uppercase_suffix:
                self.order_name = 'TIME_ORDER'
                self.from_name = 'TIME_FROM'
                self.to_name = 'TIME_TO'
            else:
                self.order_name = 'time_order'
                self.from_name = 'time_from'
                self.to_name = 'time_to'

            # temporary data frame which includes all the relative time indexes (from 1 to self.history_length)
            slices = pd.DataFrame.from_records({self.order_name: list(range(1, self.history_length+1)), 'join': 1})

            # cross join of unique IDs with slices
            self.X_out = self.X_in.groupby(self.id_name)[self.time_max_name].max().to_frame()
            self.X_out['join'] = 1
            self.X_out.reset_index(level=0, inplace=True)
            self.X_out = pd.merge(self.X_out, slices, on='join')[[self.id_name, self.time_max_name, self.order_name]]

            # adding new columns with FROM and TO values indicating the time interval the aggregations will be taken from
            self.__relative_time(self.time_max_name, self.from_name, self.order_name, 0)
            self.__relative_time(self.time_max_name, self.to_name, self.order_name, 1)

            # index the data frame by ID and order
            self.X_out.set_index([self.id_name, self.order_name], inplace=True)
            self.X_out[self.order_name] = self.X_out.index.get_level_values(1)

            # self.X_base = base for each step, remains the same for the whole time
            self.X_base = self.X_out[[self.order_name, self.from_name, self.to_name]].copy()

            # self.X_buff = buffer which will be used to temporarily add new columns before they are all joined to X_out
            self.X_buff = self.X_out.drop(self.X_out.columns, axis=1, inplace=False)

            # SQL string - beginning of select
            if self.time_granularity == 'months':
                self.strsql_timecondition = 'floor(months_between(' + self.time_max_name + \
                    ',' + self.time_name + ')) + 1'
                self.strsql_timecondition0 = 'floor(months_between(' + self.time_max_name + \
                    ',' + self.time_name + '))'
                self.strsql_timefrom = 'add_months(' + self.time_max_name + ', -(' + self.strsql_timecondition + '))'
                self.strsql_timeto = 'add_months(' + self.time_max_name + ', -(' + self.strsql_timecondition + ') + 1)'
            elif self.time_granularity == 'days':
                self.strsql_timecondition = 'floor(' + self.time_max_name + ' - ' + self.time_name + ') + 1'
                self.strsql_timecondition0 = 'floor(' + self.time_max_name + ' - ' + self.time_name + ')'
                self.strsql_timefrom = self.time_max_name + ' - ' + self.strsql_timecondition
                self.strsql_timeto = self.time_max_name + ' - ' + self.strsql_timecondition + ' + 1'
            else:
                self.strsql_timecondition = '_THIS TIME GRANULARITY IS NOT SUPPORTED BY SQL CODE GENERATOR YET_'
                self.strsql_timecondition0 = '_THIS TIME GRANULARITY IS NOT SUPPORTED BY SQL CODE GENERATOR YET_'
                self.strsql_timefrom = self.strsql_timecondition
                self.strsql_timeto = self.strsql_timecondition
            self.strsql_ = ['select t.' + self.id_name + '\n' + ',' + self.time_max_name + '\n' +
                        ',' + self.strsql_timecondition + ' as TIME_ORDER\n' +
                        ',' + self.strsql_timefrom + ' as TIME_FROM\n' +
                        ',' + self.strsql_timeto + ' as TIME_TO\n']
            self.strsql_end = ['']

            # self.nr_columns_done = count of new columns that have been already created
            self.nr_columns_done = 0
            self.nr_columns_done_print = 0

        if (resume) and (not resume_from):
            self.resume_from = self.nr_columns_done
        elif resume:
            self.resume_from = resume_from-1

        # for each slice aggregation in list, create such aggregation
        for _, s in self.slicemeta_.iterrows():

            if (not resume) or (resume and s['cumsum_features'] > self.resume_from):
                
                self.nr_columns_done_print = s['cumsum_features']

                try:

                    # aggregation of a categorical variable
                    if s['cat'] == 1:
                        self.__slice_categorical(s['col'], s['func'], s['nancat'], s['metric'], 1)

                    # general aggregation
                    elif s['segm_var'] == '':
                        self.__slice_general(s['col'], s['func'], 1)

                    # aggregation segmented by a segmentation variable
                    elif s['relative'] == 0:
                        self.__slice_segmented(s['col'], s['func'], s['segm_var'], 1)

                    # aggregation segmented by a segmentation variable relative to the general aggregation
                    else:
                        self.__slice_relative(s['col'], s['func'], s['segm_var'], 1)

                except:

                    print('Problem occurred creating column given by following parameters:', {'column':s['col'], 'func':s['func'], 'segmentation':s['segm_var'], 'metric':s['metric']})

                # if at least 20 columns in buffer, move data from buffer to output
                if len(self.X_buff.columns) >= 20:
                    self.X_out = pd.concat([self.X_out, self.X_buff], axis=1)
                    self.X_buff.drop(self.X_buff.columns, axis=1, inplace=True)

                gc.collect()

                self.nr_columns_done = s['cumsum_features']

        # move the rest of the data from buffer to the output
        if len(self.X_buff.columns) > 0:
            self.X_out = pd.concat([self.X_out, self.X_buff], axis=1)
            self.X_buff.drop(self.X_buff.columns, axis=1, inplace=True)

        self.X_out.drop(self.order_name, axis=1, inplace=True)
        del self.X_in

        gc.collect()

        # SQL string - end of select
        self.strsql_.append('from _TABLENAME_ t \n')
        self.strsql_.extend(self.strsql_end)
        self.strsql_.append(f"""
                            where {self.strsql_timecondition} <= {self.history_length}
                              and {self.strsql_timecondition} >= 1
                         group by {self.strsql_timecondition}, t.{self.id_name}, {self.time_max_name}
                        """)
        self.strsql_ = ''.join(self.strsql_)
        # print(self.strsql_)

        return self.X_out.reset_index()


class OrderAssigner(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    time_name : string
        TIME ID column name of the datasets that will be used in fit and transform procedures
    time_max_name : string
        TIME MAX column is the column of "current time" for each ID, i.e. time from which the history is calculated back from
        Example: time_max_name = application date, time_name = transaction date, only rows with time_name <= time_max_name are taken into account
    time_granularity : string
        granularity of the time intervals, can be 'years', 'months', 'weeks', 'days', 'hours', 'minutes', 'seconds' or 'order' (in such case, just the order of the row in time (sorted descending from time_max_name) is returned)
    history_length : int, optional
        how long the history should be (examplle values: 12 or 24 for months, 30 for days etc.)
    time_format : string
        the format of time columns
    partition_name : string
        if time_granularity is 'order', put the name of the column with the ID of the client (or other partitioning entity) here. the order is the calculated in each partition (e.g. for each client) separately
    uppercase_suffix : boolean, optional
        boolean if the suffix of the aggregation type should be uppercase

    Attributes
    ----------
    X_out : DataFrame
        output 
    sql_ : string
        SQL query which makes the same transformation on Oracle database

    Methods
    ----------
    fit(X, y = None) :
        checks if the columns specified in parameters are present in the dataset
        X : dataframe
            the dataframe you want the aggregations to be executed on
    transform(X) :
        execute the time since aggregations   
        X : dataframe
            the dataframe you want the aggregations to be executed on
    """

    def __init__(self, time_name, time_max_name,
                 time_granularity='months', history_length=12, time_format='%Y-%m-%d', partition_name=None, uppercase_suffix=True):
        # initiation of the instance

        # TIME ID column name of the datasets that will be used in fit and transform procedures
        # TIME MAX column is the column of "current time" for each row, i.e. time from which the history is calculated back from
        # Example: time_max_name = application date, time_name = transaction date, only rows with time_name <= time_max_name
        #          are taken into account
        self.time_name = time_name
        self.time_max_name = time_max_name

        # the format of time columns
        self.time_format = time_format
        # granularity of the time slices, can be 'months', 'weeks', 'days', 'hours', 'minutes', 'seconds' or 'order'
        self.time_granularity = time_granularity
        # how long the history should be (examplle values: 12 or 24 for months, 30 for days etc.)
        self.history_length = history_length

        # partitioning entity for order - only if granularity is 'order'
        self.partition_name = partition_name

        # boolean if the suffix of the aggregation type should be uppercase
        self.uppercase_suffix = uppercase_suffix

    def fit(self, X, y=None):
        # just check if the important columns are on place

        # boolean telling that the dataset is not valid
        cantfit = False

        # check whether the history is long enough and ID and TIME ID columns are present in the X data set
        if self.time_name not in X:
            print('Time column', self.time_name, 'not present in the dataset!')
            cantfit = True
        if self.time_max_name not in X:
            print('Time max column', self.time_max_name, 'not present in the dataset!')
            cantfit = True
        if (self.time_granularity == 'order') and ((self.partition_name is None) or (self.partition_name not in X)):
            print('Partitioning column', self.partition_name, 'not present in the dataset!')
            cantfit = True

        if not cantfit:
            self.fit_ok_ = True

        #necessary columns to copy in transform() into self.X_in
        self.cols_needed_in_copy = list(set([self.partition_name, self.time_name, self.time_max_name]) - {'',None,np.nan})

        return

    def transform(self, X):

        check_is_fitted(self, ['fit_ok_'])

        self.X_out = X.copy()
        self.X_out[self.time_name] = pd.to_datetime(self.X_out[self.time_name], format=self.time_format)
        self.X_out[self.time_max_name] = pd.to_datetime(
            self.X_out[self.time_max_name], format=self.time_format)

        # remove the future
        self.X_out = self.X_out[self.X_out[self.time_max_name] >= self.X_out[self.time_name]]

        # names of new columns
        if self.uppercase_suffix:
            self.diff_name = 'TIME_ORDER'
        else:
            self.diff_name = 'time_order'

        # calculate the time difference
        print('Calculating column '+self.diff_name+' ...')
        if self.time_granularity == 'years':
            self.X_out[self.diff_name] = self.X_out[[self.time_name, self.time_max_name]].apply(
                lambda x: relativedelta(dt1=x[1], dt2=x[0]), axis=1)
            self.X_out[self.diff_name] = self.X_out[self.diff_name].apply(lambda x: x.years)
            self.strsql_timecondition = 'floor(months_between(' + self.time_max_name + \
                ',' + self.time_name + ')/12) + 1'
        elif self.time_granularity == 'months':
            self.X_out[self.diff_name] = self.X_out[[self.time_name, self.time_max_name]].apply(
                lambda x: relativedelta(dt1=x[1], dt2=x[0]), axis=1)
            self.X_out[self.diff_name] = self.X_out[self.diff_name].apply(lambda x: 12*x.years+x.months)
            self.strsql_timecondition = 'floor(months_between(' + self.time_max_name + \
                ',' + self.time_name + ')) + 1'
        elif self.time_granularity == 'weeks':
            self.X_out[self.diff_name] = self.X_out[self.time_max_name] - self.X_out[self.time_name]
            self.X_out[self.diff_name] = self.X_out[self.diff_name].apply(lambda x: x.days//7)
            self.strsql_timecondition = 'floor((' + self.time_max_name + ' - ' + self.time_name + ')/7) + 1'
        elif self.time_granularity == 'days':
            self.X_out[self.diff_name] = self.X_out[self.time_max_name] - self.X_out[self.time_name]
            self.X_out[self.diff_name] = self.X_out[self.diff_name].apply(lambda x: x.days)
            self.strsql_timecondition = 'floor(' + self.time_max_name + ' - ' + self.time_name + ') + 1'
        elif self.time_granularity == 'hours':
            self.X_out[self.diff_name] = self.X_out[self.time_max_name] - self.X_out[self.time_name]
            self.X_out[self.diff_name] = self.X_out[self.diff_name].apply(lambda x: x.days*24 + x.seconds//3600)
            self.strsql_timecondition = 'floor((' + self.time_max_name + ' - ' + self.time_name + ')*24) + 1'
        elif self.time_granularity == 'minutes':
            self.X_out[self.diff_name] = self.X_out[self.time_max_name] - self.X_out[self.time_name]
            self.X_out[self.diff_name] = self.X_out[self.diff_name].apply(lambda x: x.days*1440 + x.seconds//60)
            self.strsql_timecondition = 'floor((' + self.time_max_name + ' - ' + \
                self.time_name + ')*1440) + 1'
        elif self.time_granularity == 'seconds':
            self.X_out[self.diff_name] = self.X_out[self.time_max_name] - self.X_out[self.time_name]
            self.X_out[self.diff_name] = self.X_out[self.diff_name].apply(lambda x: x.days*86400 + x.seconds)
            self.strsql_timecondition = 'floor((' + self.time_max_name + ' - ' + \
                self.time_name + ')*86400) + 1'
        elif self.time_granularity == 'order':
            self.X_out[self.diff_name] = self.X_out.groupby(
                self.partition_name)[self.time_name].rank(ascending=False, method='first')
            self.strsql_timecondition = 'row_number() over (partition by ' + self.partition_name + \
                ' order by ' + self.time_name + ' desc)'

        # adds 1 to time difference (basically meaning that 1 month means "within the last one month")
        if self.time_granularity != 'order':
            self.X_out[self.diff_name] = self.X_out[self.diff_name].apply(lambda x: x+1)

        # removes too old history
        self.X_out = self.X_out[(self.X_out[self.diff_name] <= self.history_length)]

        print('Column '+self.diff_name+' added.')

        # SQL Code generator

        if self.time_granularity == 'order':
            self.strsql_ = ['select * from (\nselect t.*, ' + self.strsql_timecondition + ' as TIME_ORDER\n']
            sql_wherecondition = 'where ' + self.time_name + ' <= ' + self.time_max_name + '\n' + \
                                 ') where TIME_ORDER < ' + str(self.history_length) + '\n'
        else:
            self.strsql_ = ['select t.*, ' + self.strsql_timecondition + ' as TIME_ORDER\n']
            sql_wherecondition = 'where ' + self.strsql_timecondition + ' <= ' + str(self.history_length) + '\n' + \
                                 'and ' + self.strsql_timecondition + ' >= 1\n'

        self.strsql_.append('from _TABLENAME_ t \n' + sql_wherecondition)
        self.strsql_ = ''.join(self.strsql_)

        gc.collect()

        return self.X_out
