"""
HOW TO CALL IT IN NOTEBOOK

>>> import importlib
>>> importlib.reload(scoring)
>>> importlib.reload(scoring.stability_grouping)
>>> from scoring.stability_grouping import StableGrouping

>>> sgrouping = StableGrouping(
>>>    columns=cols_pred_num,
>>>    cat_columns=cols_pred_cat,
>>>    bin_stability_threshold=0.10,
>>>    max_leaves=10,
>>>    important_minorities=[],
>>>    must_have_variables=[],
>>>    min_data_in_leaf_for_minotirites=100,
>>>    min_data_in_leaf_share=0.05,
>>>    output_folder='./documentation/stability_grouping/',
>>>    show_plots=False,
>>>    )

>>> sgrouping.fit(
>>>    X_train=data[train_mask][cols_pred_num+cols_pred_cat],
>>>    X_valid=data[valid_mask][cols_pred_num+cols_pred_cat],
>>>    y_train=data[train_mask][col_target],
>>>    y_valid=data[valid_mask][col_target],
>>>    t_train=data[train_mask][col_month],
>>>    t_valid=data[valid_mask][col_month],
>>>    w_train=data[train_mask][col_weight],
>>>    w_valid=data[valid_mask][col_weight],
>>>    progress_bar=True,
>>>   )

>>> sgrouping.save('./documentation/stability_grouping/grouping.json',)
>>> print('UNGROUPABLE VARIABLES:', sgrouping.ungroupable())
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

import gc
import os
import json
import warnings
import matplotlib

from tqdm.notebook import tqdm
from matplotlib.ticker import FuncFormatter, PercentFormatter
from sklearn.utils import assert_all_finite
from sklearn.utils.validation import check_consistent_length, column_or_1d

from .grouping import woe, Grouping
from .metrics import gini

def lgbm_trained_booster(param, lgbtrain, lgbvalid, _early_stopping_rounds = 50):
    """returns trained LGBM booster object

    Args:
        param (dict): parameters to be passed to LightGBM
        lgbtrain (lgb.Datasettype): training dataset
        lgbvalid (lgb.Dataset): validation dataset
        _early_stopping_rounds (int, optional): Early stopping rounds parameter of LightGBM. Defaults to 50.

    Returns:
        lgb.Booster: trained booster
    """

    _evals  = [lgbtrain, lgbvalid]
    _valid_names = ['TRAIN','VALID']

    bst = lgb.train(param, lgbtrain, valid_sets = _evals, verbose_eval = 0, early_stopping_rounds =_early_stopping_rounds)

    return bst

def homogenity(data, col): 
    """Analyses homogenity of column in data.
    The homogenity is caluclated as number of observations where the value of the column is equal to its mode divided by number of all observations.

    Args:
        data (pd.DataFrame): dataframe the homogenity analysis to be performed on
        col (str): name of columns of data which should be analyzed

    Returns:
        numeric: homogenity value
    """
    mask = (data[col] == data[col].mode()[0] )
    homogenity = data[mask].shape[0]/data.shape[0]
    return homogenity  

class StableGrouping():
    """Automatic binning with optimal stable Gini.
    Idea is to make most of the decisions pertaining to design of WOE bins all in one place 
    from point of view of performance and rank time-stability of the resulting bins.
    Aim is to significantly minimize the manual and time consuming interactive binning,
    while at the same time preserving compatibility with the interactive binning should
    that be still occasionally needed (e.g. for business reasons).

    ALGORITHM:
        For each variable, for n_bins in range (2,11):

        1. train a tree model (LGBM - see comment below) M, on training dataset,
           with parameters min_data_in_leaf = 5% of training data length
           (this condition can be released for special variables (e.g. hardchecks) -
           in such case the 'min_data_in_leaf' is set to 100).
        2. evaluate Gini of M on each time stamp of validation set (most frequently month):
           if we have k months, we have k ginis. Take average gini of those
           k ginis --> Gini(n_bins) c)
        3. At the same time, calculate Elena's Rank Stability Index - RSI (version 1)
           for target rat : the model M defines a certain grouping of variable
           (e.g. for n_bins = 3 we have var partitioned into 3 bins) so calculate target rate
           in each bin across the time stamp and  get the RSI as usual. (The RSI for all bins
           is the average of RSI of each constituent bin.  The RSI of a constituent bin is 100%
           if the rank of its target rate relative to other bins's target rates remained
           constant across all time stamps; otherwise, it decreases depending on how many
           times did the rank of the bin change).
    
    Notes:
        A modification was made to the way that a bin's RSI is determined: if for any
        given bin whose rank has changed at least once,  the max of [aboslute values of that
        bin's target rate changes that are associated with the rank order changes] is <
        bin_stability_threshold*(max(target rate across all bins) - min(target rate across all bins))
        then the rank order changes for this bin are ignored (as if its rank had stayed constant).
        In other words, such a bin has RSI = 100%, although its ranking had changed, because the
        change is 'acceptably small' relative to the spread of target rates across all bins.
        The bin_stability_threshold parameter can be set.

        At the end of the inner loop, the recommended # of bins for var is:
        min(argmax(Gini(n_bins) | RSI = 100% )).
        In other words, we take the maximum average gini where the RSI is 100% and find 
        the corresponding n_bins. If there are several n_bins values associated with this gini,
        we take their minimum.
        
        The package used to run the tree model is LGBM whereby only the first tree is trained,
        without further boosting. This is  used in favour of scikitlearn's own tree algorithm
        because it allows for native processing of categorical variables as well.

    Args:
        columns (list of str): list of names of continuous columns to be grouped
        cat_columns (list of str): list of names of categorical columns to be grouped
        bin_stability_threshold (float, optional): Tolerance for RSI calculation.
            See class description for details. Defaults to 0.10.
        max_leaves (int, optional): Maximal number of bins for each variable
            to be tried. Defaults to 10.
        important_minorities (list of str, optional): List of columns where we accept
            very small groups to exist in grouping (e.g. hardchecks). Defaults to [].
        must_have_variables (list of str, optional): List of columns where we accept
            unstable grouping (e.g. must be in model for business reasons). Defaults to [].
        min_data_in_leaf_for_minotirites (int, optional): Minimal group size for columns
            in important_minorities. Defaults to 100.
        min_data_in_leaf_share (float, optional): Minimial group size as share of all observations
            for columns not in important_minorities. Defaults to 0.05.
        output_folder (str, optional): Folder to print graphical outputs into. Defaults to None.
        show_plots (bool, optional): Should be graphical outputs printed to notebook? Defaults to True.
    """

    _param = {
        # Objective and validation metric:
        'num_iterations': 1, 
        'objective': 'binary',
        'num_class':  1,
        'metric': 'auc',
        # Shrinkage parameter :
        'learning_rate': 1,
        # Weak learner (Oblivious tree):
        'boosting_type': 'gbdt',
        'max_depth': -1,
        'num_leaves': 2, 
        # Other:
        'seed': 42,
        'verbosity': -1,
        'use_missing' : '+',    #<-- do not auto-arrange nan. Keep it separate
        'device_type' : 'cpu',
        'n_jobs': -1
    }

    def __init__(self, columns, cat_columns, bin_stability_threshold=0.10, max_leaves=10, important_minorities=[], must_have_variables=[], min_data_in_leaf_for_minotirites=100, min_data_in_leaf_share=0.05, output_folder=None, show_plots=True):
        """Initializes instance of the class.
        """

        self.columns = columns
        self.cat_columns = cat_columns
        self.bin_stability_threshold = bin_stability_threshold
        self.max_leaves = max_leaves
        self.must_have_variables = must_have_variables
        self.important_minorities = important_minorities
        self.min_data_in_leaf_for_minotirites = min_data_in_leaf_for_minotirites
        self.min_data_in_leaf_share = min_data_in_leaf_share
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        self.show_plots = show_plots
        self.bins_data = dict()
        self.excluded = []
        self.grouping = Grouping(
            columns=self.columns,
            cat_columns=self.cat_columns,
        )
        self.fitted = False

    def fit(self, X_train, X_valid, y_train, y_valid, t_train, t_valid, w_train=None, w_valid=None, progress_bar=False, category_limit=100):
        """Fits the optimal grouping with regards to stability and creates the graphical outputs.

        Args:
            X_train (pd.DataFrame): dataframe with predictors - training set
            X_valid (pd.DataFrame): dataframe with predictors - validation set
            y_train (pd.Series): target column - training set
            y_valid (pd.Series): target column - validation set
            t_train (pd.Series): month of observation column - training set
            t_valid (pd.Series): month of observation column - validation set
            w_train (pd.Series, optional): weight column - training set. Defaults to None.
            w_valid (pd.Series, optional): weight column - validation set. Defaults to None.
            progress_bar (bool, optional): whether to display progress bar. Defaults to False.
            category_limit (int, optional): limit of unique values of categorical variables to be processed.
                Defaults to 100.
        """

        warnings.filterwarnings('ignore')

        if type(X_train) != pd.DataFrame:
            raise ValueError('X_train should be DataFrame')
        check_consistent_length(X_train, y_train)
        check_consistent_length(X_train, t_train)
        column_or_1d(y_train)
        column_or_1d(t_train)
        if type(X_valid) != pd.DataFrame:
            raise ValueError('X_valid should be DataFrame')
        check_consistent_length(X_valid, y_valid)
        check_consistent_length(X_valid, t_valid)
        column_or_1d(y_valid)
        column_or_1d(t_valid)

        if np.any(X_train.columns.duplicated()):
            duplicities = [col_name for col_name, duplicated in zip(X_train.columns, X_train.columns.duplicated()) if duplicated]
            raise ValueError(f"Columns {list(dict.fromkeys(duplicities))} are duplicated in your Dataset.")  # list.dict hack to remove duplicated quickly
        if np.any(X_valid.columns.duplicated()):
            duplicities = [col_name for col_name, duplicated in zip(X_valid.columns, X_valid.columns.duplicated()) if duplicated]
            raise ValueError(f"Columns {list(dict.fromkeys(duplicities))} are duplicated in your Dataset.")  # list.dict hack to remove duplicated quickly

        for name, column in X_train[self.columns].iteritems():
            if np.any(np.isinf(column.values)):
                    raise ValueError(f'Column {name} containes non-finite values.')
        for name, column in X_valid[self.columns].iteritems():
            if np.any(np.isinf(column.values)):
                    raise ValueError(f'Column {name} containes non-finite values.')

        if w_train is not None:
            check_consistent_length(w_train, y_train)
            column_or_1d(w_train)
            w_train = w_train.astype('float64')
            w_train[pd.isnull(y_train)] = np.nan
        if w_valid is not None:
            check_consistent_length(w_valid, y_valid)
            column_or_1d(w_valid)
            w_valid = w_valid.astype('float64')
            w_valid[pd.isnull(y_valid)] = np.nan

        for name, column in X_train[self.cat_columns].iteritems():
            if column.nunique() > category_limit:
                raise ValueError(f'Column {name} has more than {category_limit} unique values. '\
                                 'Large number of unique values might cause memory issues. '\
                                  'This limit can be set with parameter `category_limit.')

        for name, column in X_valid[self.cat_columns].iteritems():
            if column.nunique() > category_limit:
                raise ValueError(f'Column {name} has more than {category_limit} unique values. '\
                                 'Large number of unique values might cause memory issues. '\
                                  'This limit can be set with parameter `category_limit.')

        for name in self.cat_columns:
            X_train[name] = X_train[name].astype("category")
            X_valid[name] = X_valid[name].astype("category")

        if progress_bar:
            iterator = tqdm(self.columns + self.cat_columns, leave=True, unit='cols')
        else:
            iterator = self.columns + self.cat_columns

        for column in iterator:
            is_categorical = column in self.cat_columns
            is_important_minority = column in self.important_minorities
            is_must_have = column in self.must_have_variables
            if progress_bar:
                iterator.set_description(desc=column, refresh=True)
            n_bins, exclude = self._auto_grouping(X_train[column], y_train, X_valid[column], y_valid, t_train, t_valid, is_categorical, is_important_minority, is_must_have, column, w_train, w_valid)

            if exclude:
                self.excluded.append(column)
            else:
                self.bins_data[column] = self._get_woes(n_bins, X_train[column], y_train, X_valid[column], y_valid, is_categorical, is_important_minority, w_train, w_valid)

        self.grouping.bins_data_ = self.bins_data
        self.fitted = True

        warnings.filterwarnings('default')

        return

    
    def _check_fitted(self):
        """Checks the value of fitted parameter, which is true only if fit() method was sucesfully called already.
        Otherwise, the grouping is not filled yet and this method raises Exception.

        Raises:
            Exception: Model was not fitted yet
        """
        if not self.fitted:
            raise Exception("Model was not yet fitted.")


    def get_dummy_names(self, columns_to_transform=None):
        """Inherited from Grouping object.
        Get name of dummy variables from dummy variable transformation of predictors in list columns_to_transform

        Args:
            columns_to_transform (list of str, optional): List of predictors to get dummy names for. Defaults to None.

        Returns:
            dict: Dictionary with predictor name as key and dummy names list as value
        """
        self._check_fitted()

        return self.grouping.get_dummy_names(columns_to_transform)

    def export_dictionary(self, suffix="_WOE", interval_edge_rounding=3, woe_rounding=5):
        """
        Inherited from Grouping object.

        Returns a dictionary with (woe:bin/values) pairs for fitted predictors.

        Numerical predictors are in this format:
        (woe): "[x, y)"
        Categorical predictors are in this format:
        round(woe): ["AA","BB","CC","Unknown"]

        Args:
            suffix (str, optional): suffix of WOE variables. Defaults to "_WOE".
            interval_edge_rounding (int, optional): rounding for numerical variable interval edges. Defaults to 3.
            woe_rounding (int, optional): rounding for WOE values. Defaults to 5.

        Example:
            >>> {'Numerical_1_WOE': {-0.77344: '[-inf, 0.093)',
            >>>                      -0.29478: '[0.093, 0.248)',
            >>>                       0.16783: '[0.248, 0.604)',
            >>>                       0.86906: '[0.604, 0.709)',
            >>>                       1.84117: '[0.709, inf)',
            >>>                       0.0: 'NaN'},
            >>> 'Categorical_1_WOE': { 0.34995: 'EEE, FFF, GGG, III',
            >>>                        0.03374: 'HHH',
            >>>                       -0.16117: 'CCC, DDD',
            >>>                       -0.70459: 'BBB',
            >>>                       -0.95404: 'AAA',
            >>>                        0.0: 'nan, Unknown'}}
                             
        
        """
        self._check_fitted()

        return self.grouping.export_dictionary(
            suffix=suffix, interval_edge_rounding=interval_edge_rounding, woe_rounding=woe_rounding
        )

    def transform(self, data, transform_to='woe', columns_to_transform=None, progress_bar=False):
        """Inherited from Grouping object.
        Performs transformation of `data` based on `transform_to` parameter and adds suffix to column names.

        Args:
            data (pd.DataFrame): data to be transformed
            transform_to (str, optional): Type of transformation. Possible values: `woe`,`shortnames`,`group_number`,`dummy`. Defaults to 'woe'.
            columns_to_transform (list of str, optional): List of columns of data to be transformed. Defaults to None.
            progress_bar (bool, optional): Display progress bar? Defaults to False.

        Returns:
            pd.DataFrame: transformed data
        """
        self._check_fitted()

        if columns_to_transform:
            columns_to_transform = [col for col in columns_to_transform if col in self.bins_data.keys()]
        else:
            columns_to_transform = list(self.bins_data.keys())

        return self.grouping.transform(data=data, transform_to=transform_to, columns_to_transform=columns_to_transform, progress_bar=progress_bar)

    def save(self, filename):
        """Inherited from Grouping object.
        Saves the grouping dictionary to external JSON file.

        Args:
            filename (str): name of file to save the grouping to
        """
        self._check_fitted()

        self.grouping.save(filename)

    def load(self, filename):
        """Inherited from Grouping object.
        Loads the grouping dictionary from external JSON file.

        Args:
            filename (str): name of file to load the grouping from
        """
        self.grouping.load(filename)

    def export_as_sql(self, suffix='_WOE', filename=None):
        """Inherited from Grouping object.
        Creates a SQL script for transforming data based on fitted grouping. Designed to be used with print() to replace \\n with newlines.

        Returns a string with SQL script with sets of CASE statements for transforming data.

        Args:
            suffix (str): suffix to be added to transformed predictors, default='WOE'
            filename (str): path to file for export

        Returns:
            str : SQL script with \\n for new lines.
        """
        self._check_fitted()

        self.grouping.export_as_sql(suffix, filename)

    def plot_bins(self, data, cols_pred_num, cols_pred_cat, mask, col_target, output_folder, col_weight=None):
        """Inherited from Grouping object.
        Plots the binning on statistics on given dataset.

        Args:
            data (pd.DataFrame): data to be plotted
            cols_pred_num (list of str): numerical predictors to be analyzed
            cols_pred_cat (list of str): categorical predictors to be analyzed
            mask (pd.Series): mask to be applied to the data
            col_target (str): name of target column
            output_folder (str): folder to save the outputs to
            col_weight (str, optional): name of optional weight column. Defaults to None.
        """
        self._check_fitted()

        self.grouping.plot_bins(data, cols_pred_num, cols_pred_cat, mask, col_target, output_folder, col_weight)

    def ungroupable(self):
        """Returns list of variables that could not been grouped because of isntability.

        Returns:
            list of str: list of ungroupable variables
        """
        self._check_fitted()

        return self.excluded

    def _auto_grouping(self, x_train, y_train, x_valid, y_valid, t_train, t_valid, is_categorical, is_important_minority, is_must_have, variable_name, w_train=None, w_valid=None):
        """
        Iteratively attempts to group a variable to n groups (n from 2 to self.max_leaves),
        calculates stability index (RSI) and predictive power (Gini) for such grouping,
        identifies the optimal n (maximizing RSI and Gini)
        creates charts and prints them

        Args:
            x_train (pd.Series): predictor column - training set
            y_train (pd.Series): target column - training set
            x_valid (pd.Series): predictor column - validation set
            y_valid (pd.Series): target column - validation set
            t_train (pd.Series): month of observation column - training set
            t_valid (pd.Series): month of observation column - validation set
            is_categorical (bool): is x categorical (True) or numerical (False)?
            is_important_minority (bool): has x important infreqeunt values, which might form a special group?
            is_must_have (bool): is x a must-have predictor which can be used even if not stable?
            variable_name ([type]): name of x
            w_train (pd.Series, optional): weight column - training set. Defaults to None.
            w_valid (pd.Series, optional): weight column - validation set. Defaults to None.

        Returns:
            int, bool: number of bins, boolean that the predictor is unstable and should be excluded from final predictor set
        """

        if w_train is None:
            w_train = pd.Series(1, index=y_train.index)
        if w_valid is None:
            w_valid = pd.Series(1, index=y_valid.index)
        T = pd.DataFrame({'x': x_train, 'y': y_train, 'w': w_train, 't': t_train})
        V = pd.DataFrame({'x': x_valid, 'y': y_valid, 'w': w_valid, 't': t_valid})

        if is_categorical:
            categorical_features = ['x']
        else:
            categorical_features = []

        lgbm_train_data = lgb.Dataset(T[['x']], label=T['y'], weight=T['w'], categorical_feature=categorical_features, free_raw_data=False)
        lgbm_valid_data = lgb.Dataset(V[['x']], label=V['y'], weight=V['w'], reference=lgbm_train_data)

        eval_data = V

        aucs = {}
        aucs_moving_avg = {}

        fig = plt.figure(figsize=(25,35))

        gsG = fig.add_gridspec(1,3)
        gs = gsG[0].subgridspec(4,1)
        gs2 = gsG[1].subgridspec(self.max_leaves-1,1)
        gs3 = gsG[2].subgridspec(self.max_leaves-1,1)

        x_labels = sorted(eval_data['t'].unique())

        cntgrph = 0
        for num_leaves in np.arange(2,self.max_leaves + 1):

            self._param['num_leaves'] = num_leaves

            if is_important_minority:
                self._param['min_data_in_leaf'] = self.min_data_in_leaf_for_minotirites
            else:
                self._param['min_data_in_leaf'] = int(np.ceil(self.min_data_in_leaf_share * T.shape[0]))

            bst = lgbm_trained_booster(self._param, lgbm_train_data, lgbm_valid_data)

            aucs[num_leaves] = {}
            aucs_moving_avg[num_leaves] = {}
            std = []
            graphs_time_seq = pd.DataFrame()

            sum_ = 0
            c = 0

            for month in x_labels:

                df = eval_data[eval_data['t']==month]
                
                df_pred = pd.DataFrame(bst.predict(df[['x']], num_iteration=1), index=df.index)
                df = pd.concat([df[['x','y','w']], pd.DataFrame(df_pred)], axis=1)
                df.rename(columns={0:'p'}, inplace=True)

                # calculate Gini
                djinn = gini(df['y'], df['p'], df['w'])
                sum_ += djinn
                c += 1
                aucs[num_leaves][month] = djinn
                std.append(djinn)
                aucs_moving_avg[num_leaves][month] = sum_/c

                # rank score bins by event rate. each unique score value corresponds to a specific bin out of n_leaves bins

                event_rate = pd.DataFrame(
                    df.groupby(['p'], sort=True).apply(lambda row: (row['y']*row['w']).sum()/row['w'].sum())
                ).rename(columns={0:'y_rate'})

                # get rank of each value of p
                event_rate.sort_values(by=['y_rate'], ascending=[False], inplace=True)
                event_rate.reset_index(inplace=True) #adds 'p' from index
                event_rate.reset_index(inplace=True)
                event_rate.rename(columns={'index':'rank_y'}, inplace=True) #adds 'rank_y' from index

                # rank p bins by population share
                pop_sum = df['w'].sum()
                pop_share = pd.DataFrame(
                    df.groupby(['p'], sort=True).apply(lambda row: row['w'].sum()/pop_sum)
                ).rename(columns={0:'population'})

                # get population rank of each score in month
                pop_share.sort_values(by=['population'] , ascending=[False], inplace = True)
                pop_share.reset_index(inplace=True) #adds 'p' from index
                pop_share.reset_index(inplace=True)
                pop_share.rename(columns={'index':'rank_pop'}, inplace=True) #adds 'rank_pop' from index
                pop_share['t'] = month
        
                graphs = pd.merge(event_rate, pop_share , how = 'inner', left_on = ['p'], right_on = ['p'] )
                graphs['sample_size'] = df['w'].sum()
                graphs_time_seq = graphs_time_seq.append(graphs)
            
                del event_rate, pop_share, graphs

            gc.collect()

            # overall gini

            std = np.array(std)
            aucs[num_leaves]['avg_gini'] = std.mean()
            aucs[num_leaves]['std_gini'] = (std.max()-std.min() + 0.001)/(std.mean() + 0.001)
            aucs[num_leaves]['max_gini'] = std.max()
            aucs[num_leaves]['min_gini'] = std.min()

            # modify rank_y to allow for small changes in bin event rate compared to total event rate span (controlled by bin_stability_threshold)
            graphs_time_seq.sort_values(by=['p','t'], ascending=[False,True], inplace=True)
            event_rate_span = graphs_time_seq['y_rate'].max() - graphs_time_seq['y_rate'].min()

            # for each score value (i.e. bin) get max y_rate leap that is accompanied by a change of rank. denominate this leap as percentage of 
            # y_rate span across all bins and all time periods

            rank_rate_mask = {}

            for p in graphs_time_seq['p'].unique():

                rank_rate_mask[p] = {}

                selection = graphs_time_seq[graphs_time_seq['p']==p]
                rank_mode = selection['rank_y'].mode()[0]

                event_rates = selection['y_rate'].values
                ranks = selection['rank_y'].values

                max_leap = 0

                for i in range(len(ranks)-1):

                    if ranks[i]==ranks[i+1]:
                        event_rate_delta = 0
                    else:
                        event_rate_delta = abs(event_rates[i]-event_rates[i+1])

                    max_leap = max(max_leap, event_rate_delta)

                if max_leap/event_rate_span <= self.bin_stability_threshold:
                    rank_rate_mask[p] = rank_mode
                else:
                    rank_rate_mask[p] = -1

            rank_rate_mask = pd.DataFrame(rank_rate_mask, index=[0]).T.reset_index().rename(columns={'index':'p',0:'max_leap'})
            graphs_time_seq = pd.merge(graphs_time_seq, rank_rate_mask, how='inner', left_on=['p'], right_on=['p'])
            graphs_time_seq['rank_y'] = graphs_time_seq.apply(lambda row: int(row['rank_y']) if row['max_leap']==-1 else int(row['max_leap']), axis=1)
            graphs_time_seq.drop(columns=['max_leap'], inplace=True)

            # generate charts

            # bin event rate in time
            ax = fig.add_subplot(gs2[cntgrph,0])
            ax.set_title('n_bins: '  + str(num_leaves), fontsize=16)
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels) 
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
            ax.set_ylabel('Event Rate',fontsize=16)
            ax.grid(True)
            ax.set_ylim([0.8*graphs_time_seq['y_rate'].min(), 1.2*graphs_time_seq['y_rate'].max()])

            for p in sorted(graphs_time_seq['p'].unique()):
                grf = graphs_time_seq[graphs_time_seq['p']==p].sort_values(['t'])
                ax.plot(grf['t'], grf['y_rate'], linewidth=3, linestyle='-', marker='x', markersize=15)

            ax = fig.add_subplot(gs3[cntgrph,0])
            ax.set_title('n_bins: '  + str(num_leaves), fontsize=16)
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels) 
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
            ax.set_ylabel('population %',fontsize=16)
            ax.grid(True)
            ax.set_ylim([0, 1.2*graphs_time_seq['population'].max()])

            for p in sorted(graphs_time_seq['p'].unique()):
                grf = graphs_time_seq[graphs_time_seq['p']==p].sort_values(['t'])
                ax.plot(grf['t'], grf['population'], linewidth=3, linestyle='-', marker='x', markersize=15)
    
            cntgrph += 1

            rsi_event_rate = 0
            rsi_population_rate = 0
            c = 0

            for p in sorted(graphs_time_seq['p'].unique()):
                rsi_event_rate += homogenity(graphs_time_seq[graphs_time_seq['p']==p], 'rank_y')
                rsi_population_rate += homogenity(graphs_time_seq[graphs_time_seq['p']==p], 'rank_pop')
                c += 1

            rsi_event_rate = rsi_event_rate/c
            rsi_population_rate = rsi_population_rate/c

            aucs[num_leaves]['rsi_event_rate'] = rsi_event_rate
            aucs[num_leaves]['rsi_population_rate'] = rsi_population_rate

        del bst
        gc.collect()

        aucs = pd.DataFrame(aucs).T.reset_index(drop=False).rename(columns={'index': 'n_bins'})

        # fix situation where RSI = 1 for a certain number of bins but it is <1 for a lower number of bins
        aucs['omni'] = 1
        aucs['rsi_event_rate_lag'] = aucs.groupby('omni')['rsi_event_rate'].shift(1)
        aucs['rsi_event_rate'] = aucs.apply(lambda row: row['rsi_event_rate_lag'] if (row['rsi_event_rate']==1 and row['rsi_event_rate_lag']<1) else row['rsi_event_rate'], axis=1)
        aucs.drop(columns=['omni','rsi_event_rate_lag'], inplace=True)

        aucs_moving_avg = pd.DataFrame(aucs_moving_avg).T.reset_index(drop=False).rename(columns={'index': 'n_bins'})

        #get recommended number of bins (prioritize event rate rank stability)
        max_stability = aucs['rsi_event_rate'].max()
        if (not is_important_minority) and (not is_must_have):
            if aucs['rsi_event_rate'].max() < 1:
                exclude = True
                n_bins = -1
            else:
                exclude = False
                stable_region = aucs[aucs['rsi_event_rate']==max_stability]
                max_gini = stable_region['avg_gini'].max()
                n_bins = stable_region[stable_region['avg_gini']==max_gini]['n_bins'].values.min()
        else:
            exclude = False
            stable_region = aucs[aucs['rsi_event_rate']==max_stability]
            max_gini = stable_region['avg_gini'].max()
            n_bins = stable_region[stable_region['avg_gini']==max_gini]['n_bins'].values.min()

        plot_max = aucs['max_gini'].max()
        plot_min = min(0, aucs['min_gini'].min())

        ax1 = fig.add_subplot(gs[0,0])
        ax1.set_title(variable_name+' (excluding NULL)', fontsize=16)
        ax1.set_xlabel('Time', fontsize=12, rotation=0)
        ax1.set_ylabel('Gini', fontsize=16)
        ax1.grid(True)
        ax1.set_xticks(range(len(x_labels)))
        ax1.set_xticklabels(x_labels)
        ax1.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
        ax1.set_ylim([1.5*plot_min, 1.5*plot_max])

        ax2 = fig.add_subplot(gs[1,0])
        ax2.set_title('Drift (excluding NULL)', fontsize=16)
        ax2.set_xlabel('Time' ,fontsize=12, rotation = 0)
        ax2.set_ylabel('cumulative average of Gini',fontsize=16)
        ax2.grid(True)
        ax2.set_xticks(range(len(x_labels)))
        ax2.set_xticklabels(x_labels)
        ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
        ax2.set_ylim([0.8*plot_min, 1.2*plot_max])

        ax3 = fig.add_subplot(gs[2,0])
        ax3.set_title('Gini across Time (excluding NULL)', fontsize=16)
        ax3.set_xlabel('# bins',fontsize=16)
        ax3.set_ylabel('avg Gini',fontsize=16)
        ax3.grid(True)
        ax3.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
        ax3.set_ylim([0.8*aucs['avg_gini'].min(), 1.2*aucs['avg_gini'].max()])

        ax4 = fig.add_subplot(gs[3,0])
        ax4.set_title('Rank Stability Index (V1, excluding NULL)' , fontsize=16)
        ax4.set_xlabel('# bins', fontsize=16)
        ax4.set_ylabel('SI', fontsize=16)
        ax4.grid(True)
        ax4.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
        ax4.set_ylim([0, 1])

        s = len(eval_data['t'].unique())

        for _, v in aucs.iterrows():
            ax1.plot(v.values[1:s+1], label=f'# bins: {str(int(v.values[0]))}', linewidth=3, linestyle='--', marker='x', markersize=15)
            ax1.legend(loc="lower right", fontsize=10, bbox_to_anchor=(1, 0.65))

        for _, v in aucs_moving_avg.iterrows():
            ax2.plot(v.values[1:s+1], label=f'# bins: {str(int(v.values[0]))}', linewidth=3, linestyle='--', marker='x', markersize=15)
            ax2.legend(loc="lower right", fontsize=10, bbox_to_anchor=(1, 0.65))

        ax3.plot(aucs['n_bins'], aucs['avg_gini'], label = f'recommended # bins: {str(n_bins)}',  linewidth=3, linestyle='--', marker='x', markersize=15 , color='blue')
        ax3.legend(loc="lower right", fontsize=16, bbox_to_anchor=(0.75, 0.9)) 

        ax4.plot(aucs['n_bins'], aucs['rsi_event_rate'], label = 'RSI for event rate', linewidth=3, linestyle='--', marker='x' , markersize=15 , color='purple') 
        ax4.legend(loc="lower right", fontsize=12, bbox_to_anchor=(1, 0.9)) 

        ax4.plot(aucs['n_bins'], aucs['rsi_population_rate'], label = 'RSI for population', linewidth=1, linestyle='--', marker='x' , markersize=10 , color='black') 
        ax4.legend(loc="lower right", fontsize=12, bbox_to_anchor=(1, 0.9))

        if self.output_folder is not None:
            plt.savefig(f'{self.output_folder}/{variable_name}.png')
        if self.show_plots:
            plt.show()
        plt.close()

        return n_bins, exclude


    def _get_woes(self, n_bins, x_train, y_train, x_valid, y_valid, is_categorical, is_important_minority, w_train=None, w_valid=None):
        """
        Groups given variable into given number of bins and returns grouping dictionary (ranges and WOEs).

        Args:
            n_bins (int): number of bins x should be grouped into
            x_train (pd.Series): predictor column - training set
            y_train (pd.Series): target column - training set
            x_valid (pd.Series): predictor column - validation set
            y_valid (pd.Series): target column - validation set
            is_categorical (bool): is x categorical (True) or numerical (False)?
            is_important_minority (bool): has x important infreqeunt values, which might form a special group?
            w_train (pd.Series, optional): weight column - training set. Defaults to None.
            w_valid (pd.Series, optional): weight column - validation set. Defaults to None.
            
        Returns:
            dict: dictionary with grouping metadata (ranges and WOEs of individual groups)
        """

        if w_train is None:
            w_train = pd.Series(1, index=y_train.index)
        if w_valid is None:
            w_valid = pd.Series(1, index=y_valid.index)

        nan_woe = 0

        if len(x_train[pd.isnull(x_train)]) > 0:
            nan_woe = woe(
                y = y_train[pd.isnull(x_train)],
                y_full = y_train,
                w = w_train[pd.isnull(x_train)],
                w_full = w_train,
            )

        T = pd.DataFrame({'x': x_train, 'y': y_train, 'w': w_train})
        V = pd.DataFrame({'x': x_valid, 'y': y_valid, 'w': w_valid})

        if is_categorical:
            categorical_features = ['x']
        else:
            categorical_features = []

        lgbm_train_data = lgb.Dataset(T[['x']], label=T['y'], weight=T['w'], categorical_feature=categorical_features)
        lgbm_valid_data = lgb.Dataset(V[['x']], label=V['y'], weight=V['w'])

        self._param['num_leaves'] = n_bins

        if is_important_minority:
            self._param['min_data_in_leaf'] = self.min_data_in_leaf_for_minotirites
        else:
            self._param['min_data_in_leaf'] = int(np.ceil(self.min_data_in_leaf_share * T.shape[0]))

        bst = lgbm_trained_booster(self._param, lgbm_train_data, lgbm_valid_data)

        T['p'] = pd.Series(bst.predict(T[['x']], num_iteration = bst.best_iteration), index=T.index)

        if not is_categorical:

            buckets = []

            for bucket_pd in sorted(list(T['p'].unique())):

                bucket_woe = woe(
                    y = T[T['p']==bucket_pd]['y'],
                    y_full = T['y'],
                    w = T[T['p']==bucket_pd]['w'],
                    w_full = T['w'],
                )

                buckets.append({
                    'min': T[T['p']==bucket_pd]['x'].min(),
                    'max': T[T['p']==bucket_pd]['x'].max(),
                    'woe': bucket_woe
                })

            buckets = sorted(buckets, key=lambda bucket: bucket['max'])
            
            bins = [-np.inf] + [(bucket['max']+next_bucket['min'])/2 for bucket, next_bucket in zip(buckets[:-1],buckets[1:])] + [np.inf]
            woes = [bucket['woe'] for bucket in buckets]

            grpdict = {
                'nan_woe': nan_woe,
                'dtype': 'float64',
                'bins': bins,
                'woes': woes,
            }
        
        else:

            woes = []
            bins = {}

            for bucket_idx, bucket_pd in enumerate(sorted(list(T['p'].unique()))):

                woes.append(woe(
                    y = T[T['p']==bucket_pd]['y'],
                    y_full = T['y'],
                    w = T[T['p']==bucket_pd]['w'],
                    w_full = T['w'],
                ))

                for category in T[T['p']==bucket_pd]['x'].unique():
                    bins[category] = bucket_idx

            grpdict = {
                'unknown_woe': 0,
                'dtype': 'category',
                'bins': bins,
                'woes': woes,
            }

        return grpdict