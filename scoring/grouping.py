# Home Credit Python Scoring Library and Workflow
# Copyright © 2017-2020, Pavel Sůva, Marek Teller, Martin Kotek, Jan Zeller,
# Marek Mukenšnabl, Kirill Odintsov, Elena Kuchina, Jan Coufalík, Jan Hynek and
# Home Credit & Finance Bank Limited Liability Company, Moscow, Russia.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# -*- coding: utf-8 -*-

"""
Todo
 * Refactor plots and tables in the top of interactive grouping form
 * Categorical interactive grouping - implement plots
 * Implement or move statistics table for continuous interactive  grouping (we don't see WOEs now!) from old sources (QTable or alternative)
 * Wrong sorting in QGrids (see in interactive grouping of categoricals) since we convert to floating point numbers to str to format them (add issue to github to have column based formatting?)
 * We need 'editable' property per column in QGrid and in some way to show that only "group" column is editable for interactive grouping of categorical variables.
 * Do we need to change WOE for nan manually for categorical?
 * Do we need "other" WOE in categorical grouping?
 * Special values for continuous variables (to treat them like nans, define WOE etc.)
 * Many Tab control captions / long captions looks poor
   https://github.com/jupyter-widgets/ipywidgets/issues/1905
 * Now editor recalculates WOEs for not NaN (review behaviour?). That is common topic for both: continuous and categorical predictors.
 * Non-monotnic groups (think if we need them)
 * Full logistic (think if we need them)
 * Inline comments and help generation (Sphynx is our choice)
 * Good default value for woe_coeff
 * Maybe automatic default values for size of groups and max_cat...
 * Do we need to be able to change all params both in __init__ and InteractiveGrouping.display(...)
 * Separate Interactive and non-interactive grouping in different modules
 * Move woe function to metrics module (maybe also create ER func, check that gini func is used also everywhere from metrics module)
 * precision of variables inside TextFloat for continuous interactive grouping
 * Very fast clicking in interactive grouping can cause errors (now implemented workaround, have issue on github of ipywidgets)
 * Refactor recreate controls for auto-grouping of continuous (duplicated functionality now)
 * Add debug mode with print(...) statements
 * Shiyal don't see step-wise
 * adapt pipeline to new version
 * should be nan_woe in both formats (cont/cat) for unification?

 Todo/existing bugs:
 * X axis labels are shifted on interactive grouping plot with detailed bins for categoricals
 * for continues if we have just one not nan value and nan - plot crashes
 * error in InteractiveGrouping found by Pavel: when you click Apply and Save and didn't open any categorical Context before - save crashes.
 * poor performance of "Save and Apply" action
 * During automatic grouping number of observations in group consists of rare categories can be below self.min_samples. We should think if this behaviour is ok.

"""

# %matplotlib notebook

# To prevent automatic figure display when execution of the cell ends
# %config InlineBackend.close_figures=True

import copy
import datetime
import json
import operator
import re
import time
from collections import Counter
from math import log

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qgrid
from IPython.display import HTML, Markdown, display
from matplotlib import gridspec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import assert_all_finite
from sklearn.utils.validation import check_consistent_length, column_or_1d
from tqdm.notebook import tqdm

from scoring.plot import print_binning_stats_cat, print_binning_stats_num
from scoring.scorecard import ScoreCard


def gini_grp(woe, share, def_rate):
    df = pd.concat([woe, share, def_rate], axis=1)
    df.columns = ['woe', 'share', 'def_rate']
    df = df[df['share'] > 0].sort_values('woe')
    df['bad'] = df['share'] * df['def_rate']
    df['good'] = df['share'] - df['bad']
    df['bad_pct'] = df['bad'] / df['bad'].sum()
    df['good_pct'] = df['good'] / df['good'].sum()
    df['cum_bad_pct_0'] = df['bad_pct'].cumsum()
    df['cum_bad_pct'] = (df['cum_bad_pct_0'] + df['cum_bad_pct_0'].shift(1).fillna(0)) / 2
    df['auc'] = df['good_pct'] * df['cum_bad_pct']
    return (df['auc'].sum() - 0.5) * 2


def woe(y, y_full, smooth_coef=0.001, w=None, w_full=None):
    """Weight of evidence

    Args:
        y (np.array or pandas.Series): target in current category, should contain just {0, 1}
        y_full (np.array or pandas.Series): whole target
        smooth_coef (float, optional): coefficient to avoid divizion by zero or log of zero. Defaults to 0.001.
        w (np.array or pandas.Series, optional): weight of observations in current category. Defaults to None.
        w_full (np.array or pandas.Series, optional): weight of observations in the whole. Defaults to None.

    Returns:
        float: WOE of the category
    """

    # TODO: think about default value of smooth_coef
    if smooth_coef < 0:
        raise ValueError('Smooth_coef should be non-negative')
    y = column_or_1d(y)
    y_full = column_or_1d(y_full)
    if y.size > y_full.size:
        raise ValueError('Length of y_full should be >= length of y')
    if len(set(y) - {0, 1}) > 0:
        raise ValueError('y should consist just of {0,1}')
    if not np.array_equal(np.unique(y_full), [0, 1]):
        raise ValueError('y_full should consist of {0,1}, noth should be presented')
    if w is not None and y.size != w.size:
        raise ValueError('Size of y and w must be the same')
    if w_full is not None and y_full.size != w_full.size:
        raise ValueError('Size of y_full and w_full must be the same')
    if w is None:
        w = np.ones(len(y))
    if w_full is None:
        w_full = np.ones(len(y_full))
    if y.size == 0:
        return 0.
    woe = np.log((sum((1 - y) * w) / sum(w) + smooth_coef) / (sum(y * w) / sum(w) + smooth_coef)) - \
          np.log((sum((1 - y_full) * w_full) / sum(w_full) + smooth_coef) /
                 (sum(y_full * w_full) / sum(w_full) + smooth_coef))
    return woe


def nwoe(y, y_full, group, group_full, smooth_coef=0.001, w=None, w_full=None):
    """Net weight of evidence

    Args:
        y (np.array or pandas.Series): target in current category, should contain just {0, 1}
        y_full (np.array or pandas.Series): whole target
        group (np.array or pandas.Series): binary uplift group in current category
        group_full (np.array or pandas.Series): whole group
        smooth_coef (float, optional): coefficient to avoid divizion by zero or log of zero. Defaults to 0.001.
        w (np.array or pandas.Series, optional)): sample weights in category. Defaults to None.
        w_full (np.array or pandas.Series, optional)): sample weights in the whole. Defaults to None.

    Returns:
        float: NWOE of the category
    """

    if smooth_coef < 0:
        raise ValueError('Smooth_coef should be non-negative')
    y = column_or_1d(y)
    y_full = column_or_1d(y_full)
    if y.size > y_full.size:
        raise ValueError('Length of y_full should be >= length of y')
    if group.size > group_full.size:
        raise ValueError('Length of group_full should be >= length of group')
    if len(set(y) - {0, 1}) > 0:
        raise ValueError('y should consist just of {0,1}')
    if not np.array_equal(np.unique(y_full), [0, 1]):
        raise ValueError('y_full should consist of {0,1}, noth should be presented')
    if y.size != group.size:
        raise ValueError('Size of y and group must be the same')
    if w is not None and y.size != w.size:
        raise ValueError('Size of y and w must be the same')
    if w_full is not None and y_full.size != w_full.size:
        raise ValueError('Size of y_full and w_full must be the same')
    if w is None:
        w = np.ones(len(y))
    if w_full is None:
        w_full = np.ones(len(y_full))
    if y.size == 0:
        return 0.

    nwoe = np.log((sum((1 - y[group == 1]) * w[group == 1]) / (sum(w[group == 1]) + 1) + smooth_coef) /
              (sum(y[group == 1] * w[group == 1]) / (sum(w[group == 1]) + 1) + smooth_coef)) - \
       np.log((sum((1 - y_full[group_full == 1]) * w_full[group_full == 1]) /
               sum(w_full[group_full == 1]) + smooth_coef) /
              (sum(y_full[group_full == 1] * w_full[group_full == 1]) /
               sum(w_full[group_full == 1]) + smooth_coef)) - \
       (np.log((sum((1 - y[group == 0]) * w[group == 0]) / (sum(w[group == 0]) + 1) + smooth_coef) /
               (sum(y[group == 0] * w[group == 0]) / (sum(w[group == 0]) + 1) + smooth_coef)) -
        np.log((sum((1 - y_full[group_full == 0]) * w_full[group_full == 0]) /
                sum(w_full[group_full == 0]) + smooth_coef) /
               (sum(y_full[group_full == 0] * w_full[group_full == 0]) /
                sum(w_full[group_full == 0]) + smooth_coef)))

    return nwoe

def woe_scalar(goods, bads, total_goods, total_bads, smoothing_coef=0.0):
    """Calculates WOE value for a single group with optional smoothing coefficient.
    
    Args:
        goods (scalar): number of goods in group
        bads (scalar): number of bads in group
        total_goods (scalar): number in goods in population
        total_bads (scalar): number in bads in population
        smoothing_coef (float, optional): Smoothing to skew small groups to be more similar to population average. Defaults to 0.0.
    
    Returns:
        woe (float): WOE value for group
    """
    
    if smoothing_coef < 0:
        raise ValueError("'smoothing coef should be positive.'")
    # empty group WOE is trivially 0
    if goods == bads == 0:
        return 0.0

    try:
        group_badrate = bads / (goods + bads)
        group_goodrate = goods / (goods + bads)
        total_badrate = total_bads / (total_goods + total_bads)
        total_goodrate = total_goods / (total_goods + total_bads)
        return log((group_goodrate + smoothing_coef) / (group_badrate + smoothing_coef)) - log(
            (total_goodrate + smoothing_coef) / (total_badrate + smoothing_coef)
        )
    except (ZeroDivisionError, ValueError):
        raise ValueError(
            "Some group contains only good/bad observations. WOE value doesn't exist. Try using non-zero smoothing coef"
        )

def _order_leaves(tree):
    """Returns indices of sklearn tree leaves from left to right.
    
    Args:
        tree (object): instance of sklearn.tree object used in classifiers and other models
    
    Returns:
        np.array: indeces of leaf nodes from left to right
    """
    
    leaves_list = list()
    
    def leaves(i):
        left = tree.children_left[i]
        right = tree.children_right[i]
        if left == -1 and right == -1: #node is a leaf
            leaves_list.append(i)
        else:
            leaves(left)
            leaves(right)
               
    leaves(0)
    return np.array(leaves_list)


def tree_based_grouping(x, y, group_count, min_samples, woe_smooth_coef=0.01, w=None):
    """Grouping using decision trees, accepts observations with Nan values.
    
    Args:
        x (np.array): vector of observation values
        y (np.array): vector of target consisting of 0 and 1
        group_count (int): maximum number of groups
        min_samples (int): minimum number of objects in leaf (observations in each group)
        woe_smooth_coef (float, optional): Smoothing to skew small 
            groups to be more similar to population average. Defaults to 0.0.
        w (np.array, optional): vector of observations weights. Defaults to None.
    
    Raises:
        ValueError:
    
    Returns:
        tuple: array of bin edges with -+inf at the ends,
            array of WOEs for each group,
            WOE value for NaN observations
    """
    # check if x,y,w are 1d vectors and same length
    check_consistent_length(x, y, w)
    x = column_or_1d(x)
    y = column_or_1d(y)

    if w is None:
        w = np.ones(x.size)
    else:
        w = column_or_1d(w)

    # check if y has just {0,1}
    if not np.all((y == 1) | (y == 0)):
        raise ValueError("y should only contain values 0 and 1.")

    # calculate nan_woe
    y_nan = y[np.isnan(x)]
    w_nan = w[np.isnan(x)]
    bads_nan = sum(w_nan[y_nan == 1])
    goods_nan = sum(w_nan[y_nan == 0])

    total_bads = sum(w[y == 1])
    total_goods = sum(w[y == 0])

    # remove Nan values from all vectors
    w = w[~np.isnan(x)]
    y = y[~np.isnan(x)]
    x = x[~np.isnan(x)]

    if np.all(w == 1):
        w = None

    x = x.reshape(x.shape[0], -1)

    clf = DecisionTreeClassifier(max_leaf_nodes=group_count, min_samples_leaf=min_samples)
    clf.fit(x, y, sample_weight=w)

    bins = np.array([-np.inf] + sorted(clf.tree_.threshold[clf.tree_.feature == 0]) + [np.inf])

    leaves = [l[0] for l in clf.tree_.value[_order_leaves(clf.tree_)]]
    
    woes = np.array([woe_scalar(goods, bads, total_goods, total_bads, woe_smooth_coef) for goods, bads in leaves])
    nan_woe = woe_scalar(goods_nan, bads_nan, total_goods, total_bads, woe_smooth_coef)
    
    return bins, woes, nan_woe

def tree_based_grouping_uplift(x, y, group_count, min_samples, w=None, group=None):
    """Grouping using decision trees

    Args:
        x (np.array or pandas.Series): array of values, shape (number of observations,)
        y (np.array or pandas.Series): binary target, shape (number of observations,)
        group_count (int): maximum number of groups
        min_samples (int): minimum number of objects in leaf (observations in each group)
        w (np.array or pandas.Series, optional): sample weights
        group (np.array or pandas.Series, optional): array with binary uplift group (treatment or control)

    Returns:
        np.array: array of split points (including -+inf)

    notes:
        x should not include nans
    """
    # TODO: add documentation about split points selected by DecisionTreeClassifier (Pavel email)
    check_consistent_length(x, y)
    x = column_or_1d(x)
    assert_all_finite(x)
    y = column_or_1d(y)


    if len(set(y) - {0, 1}) > 0:
        raise ValueError('y should consist just of {0,1}')

    notnan_mask = ~np.isnan(x)

    if w is not None:
        check_consistent_length(y, w)
        w = column_or_1d(w)[notnan_mask]

    x = x.reshape(x.shape[0], -1)  # (n,) -> (n, 1)

    if group is None:
        clf = DecisionTreeClassifier(max_leaf_nodes=group_count, min_samples_leaf=min_samples)
        clf.fit(x[notnan_mask], y[notnan_mask], sample_weight=w)
    else:
        import uplift.tree
        clf = uplift.tree.DecisionTreeClassifier(criterion='uplift_gini', max_leaf_nodes=group_count,
                                                      min_samples_leaf=min_samples)
        clf.fit(x[notnan_mask], y[notnan_mask], group[notnan_mask], sample_weight=w)

    # clf.tree_treshold holds splitting points for all nodes
    # each nodes splits by certain feature (in our case there is only one)
    # so we only select splitting points with feature with index 0
    # the rest are leaf notes with index -2 since they don't have spliting value
    final_bins = np.concatenate([np.array([-np.inf]),
                                 np.sort(clf.tree_.threshold[clf.tree_.feature == 0]),
                                 np.array([np.inf])])
    return _convert_to_proper_bin_dtype(x.dtype, final_bins)

def auto_group_continuous(x, y, group_count, min_samples, woe_smooth_coef, bins=None, w=None, group=None):
    """Auto grouping continuous features

    Args:
        x:
        y:
        group_count:
        min_samples:
        woe_smooth_coef:
        bins:
        w:
        group:

    Returns:
        tuple: list of intervals, array of woes, nan woe
    """

    if group is None:
        return tree_based_grouping(x.values, y, group_count=group_count, min_samples=min_samples, w=w, woe_smooth_coef=woe_smooth_coef)

    notnan_mask = x.notnull()
    if w is not None:
        w_nna = w[notnan_mask]
    else:
        w_nna = None

    if bins is None:
        bins = tree_based_grouping_uplift(x[notnan_mask], y[notnan_mask], group_count, min_samples, w=w_nna, 
                                   group=None if group is None else  group[notnan_mask])

    # temporary DataFrame since we need both x and y in grouping / aggregation
    if w is not None:
        df = pd.DataFrame({'x': x, 'y': y, 'w': w})
    else:
        w1 = np.ones(len(x))
        df = pd.DataFrame({'x': x, 'y': y, 'w': w1})

    if group is not None:
        df['group'] = group

    df.loc[pd.isnull(df['y']), 'w'] = np.nan
    bin_indices = pd.cut(df[notnan_mask]['x'], bins=bins, right=False, labels=False)
    # sg Some values can be missing in new data
    woes = np.zeros(bins.shape[0] - 1)
    if group is None:
        new_woes = df.groupby(bin_indices).apply(
            lambda rows: woe(rows['y'], df['y'], woe_smooth_coef, w=rows['w'], w_full=df['w'])).to_dict()
        np.put(woes, list(new_woes.keys()), list(new_woes.values()))
        nan_woe = woe(df[~notnan_mask]['y'], df['y'], woe_smooth_coef, w=df[~notnan_mask]['w'], w_full=df['w'])
    else:
        new_woes = df.groupby(bin_indices).apply(
            lambda rows: nwoe(rows['y'], df['y'], rows['group'], df['group'], woe_smooth_coef, w=rows['w'],
                              w_full=df['w'])).to_dict()
        np.put(woes, list(new_woes.keys()), list(new_woes.values()))
        nan_woe = nwoe(df[~notnan_mask]['y'], df['y'], df[~notnan_mask]['group'], df['group'], woe_smooth_coef,
                       w=df[~notnan_mask]['w'], w_full=df['w'])

    return bins, woes, nan_woe


def auto_group_categorical(x, y, group_count, min_samples, min_samples_cat, woe_smooth_coef, bins=None, w=None,
                           group=None):
    """Auto grouping categorical features

    Args:
        x:
        y:
        group_count:
        min_samples:
        min_samples_cat:
        woe_smooth_coef:
        bins:
        w:
        group:

    Returns:
        tuple: list of intervals, array of woes, unknown woe
    """
    # temporary DataFrame since we need both x and y in grouping / aggregation
    # print('auto_group_categorical')
    if x.dtype.name == 'category':
        x = x.astype(x.cat.categories.dtype.name)

    if w is not None:
        df = pd.DataFrame({'x': x, 'y': y, 'w': w})
    else:
        w1 = np.ones(len(x))
        df = pd.DataFrame({'x': x, 'y': y, 'w': w1})

    if group is not None:
        df['group'] = group

    df.loc[pd.isnull(df['y']), 'w'] = np.nan
    df['wy'] = df['w'] * df['y']

    if bins is None:
        if group is None:
            stats = df.groupby('x').apply(
                lambda rows: pd.Series(index=['cnt', 'cnt_bads'], data=[rows['w'].sum(), rows['wy'].sum()]))
            stats['event_rate'] = np.nan
            stats.loc[stats['cnt'] > 0, 'event_rate'] = stats['cnt_bads'] / stats['cnt']
            nan_stat = pd.Series(index=['cnt', 'cnt_bads'],
                                 data=[df[df.x.isnull()]['w'].sum(), df[df.x.isnull()]['wy'].sum()])
            if nan_stat['cnt'] > 0:
                nan_stat['event_rate'] = nan_stat['cnt_bads'] / nan_stat['cnt']
            else:
                nan_stat['event_rate'] = np.nan
        else:
            stats = df.groupby('x').apply(
                lambda rows: pd.Series(index=['cnt', 'cnt_0', 'cnt_1', 'cnt_bads_0', 'cnt_bads_1'],
                                       data=[rows['w'].sum(),
                                             rows[rows['group'] == 0]['w'].sum(),
                                             rows[rows['group'] == 1]['w'].sum(),
                                             rows[rows['group'] == 0]['wy'].sum(),
                                             rows[rows['group'] == 1]['wy'].sum(),
                                             ]))
            stats['event_rate'] = np.nan
            stats.loc[(stats['cnt_0'] > 0) & (stats['cnt_1'] > 0), 'event_rate'] = \
                stats['cnt_bads_1'] / stats['cnt_1'] - stats['cnt_bads_0'] / stats['cnt_0']

            nan_stat = pd.Series(index=['cnt', 'cnt_0', 'cnt_1', 'cnt_bads_0', 'cnt_bads_1'],
                                 data=[df[df.x.isnull()]['w'].sum(),
                                       df[df.x.isnull() & (group == 0)]['w'].sum(),
                                       df[df.x.isnull() & (group == 1)]['w'].sum(),
                                       df[df.x.isnull() & (group == 0)]['wy'].sum(),
                                       df[df.x.isnull() & (group == 1)]['wy'].sum(),
                                       ])
            if nan_stat['cnt_0'] > 0 and nan_stat['cnt_1'] > 0:
                nan_stat['event_rate'] = nan_stat['cnt_bads_1'] / \
                                         nan_stat['cnt_1'] - nan_stat['cnt_bads_0'] / nan_stat['cnt_0']
            else:
                nan_stat['event_rate'] = np.nan

        # (DG) rare mask doesn't consider treatment and control counts separetly (make review leater)
        rare_mask = stats['cnt'] < min_samples_cat
        rare_values = stats[rare_mask].index.values
        rare_df = df.join(pd.DataFrame(index=rare_values), on='x', how='inner')
        rare_w = rare_df['w'].values
        rare_wy = rare_df['wy'].values
        # sg!!!
        # cat -> statistically significant event-rate
        mapping = stats[~rare_mask]['event_rate'].to_dict()

        if group is None:
            if nan_stat['cnt'] >= min_samples_cat:
                mapping[np.nan] = nan_stat['event_rate']
            elif nan_stat['cnt'] > 0:
                rare_values = np.append(rare_values, np.nan)
                rare_w = np.append(rare_w, df[df.x.isnull()].w.values)
                rare_wy = np.append(rare_wy, df[df.x.isnull()].wy.values)

            mapping.update({v: rare_wy.sum() / rare_w.sum() for v in rare_values})
        else:
            rare_group = rare_df['group'].values

            if nan_stat['cnt'] >= min_samples_cat and nan_stat['cnt_0'] > 0 and nan_stat['cnt_1'] > 0:
                mapping[np.nan] = nan_stat['event_rate']
            elif nan_stat['cnt'] > 0:  # nan_stat['cnt'] > 0:
                rare_values = np.append(rare_values, np.nan)
                rare_w = np.append(rare_w, df[df.x.isnull()].w.values)
                rare_wy = np.append(rare_wy, df[df.x.isnull()].wy.values)
                rare_group = np.append(rare_group, df[df.x.isnull()].group.values)

            if rare_w[rare_group == 1].sum() > 0 and rare_w[
                rare_group == 0].sum() > 0 and rare_w.sum() >= min_samples_cat:
                mapping.update({v: rare_wy[rare_group == 1].sum() / rare_w[rare_group == 1].sum() -
                                   rare_wy[rare_group == 0].sum() / rare_w[rare_group == 0].sum()
                                for v in rare_values})
            else:
                mapping.update({v: 0. for v in rare_values})

        # new continuous column
        x2 = df.x.replace(mapping)


        if group is None:
            bins, woes, _ = tree_based_grouping(x2.values, y, group_count=group_count, min_samples=min_samples, w=w, woe_smooth_coef=woe_smooth_coef)
        else:
            bins = tree_based_grouping_uplift(x2, y, group_count, min_samples, w=w, group=group)

        # mapping: cat -> ER
        # bins: ER [-inf, 0.1, 0.34,+inf]]

        # sg - rewrite - now duplicated functionality with "else" below
        bin_indices = pd.cut(x2, bins=bins, right=False, labels=False)

        if group is None:
            woes = df.groupby(bin_indices).apply(lambda rows: woe(
                rows['y'], df['y'], woe_smooth_coef, w=rows['w'], w_full=df['w'])).values

        # cat -> group number Series
        # groups = pd.cut(pd.Series({value: er for value, er in mapping.items() if not (isinstance(value, float) and np.isnan(value))}),
        groups = pd.cut(pd.Series(mapping), bins=bins, right=False, labels=False)
        bins = groups.to_dict()
        if nan_stat['cnt'] == 0:
            # nan_group = pd.cut([m[np.nan]], bins=bins, right=False, labels=False)[0]
            # new group for nan with WOE=0.
            nan_group = groups.max() + 1
            woes = np.append(woes, [0.])
            bins[np.nan] = nan_group

        # WOE for values that are not present in the training set
        unknown_woe = 0

    else:
        # sg duplication here!
        # this branch was added to support "recalc WOEs on fixed bins (splits)" mode
        # {1: 2, 3: 2, 4: 0}
        df['tmp'] = np.nan
        for cat, g in bins.items():
            if type(cat) == float and np.isnan(cat):
                df.loc[df.x.isnull(), 'tmp'] = g
            else:
                df.loc[df.x == cat, 'tmp'] = g
        # sg Some values can be missing in new data
        woes = np.zeros(len(bins))

        if group is None:
            new_woes = df.groupby(df['tmp']).apply(lambda rows: woe(
                rows['y'], df['y'], woe_smooth_coef, w=rows['w'], w_full=df['w'])).to_dict()
        else:
            new_woes = df.groupby(df['tmp']).apply(lambda rows: nwoe(
                rows['y'], df['y'], rows['group'], df['group'], woe_smooth_coef, w=rows['w'], w_full=df['w'])).to_dict()

        np.put(woes, list(new_woes.keys()), list(new_woes.values()))
        unknown_woe = 0

    return bins, woes, unknown_woe


def _event_rates(x, y, bins, w=None):
    """Event rates

    Args:
        x:
        y:
        bins:
        w:

    Returns:
        np.array: event rates for the given bins
    """
    values = pd.Series([np.nan] * (len(bins) - 1), index=range(1, len(bins)))
    if w is None:
        values.update(y.groupby(np.digitize(x, bins)).mean())
        result = values.values
    else:
        wy = w * y
        df = pd.DataFrame({'x': x, 'y': y, 'w': w, 'wy': wy})
        df = df.groupby(np.digitize(df['x'], bins)).sum()
        values.update(df['wy'] / df['w'])
        result = values.values
    return result

def _convert_to_proper_bin_dtype(data_type, target):
        '''
        Converts `target` to proper dtype based on `data_type`
        float16/32 -> float16/32
        others -> float64
        '''
        if data_type is np.dtype(np.float16):
            return np.float16(target)
        elif data_type is np.dtype(np.float32):
            return np.float32(target)
        else:
            return target



class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class NumpyJSONDecoderOld(json.JSONDecoder):
    """
    For compatibility
    """

    def __init__(self):
        super(NumpyJSONDecoderOld, self).__init__(object_hook=self.hook)

    def hook(self, d):
        if 'bins' not in d:  # we need to apply changes only to second level of JSON "objects"
            return d
        if isinstance(d['bins'], list):
            d['bins'] = np.array(d['bins'])
        d['woes'] = np.array(d['woes'])
        return d


class NumpyJSONDecoder(json.JSONDecoder):
    """
    TODO: rename, it's not universal numpy decoder
    """

    def __init__(self):
        super(NumpyJSONDecoder, self).__init__(object_hook=self.hook)

    def hook(self, d):
        if 'bins' not in d and 'cat_bins' not in d:  # we need to apply changes only to second level of JSON "objects"
            return d
        if 'bins' in d:
            d['bins'] = np.array(d['bins'])
        d['woes'] = np.array(d['woes'])
        return d


class Grouping(BaseEstimator, TransformerMixin):
    """Grouping

    Args:
        columns (list of str): continuous columns
        cat_columns (list of str): categorical columns
        group_count (int, optional): maximum number of groups (default: 3)
        min_samples (int, optional): minimum number of objects in leaf (observations in each group) (default: 1)
        min_samples_cat (int, optional): minimal number of samples in category to trust event rate (default: 1)
        woe_smooth_coef (float, optional): smooth coefficient (default: 0.001)
        filename (str): filename (default: None)

    Attributes:
        bins_data: list of dict with keys: bins, woes, nan_woe
    """

    def __init__(self, columns, cat_columns, group_count=3, min_samples=1, min_samples_cat=1, woe_smooth_coef=0.001,
                 filename=None):

        self.columns = columns
        self.cat_columns = cat_columns
        self.group_count = group_count
        self.min_samples = min_samples
        self.min_samples_cat = min_samples_cat  # sg
        self.woe_smooth_coef = woe_smooth_coef
        self.filename = filename  # not needed
        if filename is not None:
            self.load(filename)

    def get_dummy_names(self, columns_to_transform=None):
        """
        Get name of dummy variables from dummy variable transformation of predictors in list columns_to_transform

        Args:
            columns_to_transform (list of str, optional): List of predictors to get dummy names for. Defaults to None.

        Returns:
            dict: Dictionary with predictor name as key and dummy names list as value
        """

        if columns_to_transform is not None:
            for column in columns_to_transform:
                if column not in self.columns + self.cat_columns:
                    raise ValueError(f'Column {column} not in grouping.')
            cols_num = [col for col in columns_to_transform if col in self.columns]
            cols_cat = [col for col in columns_to_transform if col in self.cat_columns]
        else:
            cols_num = self.columns
            cols_cat = self.cat_columns
            columns_to_transform = cols_num + cols_cat

        dummies = {}
        suffix = '_DMY'

        for name in columns_to_transform:
            if name in cols_num:
                bin_data = self.bins_data_[name]
                dummy_vars = []
                for i in range(len(bin_data['woes'])):
                    dummy_name = f'{name}{suffix}_{i}'
                    dummy_vars.append(dummy_name)
                dummy_name = f'{name}{suffix}_NaN'
                dummy_vars.append(dummy_name)
                dummies[name] = dummy_vars

            if name in cols_cat:
                bin_data = self.bins_data_[name]
                dummy_vars = []
                for i in range(len(bin_data['woes'])):
                    dummy_name = f'{name}{suffix}_{i}'
                    dummy_vars.append(dummy_name)
                dummy_name = f'{name}{suffix}_Unknown'
                dummy_vars.append(dummy_name)
                dummies[name] = dummy_vars

        return dummies

    def transform(self, data, transform_to='woe', columns_to_transform=None, progress_bar=False):
        """
        Performs transformation of `data` based on `transform_to` parameter and adds suffix to column names.

        Args:
            data (pd.DataFrame): data to be transformed
            transform_to (str, optional): Type of transformation. Possible values: `woe`,`shortnames`,`group_number`,`dummy`. Defaults to 'woe'.
            columns_to_transform (list of str, optional): List of columns of data to be transformed. Defaults to None.
            progress_bar (bool, optional): Display progress bar? Defaults to False.

        Returns:
            pd.DataFrame: transformed data
        """

        if columns_to_transform is not None:
            for column in columns_to_transform:
                if column not in self.columns + self.cat_columns:
                    raise ValueError('Column {} not in grouping.'.format(column))
            cols_num = [col for col in columns_to_transform if col in self.columns]
            cols_cat = [col for col in columns_to_transform if col in self.cat_columns]
        else:
            cols_num = self.columns
            cols_cat = self.cat_columns
            columns_to_transform = cols_num + cols_cat

        if transform_to not in {'woe','shortnames', 'group_number', 'dummy'}:
            raise ValueError("'{0}' is not a valid transform_to value "
                             "('woe', 'shortnames', 'group_number', 'dummy').".format(transform_to))
        else:
            suffix_dict = {'woe':'_WOE', 'shortnames':'_RNG', 'group_number':'_GRP', 'dummy':'_DMY'}
            suffix = suffix_dict[transform_to]

        if progress_bar:
            iterator = tqdm(data[columns_to_transform].iteritems(), total=len(columns_to_transform), leave=True,
                            unit='cols')
        else:
            iterator = data[columns_to_transform].iteritems()
        
        if transform_to != 'dummy':
            data_woe = pd.DataFrame(columns=cols_num + cols_cat)
        else:
            data_woe = pd.DataFrame(index=data.index)

        for name, column in iterator:
            # print(name)
            if progress_bar:
                iterator.set_description(desc=name, refresh=True)
            if name in cols_num:
                bin_data = self.bins_data_[name]

                if transform_to == 'woe':
                    # use standard woe values
                    target_values = np.array(bin_data['woes']).astype(np.float32)
                    target_nan = bin_data['nan_woe']

                elif transform_to == 'shortnames':
                    # use internaval as shortnames eg. (-inf,1.35]
                    target_values = ['[{:.3f}, {:.3f})'.format(bin_data['bins'][i], bin_data['bins'][i + 1]) for i in
                                     range(len(bin_data['bins']) - 1)]
                    target_nan = 'NaN'

                elif transform_to == 'group_number':
                    # use group number
                    target_values = [i for i in range(len(bin_data['bins']) - 1)]
                    target_nan = len(bin_data['bins']) - 1

                if transform_to != 'dummy':
                    tmp = pd.cut(column, bin_data['bins'], right=False, labels=False)
                    map_dict = {np.nan: target_nan, **{i: target_values[i] for i in range(len(target_values))}}
                    data_woe[name] = tmp.map(map_dict)
                
                else: #create dummy variables
                    tmp = pd.cut(column, bin_data['bins'], right=False, labels=False)
                    for i in range(len(bin_data['woes'])):
                        dummy_name = self.get_dummy_names(columns_to_transform=[name])[name][i]
                        data_woe[dummy_name] = 0
                        data_woe.loc[tmp==i, dummy_name] = 1
                    dummy_name = self.get_dummy_names(columns_to_transform=[name])[name][-1]
                    data_woe[dummy_name] = 0
                    data_woe.loc[pd.isnull(tmp), dummy_name] = 1

            if name in cols_cat:
                bin_data = self.bins_data_[name]

                if transform_to == 'woe':
                    # use standard woe values
                    target_values = np.array(bin_data['woes']).astype(np.float32)
                    target_nan = bin_data['unknown_woe']

                    map_dict = {k: target_values[v] for k, v in bin_data['bins'].items()}
                    data_woe[name] = column.map(map_dict).astype(np.float32).fillna(target_nan)

                elif transform_to == 'shortnames':
                    # use concatenated values as group names
                    groups = [[] for i in range(len(bin_data['woes']))]
                    for value, group in bin_data['bins'].items():
                        groups[group].append(str(value))
                    # cut these names to 40chars
                    target_values = [','.join(s)[:40] for s in groups]
                    # add numbering to potential duplicate group names
                    for target_name, count in Counter(target_values).items():
                        if count > 1:
                            for suf in [' '] + list(range(1, count)):
                                target_values[target_values.index(target_name)] = target_name + str(suf)
                    target_nan = 'Unknown'

                    map_dict = {k: target_values[v] for k, v in bin_data['bins'].items()}
                    data_woe[name] = column.astype(str).map(map_dict).fillna(target_nan)

                elif transform_to == 'group_number':
                    # use group number
                    s = set(bin_data['bins'].values())
                    target_values = [i for i in range(len(s))]
                    target_nan = len(s)
                    
                    map_dict = {k: target_values[v] for k, v in bin_data['bins'].items()}
                    data_woe[name] = column.map(map_dict).astype(np.float32).fillna(target_nan)
                    
                
                elif transform_to == 'dummy'  : #create dummy variables
                    s = set(bin_data['bins'].values())
                    target_values = [i for i in range(len(s))]
                    target_nan = len(s)
                    map_dict = {k: target_values[v] for k, v in bin_data['bins'].items()}
                    tmp = column.map(map_dict).astype(np.float32).fillna(target_nan)
                    for i in range(len(bin_data['woes'])):
                        dummy_name = self.get_dummy_names(columns_to_transform=[name])[name][i]
                        data_woe[dummy_name] = 0
                        data_woe.loc[tmp==target_values[i], dummy_name] = 1
                    dummy_name = self.get_dummy_names(columns_to_transform=[name])[name][-1]
                    data_woe[dummy_name] = 0
                    data_woe.loc[tmp==target_nan, dummy_name] = 1

        if transform_to == 'group_number':
            data_woe = data_woe.astype(np.int32)
        elif transform_to == 'dummy':
            data_woe = data_woe.astype(np.int16)
        elif transform_to == 'woe':
            data_woe = data_woe.astype(np.float32)

        if transform_to != 'dummy':
            renaming = {col: col + suffix for col in data_woe.columns}
        else:
            renaming = {col: col for col in data_woe.columns}

        return data_woe.rename(renaming, axis='columns')

    def saveOld(self, filename):
        # check_is_fitted(self, 'bins_data_')
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.bins_data_, f, ensure_ascii=False,
                      cls=NumpyJSONEncoder)

    def loadOld(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            self.bins_data_ = json.load(f, cls=NumpyJSONDecoderOld)
        # ugly workaround to have nan properly assigned
        for v, g in self.bins_data_.items():
            changenan = False
            nanval = 0
            if isinstance(g['bins'], dict):
                for c, b in g['bins'].items():
                    if c == 'NaN':
                        changenan = True
                        nanval = b
                if changenan:
                    g['bins'][np.nan] = nanval
                    del g['bins']['NaN']

    # sg
    def save(self, filename):
        """
        Saves the grouping dictionary to external JSON file.

        Args:
            filename (str): name of file to save the grouping to
        """
        # check_is_fitted(self, 'bins_data_')
        tmp = copy.deepcopy(self.bins_data_)
        for k, v in tmp.items():
            if isinstance(v['bins'], dict):
                v['cat_bins'] = list(v['bins'].items())
                del v['bins']

        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(tmp,
                      file,
                      ensure_ascii=False,
                      cls=NumpyJSONEncoder,
                      indent=2)

        display(f"Grouping saved on {datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')} to file {filename}")

    def load(self, filename):
        """
        Loads the grouping dictionary from external JSON file.

        Args:
            filename (str): name of file to load the grouping from
        """
        with open(filename, 'r', encoding='utf-8') as f:
            self.bins_data_ = json.load(f, cls=NumpyJSONDecoder)
            # display(self.bins_data_)
            for k, v in self.bins_data_.items():
                if 'cat_bins' in v:
                    v['bins'] = dict(v['cat_bins'])
                    del v['cat_bins']
                    # NaNs workaround
                    if "NaN" in v['bins']:
                        v[np.nan] = v['NaN']
                        del v['NaN']

                if 'dtype' in v.keys():
                    if v['dtype'] == 'float16':
                        v['bins'] = v['bins'].astype(np.float16)
                    elif v['dtype'] == 'float32':
                        v['bins'] = v['bins'].astype(np.float32)

    def _auto_grouping(self, x, y, w=None, group=None):
        # print(x.name)

        if not ((type(x) is pd.Series) or (type(x) is np.ndarray and x.ndim == 1)):
            raise ValueError("x should be a pandas.Series or 1d numpy.array")
        
        if not ((type(y) is pd.Series) or (type(y) is np.ndarray and x.ndim == 1)):
            raise ValueError("y should be a pandas.Series")


        if x.name in self.columns:
            bins, woes, nan_woe = auto_group_continuous(
                x, y, self.group_count, self.min_samples, self.woe_smooth_coef,
                bins=self.bins_data_[x.name]['bins'] if x.name in self.bins_data_ else None, w=w, group=group)
        else:
            bins, woes, unknown_woe = auto_group_categorical(
                x, y, self.group_count, self.min_samples, self.min_samples_cat, self.woe_smooth_coef,
                bins=self.bins_data_[x.name]['bins'] if x.name in self.bins_data_ else None, w=w, group=group)

        bin_data = {'bins': bins, 'woes': woes, }

        if x.name in self.columns:
            bin_data['nan_woe'] = nan_woe
        else:
            bin_data['unknown_woe'] = unknown_woe

        bin_data['dtype'] = x.dtype.name
        self.bins_data_[x.name] = bin_data

    def fit(self, X, y, w=None, progress_bar=False, category_limit=100):
        """
        Makes automatic grouping

        Args:
            X (pandas.DataFrame):
            y (pandas.Series or np.array):
            w (np.array or pandas.Series, optional): sample weights (default: None)
            progress_bar (bool, optional): progress bar? (default: False)
            category_limit (int, optional): (default 100)

        """

        if type(X) != pd.DataFrame:
            raise ValueError('X should be DataFrame')
        check_consistent_length(X, y)
        y = column_or_1d(y)

        if np.any(X.columns.duplicated()):
            duplicities = [col_name for col_name, duplicated in zip(X.columns, X.columns.duplicated()) if duplicated]
            raise ValueError(f"Columns {list(dict.fromkeys(duplicities))} are duplicated in your Dataset.")  # list.dict hack to remove duplicated quickly

        for name, column in X[self.columns].iteritems():
            if np.any(np.isinf(column.values)):
                    raise ValueError(f'Column {name} containes non-finite values.')


        if w is not None:
            check_consistent_length(w, y)
            w = column_or_1d(w).astype('float64')
            w[pd.isnull(y)] = np.nan

        for name, column in X[self.cat_columns].iteritems():
            if column.nunique() > category_limit:
                raise ValueError(f'Column {name} has more than {category_limit} unique values. '
                                 'Large number of unique values might cause memory issues. '
                                  'This limit can be set with parameter `category_limit.')

        if not hasattr(self, 'bins_data_'):
            self.bins_data_ = {}
        # So we will keep already trained bins if they exists (loaded from file or fitted before)

        if progress_bar:
            iterator = tqdm(self.columns + self.cat_columns, leave=True, unit='cols')
        else:
            iterator = self.columns + self.cat_columns

        for column in iterator:
            if progress_bar:
                iterator.set_description(desc=column, refresh=True)
            self._auto_grouping(X[column], y, w)
        return self  # sg

    def fit_uplift(self, X, y, group, w=None):
        """Makes automatic grouping

        Args:
            X (pandas.DataFrame): dataframe with features
            y (pandas.Series): Series with target
            group (pandas.Series): Series with uplift group (treatment or control)
            w (np.array or pandas.Series): sample weights (default: None)
        """

        if type(X) != pd.DataFrame:
            raise ValueError('X should be DataFrame')
        check_consistent_length(X, y)
        y = column_or_1d(y)

        if w is not None:
            check_consistent_length(w, y)
            w = column_or_1d(w).astype('float64')
            w[pd.isnull(y)] = np.nan

        if not hasattr(self, 'bins_data_'):
            self.bins_data_ = {}

        iterator = self.columns + self.cat_columns

        for column in iterator:
            self._auto_grouping(X[column], y, w, group)

        return self

    def plot_bins(self, data, cols_pred_num, cols_pred_cat, mask, col_target, output_folder, col_weight=None):
        """
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
        if col_weight is not None:
            col_weight = data[mask][col_weight]

        for col in cols_pred_num + cols_pred_cat:
            if col not in self.bins_data_.keys():
                raise ValueError(f"{col} is missing in grouping.")

        if len(self.bins_data_) > 0:
            for v, g in sorted(self.bins_data_.items(), key=operator.itemgetter(0)):
                if v in cols_pred_num:
                    display(Markdown('***'))
                    display(Markdown('### {}'.format(v)))
                    print_binning_stats_num(data[mask][[col_target, v]], v, col_target, g['bins'], g['woes'],
                                            g['nan_woe'], col_weight=col_weight
                                            , savepath=output_folder + '/predictors/' + v + '_')
                elif v in cols_pred_cat:
                    display(Markdown('***'))
                    display(Markdown('### {}'.format(v)))
                    print_binning_stats_cat(data[mask][[col_target, v]], v, col_target
                                            , g['bins'].keys(), g['bins'].values(), g['woes'], g['unknown_woe'],
                                            col_weight=col_weight
                                            , savepath=output_folder + '/predictors/' + v + '_')

    def export_as_sql(self, suffix='_WOE', filename=None):
        """Creates a SQL script for transforming data based on fitted grouping. Designed to be used with print() to replace \\n with newlines.

        Returns a string with SQL script with sets of CASE statements for transforming data.

        Args:
            suffix (str): suffix to be added to transformed predictors, default='WOE'
            filename (str): path to file for export

        Returns:
            str : SQL script with \\n for new lines.
        """
        try:
            predictors = [key + suffix for key, _ in self.bins_data_.items()]
        except:
            raise KeyError('Binning data missing or corrupted. Make sure grouping was fitted properly.')
        coefficients = np.asarray([[1.0] * len(predictors)])
        intercept = 1.0
        temporary_scorecard = ScoreCard(grouping=self,
                                        predictors=predictors,
                                        coefficients=coefficients,
                                        intercept=intercept)
        e = temporary_scorecard.to_SQL()
        e = e.replace('\n', '$')
        pattern = r'(select[\s$]*case.*),[\s$]*case[\s$]*when 1=1'
        e = re.findall(pattern, e)[0]
        e = e.replace('$', '\n') + '\n from _SOURCETABLENAME_'

        return e

    def export_dictionary(self, suffix="_WOE", interval_edge_rounding=3, woe_rounding=5):
        '''
        Returns a dictionary with (woe:bin/values) pairs for fitted predictors.

        Numerical predictors are in this format:
        round(woe): "[x, y)"

        Categorical predictors are in this format:
        round(woe): ["AA","BB","CC","Unknown"]

        Args:
            suffix (str, optional): suffix of WOE variables
            interval_edge_rounding (int, optional): rounding for numerical variable interval edges
            woe_rounding (int, optional): rounding for WOE values

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
        '''

        woe_dictionary = dict()
        #iterate over numerical predictors
        for col in self.columns:

            # skip vars that are not fitted
            if col not in self.bins_data_.keys():
                continue

            grouping_data = self.bins_data_[col]
            woe_dictionary[col + suffix] = {}
            
            #go over other intervals
            bins = grouping_data['bins']
            woes = grouping_data['woes']
            intervals = list(zip(bins, bins[1:]))
            for woe, (lower, upper) in zip(woes, intervals):
                woe_dictionary[col + suffix][round(np.float32(woe),woe_rounding)] = f"[{lower:.{interval_edge_rounding}f}, {upper:.{interval_edge_rounding}f})"
                
            #NaN WOE handled separately
            nan_woe = round(np.float32(grouping_data['nan_woe']), woe_rounding)
            if nan_woe in woe_dictionary[col + suffix].keys():
                woe_dictionary[col + suffix][nan_woe] += ' NaN'
            else:
                woe_dictionary[col + suffix][nan_woe] = 'NaN'
                
        #interate over categorical predictors
        for col in self.cat_columns:

            # skip vars that are not fitted
            if col not in self.bins_data_.keys():
                continue

            grouping_data = self.bins_data_[col]
            woe_dictionary[col + suffix] = {}
            bins = grouping_data['bins']
            woes = grouping_data['woes']
            
            groups = [list() for _ in woes]
            for value, bin_ in bins.items():
                groups[bin_].append(str(value))
            
            for woe, group in zip(woes, groups):
                woe_dictionary[col + suffix][round(np.float32(woe), woe_rounding)] = group
            
            unknown_woe = round(np.float32(grouping_data['unknown_woe']),woe_rounding)
            if unknown_woe in woe_dictionary[col + suffix].keys():
                woe_dictionary[col + suffix][unknown_woe].append("Unknown")
            else:
                woe_dictionary[col + suffix][unknown_woe] = ["Unknown"]
            
        return woe_dictionary



class Wrapper(object):
    """Very simple wrapper to make binding between elems and textboxes easier"""

    def __init__(self, val):
        self.val = val


class Context(object):
    def __init__(self, column, grouping):
        self.column = column  # column
        self.grouping = grouping

    @property
    def x(self):
        return self.grouping.train_t[self.column]

    @property
    def y(self):
        return self.grouping.train_t[self.grouping.target_column]

    @property
    def weight(self):
        if self.grouping.w_column is not None:
            w = self.grouping.train_t[self.grouping.w_column].copy()
            w[pd.isnull(self.grouping.train_t[self.grouping.target_column])] = np.nan
            return w
        else:
            w = pd.Series(data=np.ones(len(self.grouping.train_t[self.column])),
                          index=self.grouping.train_t[self.column].index)
            w[pd.isnull(self.grouping.train_t[self.grouping.target_column])] = np.nan
            return w

    @property
    def has_nan(self):
        return self.x.isnull().any()


    def validate(self):
        raise NotImplementedError

    def _create_plots(self):
        raise NotImplementedError

    def _update_data(self):
        raise NotImplementedError
    
    def _update_form(self, *args):
        raise NotImplementedError

    def update(self, tab_change=False):
        # print('update', tab_change)
        if tab_change:
            self.grouping.fig.clear()
            self._create_plots()

        valid = self.validate()

        self.grouping.validate_all()

        if valid:
            self._update_data()
        self._update_form(valid)
        self.grouping.fig.canvas.draw()


class ContinuousContext(Context):
    def __init__(self, column, bins, nan_woe, grouping, gini=np.nan, manual_woe=None):
        super(ContinuousContext, self).__init__(column, grouping)
        self.elems = [Wrapper(x1) for x1 in bins[1:-1]]
        self.nan_woe = nan_woe
        self.gini = gini

        if manual_woe is None:
            self.manual_woe = pd.DataFrame({'Man WOE': [np.NaN] * len(bins)})
        else:
            self.manual_woe = pd.DataFrame({'Man WOE': manual_woe})
        # self.manual_woe.index = self.manual_woe.index.astype(str)
        self.manual_woe.index.names = ['group']

    def _create_plots(self):
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])

        self.grouping.ew_bins_ax = self.grouping.fig.add_subplot(gs[0])
        self.grouping.ew_er_ax = self.grouping.ew_bins_ax.twinx()
        self.grouping.ew_bins_ax.set_facecolor('white')
        self.grouping.ew_er_ax.set_facecolor('white')
        self.grouping.ew_bins_ax.tick_params(axis='both', which='both', labelsize=8)
        self.grouping.ew_er_ax.tick_params(axis='both', which='both', labelsize=8)

        self.grouping.ed_bins_ax = self.grouping.fig.add_subplot(gs[1])
        self.grouping.ed_er_ax = self.grouping.ed_bins_ax.twinx()
        self.grouping.ed_bins_ax.set_facecolor('white')
        self.grouping.ed_er_ax.set_facecolor('white')
        self.grouping.ed_bins_ax.tick_params(axis='both', which='both', labelsize=8)
        self.grouping.ed_er_ax.tick_params(axis='both', which='both', labelsize=8)

        self.grouping.groups_bins_ax = self.grouping.fig.add_subplot(gs[2])
        self.grouping.groups_er_ax = self.grouping.groups_bins_ax.twinx()
        self.grouping.groups_bins_ax.set_facecolor('white')
        self.grouping.groups_er_ax.set_facecolor('white')
        self.grouping.groups_bins_ax.tick_params(axis='both', which='both', labelsize=8)
        self.grouping.groups_er_ax.tick_params(axis='both', which='both', labelsize=8)

    def ew_plot(self, bar_ax, plot_ax, valid):
        """ Equi-width plot"""
        # clear axes
        bar_ax.clear()
        plot_ax.clear()

        if not valid:
            bar_ax.text(0.5 * (bar_ax.get_xlim()[0] + bar_ax.get_xlim()[1]),
                        0.5 * (bar_ax.get_ylim()[0] +
                               bar_ax.get_ylim()[1]), 'Error',
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=20, color='red')
            return

        # data
        # |    ****** ** | ****  |         | nan
        # elems as numpy array (withoint +-inf)
        bins = np.array([x1.val for x1 in self.elems])
        # equi-width bins (without NaN bin)
        ew_bins = np.linspace(self.x.min(), self.x.max(),
                              self.grouping.bin_count + 1)
        ew_bin_width = ew_bins[1] - ew_bins[0]  # equi-width bin width
        # bins + ew_bins (with NaN bin)
        all_bins = np.sort(np.unique(np.hstack((ew_bins, bins))))
        # x2 for histogram counting
        x2 = self.x.replace(all_bins[-1], (all_bins[-2] + all_bins[-1]) / 2)
        split_points = bins.copy()  # group borders

        if self.has_nan:
            all_bins = np.append(
                all_bins, ew_bins[-1] + ew_bin_width)  # must add to max
            x2.fillna(ew_bins[-1] + ew_bin_width / 2,
                      inplace=True)  # must add to max
            split_points = np.append(split_points, ew_bins[-1])

        # draw bars
        counts, _ = np.histogram(x2, all_bins)

        bin_groups = []
        bin_colors = []
        g = c = sp = 0
        for border in all_bins:
            if len(split_points) > 0 and sp < len(split_points) and border == split_points[sp]:
                g += 1
                c += 1
                sp += 1
            bin_groups.append(g)
            bin_colors.append(
                self.grouping._bar_colors[c % len(self.grouping._bar_colors)])
        left = all_bins[:-1]
        height = counts
        width = [x1[1] - x1[0] for x1 in zip(all_bins[:-1], all_bins[1:])]
        patches = bar_ax.bar(left, height, width, color=bin_colors, hatch='', edgecolor='black',
                             linewidth=0.5, align='edge')
        if self.has_nan:
            patches[-1].set_hatch('/')

        # plot event rates
        brs = _event_rates(x2, self.y, all_bins, w=self.weight) * 100

        plot_ax.plot(np.vstack([all_bins[1:], all_bins[:-1]]).mean(axis=0), brs, marker='o',
                     color='orangered', linestyle='dotted', ms=3, linewidth=1.5)

        # split_points - plot vertical lines ( | )
        for p in split_points:
            bar_ax.axvline(x=p, color='black', linewidth=0.5)

        # show group border numbers
        for x1 in bins:
            bar_ax.text(x1, bar_ax.get_ylim()[1] * 1.05, '{:.2f}'.format(x1), horizontalalignment='center',
                        verticalalignment='bottom', color='black', rotation=90, fontdict={'size': 8})
        if self.has_nan:
            bar_ax.text((all_bins[-1] + all_bins[-2]) / 2, bar_ax.get_ylim()[1] * 1.05, 'NaN',
                        horizontalalignment='center',
                        verticalalignment='bottom', color='black', rotation=90, fontdict={'size': 8})

        # xticks по границам бинов
        bar_ax.set_xticks(ew_bins)
        bar_ax.set_xticklabels(['{:.2f}'.format(b)
                                for b in ew_bins], rotation=90, fontsize=8)

        bar_ax.set_xlim(all_bins[0], all_bins[-1])

        bar_ax.set_ylabel("count", fontsize=8)
        plot_ax.set_ylabel("event rate, %", fontsize=8)

    def ed_plot(self, bar_ax, plot_ax, valid):
        """ Equi-depth plot"""
        # clear axes
        bar_ax.clear()
        plot_ax.clear()

        if not valid:
            bar_ax.text(0.5 * (bar_ax.get_xlim()[0] + bar_ax.get_xlim()[1]),
                        0.5 * (bar_ax.get_ylim()[0] +
                               bar_ax.get_ylim()[1]), 'Error',
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=20, color='red')
            return

        # data
        # elems as numpy array (without +-inf)
        bins = np.array([x1.val for x1 in self.elems])
        ed_bins = np.unique(np.percentile(self.x[self.x.notnull()], np.linspace(
            0, 100, self.grouping.bin_count + 1), interpolation='lower'))  # equi-depth bins (without NaN bin)
        ed_bin_width = ed_bins[1] - ed_bins[0]  # equi-depth bin width
        # bins + ed_bins (with NaN bin)
        all_bins = np.sort(np.unique(np.hstack((ed_bins, bins))))
        # x2 for histogram counting
        x2 = self.x.replace(all_bins[-1], (all_bins[-2] + all_bins[-1]) / 2)
        split_points = bins.copy()  # group borders

        if self.has_nan:
            all_bins = np.append(all_bins, ed_bins[-1] + ed_bin_width)
            x2.fillna(ed_bins[-1] + ed_bin_width / 2, inplace=True)
            split_points = np.append(split_points, ed_bins[-1])

        # draw bars
        counts, _ = np.histogram(x2, all_bins)

        # bin_groups = []
        bin_colors = []
        c = sp = 0
        for border in all_bins:
            if len(split_points) > 0 and sp < len(split_points) and border == split_points[sp]:
                # g += 1
                c += 1
                sp += 1
            # bin_groups.append(g)
            bin_colors.append(
                self.grouping._bar_colors[c % len(self.grouping._bar_colors)])
        left = np.arange(all_bins.shape[0] - 1)
        height = counts
        width = 1
        patches = bar_ax.bar(left, height, width, color=bin_colors, hatch='', edgecolor='black',
                             linewidth=0.5, align='edge')
        if self.has_nan:
            patches[-1].set_hatch('/')

        # plot event rates
        brs = _event_rates(x2, self.y, all_bins, w=self.weight) * 100

        # print(np.arange(all_bins.shape[0]-1)+0.5, brs)

        plot_ax.plot(np.arange(all_bins.shape[0] - 1) + 0.5, brs, marker='o',
                     color='orangered', linestyle='dotted', ms=3, linewidth=1.5)

        # split_points - plot vertical lines ( | )
        for p in split_points:
            bar_ax.axvline(x=np.where(all_bins == p)[
                0], color='black', linewidth=0.5)

        # show group border numbers
        for x1 in bins:
            bar_ax.text(np.where(all_bins == x1)[0], bar_ax.get_ylim()[1] * 1.05, '{:.2f}'.format(x1),
                        horizontalalignment='center',
                        verticalalignment='bottom', color='black', rotation=90, fontdict={'size': 8})
        if self.has_nan:
            bar_ax.text(all_bins.shape[0] - 1.5, bar_ax.get_ylim()[1] * 1.05, 'NaN', horizontalalignment='center',
                        verticalalignment='bottom', color='black', rotation=90, fontdict={'size': 8})

        # xticks по границам бинов
        bar_ax.set_xticks(
            np.arange(all_bins.shape[0] + (-1 if self.has_nan else 0)))
        # print(np.arange(all_bins.shape[0] + (-1 if self.has_nan else 0)), ['{:.2f}'.format(b) for b in (all_bins[:-1] if self.has_nan else all_bins)])

        bar_ax.set_xticklabels(['{:.2f}'.format(b) for b in (all_bins[:-1] if self.has_nan else all_bins)], rotation=90,
                               fontsize=8)

        bar_ax.set_xlim(0, all_bins.shape[0] - 1)

        # plot_ax.set_visible(False)

        bar_ax.set_ylabel("count", fontsize=8)
        plot_ax.set_ylabel("event rate, %", fontsize=8)

    def groups_plot(self, bar_ax, plot_ax, valid):
        # Final groups plot
        # clear axes
        bar_ax.clear()
        plot_ax.clear()

        if not valid:
            bar_ax.text(0.5 * (bar_ax.get_xlim()[0] + bar_ax.get_xlim()[1]),
                        0.5 * (bar_ax.get_ylim()[0] +
                               bar_ax.get_ylim()[1]), 'Error',
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=20, color='red')
            return

        # data
        # elems as numpy array (without +-inf)
        bins = np.array([x1.val for x1 in self.elems])
        # ed_bin_width = ed_bins[1] - ed_bins[0] # equi-depth bin width
        all_bins = np.concatenate(
            [np.array([self.x.min()]), bins, np.array([self.x.max()])])
        # x2 for histogram counting
        x2 = self.x.replace(all_bins[-1], (all_bins[-2] + all_bins[-1]) / 2)
        split_points = bins.copy()  # group borders

        if self.has_nan:
            split_points = np.append(split_points, all_bins[-1])
            all_bins = np.append(all_bins, all_bins[-1] + 666)
            x2.fillna(all_bins[-1] - 333, inplace=True)

        # draw bars
        counts, _ = np.histogram(x2, all_bins)

        bin_colors = []
        c = sp = 0
        for border in all_bins:
            if len(split_points) > 0 and sp < len(split_points) and border == split_points[sp]:
                c += 1
                sp += 1
            bin_colors.append(
                self.grouping._bar_colors[c % len(self.grouping._bar_colors)])

        left = np.arange(all_bins.shape[0] - 1)
        height = counts
        width = 1

        # print(left, height, width)

        patches = bar_ax.bar(left, height, width, color=bin_colors, hatch='', edgecolor='black',
                             linewidth=0.5, align='edge')
        if self.has_nan:
            patches[-1].set_hatch('/')

        # plot event rates
        brs = _event_rates(x2, self.y, all_bins, w=self.weight) * 100

        plot_ax.plot(np.arange(all_bins.shape[0] - 1) + 0.5, brs, marker='o',
                     color='orangered', linestyle='dotted', ms=3, linewidth=1.5)

        # split_points - plot vertical lines ( | )
        for p in split_points:
            bar_ax.axvline(x=np.where(all_bins == p)[
                0], color='black', linewidth=0.5)

        # show group border numbers
        for x1 in bins:
            bar_ax.text(np.where(all_bins == x1)[0], bar_ax.get_ylim()[1] * 1.05, '{:.2f}'.format(x1),
                        horizontalalignment='center',
                        verticalalignment='bottom', color='black', rotation=90, fontdict={'size': 8})
        if self.has_nan:
            bar_ax.text(all_bins.shape[0] - 1.5, bar_ax.get_ylim()[1] * 1.05, 'NaN', horizontalalignment='center',
                        verticalalignment='bottom', color='black', rotation=90, fontdict={'size': 8})

        # xticks по границам бинов
        bar_ax.set_xticks(
            np.arange(all_bins.shape[0] + (-1 if self.has_nan else 0)))
        bar_ax.set_xticklabels(['{:.2f}'.format(b) for b in (all_bins[:-1] if self.has_nan else all_bins)], rotation=90,
                               fontsize=8)

        bar_ax.set_xlim(0, all_bins.shape[0] - 1)

        bar_ax.set_ylabel("count", fontsize=8)
        plot_ax.set_ylabel("event rate, %", fontsize=8)

    def _on_del(self, c):
        # print('_on_del')

        # Stupid workaround of fast clicking on Del
        # TODO: review
        if c.elem not in self.elems:
            return

        self.w.children = tuple(
            c1 for c1 in self.w.children if c1 not in c.dep)
        for c1 in c.dep:
            c1.close()
        # print([x1.val for x1 in self.elems], c.elem.val)
        indx = self.elems.index(c.elem)
        del self.elems[indx]
        # print(self.manual_woe.index)
        shift_position = indx + 1
        self.manual_woe[shift_position:] = self.manual_woe[shift_position:].shift(-1)
        self.update()

    def _on_value_change(self, change):
        # print(change['new'])
        change.owner.elem.val = _convert_to_proper_bin_dtype(self.x.dtype, change['new'])
        self.update()

    def _on_nan_woe_change(self, change):
        # print change['new']
        self.nan_woe = change['new']
        self.update()

    def _draw_border_control(self, elem):
        """Create control for split point"""
        del_btn = widgets.Button(description='-', layout=widgets.Layout(
            width='30px', height='30px'), button_style='danger', tooltip='Remove split point')
        del_btn.layout.align_self = 'center'
        del_btn.elem = elem

        float_text = widgets.BoundedFloatText(layout=widgets.Layout(width='100px'), value=float(str(elem.val)),
                                              min=self.x.min(), max=self.x.max(), step=None)
        float_text.elem = elem
        float_text.observe(self._on_value_change, names='value')

        v_box = widgets.VBox([float_text, del_btn],
                             layout=widgets.Layout(justify_content='center'))
        del_btn.on_click(self._on_del)

        add_btn = widgets.Button(description='+', layout=widgets.Layout(
            width='30px'), button_style='success', tooltip='Add split point')
        add_btn.on_click(self._on_add)

        add_btn.elem = elem

        del_btn.dep = [v_box, add_btn]

        return v_box, add_btn

    def _on_add(self, c):
        mi = self.x.min()
        ma = self.x.max()

        if c.elem is None:
            idx = -1
        else:
            idx = self.elems.index(c.elem)
            mi = self.elems[idx].val
        if idx < len(self.elems) - 1:
            ma = self.elems[idx + 1].val
        elem = Wrapper(mi + (ma - mi) / 2.)

        shift_position = idx + 2 if idx >= 0 else 0
        # print(shift_position)
        self.manual_woe = self.manual_woe.append({'Man WOE': np.nan}, ignore_index=True)
        self.manual_woe[shift_position:] = self.manual_woe[shift_position:].shift(1)
        self.elems.insert(idx + 1, elem)
        # display([i.val for i in self.elems])

        v_box, add_btn = self._draw_border_control(elem)

        self.w.children = self.w.children[: list(self.w.children).index(c) + 1] + (v_box, add_btn) + \
                          self.w.children[list(self.w.children).index(c) + 1:]

        self.update()

    def _on_tree_grouping(self, c):
        notnan_mask = self.x.notnull()

        if self.grouping.w_column is not None:
            w = self.grouping.train_t[notnan_mask][self.grouping.w_column]
        else:
            w = None
        bins, _, _ = auto_group_continuous(self.x[notnan_mask],
                                           self.grouping.train_t[notnan_mask][self.grouping.target_column],
                                           self.grouping.group_count_text.value, self.grouping.group_size_text.value,
                                           self.grouping.woe_smooth_coef, w=w)

        # update UI
        for c in self.w.children:
            c.close()

        children = []

        self.elems = [Wrapper(x1) for x1 in bins[1:-1]]

        add_btn = widgets.Button(description='+', layout=widgets.Layout(
            width='30px'), button_style='success', tooltip='Add split point')
        add_btn.elem = None  # elem - это ссылка элемент списка границ групп
        add_btn.on_click(self._on_add)

        children.append(add_btn)

        for elem in self.elems:
            v_box, add_btn = self._draw_border_control(elem)
            children.append(v_box)
            children.append(add_btn)

        self.w.children = children
        self.update()

    def _on_nan_woe_auto(self, c):
        nan_mask = self.x.isnull()
        nan_woe = woe(self.y[nan_mask], self.y, self.grouping.woe_smooth_coef, w=self.weight[nan_mask],
                      w_full=self.weight)

        self.nan_woe_text.value = nan_woe
        self.gr_widget.edit_cell(5,'Man WOE', np.NaN)
        self.manual_woe.iloc[-1] = np.NaN
        self.update()

    def _on_nan_woe_zero(self, c):
        self.nan_woe_text.value = 0
        self.gr_widget.edit_cell(5,'Man WOE', np.NaN)
        self.manual_woe.iloc[-1] = np.NaN
        self.update()

    def _on_manual_woe_change(self, event, current_widget):
        if event['source'] == 'api':
            return 

        self.manual_woe['Man WOE'][event['index']] = event['new']
        current_widget.df.loc[event['index'], 'Man WOE'] = event['new']
        # check if user edited NaN WOE value
        # in that case update NaN WOE value in the special box
        if current_widget.df.loc[event['index'],'group'] == 'nan':
            self.nan_woe_text.value = event['new']
            self.update()

        # display(self.grouping)
        # display(self.bins_data_)
        self.update()

    def create_tab_item(self):
        # Начальная отрисовка полей для границ групп
        children = []
        """
          +    1    +    2    +
               -         -
          1    1    2    2   None

        """
        add_btn = widgets.Button(description='+', layout=widgets.Layout(
            width='30px'), button_style='success', tooltip='Add split point')
        add_btn.elem = None  # elem - это ссылка элемент списка границ групп
        add_btn.on_click(self._on_add)

        children.append(add_btn)

        for elem in self.elems:
            v_box, add_btn = self._draw_border_control(elem)
            children.append(v_box)
            children.append(add_btn)

        # self.output = widgets.Output()
        # self.has_nan_cb = widgets.Checkbox(description='NaN group', indent=False,
        #                                   layout = widgets.Layout(width='100px'))
        # children.append(self.has_nan_cb)
        self.w = widgets.HBox(children)

        self.nan_woe_text = widgets.FloatText(value=self.nan_woe, layout=widgets.Layout(width='160px'),
                                              description='NaN WOE')

        self.nan_woe_text.observe(self._on_nan_woe_change, names='value')

        nan_woe_auto_btn = widgets.Button(description='Auto', layout=widgets.Layout(
            width='80px'), tooltip='Automatic WOE for NaN group')
        nan_woe_auto_btn.on_click(self._on_nan_woe_auto)

        nan_woe_zero_btn = widgets.Button(description='0', layout=widgets.Layout(
            width='80px'), tooltip='Zero WOE for NaN group')
        nan_woe_zero_btn.on_click(self._on_nan_woe_zero)

        # , border='1px solid red'))
        self.valid1 = widgets.Valid(
            value=True, layout=widgets.Layout(width='500px'))
        self.valid1.style.description_width = '400px'

        # table with final groups
        self.gr_widget = qgrid.show_grid(pd.DataFrame(),
                                         grid_options={'filterable': False, 'explicitInitialization': False},
                                         column_options={'editable': False},
                                         column_definitions={'Man WOE': {'editable': True}})

        self.gr_widget.on('cell_edited', self._on_manual_woe_change)

        # show gini
        self.gini_widget = widgets.Text(value=str(round(self.gini, 4)), description='Gini of variable:',
                                        style={'description_width': 'initial'}, disabled=True)

        # tree grouping
        tree_group_btn = widgets.Button(description='Group automatically', layout=widgets.Layout(width='160px')
                                        # , button_style = 'primary'
                                        , tooltip='Do tree based split')
        tree_group_btn.on_click(self._on_tree_grouping)

        return widgets.VBox([  # self.grouping.output,
            self.gini_widget,
            self.w,
            widgets.HBox(
                [self.nan_woe_text, nan_woe_auto_btn, nan_woe_zero_btn]),
            self.valid1,
            self.gr_widget,
            tree_group_btn])

    def validate(self):
        elems = np.array([x1.val for x1 in self.elems])
        valid = False
        if len(elems) > 0 and (elems.min() <= self.x.min() or elems.max() >= self.x.max()):
            msg = 'Error: group borders should be inside (min, max)'
        elif not np.array_equal(np.sort(np.unique(elems)), elems):
            msg = 'Error: Nonmonotonic group borders'
        else:
            msg = None
            valid = True

        self.valid1.value = valid
        if not valid:
            self.valid1.readout = msg

        return valid

    def _update_data(self):
        """ recalculate WOEs"""
        df = pd.DataFrame({'x': self.x, 'y': self.y, 'w': self.weight})
        bins = np.concatenate([np.array([-np.inf]), np.array([x1.val for x1 in self.elems]),
                               np.array([np.inf])])
        notnan_mask = self.x.notnull()
        bin_indices = pd.cut(df[notnan_mask]['x'],
                             bins=bins, right=False, labels=False)

        woes = pd.Series([0.] * (len(bins) - 1), index=range(0, len(bins) - 1))
        woes.update(df.groupby(bin_indices).apply(lambda rows: woe(
            rows['y'], df['y'], self.grouping.woe_smooth_coef, w=rows['w'], w_full=df['w'])))

        self.woes = woes.values

    def _update_form(self, valid):

        bins = np.concatenate([np.array([-np.inf]), np.array([x1.val for x1 in self.elems]),
                               np.array([np.inf])])
        df = pd.DataFrame({'group': self.x, 'y': self.y, 'w': self.weight})
        df['wy'] = df['y'] * df['w']
        cut_df = pd.concat([pd.cut(df[self.x.notnull()]['group'], bins=bins, right=False), df[['y', 'w', 'wy']]],
                           axis=1)
        groups_df = pd.concat([cut_df.groupby('group')[['w', 'wy']].sum().reset_index(), pd.Series(self.woes)], axis=1)
        nulls_row = pd.DataFrame([{'group': 'nan', 'w': df['w'][df['group'].isnull()].sum(),
                                   'wy': df['wy'][df['group'].isnull()].sum(), 0: self.nan_woe}])
        groups_df = pd.concat([groups_df, nulls_row], axis=0).set_index('group')
        groups_df['def_rate'] = groups_df['wy'] / groups_df['w']
        groups_df['share'] = groups_df['w'] / groups_df['w'].sum()
        groups_df.columns = ['cnt', 'wy', 'WOE', 'def_rate', 'share']
        groups_df = groups_df[['cnt', 'share', 'def_rate', 'WOE']].reset_index()
        for r in groups_df.index:
            groups_df.loc[r, 'share'] = percent_format(groups_df.loc[r, 'share'])
            groups_df.loc[r, 'def_rate'] = percent_format(groups_df.loc[r, 'def_rate'])

        groups_df['Man WOE'] = self.manual_woe['Man WOE']

        self.gini = gini_grp(groups_df['WOE'],
                             pd.to_numeric(groups_df['share'].str.strip('%').str.strip('nan').str.strip()) / 100,
                             pd.to_numeric(groups_df['def_rate'].str.strip('%').str.strip('nan').str.strip()) / 100)
        self.gini_widget.value = str(round(self.gini, 4))
        self.gr_widget.df = groups_df

        self.ew_plot(self.grouping.ew_bins_ax, self.grouping.ew_er_ax, valid)
        self.ed_plot(self.grouping.ed_bins_ax, self.grouping.ed_er_ax, valid)
        self.groups_plot(self.grouping.groups_bins_ax,
                         self.grouping.groups_er_ax, valid)

        # redraw all plots
        # neew for nice layout of split labels in top
        self.grouping.fig.tight_layout(rect=[0, 0, 1, 0.85])
        # plt.show()

        # with self.output:
        #    clear_output(wait=False)
        #    display(self.fig)

        # self.grouping.fig.canvas.draw()

    def apply(self):
        self._update_data()

        self.grouping.bins_data_[self.column]['bins'] = np.concatenate([np.array(
            [-np.inf]), np.array([x1.val for x1 in self.elems]), np.array([np.inf])])
        self.grouping.bins_data_[self.column]['woes'] = np.array(
            [woe if np.isnan(man_woe) else man_woe for woe, man_woe in zip(self.woes, self.manual_woe['Man WOE'])])
        # self.grouping.bins_data_[self.column]['woes'] = np.array(self.woes)
        self.grouping.bins_data_[self.column]['manual_woe'] = np.array(self.manual_woe['Man WOE'])
        self.grouping.bins_data_[self.column]['nan_woe'] = self.nan_woe


percent_format = lambda x: '{:,.2f}%'.format(100 * x)


class CategoricalContext(Context):
    def __init__(self, column, bins, unknown_woe, grouping, gini=np.nan, manual_woe=None):
        super(CategoricalContext, self).__init__(column, grouping)

        self.flag = False

        values_df = pd.DataFrame()
        values_df.index.names = ['value']

        # values from both: grouping + data
        # use here pd.unique(..) since np.unique(...) does something wrong when applied to pd.Series with str and nan
        values = set(pd.unique(self.x)) | set(bins.keys())
        self.unknown_woe = unknown_woe
        self.gini = gini
        if manual_woe == None:
            self.manual_woe = pd.DataFrame({'Man WOE': [np.NaN] * len(bins)})
        else:
            self.manual_woe = pd.DataFrame({'Man WOE': manual_woe})

        self.manual_woe.index = self.manual_woe.index.astype(str)
        self.manual_woe.index.names = ['group']

        for val in values:
            x_cat, y_cat, w_cat, wy_cat = ((self.x[self.x == val], self.y[self.x == val], self.weight[self.x == val],
                                            self.weight[self.x == val] * self.y[self.x == val])
                                           if type(val) != float or not np.isnan(val)
                                           else (self.x[self.x.isnull()], self.y[self.x.isnull()],
                                                 self.weight[self.x.isnull()],
                                                 self.weight[self.x.isnull()] * self.y[self.x.isnull()]))
            values_df.loc[val, 'cnt'] = w_cat.sum()
            if self.weight.sum() > 0:
                values_df.loc[val, 'share'] = percent_format(w_cat.sum() / self.weight.sum())
            else:
                values_df.loc[val, 'share'] = percent_format(0)
            if w_cat.sum() > 0:
                values_df.loc[val, 'def_rate'] = percent_format(wy_cat.sum() / w_cat.sum())
            else:
                values_df.loc[val, 'def_rate'] = np.nan
            values_df.loc[val, 'WOE'] = woe(y_cat, self.y, self.grouping.woe_smooth_coef, w=w_cat, w_full=self.weight)
            values_df.loc[val, 'group'] = ''
            if val in bins:  # group exists
                if val != val and False:
                    values_df.loc['None', 'group'] = str(bins[val])  # store str (why?)
                else:
                    values_df.loc[val, 'group'] = str(bins[val])  # store str (why?)
        self.values_df = values_df

    def _create_plots(self):
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

        self.grouping.cats_bins_ax = self.grouping.fig.add_subplot(gs[0])
        self.grouping.cats_er_ax = self.grouping.cats_bins_ax.twinx()
        self.grouping.cats_bins_ax.set_facecolor('white')
        self.grouping.cats_er_ax.set_facecolor('white')
        self.grouping.cats_bins_ax.tick_params(axis='both', which='both', labelsize=8)
        self.grouping.cats_bins_ax.tick_params(axis='x', width=0, pad=-2)
        self.grouping.cats_er_ax.tick_params(axis='both', which='both', labelsize=8)

        self.grouping.groups_bins_ax = self.grouping.fig.add_subplot(gs[1])
        self.grouping.groups_er_ax = self.grouping.groups_bins_ax.twinx()
        self.grouping.groups_bins_ax.set_facecolor('white')
        self.grouping.groups_er_ax.set_facecolor('white')
        self.grouping.groups_bins_ax.tick_params(axis='both', which='both', labelsize=8)
        self.grouping.groups_bins_ax.tick_params(axis='x', width=0)
        self.grouping.groups_er_ax.tick_params(axis='both', which='both', labelsize=8)

    def cats_plot(self, bar_ax, plot_ax, valid):
        # Original categories plot
        # clear axes
        bar_ax.clear()
        plot_ax.clear()

        if not valid:
            bar_ax.text(0.5 * (bar_ax.get_xlim()[0] + bar_ax.get_xlim()[1]),
                        0.5 * (bar_ax.get_ylim()[0] +
                               bar_ax.get_ylim()[1]), 'Error',
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=20, color='red')
            return

        chartdata = self.values_df.copy()
        chartdata['cats'] = chartdata.index.to_series()
        chartdata.sort_values(['group', 'cats'], ascending=['True', 'True'], inplace=True)

        cats = chartdata['cats']
        bins = pd.to_numeric(chartdata['group'])
        counts = chartdata['cnt']

        bin_colors = []
        cats_pos = []
        cats_pos_line = []
        splitlines = []
        c = 0
        oldbin = 0
        # print(bins)
        for i in range(0, len(cats)):
            if bins.iloc[i] > oldbin:  # sg
                c += 1
                oldbin = bins.iloc[i]  # sg
                splitlines.append(i - 0.5)
            bin_colors.append(
                self.grouping._bar_colors[c % len(self.grouping._bar_colors)])
            cats_pos.append(i - 0.5)
            cats_pos_line.append(i)
        splitlines.append(len(cats) - 0.5)

        height = counts
        width = 1

        patches = bar_ax.bar(cats_pos, height, width, color=bin_colors, hatch='', edgecolor='black',
                             linewidth=0.5, align='edge')

        # plot event rates
        brs = chartdata['def_rate'].str.strip('%').str.strip('nan').str.strip()
        brs = pd.to_numeric(brs)

        plot_ax.plot(cats_pos_line, brs, marker='o',
                     color='orangered', linestyle='dotted', ms=3, linewidth=1.5)

        # split_points - plot vertical lines ( | ), and names of the groups
        for i in range(0, len(splitlines)):
            bar_ax.axvline(splitlines[i], color='black', linewidth=0.5)
            if i > 0:
                prev_split = splitlines[i - 1]
            else:
                prev_split = -0.5
            text_pos = (splitlines[i] + prev_split) / 2
            bar_ax.text(text_pos, bar_ax.get_ylim()[1] * 1.05, i, horizontalalignment='center',
                        verticalalignment='bottom', color='black', rotation=0, fontdict={'size': 8})

        # xticks по границам бинов
        bar_ax.set_xticks(range(0, len(cats)))
        bar_ax.set_xticklabels(cats, rotation=315, fontsize=8)

        bar_ax.set_xlim(np.min(cats_pos), np.max(cats_pos) + 1)

        bar_ax.set_ylabel("count", fontsize=8)
        plot_ax.set_ylabel("event rate, %", fontsize=8)

    def groups_plot(self, bar_ax, plot_ax, valid):
        # Final groups plot
        # clear axes
        bar_ax.clear()
        plot_ax.clear()

        if not valid:
            bar_ax.text(0.5 * (bar_ax.get_xlim()[0] + bar_ax.get_xlim()[1]),
                        0.5 * (bar_ax.get_ylim()[0] +
                               bar_ax.get_ylim()[1]), 'Error',
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=20, color='red')
            return

        chartdata = self.gr_widget.df.copy()

        bins = pd.to_numeric(chartdata.index.to_series())
        counts = chartdata['cnt']

        bin_colors = []
        c = 0
        for bin in bins:
            bin_colors.append(
                self.grouping._bar_colors[c % len(self.grouping._bar_colors)])
            c += 1

        height = counts
        width = 1
        bins_pos = bins - 0.5

        patches = bar_ax.bar(bins_pos, height, width, color=bin_colors, hatch='', edgecolor='black',
                             linewidth=0.5, align='edge')

        # plot event rates
        brs = chartdata['def_rate'].str.strip('%').str.strip('nan').str.strip()
        brs = pd.to_numeric(brs)

        plot_ax.plot(bins, brs, marker='o',
                     color='orangered', linestyle='dotted', ms=3, linewidth=1.5)

        # split_points - plot vertical lines ( | )
        for b in bins:
            bar_ax.axvline(b - 0.5, color='black', linewidth=0.5)

        # xticks по границам бинов
        bar_ax.set_xticks(bins)
        bar_ax.set_xticklabels(bins, rotation=0, fontsize=8)

        bar_ax.set_xlim(np.min(bins_pos), np.max(bins_pos) + 1)

        bar_ax.set_ylabel("count", fontsize=8)
        plot_ax.set_ylabel("event rate, %", fontsize=8)

    def _on_values_df_change(self, change):
        if self.flag:
            return
        # print('_on_values_df_change')
        # print(change['new'][[]])
        self.values_df = change['new'][[]].copy().join(self.values_df)
        if self.values_df[self.values_df.columns].sort_index().equals(
                change['new'][self.values_df.columns].sort_index()):
            return

        self.values_df['group'] = change['new']['group']

        # print(change['new'], self.values_df)
        self.update()

    def _on_values_df_change_new(self, event, current_widget):
        # display(self.values_df)
        # display(current_widget.get_changed_df())
        self.values_df = current_widget.get_changed_df()

        self.update()

    def _on_unknown_woe_change(self, change):
        # print(change['new'])
        self.unknown_woe = change['new']

    def _on_manual_woe_change(self, event, current_widget):
        self.manual_woe['Man WOE'][event['index']] = event['new']
        # self.values_df = current_widget.get_changed_df()

    """
    def _group_list(self): 
        groups = list(map(int, np.unique(self.values_df['group'])))
        groups.sort()
        return groups
       
    # returns groups dropdown widget
    def _group_dropdown_options(self):
        groups = self._group_list()
        group_options = ['New group'] + list(map(str, groups))
        return group_options
    """

    def create_tab_item(self):
        self.val_widget = qgrid.show_grid(self.values_df, grid_options={'filterable': False, 'sortable': True,
                                                                        'explicitInitialization': False},
                                          column_options={'editable': False},
                                          column_definitions={'group': {'editable': True}})
        self.gr_widget = qgrid.show_grid(pd.DataFrame(), grid_options={'filterable': False, 'sortable': False,
                                                                       'explicitInitialization': False},
                                         column_options={'editable': False},
                                         column_definitions={'Man WOE': {'editable': True},
                                                             'Man WOE Value': {'editable': True}})

        if qgrid.__version__ == '1.0.2':
            self.val_widget.observe(self._on_values_df_change, names=['_df'])
        else:
            self.val_widget.on('cell_edited', self._on_values_df_change_new)
            self.gr_widget.on('cell_edited', self._on_manual_woe_change)
        # tree grouping
        tree_group_btn = widgets.Button(description='Group automatically', layout=widgets.Layout(width='160px')
                                        # , button_style = 'primary'
                                        , tooltip='Do tree based split')
        tree_group_btn.on_click(self._on_tree_grouping)

        self.unknown_woe_text = widgets.FloatText(value=self.unknown_woe, layout=widgets.Layout(width='320px'),
                                                  description='WOE for unknown values:',
                                                  style={'description_width': 'initial'})

        self.unknown_woe_text.observe(self._on_unknown_woe_change, names='value')

        # show gini
        self.gini_widget = widgets.Text(value=str(round(self.gini, 4)), description='Gini of variable:',
                                        style={'description_width': 'initial'}, disabled=True)

        control_box = widgets.VBox(
            [self.gini_widget, self.val_widget, self.gr_widget, self.unknown_woe_text, tree_group_btn])
        return control_box

    def _on_tree_grouping(self, c):

        if self.grouping.w_column is not None:
            w = self.grouping.train_t[self.grouping.w_column]
        else:
            w = None
        bins, _, unknown_woe = auto_group_categorical(self.x, self.grouping.train_t[self.grouping.target_column],
                                                      self.grouping.group_count_text.value,
                                                      self.grouping.group_size_text.value,
                                                      self.grouping.min_samples_cat_text.value,
                                                      self.grouping.woe_smooth_coef, w=w)
        # apply in values_df und UI
        self.values_df['group'] = ''
        for value, group in bins.items():
            self.values_df.loc[value, 'group'] = str(group)
        self.unknown_woe = unknown_woe
        self.update()

    def validate(self):
        return True

    def _update_data(self):
        # print('_update_data')
        groups_mask = pd.to_numeric(self.values_df['group'],
                                    errors='coerce').notnull()  # mask of groups specified as numbers
        # print(groups_mask)
        repl = pd.Series(np.sort(self.values_df[groups_mask]['group'].unique().astype(float))).astype(str).to_dict()
        # repl - dict: number -> group (str)
        repl = {v: k for k, v in repl.items()}
        # repl - dict: group (str) -> number(int)
        self.values_df.loc[groups_mask, 'group'] = self.values_df[groups_mask]['group'].astype(float).astype(
            str).replace(repl).astype(str)
        self.values_df.loc[~groups_mask, 'group'] = ''

        # move manual WOE values accordingly
        downshift = [i for key, i in repl.items() if int(float(key)) > i]
        if len(downshift) > 0:
            shift_position = min(downshift)
            self.manual_woe[shift_position:] = self.manual_woe[shift_position:].shift(-1)

    def _update_form(self, valid):
        # crazy code to disable duplicated call of _updat_data.. Also should check if we need hold_traid_notification ..
        self.flag = True
        with self.val_widget.hold_trait_notifications():
            self.val_widget.df = self.values_df
        self.flag = False

        groups_df = pd.DataFrame()
        # grouped = self.val_widget._df.groupby('group')
        groups = pd.Series(self.values_df['group'].unique())
        # print(self.values_df['group'].unique())
        groups = pd.to_numeric(groups,
                               errors='coerce')  # exclude NaN (errored values should be exluded during validation)
        groups = groups[groups.notnull()].astype(int).sort_values().astype(str)  # convert to int and sort
        # print(groups)
        for group in groups:
            values = self.values_df[self.values_df['group'] == group].index
            has_nan = values.isnull().any()
            notnan_values = values[values.notnull()]
            mask = self.x.isin(notnan_values)
            if has_nan:
                mask |= self.x.isnull()

                # repl = group_values['WOE'].to_dict()
            # print(repl)
            # repl = {k:str(v) for k, v in repl.items()}

            # group_x = self.x[mask].replace(repl).astype(float)
            # group_x = self.x[mask].replace(repl).astype(float)
            group_y = self.y[mask]
            group_w = self.weight[mask]
            group_wy = self.y[mask] * self.weight[mask]

            groups_df.loc[group, 'cnt'] = group_w.sum()
            if self.weight.sum() > 0:
                groups_df.loc[group, 'share'] = percent_format(group_w.sum() / self.weight.sum())
            else:
                groups_df.loc[group, 'share'] = percent_format(0)
            if group_w.sum() > 0:
                groups_df.loc[group, 'def_rate'] = percent_format(group_wy.sum() / group_w.sum())
            else:
                groups_df.loc[group, 'def_rate'] = np.nan
            groups_df.loc[group, 'WOE'] = woe(group_y, self.y, self.grouping.woe_smooth_coef, w=group_w,
                                              w_full=self.weight)
            # groups_df.loc[group, 'gini'] = roc_auc_score(group_y, group_x) if group_y.unique().shape[0]>1 else np.nan
        groups_df.index.names = ['group']
        # print(groups_df)

        groups_df['Man WOE'] = self.manual_woe['Man WOE']
        # print(groups_df.index)
        # print(self.manual_woe.index)
        # print(pd.concat([groups_df,self.manual_woe], axis=1))

        self.gr_widget.df = groups_df
        self.gini = gini_grp(groups_df['WOE'],
                             pd.to_numeric(groups_df['share'].str.strip('%').str.strip('nan').str.strip()) / 100,
                             pd.to_numeric(groups_df['def_rate'].str.strip('%').str.strip('nan').str.strip()) / 100)
        self.gini_widget.value = str(round(self.gini, 4))

        self.groups_plot(self.grouping.groups_bins_ax,
                         self.grouping.groups_er_ax, valid)
        try:
            self.cats_plot(self.grouping.cats_bins_ax,
                           self.grouping.cats_er_ax, valid)
        except:
            pass

    def apply(self):
        # self._update_data()
        self._update_form(True)  # In case of categorical _update_form() is needed instead of _update_data().
        # Frankly speaking we have smooth responsibility between _update_form() and _update_data() here.

        bins = self.values_df['group'].to_dict()
        bins = {k: int(v) for k, v in bins.items() if v != ''} #if there are no nan values nan group has no number,
                                                               #in this case int(v) raises
                                                               #ValueError: invalid literal for int() with base 10: ''
        
        #print(self.gr_widget._df.sort_index().columns)
        woes = self.gr_widget._df.sort_index()['WOE'].values

        self.grouping.bins_data_[self.column]['bins'] = bins
        # self.grouping.bins_data_[self.column]['woes'] = woes
        self.grouping.bins_data_[self.column]['woes'] = np.array(
            [woe if np.isnan(man_woe) else man_woe for woe, man_woe in zip(woes, self.manual_woe['Man WOE'])])
        self.grouping.bins_data_[self.column]['unknown_woe'] = self.unknown_woe


class InteractiveGrouping(Grouping):
    """Interactive grouping editor"""

    """
    @property
    def y(self):
        return self.train_t[self.target_column]
    """

    def __init__(self, columns, cat_columns, group_count=3, min_samples=1, min_samples_cat=1, woe_smooth_coef=0.001):
        super().__init__(columns, cat_columns, group_count, min_samples, min_samples_cat, woe_smooth_coef)

    def validate_all(self):

        invalid_columns = []
        # for context in self.contexts:
        #    if context.valid1.value == False:
        #        invalid_columns.append(context.column)

        for context in self.contexts:
            if not context.validate():
                invalid_columns.append(context.column)

        all_valid = len(invalid_columns) == 0
        self.valid_all.value = all_valid
        self.valid_all.readout = 'Check columns: {}'.format(
            ', '.join(invalid_columns))
        self.apply_save_btn.disabled = not all_valid

    def _on_apply_save(self, c):
        # print('save')
        for context in tqdm(self.contexts, desc='Saving...', leave=False):
            # print(context.column)
            context.apply()
            # apply
            # context._update_data()

            # self.bins_data_[context.column]['bins'] = np.concatenate([np.array(
            #    [-np.inf]), np.array([x1.val for x1 in context.elems]), np.array([np.inf])])
            # self.bins_data_[context.column]['woes'] = context.woes
            # self.bins_data_[context.column]['nan_woe'] = context.nan_woe
        # save
        # display(self.bins_data_)
        # display(self.manual_woe)
        for con in self.contexts:
            # print(con.gr_widget.df)
            self.bins_data_[con.column]['manual_woe'] = con.manual_woe['Man WOE'].values[
                                                        :len(self.bins_data_[con.column]['woes']) + 1]
            # self.bins_data_[con.column]['manual_woe'] = np.array(con.gr_widget.df['Man WOE'])

        # display(self.bins_data_)
        self.save(self.filename_text.value)

    def _initialize_data(self):
        # if nothing "fitted"
        if not hasattr(self, 'bins_data_'):
            self.bins_data_ = {}

        # simple defaults for new columns

        if self.w_column is not None:
            w = self.train_t[self.w_column]
        else:
            w = None

        for column in self.columns + self.cat_columns:
            if column not in self.bins_data_:
                self._auto_grouping(self.train_t[column], self.train_t[self.target_column], w=w)

        # creating contexts
        self.contexts = []
        for column in self.columns:
            bin_data = self.bins_data_[column]
            if 'manual_woe' in bin_data.keys():
                context = ContinuousContext(
                    column, bins=bin_data['bins'], nan_woe=bin_data['nan_woe'], grouping=self,
                    manual_woe=bin_data['manual_woe'])
            else:
                context = ContinuousContext(
                    column, bins=bin_data['bins'], nan_woe=bin_data['nan_woe'], grouping=self, manual_woe=None)
            self.contexts.append(context)

        for column in self.cat_columns:
            bin_data = self.bins_data_[column]
            if 'manual_woe' in bin_data.keys():
                context = CategoricalContext(
                    column, bins=bin_data['bins'], unknown_woe=bin_data['unknown_woe'], grouping=self,
                    manual_woe=bin_data['manual_woe'])
            else:
                context = CategoricalContext(
                    column, bins=bin_data['bins'], unknown_woe=bin_data['unknown_woe'], grouping=self, manual_woe=None)
            self.contexts.append(context)

    def _hack_headers(self):
        display(HTML("""
        <style>
        .output_wrapper button.btn.btn-default,
        .output_wrapper .ui-dialog-titlebar {
          display: none;
        }
        </style>"""))

    """
    def _on_group_count_change(self, change):
        self.group_count = change['new']

    def _on_group_size_change(self, change):
        self.min_samples = change['new']

    def _on_min_samples_cat_change(self, change):
        self.min_samples_cat = change['new']
    """

    def _export_all(self, _):
        """
        Iterates over all contexts - updates tables, draws charts
        then exports tables to .csv and charts to .png
        Export path is taken from text field in widget.
        Tries to create directories if path does not exist.
        """
        import os

        export_path = self.export_text.value

        if not os.path.exists(export_path):
            os.makedirs(export_path)

        for c in tqdm(self.contexts, leave=False):
            c.update(tab_change=True)
            c.grouping.fig.savefig(f"{export_path}/{c.column}.png")
            if isinstance(c, CategoricalContext):
                c.values_df.to_csv(f"{export_path}/{c.column}_groups.csv")
                c.gr_widget.df.to_csv(f"{export_path}/{c.column}.csv")
            else:
                c.gr_widget.df.to_csv(f"{export_path}/{c.column}.csv", index=False)


    def _initialize_form(self):
        # plt.ioff()

        self._hack_headers()
        self.fig = plt.figure(figsize=(9.5, 2.5))

        # self.output = widgets.Output(layout = widgets.Layout(height = '200px'))

        tab_items = []
        dd_options = {}
        ctx_idx = 0
        for context in self.contexts:
            tab_items.append(context.create_tab_item())
            dd_options[context.column] = ctx_idx
            ctx_idx += 1
        self.tab = widgets.Tab(tab_items)
        self.dd = widgets.Dropdown(options=dd_options)
        for i, context in enumerate(self.contexts):
            self.tab.set_title(i, context.column)
        self.tab.observe(self._on_tab_change, names='selected_index')
        self.dd.observe(self._on_dd_change, names='value')
        self.tab.selected_index = 0

        self.group_count_text = widgets.IntText(min=1, max=10, value=self.group_count,
                                                layout=widgets.Layout(width='140px'),
                                                description='Max groups')
        self.group_size_text = widgets.IntText(min=1, max=1000000, value=self.min_samples,
                                               layout=widgets.Layout(width='170px'),
                                               description='Min group size')

        self.min_samples_cat_text = widgets.IntText(min=1, max=1000000, value=self.min_samples_cat,
                                                    layout=widgets.Layout(width='170px'),
                                                    description='Min category size')

        # self.group_count_text.observe(
        #    self._on_group_count_change, names='value')
        # self.group_size_text.observe(self._on_group_size_change, names='value')
        # self.min_samples_cat_text.observe(self._on_min_samples_cat_change, names='value')
        settings_acc = widgets.Accordion(
            [widgets.HBox([self.group_count_text, self.group_size_text, self.min_samples_cat_text])])
        settings_acc.set_title(0, 'Settings')
        settings_acc.selected_index = None

        # , border='1px solid red'))
        self.valid_all = widgets.Valid(
            value=True, layout=widgets.Layout(width='500px'))
        self.valid_all.style.description_width = '400px'

        self.apply_save_btn = widgets.Button(description='Apply and Save', layout=widgets.Layout(
            width='160px', height='30px'), tooltip='Apply and Save changes')

        self.apply_save_btn.on_click(self._on_apply_save)

        self.filename_text = widgets.Text(value=self.filename, layout=widgets.Layout(width='300px'),
                                          description='Model filename')
        self.filename_text.style.description_width = '150px'

        self.export_btn = widgets.Button(description='Export all', layout=widgets.Layout(
            width='160px', height='30px'), tooltip='Export charts and tables to files')
        
        self.export_btn.on_click(self._export_all)

        self.export_text = widgets.Text(value="documentation/igrouping", layout=widgets.Layout(width='300px'),
                                          description='Export path')

        widget = widgets.VBox([self.dd, self.tab, self.valid_all, widgets.HBox(
            [self.apply_save_btn, self.filename_text, self.export_btn, self.export_text]), settings_acc])
        self._hack_readout_width()
        display(widget)
        # print('before')
        self._on_tab_change({'new': 0})
        self._on_dd_change({'new': 0})

    def _hack_readout_width(self):
        # https://github.com/jupyter-widgets/ipywidgets/issues/1937
        display(
            HTML("""
            <style>
            .widget-valid-readout.widget-readout {
                max-width: 450px
            }
            </style>
            """)
        )

    def _on_dd_change(self, change):
        self.tab.selected_index = change['new']

    def _on_tab_change(self, change):
        # print('_on_tab_change')
        self.context = self.contexts[change['new']]
        # self.fig.canvas.set_window_title(self.context.column)
        self.context.update(tab_change=True)
        self.dd.value = change['new']

    def _check_enough_values(self, data):
        """
        Return all columns of `data` that have just one non-null value.
        """
        non_unique_columns = []
        for name, col in data.iteritems():
            if col.value_counts().shape[0] < 2:
                non_unique_columns.append(name)
        
        return non_unique_columns


    def display(self, train_t, columns, cat_columns, target_column='def_6_60', w_column=None,
                oot_target_columns=['def_6_60'], oot_time_column='month', filename=None,
                equi_width_plot=True, quantile_plot=True, oot_plot=True, final_plot=True,
                groups_table=True, bin_count=20, woe_smooth_coef=0.001, group_count=None, min_samples=None,
                min_samples_cat=None):
        """Displays interactive grouping

        Args:
            train_t (pd.DataFrame): [description]
            columns (list of str): [description]
            cat_columns (list of str): [description]
            target_column (str, optional): [description]. Defaults to 'def_6_60'.
            w_column (str, optional): [description]. Defaults to None.
            oot_target_columns (list, optional): [description]. Defaults to ['def_6_60'].
            oot_time_column (str, optional): [description]. Defaults to 'month'.
            filename (str, optional): [description]. Defaults to None.
            equi_width_plot (bool, optional): [description]. Defaults to True.
            quantile_plot (bool, optional): [description]. Defaults to True.
            oot_plot (bool, optional): [description]. Defaults to True.
            final_plot (bool, optional): [description]. Defaults to True.
            groups_table (bool, optional): [description]. Defaults to True.
            bin_count (int, optional): [description]. Defaults to 20.
            woe_smooth_coef (float, optional): [description]. Defaults to 0.001.
            group_count (int, optional): [description]. Defaults to None.
            min_samples (int, optional): [description]. Defaults to None.
            min_samples_cat (int, optional): [description]. Defaults to None.
        """

        self.train_t = train_t
        self.columns = columns
        self.cat_columns = cat_columns
        self.target_column = target_column
        self.w_column = w_column
        self.oot_target_columns = oot_target_columns
        self.oot_time_column = oot_time_column

        for column in self._check_enough_values(self.train_t[self.columns]):
            self.columns.remove(column)
            print(f"Predictor {column} not displayed because it has only one non-null value.")

        if filename is None:
            self.filename = 'myIntGrouping.json'
        else:
            self.filename = filename
            self.load(filename)
        self.equi_width_plot = equi_width_plot
        self.quantile_plot = quantile_plot
        self.oot_plot = oot_plot
        self.final_plot = final_plot
        self.groups_table = groups_table
        self.bin_count = bin_count
        self.woe_smooth_coef = woe_smooth_coef
        if group_count is not None:
            self.group_count = group_count
        if min_samples is not None:
            self.min_samples = min_samples
        if min_samples_cat is not None:
            self.min_samples_cat = min_samples_cat

        self._bar_colors = plt.cm.get_cmap('Pastel1').colors
        self._plot_colors = plt.cm.get_cmap('Set1').colors

        self._initialize_data()
        self._initialize_form()
