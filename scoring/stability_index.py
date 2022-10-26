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
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import itertools  # library needed for usage of combinations function in psi_calc_df


def stability_index_value(data, var, col_target, col_base, col_month,
                          col_weight=None, exclude_nan=False, exclude_zero=False):
    """
    Calculates Elena's Stability Index (let's call it ESI, shall we? :-)). 
    (Or we also call it more humbly Rank Stability Index... it is the same).
    The index is meant to show stability of variable which is already grouped (e.g. WOE variable)
    and shows how the various categories of the variable change their order in respect to default rate.
    This implies that the index is not working well for U shaped variables.
    There are two versions of ESI, defined as follows:

    v1:
        1. Compute the bad rates in each category for each month
        2. Compute the rank of each category within each month based on the bad rate
        3. Compute the frequency of each position / rank within each category from step 2
        4. Compute the ratios of the most frequent position / rank within each category from step 3
        5. Compute average of the ratios from step 4

    v2:
        1. Compute the bad rates in each category for each month
        2. Compute the rank of each category within each month based on the bad rate
        3. Compute the frequency of each position / rank through all categories from step 2 and the corresponding ratios
        4. Compute the product of the ratios within each position / rank from step 3
        5. Compute average of the products from step 4

    Args:
        data (pandas.DataFrame): the data frame with the columns specified in the following arguments
        var (str): name of variable to calculate ESI for
        col_target (str): name of the target column in data
        col_base (str): name of the base column (0/1 whether col_target was observable) in data
        col_month (str): name of column with number of month in data 
                           (or generally of a numerical column with time information)
        col_weight (str, optional): name of weight column in data 
                            (giving various weights to individual observations) (default: None)
        exclude_nan (boolean, optional): indicates whether rows with NaN value of data[var] 
                                 should be excluded from the index computation (default: False)
        exclude_zero (boolean, optional): indicates whether rows with zero value of data[var]
                                  should be excluded from the index computation (default: False)

    Returns:
        dict: dictionary in format {'v1': ESI v1, 'v2': ESI v2}, i.e. dictionary with the two versions of ESI
    """

    # 1. Compute the bad rates in each category for each month
    if col_weight is None:
        # simple sum for target and base in each category and month
        tmp = data[(data[col_base] == 1)].groupby([col_month, var], as_index=False)[[col_target, col_base]].sum()
    else:
        # weighted sum for target and base in each category and month
        tmp = data[(data[col_base] == 1)][[col_target, col_base, col_month, col_weight, var]].copy()
        tmp[col_target] = tmp[col_target]*tmp[col_weight]
        tmp[col_base] = tmp[col_base]*tmp[col_weight]
        tmp = tmp.groupby([col_month, var], as_index=False)[[col_target, col_base]].sum()

    # excluding categories specified by parameters
    if exclude_nan:
        tmp = tmp.loc[pd.isnull(tmp[var])]
    if exclude_zero:
        tmp = tmp.loc[~(tmp[var] == 0.000000)]

    tmp['bad_rate'] = tmp[col_target] / tmp[col_base]

    # 2. Compute the rank of each category within each month based on the bad rate
    tmp = tmp[[col_month, var, 'bad_rate']]
    tmp['group_rank'] = tmp.groupby(col_month)['bad_rate'].rank(ascending=0, method='dense')

    # 3.v1 Compute the frequency of each position / rank within each category from step 2
    tmp = tmp[[col_month, var, 'group_rank']]
    tmp = tmp.groupby([var, 'group_rank'])[[col_month]].count()
    tmp.columns = ['rank_count']

    # 3.v2 Compute the corresponding ratios
    tmp2 = tmp.groupby(['group_rank']).aggregate({'rank_count': ['sum']})
    tmp2 = pd.merge(tmp, tmp2['rank_count'][['sum']], how='left', on='group_rank')
    tmp2['ratio'] = tmp2['rank_count'] / tmp2['sum']

    # 4.v2 Compute the product of the ratios within each position / rank from step 3
    tmp2 = tmp2.groupby(['group_rank']).aggregate({'ratio': ['prod']})

    # 4.v1 Compute the ratios of the most frequent position / rank within each category from step 3
    tmp = tmp.groupby([var]).aggregate({'rank_count': ['sum', 'max']})
    tmp['ratio'] = tmp['rank_count']['max'] / tmp['rank_count']['sum']

    # 5. Compute average of values from step 4
    index_value = tmp['ratio'].sum() / tmp['ratio'].count()
    index_value2 = tmp2['ratio']['prod'].sum() / tmp2['ratio']['prod'].count()

    # return both results in a dictionary
    return {'v1': index_value, 'v2': index_value2}


def psi_calc(series_ref, series_act, weights_ref=None, weights_act=None):
    """
    The function calculates PSI for variables series_ref and series_act. 
    If weights are entered weighted PSI is provided.
    The function expects that values of series_ref and series_act are bin labels.

    Args:
        series_ref (pandas.Series): series used as referent vector
        series_act (pandas.Series): series used as actual vector
        weights_ref (pandas.Series): series used to weight referent vector.Defaults to None.
        weights_act (pandas.Series): series used to weight actual vector. Defaults to None.

    Returns:
        float: value of calculated PSI
    """

    assert type(series_ref) is pd.Series, "Input variable series_ref is not Panda Series."
    assert type(series_act) is pd.Series, "Input variable series_act is not Panda Series."
    if weights_ref is not None:
        assert type(weights_ref) is pd.Series, "Input variable weights_ref is not Panda Series."
    if weights_act is not None:
        assert type(weights_act) is pd.Series, "Input variable weights_act is not Panda Series."

    # if weights are not inputed use weights of value 1
    if weights_ref is None:
        weights_ref = pd.Series([1]*len(series_ref))
    if weights_act is None:
        weights_act = pd.Series([1]*len(series_act))

    # name inputed series and weights
    series_ref.name = "sref"
    series_act.name = "sact"
    weights_ref.name = "wref"
    weights_act.name = "wact"

    # join series and weights to one data frame
    sw_df_ref = pd.concat([series_ref.reset_index(), weights_ref.reset_index()], axis=1)
    sw_df_act = pd.concat([series_act.reset_index(), weights_act.reset_index()], axis=1)

    # calculate distributions of entered series
    perc_ref = sw_df_ref.groupby([series_ref.name]).sum()[weights_ref.name]/sw_df_ref[weights_ref.name].sum()
    perc_act = sw_df_act.groupby([series_act.name]).sum()[weights_act.name]/sw_df_act[weights_act.name].sum()

    # join both series to data frame
    perc_df = pd.concat([perc_ref, perc_act], axis=1)

    # if there are different values in series_ref and series_act print warning and use only common values
    if perc_df.isnull().sum().sum() > 0:
        print(f'There are some different categories in variables {series_ref.name} and {series_act.name}.'
              ' They will be ignored in PSI.')
    nan_mask = ~perc_df.isnull().any(axis=1)

    psi_val = sum((perc_df[nan_mask][weights_ref.name]-perc_df[nan_mask][weights_act.name]) *
                  np.log(perc_df[nan_mask][weights_ref.name]/perc_df[nan_mask][weights_act.name]))

    return psi_val


def psi_calc_df(df, cols_pred_psi, col_weight=None, col_month=None, mask_dict=None):
    """
    The function takes data frame df and list of predictors cols_pred_psi and based on keyword arguments calculates for
    each predictor average weighted PSI from all two consecutive months weighted PSIs
    (e.g. let's have months 1, 2, 3, weighted PSIs are calculated
    for combinations (1,2), (2,3) and average of these values is returned); or weighted PSI for each pair of masks.

    Args:
        df (pandas.DataFrame): data frame with the columns specified in the following arguments
        col_pred_psi (list): list of variables from df for which PSI should be calculated
        col_weight (str, optional): name of variable from df with observation weights, typically set to 1. Defaults to None.
        col_month (str, optional): name of column from df with months labels. Defaults to None.
        mask_dict (dict, optional): dictionary of masks of df for which PSI should be calculated. Defaults to None.

    Returns:
        pd.DataFrame, pd.DataFrame: two sorted tables with PSI values are returned
    """

    assert type(df) is pd.DataFrame, "Input variable df is not a Data Frame."
    assert type(cols_pred_psi) is list, "Input variable cols_pres_psi is not a list."
    if col_weight:
        assert type(col_weight) is str, "Input variable col_weight is not a string."
    if col_month:
        assert type(col_month) is str, "Input variable col_month is not a string."
    if mask_dict:
        assert type(mask_dict) is dict, "Input variable mask_list is not a dictionary."
    assert col_month or mask_dict, 'You have to input at least one of the variables col_month or mask_dict'

    # calculate PSI for every pair of consecutive months, average the values and store it in df_month_out


    df_month_out = pd.DataFrame(columns=['Variable', 'PSI avg per month'])
    if col_month:
        month1 = pd.Series(df[col_month].unique()).sort_values()
        month2 = month1[1:]
        for name in cols_pred_psi:
            psi_sum = 0
            psi_cnt = 0
            for m1, m2 in zip(month1, month2):
                df_reference = df.loc[df[col_month] == m1, :]
                df_actual = df.loc[df[col_month] == m2, :]
                psi_sum += psi_calc(
                    series_ref=df_reference[name], 
                    series_act=df_actual[name],
                    weights_ref=df_reference[col_weight] if col_weight else None,
                    weights_act=df_actual[col_weight] if col_weight else None)
                psi_cnt += 1
            df_month_out.loc[len(df_month_out)] = [name, psi_sum/psi_cnt]
            # print progress
            print(
                f'Month run: variable {name} is proceed. '
                f'Done variables: {cols_pred_psi.index(name)}/{len(cols_pred_psi)-1}')

    # calculate PSI for every pair of masks and store it in df_mask_out
    df_mask_out = pd.DataFrame(columns=['Variable', 'Mask', 'PSI'])
    if mask_dict:
        for name in cols_pred_psi:
            # iterate over each unique combination of masks
            for mn1, mn2 in itertools.combinations(mask_dict.keys(), 2):
                psi_sum = psi_calc(series_ref=df[mask_dict[mn1]][name],
                                   series_act=df[mask_dict[mn2]][name],
                                   weights_ref=df[mask_dict[mn1]][col_weight] if col_weight else None,
                                   weights_act=df[mask_dict[mn2]][col_weight] if col_weight else None)
                df_mask_out.loc[len(df_mask_out)] = [name, mn1+':'+mn2, psi_sum]
            # print progress
            print(
                f'Mask run: variable {name} is proceed. Done variables: {cols_pred_psi.index(name)}/{len(cols_pred_psi) - 1}')

    return df_month_out.sort_values(by=['PSI avg per month'], ascending=False), df_mask_out.sort_values(by=['PSI'], ascending=False)
