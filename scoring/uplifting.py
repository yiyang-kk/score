
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
TODO: (for DG)
    * _get_bins (quantile) work wrong in predictor has only two values
    * uplift_curve, uplift_auc transfer to metrics.py
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings

from scoring.grouping import Grouping, nwoe


def _get_bins(df, continuous=[], categorical=[], col_treatment='treatment', col_outcome='outcome', n_bins=4,
              grouping='quantile', min_samples=5000, min_samples_cat=5000):
    """Finds n_bins bins of equal size for each feature or split using the upliftree for each feature in dataframe.

    Args:
        df (pandas.DataFrame): dataframe with features
        continous (list of str, optional): list of continous (numerical) features you would like to consider for splitting into n_bins (the ones you want to evaluate NWOE, NIV etc for). Defaults to None.
        categorical (list of str, optional): list of categorical features you would like to consider for splitting into n_bins (the ones you want to evaluate NWOE, NIV etc for). Defaults to None.
        col_treatment (str, optional): Name of treatment column. Defaults to 'treatment'.
        col_outcome (str, optional): Name of outcome column. Defaults to 'outcome'.
        n_bins (int, optional): Number of even sized (no. of data points) n_bins to use for each feature (this is chosen based on both df and col datasets). Defaults to 4.
        grouping (str, optional): type of grouping, 'quantile' or 'tree'. Defaults to 'quantile'
        min_samples (int, optional): leaf for continuous features. Defaults to 5000.
        min_samples_cat (int, optional): min samples leaf for categorical features. Defaults to 5000.

    Returns:
        pandas.DataFrame, dict: DataFrame with bins for each feature for each index, dictionary with bins
    """

    if grouping == 'quantile':
        all_bins = {}
        index_list = []
        if continuous != []:
            for feat in continuous:
                index, bins = pd.qcut(df[feat], n_bins, labels=False, retbins=True, duplicates='drop')

                if bins.shape[0] < 3:
                    warnings.warn(f'Zero n_bins for column "{feat}" -> niv=0')
                index_list.append(index)

                # calc woes and nan_woe like in autogrouping
                notnan_mask = df[feat].notnull()
                w1 = np.ones(len(df[feat]))
                df_new = pd.DataFrame({'x': df[feat], 'y': df[col_outcome], 'w': w1, 'group': df[col_treatment]})
                df_new.loc[pd.isnull(df_new['y']), 'w'] = np.nan
                bin_indices = pd.cut(df_new[notnan_mask]['x'], bins=bins, right=False, labels=False)
                woes = np.zeros(bins.shape[0] - 1)
                new_woes = df_new.groupby(bin_indices).apply(
                    lambda rows: nwoe(rows['y'], df_new['y'], rows['group'], df_new['group'], 0.001, w=rows['w'],
                                      w_full=df_new['w'])).to_dict()
                np.put(woes, list(new_woes.keys()), list(new_woes.values()))
                nan_woe = nwoe(df_new[~notnan_mask]['y'], df_new['y'], df_new[~notnan_mask]['group'], df_new['group'],
                               0.001, w=df_new[~notnan_mask]['w'], w_full=df_new['w'])

                feat_bins = {}
                feat_bins['bins'] = bins
                feat_bins['woes'] = woes
                feat_bins['nan_woe'] = nan_woe
                all_bins[feat] = feat_bins

        if categorical != []:
            for feat in categorical:
                index = df[feat].copy().astype(str)
                index_copy = index.copy()
                stat = index.value_counts()

                cat_cutoff = int(df.shape[0] / n_bins)
                rare = stat[stat < cat_cutoff].index.values
                rare_mask = index.isin(rare)

                if rare_mask.sum() > 0 and rare_mask.sum() < cat_cutoff:
                    warnings.warn(f'Rare category "Other" for column "{feat}"')
                    index = index[~rare_mask]
                else:
                    index[rare_mask] = 'rare_value'
                # index_list.append(index)

                bins = {}
                unique = np.unique(index)
                for i, val in enumerate(unique):
                    if val == 'rare_value':
                        for elem in rare:
                            bins[elem] = len(unique) - 1
                        continue
                    bins[val] = i

                bin_indices = index_copy.replace(bins)
                index_list.append(bin_indices)
                w1 = np.ones(len(df[feat]))
                df_new = pd.DataFrame({'x': df[feat], 'y': df[col_outcome], 'w': w1, 'group': df[col_treatment]})
                df_new.loc[pd.isnull(df_new['y']), 'w'] = np.nan
                df_new['wy'] = df_new['w'] * df_new['y']
                woes = df_new.groupby(bin_indices).apply(
                    lambda rows: nwoe(rows['y'], df_new['y'], rows['group'], df_new['group'], 0.001, w=rows['w'],
                                      w_full=df_new['w'])).values
                unknown_woe = 0

                feat_bins = {}
                feat_bins['bins'] = bins
                feat_bins['woes'] = woes
                feat_bins['unknown_woe'] = unknown_woe
                all_bins[feat] = feat_bins

        indexes = pd.concat(index_list, axis=1)

    elif grouping == 'tree':
        grouping = Grouping(continuous, categorical, group_count=n_bins, min_samples=min_samples,
                            min_samples_cat=min_samples_cat)
        grouping.fit_uplift(df[continuous + categorical], df[col_outcome], df[col_treatment])
        indexes = grouping.transform(df[continuous + categorical], suffix='',
                                     transform_to='shortnames')[continuous + categorical]
        all_bins = grouping.bins_data_

    else:
        raise ValueError('Value of grouping is not correct. '
                         'Use grouping="quantile" or grouping="tree"')

    return indexes, all_bins


def niv(df, continuous=[], categorical=[], col_treatment='treatment', col_outcome='outcome', n_bins=4, bins=None,
        grouping='quantile', min_samples=5000, min_samples_cat=5000):
    """Net Information Value (NIV)

    Args:
        df (pandas.DataFrame): original dataframe with continuous and categorical feature columns
        continuous (list of str, optional): List of continuous features. Defaults to None.
        categorical (list of str, optional): List of categorical features. Defaults to None.
        col_treatment (str, optional): Name of column with treatment labels. Defaults to 'treatment'.
        col_outcome (str, optional): Name of column with group labels Defaults to 'outcome'.
        n_bins (int, optional): number of n_bins (if you don'df have n_bins). Defaults to 4.
        bins (dict, optional): dictionary with bins. Defaults to None.
        grouping (str, optional): type of grouping, 'quantile' or 'tree'. Defaults to 'quantile'
        min_samples (int, optional): leaf for continuous features. Defaults to 5000.
        min_samples_cat (int, optional): min samples leaf for categorical features. Defaults to 5000.

    Returns:
        dict:  Net Information Values
    """

    feats = (continuous + categorical)

    if feats == []:
        raise ValueError('Continuous and categorical parameters are empty')

    if bins is None:
        indexes, _ = _get_bins(df, continuous, categorical, col_treatment, col_outcome, n_bins=n_bins,
                               grouping=grouping, min_samples=min_samples, min_samples_cat=min_samples_cat)
    else:
        data_woe = pd.DataFrame(columns=(continuous + categorical))
        iterator = tqdm(df[(continuous + categorical)].iteritems(), total=len((continuous + categorical)), leave=True,
                        unit='cols')
        for name, column in iterator:
            bin_data = bins[name]
            target_values = [i for i in range(len(bin_data['bins']) - 1)]
            target_nan = len(bin_data['bins']) - 1
            tmp = pd.cut(column, bin_data['bins'], right=False, labels=False)
            map_dict = {np.nan: target_nan, **{i: target_values[i] for i in range(len(target_values))}}
            data_woe[name] = tmp.map(map_dict)

        renaming = {col: col for col in data_woe.columns}
        indexes = data_woe.rename(renaming, axis='columns')

    niv_dict = {}

    for feat in tqdm(feats):
        index = indexes[feat]

        df_new = pd.DataFrame({'idx': index.values, 'otc': df[col_outcome], 'trt': df[col_treatment]})

        stat = df_new.groupby(['idx', 'trt']).agg(['count', 'sum'])
        stat.columns = ['n', 'n_otc']

        n_t1 = stat[stat.index.get_level_values('trt') == 1]['n'].reset_index(level=1, drop=True) + 1
        n_o1_t1 = stat[stat.index.get_level_values('trt') == 1]['n_otc'].reset_index(level=1, drop=True) + 1
        n_o0_t1 = (n_t1 - n_o1_t1) + 1
        n_t0 = stat[stat.index.get_level_values('trt') == 0]['n'].reset_index(level=1, drop=True) + 1
        n_o1_t0 = stat[stat.index.get_level_values('trt') == 0]['n_otc'].reset_index(level=1, drop=True) + 1
        n_o0_t0 = (n_t0 - n_o1_t0) + 1

        p_t1 = n_o1_t1 / n_o1_t1.sum()
        p_t0 = n_o0_t1 / n_o0_t1.sum()
        p_c1 = n_o1_t0 / n_o1_t0.sum()
        p_c0 = n_o0_t0 / n_o0_t0.sum()

        niv_value = ((p_t1 * p_c0 - p_t0 * p_c1) * np.log((p_t1 * p_c0) / (p_t0 * p_c1))).sum()

        niv_dict[feat] = niv_value

    return niv_dict


def uplift_curve(df, col_scores, col_treatment='treatment', col_outcome='outcome', n_bins=100, type_curve='uplift',
                 cumulative=True):
    """Uplift curve

    Args:
        df (pandas.DataFrame): Dataframe with labels: [col_outcome, col_treatment, col_scores]
        col_scores (list of str): List names of columns with score, example: ['rf_scores', 'xgb_scores']
        col_treatment (str, optional): Name of column with treatment labels. Defaults to 'treatment'.
        col_outcome (str, optional): Name of column with group labels Defaults to 'outcome'.
        n_bins (int, optional): How many n_bins to use (from 10 to 100). Defaults to 100.
        type_curve (str, optional): type of curve
            'uplift' - absolute difference response rate
            'gain' - absolute difference (in the number of objects) between treatment and control group
            'gain_perc' - difference in percent of eating from the total response rate.
            Classic curve from original article "Uplift modeling" by Szymon Jaroszewicz,
            National Institute of Telecommunications Warsaw, Poland
            Defaults to 'uplift'.
        cumulative (bool, optional): only for type_curve:'uplift', calculate bins cumulatively or not. Defaults to True.

    Returns:
        list: List of [("name of curves", array([points uplift curves])), .. ]

    """

    if type_curve in ['gain', 'gain_perc']:
        koef = df[col_treatment].sum() / (1 - df[col_treatment]).sum()
    curves = []

    for col_score in col_scores:
        # make array of bins (0,0,...,1,1,...,n_bins-1, n_bins-1)
        index = np.repeat(np.arange(n_bins), int(df.shape[0] / n_bins))
        # supplement to full
        index = np.concatenate([index, np.full(df.shape[0] - index.shape[0], n_bins - 1)])
        # outcome'ы, сгруппированные по скору и тритменту
        grouped = df.sort_values(by=col_score).groupby([-index, col_treatment])[col_outcome]
        # count'ы в каждом бине и группе
        cnt = grouped.count()
        # print(cnt)
        # число событий в каждом бине и группе
        cnt_target = grouped.sum()

        if type_curve == 'uplift' and cumulative == False:
            # число событий в каждом бине тритмента
            cnt_target1 = cnt_target[cnt_target.index.get_level_values(col_treatment) == 1].reset_index(level=1,
                                                                                                        drop=True)
            # число событий в каждом бине колнтрола
            cnt_target0 = cnt_target[cnt_target.index.get_level_values(col_treatment) == 0].reset_index(level=1,
                                                                                                        drop=True)
        else:
            # число событий в каждом куммулятивном бине тритмента
            cnt_target1 = cnt_target[cnt_target.index.get_level_values(col_treatment) == 1].reset_index(level=1,
                                                                                                        drop=True).cumsum()
            # число событий в каждом куммулятивном бине колнтрола
            cnt_target0 = cnt_target[cnt_target.index.get_level_values(col_treatment) == 0].reset_index(level=1,
                                                                                                        drop=True).cumsum()

        if type_curve == 'uplift':
            if cumulative:
                # число объектовм куммулятивном бине тритмента
                cnt1 = cnt[cnt.index.get_level_values(col_treatment) == 1].reset_index(level=1, drop=True).cumsum()
                cnt0 = cnt[cnt.index.get_level_values(col_treatment) == 0].reset_index(level=1, drop=True).cumsum()
            else:
                cnt1 = cnt[cnt.index.get_level_values(col_treatment) == 1].reset_index(level=1, drop=True)
                cnt0 = cnt[cnt.index.get_level_values(col_treatment) == 0].reset_index(level=1, drop=True)
            res = cnt_target1 / cnt1 - cnt_target0 / cnt0

        elif type_curve == 'gain':
            cnt1 = cnt[cnt.index.get_level_values(col_treatment) == 1].reset_index(level=1, drop=True).cumsum()
            cnt0 = cnt[cnt.index.get_level_values(col_treatment) == 0].reset_index(level=1, drop=True).cumsum()
            res = cnt_target1 - cnt_target0 * (cnt1 / cnt0)
        elif type_curve == 'gain_perc':
            res = (cnt_target1 - cnt_target0 * koef) / cnt.sum()
        else:
            raise ValueError('Value of type_curve is not correct. '
                             'Use type_curve="uplift", type_curve="gain" or type_curve="gain_perc"')

        curves.append((col_score, np.insert(res.values, 0, 0)))

    return curves


def uplift_auc(curve, bins):
    """Calculate area under uplift curve

    Args:
        curve (tuple): curve from uplift_curve func
        bins (int): the same as in plift_curve func

    Returns:
        float: area under uplift curve
    """
    return np.trapz(curve[1]) - (0.5 * bins * curve[1][-1])


def plot_uplift_curve(curves, type_curve='uplift', cumulative=True):
    """Plot Uplift Curves

    Args:
        curves (list): list of curves from uplift_curve function
        type_curve (str, optional): type of curve, the same as in uplift_curve function. Defaults to 'uplift'.
        cumulative (bool, optional): only for type_curve:'uplift', calculate bins cumulatively or not. Defaults to True.

    Returns:
        matplotlib.axes._subplots.AxesSubplot: The matplotlib axes handle.

    """

    bar_colors = plt.cm.get_cmap('Pastel1').colors

    linestyles = ['-', '--', '-.', ':']

    ylabel = {'uplift': 'delta response rate',
              'gain': 'absolute difference of groups',
              'gain_perc': 'cumulative gain [% total]',
              }

    fig, ax = plt.subplots(figsize=(15, 10))

    if type_curve == 'uplift' and cumulative == True:
        plt.title('Uplift curve', fontsize=20)
    elif type_curve == 'uplift' and cumulative == False:
        plt.title('Uplift bars', fontsize=20)
    elif type_curve == 'gain':
        plt.title('Gain curve', fontsize=20)
    elif type_curve == 'gain_perc':
        plt.title('Gain_perc bars', fontsize=20)
    else:
        raise ValueError('Value of type_curve is not correct. '
                         'Use type_curve="uplift", type_curve="gain" or type_curve="gain_perc"')

    bins = len(curves[0][1]) - 1

    for i, curve in enumerate(curves):
        if type_curve == 'uplift' and cumulative == False:
            width = 0.98 / len(curves)
            left = np.arange(bins + 1) + i * width - 0.5 + width / 2
            ax.bar(left, curve[1], width, align='center', alpha=.5,
                   label=f'{curve[0]} ({round(uplift_auc(curve, bins), 3)})')

        else:
            plt.plot(curve[1], linewidth=2.0, label=f'{curve[0]} ({round(uplift_auc(curve, bins), 3)})')
            plt.plot([0, bins], [0, curves[i][1][-1]], linestyle=linestyles[i % len(linestyles)], lw=2, color='r',
                     alpha=.5)

    if type_curve == 'uplift' and cumulative == False:
        xticks = [x + 1 for x in range(bins)]
        ax.set_xticks(xticks)

    plt.legend(loc='best', bbox_to_anchor=(1, .0), fontsize=20)

    if type_curve in ylabel.keys():
        plt.ylabel(ylabel[type_curve], fontsize=20)
    else:
        raise ValueError('Value of type_curve is not correct. '
                         'Use type_curve="uplift", type_curve="gain" or type_curve="gain_perc"')

    if bins == 100:
        plt.xlabel('percent targeted', fontsize=20)
    else:
        plt.xlabel('bins', fontsize=20)
    plt.grid(True)

    return ax


def plot_bins(df, continuous=[], categorical=[], col_treatment='treatment', col_outcome='outcome', n_bins=4, bins=None,
              niv_dict=None, grouping='quantile', min_samples=5000, min_samples_cat=5000):
    """Plots bins with delta response rate (between treatment and control group)

    Args:
        df (pandas.DataFrame): Dataframe
        continous (list of str, optional): List of continuous features. Defaults to None.
        categorical (list of str, optional): List of categorical features. Defaults to None.
        col_treatment (str, optional): Name of treatment column. Defaults to 'treatment'.
        col_outcome (str, optional): Name of outcome column. Defaults to 'outcome'.
        n_bins (int, optional): number of n_bins (if you don'df have n_bins). Defaults to 4.
        grouping (str, optional): type of grouping, 'quantile' or 'tree'. Defaults to 'quantile'
        min_samples (int, optional): leaf for continuous features. Defaults to 5000.
        min_samples_cat (int, optional): min samples leaf for categorical features. Defaults to 5000.

    Returns:
        matplotlib.axes._subplots.AxesSubplot: The matplotlib axes handle.
    """

    bar_colors = plt.cm.get_cmap('Pastel1').colors

    if bins is None:
        indexes, all_bins = _get_bins(df, continuous, categorical, col_treatment, col_outcome, n_bins=n_bins,
                                      grouping=grouping, min_samples=min_samples, min_samples_cat=min_samples_cat)
    else:
        data_woe = pd.DataFrame(columns=(continuous + categorical))
        iterator = tqdm(df[(continuous + categorical)].iteritems(), total=len((continuous + categorical)), leave=True,
                        unit='cols')
        for name, column in iterator:
            bin_data = bins[name]
            target_values = [i for i in range(len(bin_data['bins']) - 1)]
            target_nan = len(bin_data['bins']) - 1
            tmp = pd.cut(column, bin_data['bins'], right=False, labels=False)
            map_dict = {np.nan: target_nan, **{i: target_values[i] for i in range(len(target_values))}}
            data_woe[name] = tmp.map(map_dict)

        renaming = {col: col for col in data_woe.columns}
        indexes = data_woe.rename(renaming, axis='columns')
        all_bins = bins

    if niv_dict is None:
        niv_dict = niv(df, continuous=continuous, categorical=categorical, col_treatment=col_treatment,
                       col_outcome=col_outcome, n_bins=n_bins, bins=None, grouping=grouping,
                       min_samples=min_samples, min_samples_cat=min_samples_cat)
        # col_outcome=col_outcome, bins=all_bins, grouping=grouping)

    if continuous != []:
        for feat in continuous:
            bins = []
            for bin in all_bins[feat]['bins']:
                bins.append(round(bin, 4))

            df_new = pd.DataFrame({'x': indexes[feat], 'y': df[col_outcome], 'group': df[col_treatment]})

            flag_nan = False
            if all_bins[feat]['nan_woe'] != 0:
                bins.append('nan')
                flag_nan = True
                df_new['x'] = df_new['x'].fillna('NaN')

            grouped = df_new.groupby(['x', 'group'])['y'].agg(['count', 'mean'])
            grouped = grouped.reset_index()
            delta_response_rate = grouped[grouped['group'] == 1]['mean'].values - \
                                  grouped[grouped['group'] == 0]['mean'].values
            bin_count = df_new.groupby(['x'])['y'].agg(['count'])['count'].values

            left = np.arange(len(bins) - 1)
            height = bin_count
            width = 1

            bin_colors = []
            for i, _ in enumerate(bin_count):
                bin_colors.append(bar_colors[i % len(bar_colors)])

            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.bar(left, height, width, alpha=0.7, edgecolor='black', linewidth=1.5, align='edge',
                    color=bin_colors)

            xticks = [x for x in range(len(bins))]
            if flag_nan:
                xticks[-1] -= 0.5
            ax1.set_xticks(xticks)
            ax1.set_xticklabels(bins, fontsize=17)
            ax1.grid(True, color='black', alpha=0.5)
            ax1.set_xlabel('Bins', fontsize=17)
            ax1.set_ylabel('Count', fontsize=17)
            ax1.set_title(f'{str(feat)}, niv:{round(niv_dict[feat], 6)}', loc='center', fontsize=20)

            ax2 = ax1.twinx()
            ax2.plot(left + 0.5, delta_response_rate, linewidth=3.5, marker='o', color='orangered', linestyle='dotted',
                     ms=3)
            ax2.set_ylabel('Delta response rate', fontsize=17)

            l = ax1.get_ylim()
            l2 = ax2.get_ylim()
            f = lambda x: l2[0] + (x - l[0]) / (l[1] - l[0]) * (l2[1] - l2[0])
            ticks = f(ax1.get_yticks())
            ticks = np.array(ticks).round(4)
            ax2.yaxis.set_major_locator(ticker.FixedLocator(ticks))

    if categorical != []:
        for feat in categorical:
            bins = list(set(all_bins[feat]['bins'].values()))

            if df[feat].notnull().all():
                new_bins = {}
                for k, v in all_bins[feat]['bins'].items():
                    if pd.isnull(k):
                        continue
                    new_bins.update({k: v})
                bins = list(set(new_bins.values()))

            df_new = pd.DataFrame({'x': indexes[feat], 'y': df[col_outcome], 'group': df[col_treatment]})
            grouped = df_new.groupby(['x', 'group'])['y'].agg(['count', 'mean'])
            grouped = grouped.reset_index()
            delta_response_rate = grouped[grouped['group'] == 1]['mean'].values - \
                                  grouped[grouped['group'] == 0]['mean'].values
            bin_count = df_new.groupby(['x'])['y'].agg(['count'])['count'].values

            left = np.arange(len(bins))
            height = bin_count
            width = 1

            bin_colors = []
            for i, _ in enumerate(bin_count):
                bin_colors.append(bar_colors[i % len(bar_colors)])

            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.bar(left, height, width, alpha=0.7, hatch='', edgecolor='black', linewidth=1.5, align='edge',
                    color=bin_colors)
            xticks = [x + 0.5 for x in range(len(bins))]
            ax1.set_xticks(xticks)
            ax1.set_xticklabels(bins, fontsize=17)
            ax1.grid(True, color='black', alpha=0.5)
            ax1.set_xlabel('Bins', fontsize=17)
            ax1.set_ylabel('Count', fontsize=17)
            ax1.set_title(f'{str(feat)}, niv:{round(niv_dict[feat], 6)}', loc='center', fontsize=20)

            real_bins = []
            for i, bin in enumerate(bins):
                real_bins.append([])
                for k, v in all_bins[feat]['bins'].items():
                    if v == bin:
                        real_bins[i].append(k)

            ax3 = ax1.twiny()
            ax3.set_xlim(ax1.get_xlim())
            ax3.set_xticks(xticks)
            ax3.set_xticklabels(real_bins)
            ax3.set_xlabel("real values in bins")

            ax2 = ax1.twinx()
            ax2.plot(left + 0.5, delta_response_rate, linewidth=3.5, marker='o', color='orangered', linestyle='dotted',
                     ms=3)
            ax2.set_ylabel('Delta response rate', fontsize=17)

            l = ax1.get_ylim()
            l2 = ax2.get_ylim()
            f = lambda x: l2[0] + (x - l[0]) / (l[1] - l[0]) * (l2[1] - l2[0])
            ticks = f(ax1.get_yticks())
            ticks = np.array(ticks).round(4)
            ax2.yaxis.set_major_locator(ticker.FixedLocator(ticks))

    return ax1
