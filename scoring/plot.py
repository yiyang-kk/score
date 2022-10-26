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

# -*- coding: utf-8 -*-
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from scipy.special import logit, expit
import matplotlib.colors as colors
import matplotlib.cm as cmx
from os import path
#rom matplotlib.ticker import FormatStrFormatter


def loss_curve(evals_result, datasets=['train','test'], metric='logloss', title='loss curve', figsize=(15,15)):
    plt.figure(figsize=figsize)
    plt.title(title)

    for i, ds in enumerate(datasets):
        plt.plot(np.arange(1, len(evals_result['validation_{}'.format(i)][metric])+1),  evals_result['validation_{}'.format(i)][metric], label=ds)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    if 'test' in datasets:
        test_values=[float(v) for v in evals_result['validation_{}'.format(datasets.index('test'))][metric]]
        min_point=sorted(enumerate(test_values), key=lambda x: x[1], reverse=(metric=='auc'))[0]
        plt.axhline(min_point[1], color='k', linestyle='--')
        plt.annotate('{:d};{:0.3f}'.format(min_point[0], min_point[1]), xy=(min_point[0],min_point[1]), xytext=(-30,10), textcoords='offset points')
        plt.plot(min_point[0], min_point[1], marker='o', color='k')




    plt.legend(loc="upper right")

    plt.show()


def _plot_badrates(df, ax, month_col, segment_col=None, segment_names=None, zero_ylim=True, weighted=False, title=''):
    """
    Plot counts and bad rates on months
    Function to set the plot_dataset() plots - as the code is repeating
    Args:
        df (pd.DataFrame): dataset input from plot_dataset
        ax (str): subplot
        month_col (str): name of month/other time interval column
        segment_col (str): name of the segment by which we want to group the plot
        segment_names (list): a list of the unique segment names from segment_col
        zero_ylim (bool): the Y axis will or will not contain zero for bad rate
        weighted (bool): True if the plot is with weight_col, False if not
        title (str): the title - different for weighted and not weighted

    Returns:
        plot
    """
    x_axis = df.index.get_level_values(month_col).unique()

    ax2 = ax.twinx()
    if segment_col:
        length = x_axis.shape[0]  # for the 'bottom' values for stacked bar chart
        the_bottom = np.zeros(length)
        for name in segment_names:
            df_1 = df[df[segment_col] == name]['count']
            df_1 = df_1.reindex(x_axis).fillna(0)  # getting 0 for the missing values, handy for test/train/oot etc.
            df_2 = df[df[segment_col] == name]['bad rate']
            df_2 = df_2.reindex(x_axis)
            ax.bar(x_axis, df_1, label='count ' + str(name), bottom=the_bottom, alpha=0.8)
            df_2.plot(ax=ax2, marker='o', label='bad rate ' + str(name))
            the_bottom += df_1
        plt.legend(loc='best')
    else:
        df_1 = df['count']
        df_1 = df_1.reindex(x_axis).fillna(0)
        df_2 = df['bad rate']
        df_2 = df_2.reindex(x_axis)
        ax.bar(x_axis, df_1)
        df_2.plot(ax=ax2, marker='o', label='count', color='r')

    if zero_ylim:
        ax2.set_ylim(0, 1.05 * max(df['bad rate']))

    for tl in ax2.get_yticklabels():
        tl.set_color('r')

    ax.set_xlabel('time')
    ax.set_xticks(range(len(x_axis)))
    ax.set_xticklabels(x_axis, rotation=90)
    plt.xlim(-0.5, len(x_axis) - 0.5)

    if weighted:
        ax.set_ylabel('weighted bad rate', color='b')
        ax2.set_ylabel('weighted count', color='r')
    else:
        ax.set_ylabel('bad rate', color='b')
        ax2.set_ylabel('count', color='r')

    for tl in ax.get_yticklabels():
        tl.set_color('b')
    ax.set_title(title)
    return ax

def plot_dataset(data, month_col='month', def_col='def_6_60', base_col=None, segment_col=None,
                 output_folder=None, filename="data_badrate.png", weight_col=None, zero_ylim=False):
    """

    Args:
        data (pd.DataFrame): a dataset for plotting
        month_col (str): name of month/other time interval column
        def_col (str): name of target column
        base_col (str): name of base column (if any)
        segment_col (str): name of the segment by which we want to group the plot
        savefile (str): the file name with the path for saving
        weight_col (str): name of the weigth column
        zero_ylim (bool): the Y axis will or will not contain zero for bad rate

    Returns:
        plot or sets of plot for badrate and count of contracts by chosen time interval

    """

    if base_col is None:
        if segment_col:
            gr = data.groupby([month_col, segment_col], axis=0)
        else:
            gr = data.groupby(month_col, axis=0)
        res = gr.apply(lambda x: pd.Series(data=(len(x), 1.*len(x[x[def_col] == 1]) / len(x),), index=['count', 'bad rate']))
    else:
        if segment_col:
            gr = data[data[base_col] == 1].groupby([month_col, segment_col], axis=0)
        else:
            gr = data[data[base_col] == 1].groupby(month_col, axis=0)
        res = gr.apply(lambda x: pd.Series(data=(len(x[(x[base_col] == 1)]), 1.*len(x[(x[def_col] == 1)&(x[base_col] == 1)])
                                                 / len(x[(x[base_col] == 1)])), index=['count', 'bad rate']))
    if segment_col: # get rid of multiIndex
        segment_names = data[segment_col].unique()
        res.reset_index(level=segment_col, inplace=True)
    else:
        segment_names = []

    res.index = res.index.astype(str)
    matplotlib.rcParams.update({'font.size': 12})

    if weight_col is not None:
        if base_col is None:
            res_w = gr.apply(lambda x: pd.Series(data=(x[weight_col].sum(), x[weight_col][x[def_col] == 1].sum()), index=['count', 'bad rate']))
            res_w['bad rate'] = res_w['bad rate'] / res_w['count']
        else:
            res_w = gr.apply(lambda x: pd.Series(data=(x[weight_col][(x[base_col] == 1)].sum(), x[weight_col][(x[def_col] == 1) & (x[base_col] == 1)].sum()), index=['count', 'bad rate']))
            res_w['bad rate'] = res_w['bad rate'] / res_w['count']

        if segment_col:
            res_w.reset_index(level=segment_col, inplace=True)
        res_w.index = res_w.index.astype(str)

        fig, (ax, bx) = plt.subplots(2, 1)  # two plots in case of weighted
        fig.set_figheight(9)

        _plot_badrates(res_w, bx, month_col, segment_col, segment_names, zero_ylim, weighted=True, title='Weighted')
    else:
        fig, ax = plt.subplots()  # one plot in case of not weighted

    for tl in ax.get_yticklabels():
        tl.set_color('b')
    _plot_badrates(res, ax, month_col, segment_col, segment_names, zero_ylim, weighted=False, title='Count & Bad Rate')

    if output_folder:
        plt.savefig(path.join(output_folder, filename), bbox_inches='tight', dpi=72)
    plt.show()
    plt.show()



def plot_distrib(t, columns, by=None, bins=20, title=''):
    """
    Графики распределения/Distribution plot
    Могут быть либо МНОГО атрибутов, либо группировочный атрибут, но не одновременно
    There can be either more attributes OR a grouping attribute but not both at once
    
    Args:
        t: data
        columns: either list of attributes of one attribute
        by: the group-by attribute
        bins:
        title:
    """
    if type(columns) == list and len(columns) > 1 and by is not None:
        raise ValueError('Can not plot distribution of multiple variables with grouping')

    if type(columns) != list:
        columns = [columns]

    fig, ax = plt.subplots()

    plt.title(title)

    if by is not None:
        jet = plt.get_cmap('jet')
        values = range(len(t[by].unique()))
        cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

        for idx, v in enumerate(np.sort(t[by].unique())):
            hist, bins=np.histogram(t[t[by]==v][columns[0]], bins=bins)
            center = (bins[:-1] + bins[1:]) / 2
            colorVal = scalarMap.to_rgba(idx)
            plt.plot(range(len(center)), 100.*hist/hist.sum(), label = v,  color=colorVal)
            handles, labels = ax.get_legend_handles_labels()
            plt.legend(handles, labels)
            plt.xlim(t[columns[0]].min(),t[columns[0]].max())
            plt.xticks(range(len(center)), ['{:.3f}'.format(c) for c in center], rotation=90)
            plt.xlabel('value')
            plt.ylabel('%')
            idx+=1
    else:
        jet = plt.get_cmap('jet')
        values = range(len(columns))
        cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

        for idx, c in enumerate(columns):
            hist, bins=np.histogram(t[c], bins=bins)
            #print c, bins, hist
            center = (bins[:-1] + bins[1:]) / 2
            colorVal = scalarMap.to_rgba(idx)
            plt.plot(range(len(center)), 100.*hist/hist.sum(), label = c,  color=colorVal)
            handles, labels = ax.get_legend_handles_labels()
            plt.legend(handles, labels)
            plt.xlim(0, len(center)-1)
            plt.xticks(range(len(center)), ['{:.3f}'.format(c) for c in center], rotation=90)
            plt.xlabel('value')
            plt.ylabel('%')

            idx+=1
    #ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.show()

def plot_calib(score, target,  bins = 20, savepath = None, filename='calibration.png', swap_probability=True):
    """   
    Plots calibration plot (mean target vs mean score by score bins)
    График скалиброванности на вероятность

    Args:
        score (np.array): score
        target (np.array): true values of target
        bins (int, optional): number of bins to bin score to. Defaults to 20.
        savepath (str, optional): folder to save the plot to (including trailing slash). Plot won't save if None. Defaults to None.
        filename (str, optional): filename for saved plot. Defaults to 'calibration.png'.
        swap_probability (bool, optional): if score is probability of 1-target instead probability of target. Defaults to True.
    """

    bins = np.percentile(score, np.linspace(0,100, bins + 1))

    scores = []
    brs = []
    for b in zip(bins[:-1], bins[1:]):
        if swap_probability:
            scores += [1 - score[(score>=b[0]) & (score<b[1])].mean()]
        else:
            scores += [score[(score>=b[0]) & (score<b[1])].mean()]
        brs += [target[(score>=b[0]) & (score<b[1])].mean()]

    plt.scatter(scores, brs)
    upperlimit = np.nanmax(scores + brs)
    plt.xlim([0, upperlimit])
    plt.ylim([0, upperlimit])
    plt.plot(np.linspace(0, upperlimit, 1000), np.linspace(0, upperlimit, 1000) , color='red')
    plt.grid()
    plt.ylabel('default rate')
    plt.xlabel('prediction')
    if savepath is not None:
        plt.savefig(savepath + filename, bbox_inches='tight', dpi = 72)
    plt.show()


def plot_kolmogorov_smirnov(score, target, bins=200, weights=None, savepath=None, filename='kolmogorov_smirnov.png'):
    """Plots cumulative distribution functions used for Kolmogorov-Smirnov statistics. 
        Plots with matplotlib to notebook.
    
    Args:
        score (pd.Series): score values
        target (pd.Series): target vallues has to be 0 or 1
        bins (int, optional): number of bins, default = 200
        weights (pd.Series, optional): weights values, has have the same index as target
        savepath (str, optional): path to folder to save the exported plot, if None doesn't save to file
        filename(str, optional): name of exported file, default = 'kolmogorov_smirnov.png'

    """
    if weights is None:
        weights = pd.Series([1.0] * len(score), index = score.index)


    n, bins , patches = plt.hist(score[target == 1] , bins=bins, weights=weights[target == 1], density= True, histtype='step', cumulative=True, color='red', label = 'Bads')
    n,    _ , patches = plt.hist(score[target == 0] , bins=bins, weights=weights[target == 0], density= True, histtype='step', cumulative=True, color='green', label = 'Goods')
    plt.title('Cumulative distributions for Goods/Bads')
    plt.xlabel('Score')
    plt.ylabel('Cumulative')
    plt.legend(loc='lower right')

    if savepath is not None:
        plt.savefig(savepath + filename, bbox_inches='tight', dpi = 72)
    plt.show()
    plt.clf()
    plt.close()


from numpy import inf
from scoring.metrics import gini

def print_binning_stats_num(data2, var, target, bins, woes, nan_woe, col_weight=None , smooth_coef=0.001, savepath = None, ntbOut=True):

    data2['BIN_'+var] = pd.cut(data2[var], bins, right=False, precision=4).values.add_categories('nan')
    data2['BIN_'+var].fillna(value='nan', inplace=True)
    data2['WOE_'+var] = pd.cut(data2[var], bins, right=False, precision=4, labels=False)
    data2['WOE_'+var] = data2['WOE_'+var].replace(range(len(woes)), woes)
    data2['WOE_'+var].fillna(value=nan_woe, inplace=True)

    if col_weight is None:
        col_weight = pd.Series(name='weight')
        data2 = pd.concat([data2, col_weight], axis=1)
        data2[col_weight.name] = 1.0
    else:
        data2 = pd.concat([data2, col_weight], axis=1)

    ## _old stuff without support for weighted data left just in case
    # f_old = {target: [len, np.sum, np.mean], 'WOE_'+var: max}

    def f(x):
        d = dict()
        d['len'] = x[col_weight.name].sum()
        d['sum'] = (x[target] * x[col_weight.name]).sum()
        if d['len'] > 0:
            d['mean'] = (x[target] * x[col_weight.name]).sum() / x[col_weight.name].sum()
        else:
            d['mean'] = np.nan
        d['woe'] = x['WOE_' + var].max()
        return pd.Series(d, index=['len', 'sum', 'mean', 'woe'])

    # woe_old = data2[[target,'BIN_'+var, 'WOE_'+var]].groupby('BIN_'+var).agg(f_old)
    woe = data2[data2[target].notnull()][[target,'BIN_'+var, 'WOE_'+var, col_weight.name]].groupby('BIN_'+var).apply(f)
    woe['DIST_BAD'] = (woe['sum']+smooth_coef)/(woe['sum'].sum()+smooth_coef)
    woe['DIST_GOOD'] = (woe['len'] - woe['sum']+smooth_coef)/(woe['len'].sum()-woe['sum'].sum()+smooth_coef)
    woe['WOE_check'] = np.log((woe['DIST_GOOD'])/(woe['DIST_BAD']))
    woe['IV'] = (woe['DIST_GOOD']-woe['DIST_BAD'])*woe['WOE_check']
    woe['SHARE'] = woe['len']/woe['len'].sum()
    # woe.columns = ['_'.join(col).rstrip('_') for col in woe.columns.values]
    # TO DO fix renaming using pd.df.rename
    # woe.columns = ['CNT_TOTAL', 'CNT_DEFAULT', 'DEFAULT_RATE', 'WOE', 'DIST_BAD', 'DIST_GOOD', 'WOE_check', 'IV', 'SHARE']
    woe = woe.rename({'len': 'CNT_TOTAL',
                     'sum': 'CNT_DEF',
                     'mean': 'DEF_RATE',
                     'woe': 'WOE',
                     'DIST_BAD': 'DIST_BAD',
                     'DIST_GOOD': 'DIST_GOOD',
                     'WOE_check': 'WOE_check',
                     'IV': 'IV',
                     'SHARE': 'SHARE'}, axis='columns')
    labels = woe.index

    woe = woe[['CNT_TOTAL','CNT_DEF','DEF_RATE','DIST_BAD','DIST_GOOD','WOE','WOE_check','IV','SHARE']]
    if ntbOut:
        display(woe[['CNT_TOTAL','CNT_DEF','DEF_RATE','DIST_BAD','DIST_GOOD','WOE','WOE_check','SHARE']].round(4))
        print('IV: {0:.4f}'.format(woe['IV'].sum()))
        print('Gini: {0:.2f}'.format(-gini(data2[target], data2['WOE_'+var])*100))

    fig, ax1 = plt.subplots(figsize=(10,5))
    fig.patch.set_facecolor('white')


    plt.xticks(range(0, len(woe)), labels, rotation=90)
    ax1.plot(woe['WOE'].reset_index(drop=True))
    ax2 = ax1.twinx()
    ax2.bar(woe['CNT_TOTAL'].reset_index(drop=True).index, woe['CNT_TOTAL'].reset_index(drop=True), width=0.5, alpha=0.5)
    ax2.grid(False)
    ax1.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Bin')
    ax1.set_ylabel('WOE')
    ax2.set_ylabel('Count')
    plt.title(var)
    if savepath is not None:
        woe[['CNT_TOTAL', 'CNT_DEF', 'DEF_RATE', 'WOE', 'DIST_BAD', 'DIST_GOOD', 'WOE_check', 'IV', 'SHARE']].round(4).to_csv(savepath+'binning.csv')
        plt.savefig(savepath+'binning.png', bbox_inches='tight', dpi = 72)
    if ntbOut:
        plt.show()
    plt.clf()
    plt.close()

def print_binning_stats_cat(data2, var, target, bins_cat, bins_bin, woes, unknown_woe, col_weight=None , smooth_coef=0.001, savepath = None, ntbOut=True):

    bins_cat = [np.nan if (x == 'NaN' or x == 'nan') else x for x in list(bins_cat)]
    nan_included = False
    for b in bins_cat:
        if pd.isnull(b): nan_included = True
    data2['BINNUM_'+var] = data2[var].replace(list(bins_cat),list(bins_bin))
    if nan_included:
        data2.loc[(~data2[var].isin(list(bins_cat))) & (~pd.isnull(data2[var])),'BINNUM_'+var] = -1
    else:
        data2.loc[~data2[var].isin(list(bins_cat)),'BINNUM_'+var] = -1
    sorted_bins = np.unique([-1] + list(bins_bin))
    bin_names = list()
    for b in sorted_bins:
        if b == -1:
            s = 'unknown values'
        else:
            s = ''
            for i in range(0,len(list(bins_bin))):
                if list(bins_bin)[i] == b:
                    if len(s) > 0:
                        s = s + ', ' + str(list(bins_cat)[i])
                    else:
                        s = str(list(bins_cat)[i])
        bin_names.append(s)
    data2['WOE_'+var] = data2['BINNUM_'+var].replace(sorted_bins,[unknown_woe]+list(woes))
    data2['BIN_'+var] = data2['BINNUM_'+var].replace(sorted_bins,bin_names)

    if col_weight is None:
        col_weight = pd.Series(name='weight')
        data2 = pd.concat([data2, col_weight], axis=1)
        data2[col_weight.name] = 1.0
    else:
        data2 = pd.concat([data2, col_weight], axis=1)

    ## _old stuff without support for weighted data left just in case
    #f_old = {target: [len, np.sum, np.mean], 'WOE_'+var: max}

    def f(x):
        d = dict()
        d['len'] = x[col_weight.name].sum()
        d['sum'] = (x[target] * x[col_weight.name]).sum()
        d['mean'] = (x[target] * x[col_weight.name]).sum() / x[col_weight.name].sum()
        d['woe'] = x['WOE_' + var].max()
        return pd.Series(d, index=['len', 'sum', 'mean', 'woe'])

    #woe_old = data2[[target,'BIN_'+var, 'WOE_'+var]].groupby('BIN_'+var).agg(f)
    woe = data2[data2[target].notnull()][[target,'BIN_'+var, 'WOE_'+var, col_weight.name]].groupby('BIN_'+var).apply(f)
    woe['DIST_BAD'] = (woe['sum']+smooth_coef)/(woe['sum'].sum()+smooth_coef)
    woe['DIST_GOOD'] = (woe['len'] - woe['sum']+smooth_coef)/(woe['len'].sum()-woe['sum'].sum()+smooth_coef)
    woe['WOE_check'] = np.log((woe['DIST_GOOD'])/(woe['DIST_BAD']))
    woe['IV'] = (woe['DIST_GOOD']-woe['DIST_BAD'])*woe['WOE_check']
    woe['SHARE'] = woe['len']/woe['len'].sum()
    # woe.columns = ['_'.join(col).rstrip('_') for col in woe.columns.values]
    #woe.columns = ['CNT_TOTAL', 'CNT_DEF', 'DEF_RATE', 'WOE', 'DIST_BAD', 'DIST_GOOD', 'WOE_check', 'IV', 'SHARE']
    woe = woe.rename({'len': 'CNT_TOTAL',
                     'sum': 'CNT_DEF',
                     'mean': 'DEF_RATE',
                     'woe': 'WOE',
                     'DIST_BAD': 'DIST_BAD',
                     'DIST_GOOD': 'DIST_GOOD',
                     'WOE_check': 'WOE_check',
                     'IV': 'IV',
                     'SHARE': 'SHARE'}, axis='columns')
    labels = woe.index

    if ntbOut:
        display(woe[['CNT_TOTAL', 'CNT_DEF', 'DEF_RATE', 'WOE', 'DIST_BAD', 'DIST_GOOD', 'WOE_check', 'IV', 'SHARE']].round(4))
        print('IV: {0:.4f}'.format(woe['IV'].sum()))
        print('Gini: {0:.2f}'.format(-gini(data2[target], data2['WOE_'+var])*100))

    fig, ax1 = plt.subplots(figsize=(10,5))
    fig.patch.set_facecolor('white')


    plt.xticks(range(0, len(woe)), [i[:30] for i in labels], rotation=90)
    ax1.plot(woe['WOE'].reset_index(drop=True))
    ax2 = ax1.twinx()
    ax2.bar(woe['CNT_TOTAL'].reset_index(drop=True).index, woe['CNT_TOTAL'].reset_index(drop=True), width=0.5, alpha=0.5)
    ax2.grid(False)
    ax1.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Bin')
    ax1.set_ylabel('WOE')
    ax2.set_ylabel('Count')
    plt.title(var)
    if savepath is not None:
        woe[['CNT_TOTAL', 'CNT_DEF', 'DEF_RATE', 'WOE', 'DIST_BAD', 'DIST_GOOD', 'WOE_check', 'IV', 'SHARE']].round(4).to_csv(savepath+'binning.csv')
        plt.savefig(savepath+'binning.png', bbox_inches='tight', dpi = 72)
    if ntbOut:
        plt.show()
    plt.clf()
    plt.close()

import seaborn as sns

def stability_chart(binvar,target,base,obs,month,savepath=None,ntbOut=True,weight=None, grouping=None):
    """Plots bad-rate and population stability charts for a binned variable. 

    Args:
        binvar (pd.Series): binned variable to plot
        target (pd.Series): binary target
        base (pd.Series): base for bad-rate
        obs (pd.Series): base for population
        month (pd.Series): time Series
        savepath (str, optional): Path to export file. Defaults to None.
        ntbOut (bool, optional): Outputs to notebook if True. Defaults to True.
        weight (pd.Series, optional): weight Series. Defaults to None.
        grouping (scoring.Grouping, optional): Grouping object for representing bins with interval instead of transformed values. Defaults to None.

    """
    
    if grouping:
        grouping_dictionary = grouping.export_dictionary()

    def replace_legend(woe):
        """
        Tries to replace a woe value with interval or list of values if grouping was provided.
        If no grouping was provided returns original value.

        Args:
            woe (float): woe value to be replaced
        """

        # values from grouping_dictionary() are float32 but values in calculated tables
        # in stability_chart() are defaulted to float64 and therefore even with the same
        # value have different hashes and need to unified with other float32 values 
        woe = np.float32(woe)

        if grouping:
            new_value = grouping_dictionary[binvar.name][round(woe,5)]

            # if value is list of strings lets join them
            if type(new_value) is list:
                group_name = [str(i) for i in new_value]
                # split into chunks of five
                group_name = [",".join(group_name[i:i+5]) for i in range(0,len(group_name),5)]
                # each chuck get a new line
                group_name = "\n".join([str(i) for i in group_name])

                return group_name
            else:
                return new_value
        else:
            return woe



    if obs.name == base.name:
        obs = obs.copy()
        obs.name = '_obs'
    data_chart = pd.concat([binvar,month,target,base,obs],axis=1)
    if weight is None:
        data_chart['w'] = 1.0
    else:
        data_chart['w'] = weight
    unique_months = pd.DataFrame(pd.Series(list(month.unique())).rename('month'))
    unique_values = pd.DataFrame(pd.Series(list(binvar.unique())).rename('binvar'))
    unique_months['k'] = 1
    unique_values['k'] = 1
    data_chart = data_chart.rename(columns={binvar.name : 'binvar',
                                            month.name : 'month',
                                            target.name : 'target',
                                            base.name : 'base',
                                            obs.name : 'obs',
                                            'w': 'w'})
    # data_chart.columns = ['binvar','month','target','base','obs', 'w']
    for i in ['target','base','obs']:
        data_chart[i] = data_chart[i] * data_chart['w']
    data_chart = pd.merge(data_chart, pd.merge(unique_months,unique_values,how='outer',on='k'), how='right', on=['month','binvar']).fillna(0)
    data_chart.drop(['k'],axis=1,inplace=True)
    sns.set(rc={"figure.figsize": (18, 12)})

    tmp = data_chart.groupby(['binvar','month'])[['target','base','obs']].sum()
    tmp2 = tmp.reset_index().groupby('month')['obs'].sum()
    tmp3 = tmp.join(tmp2,how='inner',rsuffix='_all').reset_index(level='month')
    tmp3['bad_rate'] = tmp3['target']/tmp3['base']
    tmp3['pop_share'] = tmp3['obs']/tmp3['obs_all']

    plt.subplot(221)
    for i in tmp3.index.unique():
        subset = tmp3[tmp3.index == i].sort_values('month')
        plt.plot(range(len(subset['month'].unique())),subset['bad_rate'],label=replace_legend(i),linewidth=4)
    plt.xticks(range(len(subset['month'].unique())), np.sort(subset['month'].unique()), rotation=45,fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(binvar.name+str(': bad rate'),fontsize=16)

    plt.subplot(222)
    for i in tmp3.index.unique():
        subset = tmp3[tmp3.index == i].sort_values('month')
        plt.plot(range(len(subset['month'].unique())),subset['pop_share'],label=replace_legend(i),linewidth=4)
    plt.xticks(range(len(subset['month'].unique())), np.sort(subset['month'].unique()), rotation=45,fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(binvar.name+str(': population share'),fontsize=16)
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5),fontsize=16)
    if savepath is not None:
        plt.savefig(savepath+'stability.png', bbox_inches='tight', dpi = 72)
    if ntbOut:
        plt.show()
    plt.clf()
    plt.close()



def transmatrix(oldscore,newscore,target,base,obs,draw_default_matrix=True,draw_transition_matrix=True,savepath=None,quantiles_count=10):
    newscore_dec = pd.DataFrame(pd.qcut(newscore,quantiles_count,labels=False,duplicates='drop'))
    oldscore_dec = pd.DataFrame(pd.qcut(oldscore,quantiles_count,labels=False,duplicates='drop'))
    dec_data = pd.concat([oldscore_dec,newscore_dec,target,base,obs],axis=1)
    dec_data.columns = ['oldscore','newscore','target','base','obs']
    dec_data_agg = dec_data.groupby(['oldscore','newscore'])['target','base','obs'].sum()
    dec_data_agg2 = dec_data.groupby(['oldscore'])['obs'].sum()
    dec_data_all = dec_data_agg.reset_index().join(dec_data_agg2,on=['oldscore'],rsuffix='_all').set_index(['oldscore','newscore'])
    dec_data_all['default rate'] = dec_data_all['target']/dec_data_all['base']
    dec_data_all['share'] = dec_data_all['obs']/dec_data_all['obs_all']
    if draw_default_matrix:
        matrix_DR = np.matrix(dec_data_all.unstack()[['default rate']])
        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(matrix_DR, annot=True, cmap=sns.cubehelix_palette(light=1, as_cmap=True))
        ax.set_ylabel('old score decile')
        ax.set_xlabel('new score decile')
        plt.title('Default rate by deciles')
        if savepath is not None:
            plt.savefig(savepath+'matrix_default.png', bbox_inches='tight', dpi = 72)
        plt.show()
        plt.clf()
        plt.close()
    if draw_transition_matrix:
        matrix_OS = np.matrix(dec_data_all.unstack()[['share']])
        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(matrix_OS, annot=True, cmap=sns.cubehelix_palette(light=1, as_cmap=True))
        ax.set_ylabel('old score decile')
        ax.set_xlabel('new score decile')
        plt.title('Transition matrix by deciles')
        if savepath is not None:
            plt.savefig(savepath+'matrix_transition.png', bbox_inches='tight', dpi = 72)
        plt.show()
        plt.clf()
        plt.close()

def plot_score_dist(data, score_name, target_name, base_name = None, weight_name = None,
                    n_bins = 25, min_score = None, max_score = None, savefile = None,
                    labels = ["good", "bad"], legend_loc = "upper left"):
    """
    Plots charts with distribution/density of score values.
    
    Args:
        data (pandas.DataFrame): dataframe with score and target
        score_name (str): name of column with score
        target_name (str): name of column with target
        base_name (str, optional): name of column with base (0/1 indicator whether the observation should be taken into account) - zeros will be filtered out (default: None)
        weight_name (str, optional): name of column with weight (weight for each row) (default: None)
        n_bins (int, optional): number of bins the score should be binned to (default: 25)
        min_score (float, optional): minimal score value for binning (default: None)
        max_score (float, optional): maximal score value for binning (default: None)
        savefile (str, optional): filename to save the charts to (should be of type .png) (default: None)
        labels (list of str, optional): list of two strings - labels for levels [0, 1] of col_target (default: ['good','bad'])
        legend_loc (str, optional): location of legend in the charts (default: 'upper left')
    
    Returns:
        pandas.DataFrame: dataframe with the underlying data for the plot
    """

    if (weight_name is None) and (base_name is None):
        data = data[[score_name, target_name]].copy()
    elif (base_name is None):
        data = data[[score_name, target_name, weight_name]].copy()
    elif (weight_name is None):
        data = data[data[base_name] == 1][[score_name, target_name]].copy()
    else:
        data = data[data[base_name] == 1][[score_name, target_name, weight_name]].copy()

    if min_score is None:
        min_score = min(data[score_name])
    if max_score is None:
        max_score = max(data[score_name])

    bin_border = []
    for i in range(0,n_bins):
        bin_border += [min_score + i * (max_score - min_score) / n_bins]
    bin_border += [max_score + 0.00001]

    data['bin'] = np.zeros(len(data)).astype(int)
    bin_str = []
    for i in range(0,n_bins):
        data['bin'] = np.where((data[score_name] >= bin_border[i]) & (data[score_name] < bin_border[i+1]), i+1, data['bin'])
        bin_str += ['[' + str(round(bin_border[i],2)) + ';' + str(round(bin_border[i+1],2)) + ')']

    if weight_name is None:
        data_grp = data[['bin',target_name, score_name]].groupby([target_name, 'bin'])[[score_name]].count()
    else:
        data_grp = data[['bin',target_name, weight_name]].groupby([target_name, 'bin'])[[weight_name]].sum()

    bins_base = pd.DataFrame(np.arange(1,n_bins+1), columns=['bin'])
    bins_base.set_index('bin', inplace = True)
    good_hist = data_grp.loc[0]
    good_hist.columns = ['good_cnt']
    bad_hist = data_grp.loc[1]
    bad_hist.columns = ['bad_cnt']
    dt_hist = bins_base.join(good_hist, how = 'left').join(bad_hist, how = 'left').fillna(0)
    dt_hist['good_cnt_norm'] = dt_hist['good_cnt'] / (dt_hist['good_cnt'] + dt_hist['bad_cnt'])
    dt_hist['bad_cnt_norm'] = dt_hist['bad_cnt'] / (dt_hist['good_cnt'] + dt_hist['bad_cnt'])

    fig = plt.subplots(figsize = (15,6))
    plt.subplot(121)
    plt.bar(range(1,n_bins+1), dt_hist['bad_cnt'], label = labels[1], color = 'r')
    plt.bar(range(1,n_bins+1), dt_hist['good_cnt'], bottom = dt_hist['bad_cnt'].values, label = labels[0], color = 'b')
    plt.xticks(range(1,n_bins+1), bin_str, rotation = 90)
    plt.xlabel('Score')
    plt.ylabel('Frequency')

    plt.legend(loc = legend_loc)

    plt.subplot(122)
    plt.bar(range(1,n_bins+1), dt_hist['bad_cnt_norm'], label = labels[1], color = 'r')
    plt.bar(range(1,n_bins+1), dt_hist['good_cnt_norm'], bottom = dt_hist['bad_cnt_norm'].values, label = labels[0], color = 'b')
    plt.xticks(range(1,n_bins+1), bin_str, rotation = 90)
    plt.xlabel('Score')
    plt.ylabel('Normalized frequency')

    plt.legend(loc = legend_loc)

    if savefile is not None:
        plt.savefig(savefile, bbox_inches='tight', dpi = 72)
    plt.show()

    return dt_hist

def score_calibration(data, score, target, weight = None, shift = 0, scale = 1, ispd = False, bins = 30, savefile = None, vertical_lines = None):
    """Prints score ditribution and calibration chart - shows score histogram along with real and
    predicted bad rates of the quantiles.
    Score must be in form of logit (i.e. logistically transformed probability of default) or
    probability (in that case, use parameter ispd).
    
    Args:
        data (pd.DataFrame): data frame with score and target, must be prefiltered by mask keeping only observable rows
        score (str): name of score column in data
        target (str): name of target column in data
        weight (str, optional): name of weight column in data (default: None)
        shift (int, optional): shift (intercept) applied to score (default: 0)
        scale (int, optional): scale (slope) applied to score (default: 1)
        ispd (boolean, optional): indicator if score is in form of PD (True) or logit (False) (default: False)
        bins (int, optional): number of bins for histogram (default: 30)
        savefile (str, optional): path to file to save the output (output is not saved if savefile not specified) (default: None)
        vertical_lines (list of float, optional): list of reference vertical lines to be drawn into the chart (default: None)
    """
    dt = data.copy()

    if weight is None:
        weight = '_weight'
        dt[weight] = 1
    dt['_wt'] = dt[weight] * dt[target]

    if len(dt[(dt[target].notnull()) & (dt[score].isnull())]) > 0:
        def_rx_nohit = dt[(dt[target].notnull()) & (dt[score].isnull())]['_wt'].sum() / dt[(dt[target].notnull()) & (dt[score].isnull())][weight].sum()
    else:
        def_rx_nohit = np.nan

    mask = dt[target].notnull() & dt[score].notnull()

    if not ispd:
        dt['_pd'] = expit(scale * dt[score] + shift)
        dt[score] = scale * dt[score] + shift
    else:
        dt['_pd'] = expit(scale * logit(dt[score]) + shift)
        dt[score] = scale * logit(dt[score]) + shift

    bin_ranges = [dt[mask][score].min() + k*(dt[mask][score].max()-dt[mask][score].min())/bins for k in range(bins+1)]
    bin_means = [(bin_ranges[k] + bin_ranges[k+1])/2 for k in range(bins)]
    bin_ranges[-1] = np.inf
    def_rx = []
    cnts = []
    for k in range(bins):
        if dt[mask & (dt[score] >= bin_ranges[k]) & (dt[score] < bin_ranges[k+1])][weight].sum() > 0:
            def_rx.append(dt[mask & (dt[score] >= bin_ranges[k]) & (dt[score] < bin_ranges[k+1])]['_wt'].sum() / dt[mask & (dt[score] >= bin_ranges[k]) & (dt[score] < bin_ranges[k+1])][weight].sum())
            cnts.append(dt[mask & (dt[score] >= bin_ranges[k]) & (dt[score] < bin_ranges[k+1])][weight].sum())
        else:
            def_rx.append(np.nan)
            cnts.append(0)

    _, ax1 = plt.subplots()
    plt.bar(bin_means, cnts, color = 'r', width = (bin_ranges[1] - bin_ranges[0])*0.9)
    plt.xlabel('score')
    plt.ylabel('frequncy')

    ax2 = ax1.twinx()
    ax2.plot(bin_means, def_rx, 'o-', color = 'b')
    if def_rx_nohit:
        ax2.plot(bin_means, np.ones(len(bin_means)) * def_rx_nohit, '--', color = 'g')
    if vertical_lines is not None:
        if type(vertical_lines)!=list:
            vertical_lines = [vertical_lines]
        for line_y in vertical_lines:
            ax2.plot([line_y,line_y],[0,1], '--', color = 'c')

    sc_to_prob = 1 / (1 + np.exp(-np.array(bin_means)))
    ax2.plot(bin_means, sc_to_prob, '--', color = 'black')

    plt.xlabel('score')
    plt.ylabel('default rate')
    plt.title(score)

    if savefile is not None:
        plt.savefig(savefile, bbox_inches='tight', dpi = 72)
    plt.show()

    dt['_wpd'] = dt[weight] * dt['_pd']

    from .metrics import gini

    print(f'Avg predicted default rate: {100*dt[mask]["_wpd"].sum()/dt[mask][weight].sum():5.2f}%')
    print(f'Avg hit default rate:       {100*dt[mask]["_wt"].sum()/dt[mask][weight].sum():5.2f}%')
    print(f'Avg non-hit default rate:   {100*def_rx_nohit:5.2f}%')
    print(f'Hit Gini: {100*gini(dt[mask][target], dt[mask][score], dt[mask][weight]):5.2f}')


def confusion_chart(data, col_score, col_target, col_weight=None, reference_ar=0.50, def_by_score_ascending = None, savefile = None):
    """Draws a graphical version of confusion matrix which displays "false reject" (rejects of good clients) and "false approval" (approval of bad clients)
    rates and their dependency on desired overall approval rate. For one given level of desired overall approval rate (argument reference_ar), this function
    also returns the false reject and false approval rates as numbers (in this order).
    
    Args:
        data (pd.DataFrame): underlying dataset for the analysis, must contain score that will be used for rejection and target variable
        col_score (str): name of score variable that will be used for rejection
        col_target (str): name of target variable that tells whether the client was good (0) or bad (1)
        col_weight (str, optional): name of weight variable telling importance of each observation - if not filled, all observations have the same importance (default: None)
        reference_ar (float, optional): desired overall approval rate (default: 0.50)
        def_by_score_ascending (boolean, optional): True if the score grows with probability default. False if the score decreases with PD. None for the function to determine this automatically. (default: None)
        savefile (str, optional): Path where the chart should be saved to. If empty, the chart will not be saved. This path should include also the file name. (default: None)
    
    Returns:
        float, float: false reject and false approval rates for reference_ar
    """
    if col_weight is not None:
        data = data[[col_score, col_target, col_weight]].copy()
    else:
        data = data[[col_score, col_target]].copy()
        col_weight = '_weight'
        data[col_weight] = 1

    if def_by_score_ascending is None:
        correlation_default_score = np.corrcoef(data[col_score], data[col_target])[0][1]
        if correlation_default_score >= 0:
            def_by_score_ascending = True
        else:
            def_by_score_ascending = False

    data.sort_values(col_score, inplace=True, ascending=1-def_by_score_ascending)
    data['cumulative_weight'] = data[col_weight].cumsum()
    data['cumulative_bads'] = (data[col_weight]*data[col_target]).cumsum()
    sum_weight = data[col_weight].sum()
    sum_bads = (data[col_weight]*data[col_target]).sum()

    data['rejected_bads'] = data['cumulative_bads']
    data['rejected_goods'] = data['cumulative_weight']-data['cumulative_bads']
    data['approved_bads'] = sum_bads - data['cumulative_bads']
    data['approved_goods'] = (sum_weight-sum_bads) - (data['cumulative_weight']-data['cumulative_bads'])

    data['false_reject'] = data['rejected_goods'] / (data['rejected_goods'] + data['approved_goods'])
    data['true_reject'] =1 - data['false_reject']
    data['false_approve'] = data['approved_bads'] / (data['approved_bads'] + data['rejected_bads'])
    data['true_approve'] =1 - data['false_approve']

    data['reject_rate'] = data['cumulative_weight']/sum_weight
    data['approve_rate'] = 1 - data['reject_rate']

    plt.plot(data['approve_rate'], data['false_reject'], label = '% of falsely rejected')
    plt.plot([reference_ar, reference_ar], [0,1], color='grey')
    plt.plot(data['approve_rate'], data['false_approve'], label = '% of falsely approved')
    plt.xlabel('% approved')
    plt.legend(loc = 'center left')
    plt.xlim([0,1])
    plt.ylim([0,1])

    if savefile is not None:
        plt.savefig(savefile, bbox_inches='tight', dpi = 72)
    plt.show()

    false_reject = data[data['approve_rate']<=reference_ar].iloc[0]['false_reject']
    false_approve = data[data['approve_rate']<=reference_ar].iloc[0]['false_approve']

    print(f'Approval rate {reference_ar*100:.2f}%')
    print(f'False reject rate {false_reject*100:.2f}%')
    print(f'False approval rate {false_approve*100:.2f}%')

    return false_reject, false_approve

def expected_default_rate(data, col_score, col_target, col_weight=None, reference_ar=0.50, def_by_score_ascending = None, savefile = None):
    """Draws a chart of expected default rate above cutoff (i.e. for approved clients) and below cutoff (i.e. for rejected clients). The chart shows
    these numbers for all theoretical cutoffs (from 0% to 100% expected approval rate). The function returns default rate for approved and rejected
    clients as numbers (on this order) for specified expected approval rate (argument reference_ar).
    
    Args:
        data (pd.DataFrame): underlying dataset for the analysis, must contain score that will be used for rejection and target variable
        col_score (str): name of score variable that will be used for rejection
        col_target (str): name of target variable that tells whether the client was good (0) or bad (1)
        col_weight (str, optional): name of weight variable telling importance of each observation - if not filled, all observations have the same importance (default: {None})
        reference_ar (float, optional): desired overall approval rate (default: {0.50})
        def_by_score_ascending (boolean, optional): True if the score grows with probability default. False if the score decreases with PD. None for the function to determine this automatically. (default: {None})
        savefile (str, optional): Path where the chart should be saved to. If empty, the chart will not be saved. This path should include also the file name. (default: {None})
    
    Returns:
        float, float: default rate for approved clients and default rate for rejected clients
    """
    if col_weight is not None:
        data = data[[col_score, col_target, col_weight]].copy()
    else:
        data = data[[col_score, col_target]].copy()
        col_weight = '_weight'
        data[col_weight] = 1

    if def_by_score_ascending is None:
        correlation_default_score = np.corrcoef(data[col_score], data[col_target])[0][1]
        if correlation_default_score >= 0:
            def_by_score_ascending = True
        else:
            def_by_score_ascending = False

    data.sort_values(col_score, inplace=True, ascending=1-def_by_score_ascending)
    data['cumulative_weight'] = data[col_weight].cumsum()
    data['cumulative_bads'] = (data[col_weight]*data[col_target]).cumsum()
    data['sum_weight'] = data[col_weight].sum()
    data['sum_bads'] = (data[col_weight]*data[col_target]).sum()

    data['bad_rate_under_cutoff'] = data['cumulative_bads']/data['cumulative_weight']
    data['bad_rate_over_cutoff'] = (data['sum_bads']-data['cumulative_bads'])/(data['sum_weight']-data['cumulative_weight'])
    data.loc[data['sum_weight'] <= data['cumulative_weight'], 'bad_rate_over_cutoff'] = 0

    data['reject_rate'] = data['cumulative_weight']/data['sum_weight']
    data['approve_rate'] = 1 - data['reject_rate']

    plt.plot(data['approve_rate'], data['bad_rate_over_cutoff'], label = 'bad rate of approved')
    y_max = 1.1*np.max(
        [np.percentile(data[~np.isnan(data['bad_rate_under_cutoff'])]['bad_rate_under_cutoff'],99),
         np.percentile(data[~np.isnan(data['bad_rate_over_cutoff'])]['bad_rate_over_cutoff'],99)]
        )
    plt.plot([reference_ar, reference_ar], [0,y_max], color='grey')
    plt.plot(data['approve_rate'], data['bad_rate_under_cutoff'], label = 'bad rate of rejected')
    plt.xlabel('% approved')
    plt.legend(loc = 'upper left')
    plt.xlim([0,1])
    plt.ylim([0,y_max])

    if savefile is not None:
        plt.savefig(savefile, bbox_inches='tight', dpi = 72)
    plt.show()

    bad_rate_under_cutoff = data[data['approve_rate']<=reference_ar].iloc[0]['bad_rate_under_cutoff']
    bad_rate_over_cutoff = data[data['approve_rate']<=reference_ar].iloc[0]['bad_rate_over_cutoff']

    print(f'Approval rate {reference_ar*100:.2f}%')
    print(f'Bad rate of approved {bad_rate_over_cutoff*100:.2f}%')
    print(f'Bad rate of rejected {bad_rate_under_cutoff*100:.2f}%')

    return bad_rate_over_cutoff, bad_rate_under_cutoff
