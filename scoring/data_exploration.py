
# Home Credit Python Scoring Library and Workflow
# Copyright © 2017-2020, Pavel Sůva, Marek Teller, Martin Kotek, Jan Zeller,
# Marek Mukenšnabl, Kirill Odintsov, Elena Kuchina, Jan Coufalík, Jan Hynek,
# Nada Horka, Lubor Pacak, Kamil Yazigee and
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


import os.path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.weightstats import DescrStatsW
from IPython.display import Markdown, display, FileLink, HTML
from tqdm.notebook import tqdm


def calculate_numerical(varCol, targetCol, weightCol=None):
    """Stats about pd.Series with numerical values

    Args:
        varCol (pd.Series): column with variable
        targetCol (pd.Series): columns with target
        weightCol (pd.Series, optional): column with weight. Defaults to None.

    Returns:
        pd.DataFrame : DataFrame containing useful stats about pd.Series with numerical values. Requires a pd.Series with
        target values. Optional 'weightCol' with real number weights for each row.
    """
    varName = varCol.name
    targetName = targetCol.name

    # discard rows with empty target value
    targetMask = pd.notnull(targetCol)
    varCol = varCol[targetMask]
    targetCol = targetCol[targetMask]

    if weightCol is not None:
        weightCol = weightCol[targetMask]
        weightName = weightCol.name
        weighted = True
    else:
        weightName = 'weight'
        weightCol = pd.Series(name=weightName)
        weighted = False

    # TODO count NaNs as a valid category
    data = dict()
    data['All'] = pd.concat([varCol, targetCol], axis=1).fillna(value=np.NaN)
    data['All'] = pd.concat([data['All'], weightCol], axis=1).fillna(value={'weight': 1})
    data['T0'] = data['All'][data['All'][targetName].values == 0]
    data['T1'] = data['All'][data['All'][targetName].values == 1]

    output = dict()
    for key, subdata in data.items():
        if subdata[varName].count() == 0:
            output[key] = pd.DataFrame(index=[key])
            continue
        desc = DescrStatsW(subdata[varName], weights=subdata[weightName])
        desc_noNaNs = DescrStatsW((subdata.dropna())[varName], weights=(subdata.dropna())[weightName])
        output[key] = pd.DataFrame(index=[key])
        output[key]['Mean'] = desc_noNaNs.mean
        output[key]['Std'] = desc_noNaNs.std
        output[key]['Count'] = len(subdata)
        if weighted:
            output[key]['Sum of W'] = subdata[weightName].sum()
        output[key]['CntNaN'] = subdata[varName].isnull().sum()
        output[key]['Pct100NaN'] = round(int(output[key]['CntNaN']) / int(output[key]['Count']) * 100, 2)
        if weighted:
            if key == 'All':
                output[key]['CorrTarget'] = DescrStatsW((subdata.dropna())[[varName, targetName]], weights=(subdata.dropna())[weightName]).corrcoef[0][1]
            else:  # for only 0 or only 1, the correlation with target does not make sense and throws warning and NaN result
                output[key]['CorrTarget'] = np.NaN
        else:
            output[key]['CorrTarget'] = subdata[[varName, targetName]].corr(method='spearman', min_periods=1).iloc[0, 1]
        output[key]['Min'] = subdata[varName].min()
        output[key]['Max'] = subdata[varName].max()
        quantilesAll = pd.DataFrame(desc.quantile([0.1, 0.25, 0.50, 0.75, 0.9], return_pandas=True))
        quantilesAll.columns = [key]
        output[key] = pd.concat([output[key], quantilesAll.transpose()], axis=1)
        if weighted:
            output[key] = output[key][['Mean', 'Std', 'Count', 'Sum of W', 'CntNaN', 'Pct100NaN', 'CorrTarget', 'Min', 0.1, 0.25, 0.50, 0.75, 0.9, 'Max']]
            output[key].columns = ['Mean', 'Std.Dev.', 'Count', 'Sum of W', '# of NaN', '% of NaN', 'Corr w. Target', 'Min', '10%', '25%', 'Median', '75%', '90%', 'Max']
        else:
            output[key] = output[key][['Mean', 'Std', 'Count', 'CntNaN', 'Pct100NaN', 'CorrTarget', 'Min', 0.1, 0.25, 0.50, 0.75, 0.9, 'Max']]
            output[key].columns = ['Mean', 'Std.Dev.', 'Count', '# of NaN', '% of NaN', 'Corr w. Target', 'Min', '10%', '25%', 'Median', '75%', '90%', 'Max']

    return pd.concat([output[key] for key in ['All', 'T1', 'T0']])


def calculate_categorical(varCol, targetCol, weightCol=None):
    """Stats about pd.Series with categorical values

    Args:
        varCol (pd.Series): column with variable
        targetCol (pd.Series): columns with target
        weightCol (pd.Series, optional): column with weight. Defaults to None.

    Returns:
        pd.DataFrame : DataFrame containing useful stats about pd.Series with categorical values. Requires a pd.Series with target values. Optional 'weightCol' with real number weights for each row.
    """
    # extract column names
    varCol = varCol.astype('category')
    varName = varCol.name
    targetName = targetCol.name

    # discard rows with empty target value
    targetMask = pd.notnull(targetCol)
    varCol = varCol[targetMask]
    targetCol = targetCol[targetMask]

    if weightCol is not None:
        weightCol = weightCol[targetMask]
        weightName = weightCol.name
    else:
        weightName = 'weight'
        weightCol = pd.Series(name=weightName)
  
    varCol = varCol.cat.add_categories(['_NaN'])
    data = pd.concat([varCol, targetCol], axis=1).fillna(value='_NaN')
    if len(data[data[varCol.name] == "_NaN"]) == 0:
        data[varCol.name] = data[varCol.name].cat.remove_categories('_NaN')

    data = pd.concat([data, weightCol], axis=1).fillna(value={'weight': 1})

    # helper to hold values required for calculations but not to be displayed
    helper = pd.DataFrame(index=data[varCol.name].cat.categories)

    # output dataframe in the format for display. Categories in rows and statistics in columns
    output = pd.DataFrame(index=data[varCol.name].cat.categories)
    output['Count'] = data.groupby(varName)[weightName].sum()
    helper['Totalcount'] = output['Count'].sum()
    output['Count(%)'] = (output['Count'] / helper['Totalcount'] * 100).round(decimals=2)
    output = pd.concat([output, pd.crosstab(index=data[varName], columns=data[targetName], dropna=False, values=data[weightName], aggfunc=sum)], axis=1)
    output.rename(columns={1.0: 'Bad', 0.0: 'Good'}, inplace=True)
    output['Bad(%)'] = (output['Bad'] / output['Count'] * 100).round(decimals=2)
    output['Good(%)'] = (output['Good'] / output['Count'] * 100).round(decimals=2)
    helper['TotalGood'] = output['Good'].sum()
    helper['TotalBad'] = output['Bad'].sum()
    output['Good(%)T'] = (output['Good'] / helper['TotalGood'] * 100).round(decimals=2)
    output['Bad(%)T'] = (output['Bad'] / helper['TotalBad'] * 100).round(decimals=2)
    output['WOE'] = np.log((output['Good'] / helper['TotalGood']) / (output['Bad'] / helper['TotalBad'])).round(decimals=4)
    output['WOE'] = np.where(np.isinf(output['WOE']), 0, output['WOE'])
    output['IV'] = ((output['Good'] / helper['TotalGood'] - output['Bad'] / helper['TotalBad']) * output['WOE']).round(decimals=4)
    output['IV'] = np.where(output['IV'] < 0, 0, output['IV'])
    output['IV (Total)'] = output['IV'].sum()
    output.index.name = varName

    return output


def calculate_quantiles(varCol, targetCol, weightCol=None, bins=10):
    """Quantile info about pd.Series with numerical values

    Args:
        varCol (pd.Series): column with variable
        targetCol (pd.Series): columns with target
        weightCol (pd.Series, optional): column with weight. Defaults to None.
        bins (int, optional): number of bins. Defaults to 10.

    Returns:
        pd.DataFrame : DataFrame containing quantile information about pd.Series with numerical values. Requires a ps.Series with target values. Optional 'weightCol' with real number weights for each row.
    """
    varName = varCol.name
    targetName = targetCol.name

    # discard rows with empty target value
    targetMask = pd.notnull(targetCol)
    varCol = varCol[targetMask]
    targetCol = targetCol[targetMask]

    if weightCol is not None:
        weightCol = weightCol[targetMask]
        weightName = weightCol.name
    else:
        weightName = 'weight'
        weightCol = pd.Series(name=weightName)

    data = pd.concat([varCol, targetCol], axis=1).fillna(value=np.NaN)
    data = pd.concat([data, weightCol], axis=1).fillna(value={'weight': 1})
    dataDesc = DescrStatsW(data[varName], weights=data[weightName])
    quantilePoints = [i / bins for i in range(bins)][1:]
    bins = list(dataDesc.quantile(quantilePoints, return_pandas=True))
    # bins = [varCol.min() - 0.01] + bins + [varCol.max() + 0.01]  # Little hacky, to include min and max in proper bins
    bins = [-np.inf] + bins + [np.inf]

    data['binRange'] = pd.cut(x=data[varName], bins=bins, duplicates = 'drop')

    output = pd.DataFrame(pd.crosstab(index=data['binRange'], columns=data[targetName], values=data[weightName], aggfunc=sum, dropna=False))
    output.columns = ['Good', 'Bad']
    output['Good(%)'] = (output['Good'] / (output['Good'] + output['Bad']) * 100).round(decimals=2)
    output['Bad(%)'] = (output['Bad'] / (output['Good'] + output['Bad']) * 100).round(decimals=2)
    output['0_TotCnt'] = output['Good'].sum()
    output['1_TotCnt'] = output['Bad'].sum()
    output['TotBadRate'] = (output['1_TotCnt'] / (output['0_TotCnt'] + output['1_TotCnt'])).round(decimals=4)
    output['TotalCount'] = output['0_TotCnt'] + output['1_TotCnt']
    output['0_ColPct100'] = (output['Good'] / output['0_TotCnt'] * 100).round(decimals=2)
    output['1_ColPct100'] = (output['Bad'] / output['1_TotCnt'] * 100).round(decimals=2)
    output['0_Pct100'] = (output['Good'] / output['TotalCount'] * 100).round(decimals=2)
    output['1_Pct100'] = (output['Bad'] / output['TotalCount'] * 100).round(decimals=2)
    output['Cnt'] = output['Good'] + output['Bad']
    output['Pct100'] = (output['Cnt'] / output['TotalCount'] * 100).round(decimals=2)
    output['CumCount'] = output['Cnt'].cumsum()
    output['CumPct100'] = output['Pct100'].cumsum()
    output = output.join((pd.Series(np.log((output['Good'] / output['0_TotCnt']) / (output['Bad'] / output['1_TotCnt'])), name='WOE').round(decimals=4)))
    output['WOE'] = np.where(np.isinf(output['WOE']), 0, output['WOE'])
    output['IV'] = (((output['Good'] / output['0_TotCnt']) - (output['Bad'] / output['1_TotCnt'])) * output['WOE']).round(decimals=4)
    output['IV'] = np.where(output['IV'] < 0, 0, output['IV'])
    output['IV (total)'] = output['IV'].sum()
    output.index.name = varName
    return output


def draw_histogram(varCol, targetCol, weightCol=None, ntbOut=True, htmlOut=False, bins=50, outFolder='exp'):
    """Draws a histogram for varCol grouped by binary targerCol.

    Args:
        varCol (pd.Series): observation values
        targetCol (pd.Series): binary target
        weightCol (pd.Series, optional): values of weights of observations. Defaults to None.
        ntbOut (bool, optional): . If True outputs plos to notebook. Defaults to True.
        htmlOut (bool, optional): If True outputs to .png file on disk. Defaults to False.
        bins (int, optional): Number of bins for histogram. Defaults to 50.
        outFolder (str, optional): Base folder path for export of file. Defaults to 'exp'.
    """

    targetMask = pd.notnull(targetCol)
    varCol = varCol[targetMask]
    targetCol = targetCol[targetMask]

    notNaNMask = pd.notna(varCol)
    varCol = varCol[notNaNMask]
    targetCol = targetCol[notNaNMask]

    increment = (varCol.max()-varCol.min())/bins
    minimum = varCol.min()
    bins_edges = [minimum + i*increment for i in range(bins+1)]

    if weightCol is None:
        colors = ['slategrey', 'gray', 'green']
        f, ((ax_box, bx_box), (ax_hist, bx_hist)) = plt.subplots(2, 2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
        f.set_figwidth(15)
        sns.boxplot(x=varCol, ax=ax_box, fliersize=1, linewidth=1, color=colors[0])
        sns.histplot(x=varCol, bins=bins_edges, ax=ax_hist, kde=False, color=colors[0])
        ax_box.set(xlabel='')

        data = pd.concat([varCol, targetCol.astype('category')], axis=1)
        sns.boxplot(x=varCol.name, y=targetCol.name, data=data,  ax=bx_box, fliersize=1, orient='h', linewidth=1,
                    palette=colors[1:])
        bx_box.set(xlabel='')

        sns.histplot(x=varCol, hue=data[targetCol.name], bins=bins_edges, ax=bx_hist, kde=False, palette=colors[1:])
        bx_hist.set(ylim=ax_hist.get_ylim(), xlabel='{var} by {target}'.format(var=varCol.name, target=targetCol.name))

    else:
        weightCol = weightCol[targetMask & notNaNMask]

        colors = ['darkgray', 'gray', 'orchid']

        f, (ax_hist, bx_hist) = plt.subplots(1, 2)
        f.set_figwidth(15)
        sns.histplot(x=varCol, ax=ax_hist, bins=bins_edges, weights=weightCol, kde=False, color=colors[0])
        ax_hist.set(xlabel=varCol.name)
        data = pd.concat([varCol, weightCol, targetCol.astype('category')], axis=1)
        sns.histplot(x=data[varCol.name], hue=data[targetCol.name], bins=bins_edges, weights=data[weightCol.name],
                     ax=bx_hist, kde=False, palette=colors[1:])
        bx_hist.set(ylim=ax_hist.get_ylim(), xlabel='{var} by {target}'.format(var=varCol.name, target=targetCol.name))

    if htmlOut:
        try:
            filepath = os.path.join(outFolder, varCol.name)
            plt.savefig('{}_histogram.png'.format(filepath), format='png', bbox_inches='tight')
        except Exception:
            pass

    if ntbOut:
        plt.show()
    plt.clf()
    plt.close()


def draw_frequencies(freqData, altColors=False, ntbOut=True, htmlOut=False, outFolder='exp'):
    """Draws a histogram for categorical data. Grouped by Binary target.

    Args:
        freqData (pd.Dataframe): Data to be plotted.
        altColors (bool, optional): If True uses alternative color scheme. Defaults to False.
        ntbOut (bool, optional): . If True outputs plos to notebook. Defaults to True.
        htmlOut (bool, optional): If True outputs to .png file on disk. Defaults to False.
        outFolder (str, optional): Base folder path for export of file. Defaults to 'exp'.
    """
    if altColors:
        colors = ['darkgray', 'MediumSeaGreen', 'Salmon']
    else:
        colors = ['gray', 'green', 'red']

    f, (ax_abs, ax_bytar) = plt.subplots(1, 2, sharey=True)
    f.set_figwidth(15)
    freqData['cat'] = freqData.index
    sns.barplot(x='cat', y='Count', data=freqData, ax=ax_abs, color=colors[0])
    ax_abs.set(ylabel='')
    ax_abs.set_xticklabels(ax_abs.get_xticklabels(), rotation=90)

    sns.barplot(x='cat', y='Good', data=freqData, ax=ax_bytar, alpha=0.6, color=colors[1])
    sns.barplot(x='cat', y='Bad', data=freqData, ax=ax_bytar, alpha=0.6, color=colors[2])
    ax_bytar.set(ylabel='')
    ax_bytar.set_xticklabels(ax_bytar.get_xticklabels(), rotation=90)

    if htmlOut:
        try:
            filepath = os.path.join(outFolder, freqData.index.name)
            plt.savefig('{}_frequencies.png'.format(filepath), format='png', bbox_inches='tight')
        except Exception as e:
            display(freqData.index.name, e)
    if ntbOut:
        plt.show()
    plt.clf()
    plt.close()


def draw_badrates_woe_iv(binnedData, altColors=False, ntbOut=True, htmlOut=False, outFolder='exp'):
    """Draws plot for badrates, woe and IV values.

    Args:
        binnedData (pd.DataFrame): Data from  calculate functions.
        altColors (bool, optional): If True uses alternative color scheme. Defaults to False.
        ntbOut (bool, optional): . If True outputs plos to notebook. Defaults to True.
        htmlOut (bool, optional): If True outputs to .png file on disk. Defaults to False.
        outFolder (str, optional): Base folder path for export of file. Defaults to 'exp'.
    """
    if altColors:
        colors = ['MediumSeaGreen', 'Salmon', 'RoyalBlue', 'Orange']
    else:
        colors = ['green', 'red', 'darkblue', 'darkorange']

    f, (ax_badrate, ax_iv) = plt.subplots(1, 2)
    f.set_figwidth(15)
    binnedData[['Good(%)', 'Bad(%)']].plot.bar(ax=ax_badrate, stacked=True, color=colors[0:2])
    ax_badrate.set_xlabel('Bad rates')
    ax_badrate.legend(loc=8, bbox_to_anchor=(0.5, 1), ncol=2)

    binnedData['IV'].plot.bar(ax=ax_iv, color=colors[2], label='IV', alpha=0.6, width=0.8)
    ax_iv.tick_params(axis='y', labelcolor=colors[2])
    ax_iv.legend(loc=8, bbox_to_anchor=(0.4, 1), ncol=2)

    ax_woe = ax_iv.twinx()
    binnedData['WOE'].plot.bar(ax=ax_woe, color=colors[3], label='WOE', alpha=0.6, width=0.3)
    ax_woe.tick_params(axis='y', labelcolor=colors[3])
    ax_woe.legend(loc=8, bbox_to_anchor=(0.6, 1), ncol=2)

    ax_iv.set_xlabel('WOE and IV')
    if htmlOut:
        try:
            filepath = os.path.join(outFolder, binnedData.index.name)
            plt.savefig('{}_badrates_woe_iv.png'.format(filepath), format='png', bbox_inches='tight')
        except Exception as e:
            display(binnedData.index.name, e)

    if ntbOut:
        plt.show()
    plt.clf()
    plt.close()


def explore_numerical(varCol, targetCol, weightCol=None, ntbOut=True, htmlOut=False, outFolder='exp', bins=10):
    """Calculates a histogram grouped by binary target as well as bad rate, WOE and IV values for all groups.

    Args:
        varCol (pd.Series): Predictor Series.
        targetCol (pd.Series): Binary target Series.
        weightCol (pd.Series, optional): Weights Series. Defaults to None.
        ntbOut (bool, optional): . If True outputs plos to notebook. Defaults to True.
        htmlOut (bool, optional): If True outputs to .png file on disk. Defaults to False.
        outFolder (str, optional): Base folder path for export of file. Defaults to 'exp'.
        bins (int, optional): Number of bins. Defaults to 10.
    """

    if weightCol is not None:
        altColors = True
        varCol = pd.Series(varCol, name=varCol.name + '_weighted')
        calculatedData = calculate_numerical(varCol, targetCol, weightCol=weightCol)
        quantiles = calculate_quantiles(varCol, targetCol, weightCol=weightCol, bins=bins)
    else:
        altColors = False
        calculatedData = calculate_numerical(varCol, targetCol)
        quantiles = calculate_quantiles(varCol, targetCol, bins=bins)

    if ntbOut:
        display(Markdown('## {}'.format(varCol.name)))
        display(calculatedData)
        display(quantiles[['Good(%)', 'Bad(%)', 'Bad', 'Good', 'TotBadRate', '0_ColPct100', '1_ColPct100', '0_Pct100', '1_Pct100', 'WOE', 'IV']])
    if htmlOut:
        if not os.path.exists(outFolder):
            os.makedirs(outFolder)
        filename = os.path.join(outFolder, varCol.name) + '.html'
        with open(filename, 'w', encoding='utf-8') as f:
            text = "generated: " + datetime.now().strftime("%Y-%m-%d %H:%M") + "<br />\n" + "<br />\n"
            text += "variable name is: " + varCol.name + "<br />\n"
            text += "target name is: " + targetCol.name + "<br />\n" + "<br />\n"
            text += calculatedData.to_html() + "<br />\n"+"<br />"
            text += quantiles.to_html() + "<br />\n"+"<br />"

            text += "<br /> Histogram " + "<br />\n"
            text += ('<img src = ' + varCol.name + '_histogram.png alt ="image not found">\n')

            text += "<br /> Bad rates, WOE and IV " + "<br />\n"
            text += '<img src = ' + varCol.name + '_badrates_woe_iv.png alt ="image not found">\n'
            f.write(text)

    draw_histogram(varCol, targetCol, ntbOut=ntbOut, weightCol=weightCol, htmlOut=htmlOut, outFolder=outFolder)
    draw_badrates_woe_iv(quantiles, altColors=altColors, ntbOut=ntbOut, htmlOut=htmlOut, outFolder=outFolder)


def explore_categorical(varCol, targetCol, weightCol=None, ntbOut=True, htmlOut=False, outFolder='exp'):
    """Calculates a histogram grouped by binary target as well as bad rate, WOE and IV values for all groups.

    Args:
        varCol (pd.Series): Predictor Series.
        targetCol (pd.Series): Binary target Series.
        weightCol (pd.Series, optional): Weights Series. Defaults to None.
        ntbOut (bool, optional): . If True outputs plos to notebook. Defaults to True.
        htmlOut (bool, optional): If True outputs to .png file on disk. Defaults to False.
        outFolder (str, optional): Base folder path for export of file. Defaults to 'exp'.
    """
    if weightCol is not None:
        altColors = True
        varCol = pd.Series(varCol, name=varCol.name + '_weighted')
        calculatedData = calculate_categorical(varCol, targetCol, weightCol=weightCol)
    else:
        altColors = False
        calculatedData = calculate_categorical(varCol, targetCol)

    if varCol.nunique() < 30:
        too_many_values = False
    else:
        too_many_values = True
    if ntbOut:
        if too_many_values:
            display(Markdown('## {}'.format(varCol.name)))
            display(Markdown('Number of unique vales is too big to output to notebook. Table will be exported to HTML if enabled. Plots will not be generated.'))
            ntbOut = False
            
        else:
            display(Markdown('## {}'.format(varCol.name)))
            display(calculatedData)

    if htmlOut:
        if not os.path.exists(outFolder):
            os.makedirs(outFolder)
        filename = os.path.join(outFolder, varCol.name) + '.html'
        with open(filename, 'w', encoding='utf-8') as f:
            text = "generated: " + datetime.now().strftime("%Y-%m-%d %H:%M") + "<br />\n" + "<br />\n"
            text += "variable name is: " + varCol.name + "<br />\n"
            text += "target name is: " + targetCol.name + "<br />\n" + "<br />\n"
            text += calculatedData.to_html() + "<br />\n" + "<br />"
            if too_many_values:
                text += 'Too many values to generate plots.'
            else:
                text += "<br /> Frequencies " + "<br />\n"
                text += ('<img src = ' + varCol.name + '_frequencies.png alt ="image not found">\n')
                text += "<br /> Bad rates, WOE and IV " + "<br />\n"
                text += '<img src = ' + varCol.name + '_badrates_woe_iv.png alt ="image not found">\n'
            f.write(text)
    if not too_many_values:
        draw_frequencies(calculatedData, altColors=altColors, ntbOut=ntbOut, htmlOut=htmlOut, outFolder=outFolder)
        draw_badrates_woe_iv(calculatedData, altColors=altColors, ntbOut=ntbOut, htmlOut=htmlOut, outFolder=outFolder)


def explore_df(df, col_m, col_t, cols_p):
    """Generates a short exploratory report of DataFrame

    Args:
        df (pd.DataFrame): Data to be explored.
        col_m (str): Name of month column
        col_t (str): Name of target column
        cols_p ([str]): List of predictor columns

    Returns:
        [type]: [description]
    """

    pr = []

    pr.append('--------------------------------------------------------------------------------')
    pr.append('\n--------------------------------------------------------------------------------')
    pr.append('\nReport generated: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    pr.append('\nRows count: ' + str(len(df)))

    col_list = [col_m] + [col_t] + cols_p

    for app in col_list:
        if df[app].dtype.name in {'object', 'category'} or df[app].nunique() <= 10:
            pr.append('\n--------------------------------------------------------------------------------')
            pr.append('\n--------------------------------------------------------------------------------')
            pr.append('\n' + str(df[app].groupby(df[app]).count()))
            if df[app].dtype == 'int64' or df[app].dtype == 'float64':
                pr.append('\n------')
                pr.append('\nAvg: ' + str(df[app].mean()))
            pr.append('\n------')
            pr.append('\nNulls: ' + str(df[app].isnull().sum()))
        elif 'int' in df[app].dtype.name or 'float' in df[app].dtype.name:
            pr.append('\n--------------------------------------------------------------------------------')
            pr.append('\n--------------------------------------------------------------------------------')
            pr.append('\n' + str(app))
            pr.append('\n' + str(df[app].quantile(np.linspace(0, 1, 11))))
            pr.append('\n------')
            pr.append('\nMin: ' + str(df[app].min()))
            pr.append('\nMax: ' + str(df[app].max()))
            pr.append('\nAvg: ' + str(df[app].mean()))
            pr.append('\n------')
            pr.append('\nNulls: ' + str(df[app].isnull().sum()))
        else:
            pr.append('\n--------------------------------------------------------------------------------')
            pr.append('\n--------------------------------------------------------------------------------')
            pr.append('\n' + str(app))
            pr.append('\nUnknown datatype')

    st = ''.join(pr)

    return st



def join_explorations(col_names, filename = '_exploration.html', outFolder='exp', weighted=False):
    """Joins html files generated from explore_numerical() and explore_categorical()
    and creates a file with all predictors and table of contents.

    Args:
        col_names (list of str): list of predictors, html files have to exist
        filename (str, optional): name of file to export to. Defaults to '_exploration.html'.
        outFolder (str, optional): where to save the generated file. Defaults to 'exp'.
        weighted (bool, optional): boolean, True for joining files with weighted names. Defaults to False.
    """
    if weighted:
        col_names = [col + '_weighted' for col in col_names]
        filename = filename.replace('.html', '_weighted.html')

    try:
        with open(os.path.join(outFolder, filename), 'w', encoding='utf-8') as file_out:
            file_out.write('''<!DOCTYPE html>\n''')
            file_out.write('''<html lang="en">\n''')
            file_out.write('''<meta charset="utf-8"/>\n''')
            file_out.write('''<a name="table">Table of contents</a></br>\n''')
            for col_name in col_names:
                file_out.write('''<a href="#{col_name}">{col_name}</a></br>\n'''.format(col_name=col_name))
            for col_name in col_names:
                with open(os.path.join(outFolder, col_name + '.html'), 'r', encoding='utf-8') as file_in:
                    file_out.write('''<a name="{col_name}"><h1>{col_name}</h1></a>'''.format(col_name=col_name))
                    file_out.write('''<a href="#table">Return to table of contents</a></br>''')
                    file_out.write(file_in.read())
                    file_out.write('</br>')
        display(HTML('''<a href={} target="_blank">{}</a>'''.format(os.path.join(outFolder, filename), 'Link to Exploration file:  <b>{}</b>'.format(os.path.join(outFolder, filename)))))
    except Exception as e:
        display(e, os.path.join(outFolder, col_name + '.html'))


def nan_share_development(df, col_month, make_images=True, show_images=True, output_path=None):    
    """Returns table with NaN share of each variable in dataframe df in each month (given by col_month).
    Optionally, draws images - either saves them to files or shows them in notebook
    
    Arguments:
        df (pandas.DataFrame): dataframe the nan development should be calculated for
        col_month (str): name of column of df with the month number
    
    Keyword Arguments:
        make_images (boolean): whether images of nan share for each variable should be generated (default: {True})
        show_images (boolean): whether images (if make_images == True) should be shown in Jupyter environment (default: {True})
        output_path (str): which folder should be the table and the images (if make_images == True) saved to. This string should end with slash ("/") (default: {None})
    
    Returns:
        pandas.DataFrame: table with nan shares in time for each variable

    """
    print('Calculating NaN shares...')
    na_tab = df.groupby(col_month).apply(lambda x: x.isnull().mean()).transpose().round(3)
    if output_path is not None:
        na_tab.to_csv(output_path+'nan_share.csv')
    
    if (make_images and (show_images or (output_path is not None))):
        print('Making images...')
        for i,r in tqdm(na_tab.iterrows(), total=na_tab.shape[0]):
            plt.plot(list(r),linewidth=4)
            plt.xticks(range(len(na_tab.columns)),na_tab.columns,rotation=45,fontsize=12)
            plt.ylim(0,1)
            plt.title(i+': NaN share')
            if output_path is not None:
                plt.savefig(output_path+i+'_NaN.png',bbox_inches='tight',dpi=72)
            if show_images:
                plt.show()
            plt.close()
    
    return na_tab


def dynamic_diversity(df, ID):
    """For cases, when user joins two tables with different granularity (e.g. main table with application with some array from vector).
    For each variable, it shows how many different values the variable can have per one application (variable with application granularity is passed as argument ID)
    
    Args:
        df (pandas.DataFrame): data frame the dynamic diversity should be calculated for
        ID (str): name of column with observation (loan application) identifier

    Returns:
        pandas.DataFrame: table with dynamic diversity
    """
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    Max_uni_per_cl=df.groupby(ID).nunique().max()
    Min_uni_per_cl=df.groupby(ID).nunique().min()
    Mean_uni_per_cl=df.groupby(ID).nunique().mean()
    Unique_cnt=df.nunique()
        
    final_table = pd.concat([mis_val, mis_val_percent,Max_uni_per_cl,Min_uni_per_cl, Mean_uni_per_cl,Unique_cnt], axis=1)
    final_table_ren_columns = final_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values', 2 : 'Max unique values per client',  3 : 'Min unique values per client', 4 : 'AVG unique values per client', 5 : '# unique values per column' })
    return final_table_ren_columns


def metadata_table(dt, max_unique_list_len = 10):
    """Prints some metadata about each column - fill share, number of unique values, and some examples of unique values. Useful after feature engineering to decide which features make sense and which can be dropped.
    
    Arguments:
        dt (pandas.DataFrame): dataframe the metada should be generated for
    
    Keyword Arguments:
        max_unique_list_len (int): for each column, a sample of unique values of the column is generated. This parameter says what is the maximal number of unique values to be shown. (default: {25})
    
    Returns:
        pandas.DataFrame: table with metadata
    """
    col_meta = []
    for c in dt.columns:
        unique_list_len = dt[c].nunique()
        if unique_list_len > max_unique_list_len: unique_list_len = max_unique_list_len
        c_entry = {'name':c,
                   'type':dt[c].dtype,
                   'nunique':dt[c].nunique(),
                   'value examples':', '.join(list(dt[c].unique().astype(str)[0:unique_list_len])),
                   'fill pct': 100*dt[c].count()/len(dt[c])}
        col_meta.append(c_entry)
    col_meta_pd = pd.DataFrame(col_meta)
    col_meta_pd['type'] = col_meta_pd['type'].astype('str')
    return col_meta_pd