
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


# coding: utf-8

# from IPython import get_ipython
# get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score

from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from pandas.api.types import is_numeric_dtype

def psi_in_time(data, data_expected, pred, col_month, q=10, missing_value=0.0001, plot=True, show=True, output_folder=None):
    """Calculates PSI (Population Stability Index) of a variable by month. Reference data set (where the default distribution of the variable is taken from) is passed as one of the arguments.

    Args:
        data (pd.DataFrame): data to calculate PSI on
        data_expected (pd.DataFrame): reference data where reference distribution is taken from
        pred (str): name of variable to calculate PSI of
        col_month (str): name of column with number of month (PSI is calcualted for each month separately)
        q (int, optional): Number of categories to split variable pred to during the PSI calculation. Defaults to 10.
        missing_value (float, optional): Value to fill missings with. Defaults to 0.0001.
        plot (bool, optional): Plot PSI chart? Defaults to True.
        show (bool, optional): Show the plotted chart on output? Defaults to True.
        output_folder (str, optional): Where to save the results. Defaults to None.

    Returns:
        pd.DataFrame: table with PSI values
    """

    data_stability_expected = pd.DataFrame()
    data_stability = pd.DataFrame()

    if is_numeric_dtype(data[pred]):
        bins = np.unique(np.percentile(data_expected.loc[data_expected[pred].notnull(), pred], np.linspace(0, 100, q+1)))
        bins = [-np.inf]+sorted(list(bins))[1:-1]+[np.inf]
        data_stability_expected[pred] = pd.cut(data_expected[pred], bins=bins).values.add_categories('missing').fillna(value='missing')
        data_stability[pred] = pd.cut(data[pred], bins=bins).values.add_categories('missing').fillna(value='missing')
    else:
        bins = data[pred].unique()
        data_stability_expected[pred] = data_expected[pred].astype('category').values.add_categories('missing').fillna(value='missing')
        data_stability[pred] = data[pred].astype('category').values.add_categories('missing').fillna(value='missing')

    data_stability_expected = data_stability_expected.set_index(data_expected.index)
    data_stability_expected['expected'] = data_stability_expected.index
    data_stability_expected = data_stability_expected.groupby(pred).count().T
    data_stability_expected.columns = list(data_stability_expected)
    data_stability_expected['All'] = data_stability_expected.sum(axis=1)
    data_stability_expected
    
    data_stability = data_stability.set_index(data.index)
    data_stability['index'] = data_stability.index
    data_stability[col_month] = data[col_month]    

    pivot = pd.pivot_table(data_stability,
                           values='index', 
                           index=col_month, 
                           columns=pred, 
                           aggfunc='count', 
                           margins=True, 
                           dropna=False, 
                           margins_name='All')    

    pivot = pivot.append(data_stability_expected)
    pivot = pivot.apply(lambda x: x/x['All'], axis=1)
    pivot.loc['expected',:] = pivot.loc['expected',:].replace(0, missing_value).fillna(missing_value)
    pivot = pivot.drop('All', axis=1)
    pivot = pivot.drop('All', axis=0)


    psi = ((pivot-pivot.loc['expected',:]) * np.log(pivot/pivot.loc['expected',:])).sum(axis=1)
    # psi2 = pivot.apply(lambda x: entropy(x.fillna(missing_value), pivot.loc['expected',:]) + entropy(pivot.loc['expected',:], x.fillna(missing_value)), axis=1)
    bhattacharyya_metric = np.sqrt(1-np.sqrt(pivot*pivot.loc['expected',:]).sum(axis=1))
    jensenshannon_metric = pivot.apply(lambda x: jensenshannon(x.fillna(0), pivot.loc['expected',:]), axis=1)


    pivot['PSI'] = psi
    pivot['bhattacharyya'] = bhattacharyya_metric
    pivot['jensenshannon'] = jensenshannon_metric

    if plot and len(pivot) > 0:
        
        plt.figure(figsize = (10,6))
        plt.title(pred)
        plt.plot(pivot.drop(['expected'])['PSI'], color='b', label='PSI')
        plt.plot(pivot.drop(['expected'])['bhattacharyya'], color='r', label='Bhattacharyya')
        plt.plot(pivot.drop(['expected'])['jensenshannon'], color='y', label='Jensen-Shannon')
        plt.axhline(0.1, linestyle='--', color='y')
        plt.axhline(0.25, linestyle='--', color='r')
        plt.axhline(0.25, linestyle='--', color='b')
        # plt.xticks(pivot.drop(['expected']).index, [col.strftime('%Y-%m') for col in pivot.drop(['expected']).index], rotation=90)
        plt.xticks(pivot.drop(['expected']).index, [col for col in pivot.drop(['expected']).index], rotation=90)
        plt.legend(loc='best')
        
   
        if output_folder is not None:
            plt.savefig(output_folder+'/psi_'+str(pred)+'.png', format='png', dpi=72, bbox_inches='tight')
        if show:
            plt.show()   
        
        plt.close() 
    
    return pivot


def psi(x, y, bins=10):
    """
    PSI metric for month y relatively to the month x
    """
    bins=np.percentile(x, np.linspace(0,100,bins))
    
    freqs,bins=np.histogram(x, bins=bins)
    freqs=freqs*1.0/np.sum(freqs)
    
    freqs2, _=np.histogram(y, bins)
    freqs2=freqs2*1.0/np.sum(freqs2)
    
    ps=np.sum((freqs-freqs2)*np.log(freqs/freqs2))
    return ps



def psi_comparison_chart(months, psi_array, labels_array, title='PSI', size=(12, 8)):
    """Draws histogram of psi:
    
    Args:
        months (array like): months for which psi is calculated
        psi_array (array like): array of psi arrays, every psi array contains psi fro each month from months
        labels_array (array like): array of strings - labels for every psi array on histogram
        title (str, optional): default 'PSI', title of the histogram
        size (tuple, optional): default (12,8), size of the histogram
    """
    #if len(psi_array) == 1:
    #    psi_chart(months, psi_array[0], title=labels_array[0])
    #else:
    plt.figure(figsize=size)
    plt.grid(zorder=0)
    X = np.linspace(0, len(months), len(months))
    X1 = np.linspace(0, len(months) + 1, len(months) + 1)
    for i in range(len(psi_array)):
        plt.bar(X - (1/float(len(psi_array)))*i, psi_array[i], width=1/float(len(psi_array)), label=labels_array[i],
                zorder=3)
    #plt.bar(X, psi2,  width=0.5, color='r', label=label2, zorder=3)
    plt.plot(X1, [0.1]*(len(X) + 1), 'black')
    plt.plot(X1, [0.25]*(len(X) + 1), 'r')

    plt.title(title, fontsize=18)
    plt.xticks(X, months, rotation=45)
    plt.xlim(0, len(months)+0.5)
    plt.ylim(0, 0.3)
    plt.xlabel('Months', fontsize=13)
    plt.ylabel('PSI', fontsize=13)
    plt.legend()



def psi_hist(data, scores, months, month_col, pivot=0, score_names=None, title="PSI", bins=10):
    """
    Args:
        data (pandas.DataFrame): must contain columns for every score from scores and months_col column
        scores (array like): array of score names in data dataframe for which psi will be calculated
        months (array like): months for which psi is calculated
        months_col (str):  name of the column in data dataframe which contains month for each row
        score_names (array of strings, optional): default None,   labels for histogram, if None then index number of score will be used as label
        title: (str, optional): title of the histogram
    """
    
    psi_results = []
    for s in scores:
        psi_s = []
        x = data.loc[(data[month_col] == months[pivot]) &
                     (data[s].notnull()), s]
        for m in months:
            y = data.loc[(data[month_col] == m) &
                         (data[s].notnull()), s]
            psi_s.append(psi(x, y, bins=bins))
        psi_results.append(psi_s)
    if not score_names:
        score_names = range(len(scores))
    psi_comparison_chart(months, psi_results, score_names, title)
    




