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


from sklearn.metrics import roc_curve, auc
from scoring.metrics import lift
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# from IPython import get_ipython
# get_ipython().magic('matplotlib inline')
sns.set(palette='muted')

def draw_ROC_curve(target, score, title="", draw=True):
    fpr, tpr, thresholds = roc_curve(target, score, drop_intermediate=False)
    
    AUC = auc(fpr, tpr)
    
    if draw:
        X = np.linspace(0, 1, 100)
        plt.figure(figsize = (8,8))
        plt.title(title)
        plt.plot(fpr, tpr, linewidth=2.5, label='ROC-curve, area = %0.5f' % AUC)
        plt.plot(X, X, 'r--', label='random_classifier')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        plt.show()
    
    
    #returns AUC score
    return AUC

def gini_score(target, score):
    return (2*draw_ROC_curve(target, score, draw=False) - 1)


def gini_monthly(df, months_column, base, target, score_column):
    '''Returns dataframe which contains ginis and counts for every month 
    if all base==0 for month, gini = 0
    '''
    tmp_1 = df.loc[(df[score_column].notnull())]
    m = np.sort(np.unique(tmp_1[months_column]))
    counts = len(tmp_1.loc[tmp_1[base] == 1])
    gini = gini_score(1 - tmp_1.loc[tmp_1[base] == 1, target], tmp_1.loc[tmp_1[base] == 1, score_column])
    overall_df = pd.DataFrame()
    overall_df['months'] = ['%s - %s' % (m[0], m[-1])]
    overall_df['ginis_%s' % target[4:]] = [gini * 100]
    overall_df['counts_%s' % target[4:]] = [counts]
    overall_df.index = ['overall']
    counts_m = []
    gini_m = []
    for i in m:
        tmp = df.loc[(df[months_column] == i) & (df[base] == 1) & (df[score_column].notnull())]
        counts_m.append(len(tmp))
        #print len(tmp[tmp[target] == 1])
        if len(tmp[tmp[target] == 1])==0:
            gini_m.append(0)
        else:
            gini_m.append(gini_score(1 - tmp[target], tmp[score_column]))
    
    res = pd.DataFrame()
    res['months'] = m
    res['ginis_%s' % target[4:]] = [k * 100 for k in gini_m]
    res['counts_%s' % target[4:]] = counts_m
    res = res.append(overall_df)
    return res

def lift_monthly(df, months_column, base, target, score_column, perc=10.):
    '''Returns dataframe which contains lifts perc% and counts for every month 
    if all base==0 for month, gini = 0
    '''
    tmp_1 = df.loc[(df[score_column].notnull())]
    m = np.sort(np.unique(tmp_1[months_column]))
    counts = len(tmp_1.loc[tmp_1[base] == 1])
    lift_score = lift(tmp_1.loc[tmp_1[base] == 1, target], tmp_1.loc[tmp_1[base] == 1, score_column], perc)
    overall_df = pd.DataFrame()
    overall_df['months'] = ['%s - %s' % (m[0], m[-1])]
    overall_df['lift_%.2f%%_%s' % (perc, target[4:])] = [lift_score]
    overall_df['counts_%s' % target[4:]] = [counts]
    overall_df.index = ['overall']
    counts_m = []
    lift_m = []
    for i in m:
        tmp = df.loc[(df[months_column] == i) & (df[base] == 1) & (df[score_column].notnull())]
        counts_m.append(len(tmp))
        #print len(tmp[tmp[target] == 1])
        if len(tmp[tmp[target] == 1])==0:
            lift_m.append(0)
        else:
            lift_m.append(lift(tmp[target], tmp[score_column], perc))
    
    res = pd.DataFrame()
    res['months'] = m
    res['lift_%.2f%%_%s' % (perc, target[4:])] = [k for k in lift_m]
    res['counts_%s' % target[4:]] = counts_m
    res = res.append(overall_df)
    return res

def plot_gini_monthly(months, ginis, counts, labels,
                      title='Gini comparison',
                      y_bot=None, y_top=None, size=(8, 6), ylabel='Gini'):
    '''
    Draws ginis plots with count bars
    Args:
        months (array like): arraylike, list of months used as xticks
        ginis (array like): list of monthly gini arrays
        counts (array like): arraylike, data fro count bars
        labels (array like): arraylike, must be the same size as ginis
        title (str, optional):
        y_bot (float, optional):
        y_top (float, optional):
        size (tuple, optional):
        ylabel (str, optional):
    '''
    
    plt.figure(figsize=size)
    plt.title(title, fontsize=16)
    plt.xticks(range(len(months)), months, rotation=45)
    plt.xlabel('Months', fontsize=13)
    plt.grid(zorder=1)
    
    ax_01 = plt.subplot()
    ax_01.bar(range(len(months)), counts, color='lightgray', zorder=2)
    
    ax_02 = ax_01.twinx()
    for i in range(len(ginis)):
        ax_02.plot(range(len(ginis[i])), ginis[i], linewidth=3.0, label = labels[i])
    ax_02.legend(loc='best')
    ax_02.set_ylim(bottom = y_bot, top = y_top)
    ax_01.set_ylabel(u'Sum base', fontsize=13)
    ax_02.set_ylabel(ylabel, fontsize=13)
    
    #plt.imshow(ax_02, cmap=plt.get_cmap('Set3'))
    
    plt.show()
    