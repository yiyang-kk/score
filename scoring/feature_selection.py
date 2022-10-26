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

import random
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score



def get_shap_feature_importance(names_columns, shap_values):
    """Simple wrapper which concatenate names and shap importances

    Concatenate names and shap_values in sorted list pair ('name of feat', its shap weight)

    Args:
        names_columns (list of str): names of columns
        shap_values (list of float): list of weights, e.g. shap_values
    
    Returns:
        fi_columns (list of tuple): sorted list of pair [('name of feat', its weight),(),...]
    """
    # in the library SHAP is used .mean(0), but we take .sum(0) so that the numbers are not too small
    fi_shap = np.abs(shap_values).sum(0)

    fi_shap_list = [(names_columns[i], fi_shap[i]) for i in range(len(names_columns))]
    fi_shap_list = sorted(fi_shap_list, key=lambda kv: kv[1], reverse=True)

    return fi_shap_list


def boost_feature_selection(params, df, col_target, fi_columns, col_weight = None, base_columns=[], train_mask=None, test_mask=None,
                            oot_mask=None, n_seed=4, n_fold=4, step=1, boost='xgb'):
    """Feature selection for XGBoost and LightGBM classifier

    This function adding features from fi_columns to the base_columns one by one,
    building xgb/lgb model on each feature set and calculates AUC (for each mask)
    by default: calculates 'cv-auc-mean'

    Args:
        params (dict): XGBoost/LightGBM parameters
        df (pandas.DataFrame): DataFrame with all data (with target feat)
        col_target (str): name of target data feat from df table
        fi_columns (list of tuple): list of tuples ('name of feature', its importance)
            e.g. Use func get_shap_feature_importance(names_columns, shap_values) to form a list
        col_weight (str, optional): name of weight data feat from df table
            Defaults to None.
        base_columns (list, optional): list of columns that will participate in the model initially
            Defaults to [].
        train_mask (pandas.Series, optional): boolean mask for the training set
            Defaults to None.
        test_mask (pandas.Series, optional): boolean mask for the testing/validation set
            Defaults to None.
        oot_mask (pandas.Series, optional): boolean mask for the out of time set
            Defaults to None.
        n_seed (int, optional): the number of seeds for averaging
            Defaults to 4.
        n_fold (int, optional): the number of folds for CV
            Defaults to 4.
        step (int, optional): how many features to add at each step
            Defaults to 1.
        boost (str, optional): {'xgb', 'lgb'}
            'xgb' - XGBoost
            'lgb' - LightGBM
            Defaults to 'xgb'.

        Returns:
            dict: dictionary with elements like : {'NameOfMask-auc-mean' : [list of values of AUC]}

        Notes:
            You can specify only one or two masks
            If you don't use mask function calculates 'cv-auc-mean'
    """

    if boost == 'xgb':
        boosting_classifier = xgb
        customize_data = xgb.DMatrix
        train_auc_mean = 'train-auc-mean'
    elif boost == 'lgb':
        boosting_classifier = lgb
        customize_data = lgb.Dataset
        train_auc_mean = 'auc-mean'
    else:
        warnings.warn(f'Use "xgb" or "lgb" classifier.\nNow, by default boost="xgb"')
        boosting_classifier = xgb

    # mask calculation
    if (train_mask is None) and (test_mask is None) and (oot_mask is None):
        train_mask = np.repeat(True, len(df.index))
    elif (train_mask is None) and (oot_mask is None):
        train_mask = ~test_mask
    elif (train_mask is None) and (test_mask is None):
        train_mask = ~oot_mask
    elif train_mask is None:
        train_mask = ~(test_mask | oot_mask)

    # sort features by weights and drop features from base_columns
    fi_columns = sorted(fi_columns, key=lambda kv: kv[1], reverse=True)

    cv_auc_mean, test_auc, oot_auc = [], [], []
    name_cols = [x[0] for x in fi_columns]
    b_columns = base_columns.copy()

    # preliminary calculation for optimization
    df_train_mask = df[train_mask]
    df_train_mask_target = df_train_mask[col_target]
    if test_mask is not None:
        df_test_mask_target = df[test_mask][col_target]
    if oot_mask is not None:
        df_oot_mask_target = df[oot_mask][col_target]

    if col_weight is not None:
        df_train_mask_weight = df_train_mask[col_weight]
        if test_mask is not None:
            df_test_mask_weight = df[test_mask][col_weight]
        if oot_mask is not None:
            df_oot_mask_weight = df[oot_mask][col_weight]
    else:
        df_train_mask_weight = None
        df_test_mask_weight = None
        df_oot_mask_weight = None

    # main loop
    for i in tqdm(range(0, len(name_cols), step)):
        b_columns += name_cols[i:i + step]

        # calculating the number of trees and calculating cv-auc-mean
        data = customize_data(df_train_mask[b_columns], df_train_mask_target, weight=df_train_mask_weight)
        seed = random.randint(0, 1000)
        cv_res = boosting_classifier.cv(params, data, early_stopping_rounds=10, num_boost_round=10000,
                                        nfold=n_fold, stratified=True, seed=seed, verbose_eval=False)
        cv_auc_mean.append(max(cv_res[train_auc_mean]))

        # averaging over n_seed seed
        boost_pred = 0
        for i_seed in range(n_seed):

            seed = random.randint(0, 1000)
            params['seed'] = seed
            booster = boosting_classifier.train(params, data, num_boost_round=len(cv_res[train_auc_mean]))

            if boost == 'xgb':
                data_to_predict = customize_data(df[b_columns])
            else:
                data_to_predict = df[b_columns]

            boost_pred += 1 - booster.predict(data_to_predict)

        boost_pred /= n_seed

        # calculating test-auc-mean and oot-auc-mean
        if test_mask is not None:
            auc_score = roc_auc_score(df_test_mask_target, -boost_pred[test_mask], sample_weight=df_test_mask_weight)
            test_auc.append(auc_score)
        if oot_mask is not None:
            auc_score = roc_auc_score(df_oot_mask_target, -boost_pred[oot_mask], sample_weight=df_oot_mask_weight)
            oot_auc.append(auc_score)

    aucs = {'cv-auc-mean': cv_auc_mean}
    if test_mask is not None:
        aucs['test-auc-mean'] = test_auc
    if oot_mask is not None:
        aucs['oot-auc-mean'] = oot_auc

    return aucs


def plot_feature_selection(fi_columns, aucs, step=1):
    """Plotting dependence AUC from adding feature.

    e.g. use aucs from func xgb_feature_selection

    Args:
        fi_columns (list): list of tuples ('name of feature feat', its weight)
            e.g. Use func get_shap_feature_importance(names_columns, shap_values) to form a list
        aucs (dict): dict from boost_feature_selection function
        step (int, optional): how many features to add at each step. Defaults to 1.

    Returns:
        matplotlib.axes._subplots.AxesSubplot: The matplotlib axes handle.
    """

    fi_columns = sorted(fi_columns, key=lambda kv: kv[1], reverse=True)

    fi_col_wout_wight = [f"{x[0]} ({int(x[1])})" for x in fi_columns[::step]]

    fig, ax = plt.subplots(figsize=(20, 10))

    for auc_list in aucs.items():
        plt.xticks(range(len(fi_col_wout_wight)), fi_col_wout_wight, rotation='vertical', fontsize=15)
        plt.plot(auc_list[1], linewidth=2.0, label=auc_list[0])

    plt.title('Feature selection results', fontsize=20)
    plt.legend(loc='lower right', bbox_to_anchor=(1, 0.5), fontsize=20)
    plt.ylabel('AUC', fontsize=20)
    plt.grid(True)
    
    return ax
