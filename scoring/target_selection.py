
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


import numpy as np
import pandas as pd
import xgboost as xgb
from scoring.grouping import Grouping
from sklearn.metrics import roc_auc_score

class XgbTargetSelection():
    """
    Helps to choose the best targets if from multiple candidates. Use fit() method:

    - for each target, it trains xgboost model (all categorical predictors are grouped and WOE-transformed before the
      boosting) using training and validation (for stopping) set
    - then Gini is measured for each such model using all possible targets and testing datasets

    So at the end, we have #targets x #targets Gini values in a matrix. Then the analyst can choose which target
    (for training) is the best (usually because the Gini of its model is high also for the other targets)
        
    Args:
        xgb_params (dict, optional): dictionary of params that will be used by xgboost.train() method
        (default: {'eta': 0.1, 'max_depth': 3, 'objective': 'binary:logistic', 'eval_metric': 'auc', 'min_child_weight': 30, 'subsample': 0.85})
            (default: None)
        num_boost_round (int, optional): max. number of trees in xgboost model (default: 500)
        early_stopping_rounds (int, optional): early stopping parameter for xgboost training - if in such number steps the eval metric does not increase, the training stops (default: 25)
        group_count (int, optional): number of groups to be created from categorical variables for WOE transformation (default: 5)
        min_samples_cat (int, optional): minimal size of a group in grouping of categorical variables for WOE transformation (default: 200)
    """
    
    def __init__(self, xgb_params=None, num_boost_round=500, early_stopping_rounds=25, group_count=5, min_samples_cat=200):
        """
        Initilization.
        """
        
        self.xgb_params = {'eta': 0.1,
          'max_depth': 3,
          'objective': 'binary:logistic',
          'eval_metric': 'auc',
          'min_child_weight': 30,
          'subsample': 0.85}
        if xgb_params is not None:
            for p in xgb_params:
                self.xgb_params[p] = xgb_params[p]
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.group_count = group_count
        self.min_samples_cat = min_samples_cat
        
    def fit(self, data_train, data_valid, data_test, target_definitions, cols_pred_num, cols_pred_cat):
        """
        Fitting of all xgboost models and calculating their Gini. Their is a loop through all targets from target_definitions parameter to train the model,
        and a nested loop through the same targets again to measure Gini of each model using each target on testing dataset.
        
        Arguments:
            data_train (pd.DataFrame): training data set, used for xgboost training
            data_valid (pd.DataFrame): validation data set, used for xgboost early stopping
            data_test (pd.DataFrame): testing data set, will be scored and used for Gini evaluation
            target_definitions (list of dict): List of dictionaries in format `{['target': str, 'base': str},]`. Each dictionary corresponds to one target to try and should have two entries. Name of target column and name of corresponding base column.
            cols_pred_num (list of str): names of numerical predictors. Will enter xgboost as they are.
            cols_pred_cat (list of str): names of categoricl predictors. Will be grouped and WOE transformed before boosting.
        
        Returns:
            pd.DataFrame: Gini values on test data set considering various targets used for training (rows) and various targets used for Gini measurements (columns)
        """
    
        self.cols_pred_num = cols_pred_num
        self.cols_pred_cat = cols_pred_cat
        
        self.ginis = []
        
        for target_model in target_definitions:
            scored_test = self.xgb_onemodel(data_train[data_train[target_model['base']]==1], 
                                            data_valid[data_valid[target_model['base']]==1], 
                                            data_test,
                                            target_model['target'])
            for target_gini in target_definitions: 
                gini = 2 * roc_auc_score(data_test[data_test[target_gini['base']]==1][target_gini['target']],
                                         scored_test[data_test[target_gini['base']]==1]) - 1
                self.ginis.append({'TRAIN TARGET': target_model['target'],
                                   'GINI TARGET': target_gini['target'],
                                   'GINI': gini})
                
        self.ginis = pd.DataFrame(self.ginis).pivot(index='TRAIN TARGET', columns='GINI TARGET')['GINI']
        
        return self.ginis
    
    def xgb_onemodel(self, data_train, data_valid, data_test, target):
        """
        Trains one xgboost model, called by fit(). Returns scored testing data set which is then used in fit() to measure Gini.
        
        Args:
            data_train (pd.DataFrame): training data set - must be already restricted to observations which are in base for target which is used for training
            data_valid (pd.DataFrame): validation data set - must be already restricted to observations which are in base for target which is used for training
            data_test (pd.DataFrame): testing data set - whole testing sample, will be returned scored
            target (str): name of the target column to be used in model training
        
        Returns:
            pd.DataFrame: scored data_test
        """
    
        y_train = data_train[target]
        y_valid = data_valid[target]
    
        grouping_train = data_train[self.cols_pred_cat]
    
        print('Training Grouping for target',target)
        cat_grouping = Grouping(columns = [],
                        cat_columns = sorted(self.cols_pred_cat),
                        group_count = self.group_count,
                        min_samples_cat = self.min_samples_cat) 
        cat_grouping.fit(grouping_train,
                         y_train)
    
        print('Transforming Grouping for target',target)
        cat_train_transformed = cat_grouping.transform(data_train[self.cols_pred_cat])
        data_train = data_train.join(cat_train_transformed)
        cat_valid_transformed = cat_grouping.transform(data_valid[self.cols_pred_cat])
        data_valid = data_valid.join(cat_valid_transformed)
        cat_test_transformed = cat_grouping.transform(data_test[self.cols_pred_cat])
        data_test = data_test.join(cat_test_transformed)
    
        cols_final = []
        for c in cat_train_transformed.columns: cols_final.append(c)
        for c in self.cols_pred_num: cols_final.append(c)
    
        X_train = data_train[cols_final]
        X_valid = data_valid[cols_final]
    
        evals_result = {}
    
        booster = xgb.train(params = self.xgb_params,
                            dtrain = xgb.DMatrix(X_train, y_train),
                            num_boost_round = self.num_boost_round,
                            early_stopping_rounds = self.early_stopping_rounds,
                            evals = ((xgb.DMatrix(X_train, y_train),'train'),
                                     (xgb.DMatrix(X_valid, y_valid),'validate')
                                    ), 
                            evals_result = evals_result,
                            verbose_eval = 0)
    
        print('Transforming Booster for target',target)
        scored_test = booster.predict(xgb.DMatrix(data_test[cols_final]), ntree_limit=booster.best_ntree_limit)
           
        return scored_test
