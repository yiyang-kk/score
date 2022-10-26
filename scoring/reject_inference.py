
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


import random
import numpy as np
import pandas as pd
from sklearn.utils import check_array, as_float_array, assert_all_finite
from sklearn.utils.validation import check_is_fitted, check_consistent_length, column_or_1d
from IPython.display import display, Markdown
import matplotlib.pyplot as plt

class RejectInferenceCharts():
    """
    Class for displaying reject inference charts and potential issues in a dataset.
    
    Args:
        target (str): name of target variable
        reject (str): name of reject target variable
        predictors (list of str): list of predictors where we want to identify potential reject inference
        weight (str, optional): name of weight variable. If not defined, then weights = 1 are used for each observation. Defaults to None.
        max_woe (float, optional): censoring threshold for WoE. If any abs(calculated WoE) > max_woe, then it will be censored by this value. Defaults to 10.

    Properties:
        predictorinfo\_ (list of dictionaries): Created by fit() method. For each predictor, there is a dictionary with data needed for the output
   
    Example:
        >>> rejectinference = RejectInferenceCharts('TARGET', 'REJECT_FLAG', ['X1','X2','X3'], 'WEIGHT')
        >>> rejectinference.fit(data1ing, data_data2ation)
        >>> rejectinference.display(output_notebook = True, output_folder = 'rejectInference')
    """
    
    def __init__(self, target, reject, predictors, weight=None, max_woe=10):
        """
        Initialization method.
        """
        self.target = target
        self.reject = reject
        self.predictors = list(predictors)
        if weight is not None:
            self.weight = weight
        else:
            # if weight column is not defined, we will create a synthetic one later, it will be called 'w__'
            self.weight = 'w__'
        self.max_woe = max_woe
        self.woe_postfix = '_WOE'
    
    def __woe(self, y, y_full, w, w_full, smooth_coef=0.001, max_woe=10):
        """
        This method returns Weight of Evidence value of a subset.
        This value might be adjusted by weights, smoothing and censoring.
        
        Args:
            y (array): array of 0 and 1 (might also include NaN) representing targets within the subset
            y_full (array): array of 0 and 1 (might also include NaN) representing targets within whole set
            w (array): array of non-negative numbers (might include NaN on the same places as y) representing weights of observations in the subset
            w_full (array): array of non-negative numbers (might include NaN on the same places as y_full) representing weights of observations in whole set
            smooth_coef (float, optional): smoothing coefficient, the larger it is, the closer to zero the WoE value will be. Defaults to 0.001.
            max_woe (float, optional): censoring threshold, WoE which is in absolute value larger than this are censored to this number (with corresponding signum). Defaults to 10.
        
        Returns:
            float. Weight of Evidence in subset represented by y in whole set represented by y_full.
        """
        # treat NaNs: delete missing targets and impute missing weights by 1
        w = w[~np.isnan(y)]
        w[np.isnan(w)] = 1
        y = y[~np.isnan(y)]
        w_full = w_full[~np.isnan(y_full)]
        w_full[np.isnan(w_full)] = 1
        y_full = y_full[~np.isnan(y_full)]
        # logarithm of odds ratio in subset
        if sum(w) == 0:
            return np.nan
        lnodds = np.log( (sum((1-y)*w)/sum(w) + smooth_coef) / (sum(y*w)/sum(w) + smooth_coef) )
        # logarithm of odds ratio in whole set
        lnodds_full = np.log( (sum((1-y_full)*w_full)/sum(w_full) + smooth_coef) / (sum(y_full*w_full)/sum(w_full) + smooth_coef) )
        # weight of evidence = difference between the two logarithms of odds ratios
        woe = lnodds - lnodds_full
        # censoring in case woe gone too wild
        if np.absolute(woe) > max_woe:
            woe = np.sign(woe) * max_woe
        return woe
    
    def __weightedRatio(self, y, y_full, w, w_full, inf_imputation=0):
        """
        This method returns ratio of target within the subset and target within whole set.
        This value might be adjusted by weights.
        
        Args:
            y (array): array of 0 and 1 (might also include NaN) representing targets within the subset
            y_full (array): array of 0 and 1 (might also include NaN) representing targets within whole set
            w (array): array of non-negative numbers (might include NaN on the same places as y) representing weights of observations in the subset
            w_full (array): array of non-negative numbers (might include NaN on the same places as y_full) representing weights of observations in whole set
            inf_imputation (float, optional): number which will be used to impute values where denominator (weighted sum of target within whole set) is zero. Defaults to 0.
            
        Returns:
            float. Ratio of weighted sum of y and y_full.
        """
        # treat NaNs: delete missing targets and impute missing weights by 1
        w = w[~np.isnan(y)]
        w[np.isnan(w)] = 1
        y = y[~np.isnan(y)]
        w_full = w_full[~np.isnan(y_full)]
        w_full[np.isnan(w_full)] = 1
        y_full = y_full[~np.isnan(y_full)]
        # weighted sum of target in subset = numerator
        numerator = sum(np.multiply(y,w))
        # weighted sum of target in whole set = denomitor
        denominator = sum(np.multiply(y_full,w_full))
        if denominator == 0:
            # treat division by zero by imputation of value in case denominator == 0
            ratio = inf_imputation
        else:
            # final value is ratio of numeratior and denominator
            ratio = numerator / denominator
        return ratio
    
    def __groupDataset(self, dt, var):
        """
        This method groups dataset by a certain variable (name of it is given by argument var) and data1/data2ation sample. Within each group, it calculates four values:
            - WoE of target
            - WoE of reject target
            - ratio of target (group/whole set)
            - ratio of reject target (group/whole set)
            
        Args:
            dt (pandas.DataFrame): dataset, must include the following columns: self.sample, self.target, self.reject, var
            var (str): name of variable which is used to group the set by
            
        Returns:
            pandas.DataFrame: The data set after grouping and aggregations. It has 4 or 8 columns (depending on if data2ation sample exists).
        """
        # groupbing the data by predictor and sample. for each of the group WoEs and ratios will be calculated
        grouped = dt.groupby([var,self.sample])
        twoe = grouped.apply(lambda grp: self.__woe(y=grp[self.target], y_full=dt[self.target], w=grp[self.weight], w_full=dt[self.weight], max_woe=self.max_woe))
        rwoe = grouped.apply(lambda grp: self.__woe(y=grp[self.reject], y_full=dt[self.reject], w=grp[self.weight], w_full=dt[self.weight], max_woe=self.max_woe))
        t = grouped.apply(lambda grp: self.__weightedRatio(y=grp[self.target], y_full=dt[self.target], w=grp[self.weight], w_full=dt[self.weight]))
        r = grouped.apply(lambda grp: self.__weightedRatio(y=grp[self.reject], y_full=dt[self.reject], w=grp[self.weight], w_full=dt[self.weight]))
        # now we have 4 arrays (WoE of targer, WoE of reject, ratio of target, ratio of reject), we give them proper names
        twoe.rename(self.target+self.woe_postfix, inplace=True)
        rwoe.rename(self.reject+self.woe_postfix, inplace=True)
        t.rename(self.target, inplace=True)
        r.rename(self.reject, inplace=True)
        # concatenating the arrays into one data frame and then pivoting the sample from rows to columns (unstack method)
        grp = pd.concat([t, r, twoe, rwoe], axis=1).unstack(level=1)
        # casting the index as string because it's needed for consistency of charts
        grp.index = grp.index.astype(str)
        return grp
    
    def __textOutput(self, var_name, inf_data1, inf_data2, instability):
        """
        This method creates the text output with warnings.
        
        Args:
            var_name (str): variable name
            inf_data1 (boolean): if there is reject inference in data1 sample
            inf_data2 (boolean): if there is reject inference in data2ation sample
            instability (boolean): if there is inter-sample WoE ordering instability
        """
        outputtext = ''
        if inf_data1 and self.data2exist:
            # we have two samples and there is reject inference in data1ing sample
            outputtext = outputtext + str(var_name) + ': Possible reject Inference in data1 sample\n'
        if inf_data1 and (not self.data2exist):
            # we have one sample and there is reject inference in it
            outputtext = outputtext + str(var_name) + ': Possible reject Inference\n'
        if inf_data2:
            # we have two samples and there is reject inference in data2ation sample
            outputtext = outputtext + str(var_name) + ': Possible reject Inference in data2 sample\n'
        if instability:
            # we have two samples and WoE of target differs between them
            outputtext = outputtext + str(var_name) + ': WOE order difference between data1 and data2 samples\n'
        return outputtext
    
    def fit(self, data1, data2=None):
        """
        This method is called by the user as first. It calculates the WoEs and ratios for target and reject target and saves them into internal data structure which is used in display() method.
        
        Args:
            data1 (pandas.DataFrame): first data sample. Must include columns defined in arguments of __init__() method
            data2 (pandas.DataFrame, optional): second data sample (for verification). Must include columns defined in arguments of __init__() method. Defaults to None.
        """
        dt = pd.DataFrame(data1)
        # because we will append data1 and data2ation set together, we add new column defining the data1/data2ation split
        self.sample = 's__'
        dt[self.sample] = 'data1'
        if data2 is not None:
            # appending data2ation sample to data1 sample, creating boolean variable saying whether there is data2ation sample
            self.data2exist = True
            dt = dt.append(data2,sort=False)
            dt.loc[pd.isnull(dt[self.sample]),self.sample] = 'data2'
        else:
            # no data2ation sample exists, so we create boolean variable telling us so
            self.data2exist = False
        if self.weight not in dt:
            # adding weight column if it does not exist in the data
            dt[self.weight] = 1
        self.predictorinfo_ = []
        for p in self.predictors:
            # group dataset by each predictor and then try to sort it by target and reject ratio.
            # if the sorted orders are different, it might be reject inference, so we create a boolean variable telling us so.
            grp = self.__groupDataset(dt[[self.sample,p,self.target,self.reject,self.weight]],p)
            bytargett = grp.sort_values([(self.target,'data1'),(self.reject,'data1')])
            byrejectt = grp.sort_values([(self.reject,'data1'),(self.target,'data1')])
            if sum(bytargett.index != byrejectt.index)>0:
                inferencet = True
            else:
                inferencet = False
            if self.data2exist:
                # if data2ation sample exists, we do the same test as we did with data1 sample
                bytargetv = grp.sort_values([(self.target,'data2'),(self.reject,'data2')])
                byrejectv = grp.sort_values([(self.target,'data2'),(self.reject,'data2')])
                if sum(bytargetv.index != byrejectv.index)>0:
                    inferencev = True
                else:
                    inferencev = False
                # additionaly, we check the target sorting on data1 and data2ation and compare them
                # if the orders are different, then our WoE values from data1ing sample might be overfitted
                # so we create another warning
                if sum(bytargett.index != bytargetv.index)>0:
                    instability = True
                else:
                    instability = False
            else:
                inferencev = False
                instability = False
            # appending all the information to a list of dictionaries which will be used for visualisation later
            bytargett.index = bytargett.index.astype(np.float)
            bytargett.sort_index(inplace=True)
            bytargett.index = bytargett.index.astype(str)
            self.predictorinfo_.append({'predictor':p,
                                      'grp':bytargett,
                                      'inference_data1':inferencet,
                                      'inference_data2':inferencev,
                                      'instability':instability})    
        print('Reject inference analysis prepared. Use method display() to show the results.')
        return
    
    def display(self, predictors_subset = None, issues_only = False, tables = True, charts = True, warnings = True, output_notebook = True, output_folder = None):
        """
        This methods is called after fit(). It displays the results in a way defined by the user in kwargs.
        
        Args:
            predictors_subset (list, optional): if the user dont want to display result for all predictors which they defined when calling __init__(), they might define subset here. Defaults to None.
            issues_only (boolean, optional): display output only for such predictors which have some problems (as reject inference or inter-sample instability of WoE ordering). Defaults to False.
            tables (boolean, optional): True if user wants to display complete tables with WoEs and ratios generated by fit() method. Defaults to True.
            charts (boolean, optional): True if user wants to display charts with WoEs to visualise the potential issues. Defaults to True.
            warnings (boolean, optional): True if user wants to display explicit warnings of problems found. Defaults to True.
            output_notebook (boolean, optional): True if user wants the output to be drawn into the Jupyter Notebook. False if user wants the output to be drawn to output folder. Defaults to True.
            output_folder (str, optional): if specified, the output is also saved as files into the specified folder. Defaults to None.
        """
        check_is_fitted(self, 'predictorinfo_')
        # we decide whether we will create the output for all predictors defined in __init__(), or if user defined a subset now
        if predictors_subset is not None:
            predictors = list(predictors_subset)
        else:
            predictors = self.predictors
        # if the user wants to output of warnings into file, we create such file (if it exists, we rewrite it by empty file now)
        if (output_folder is not None) and warnings:
            open(output_folder+'/inference_warnings.txt', 'w').close()
        for p in predictors:
            # find dictionary belonging to predictor p in our list of dictionaries
            varinfo = next(item for item in self.predictorinfo_ if item['predictor'] == p)
            # find out whether the predictor has any warnings as the user might have specified that they want to see only such predictors
            if (issues_only and varinfo['inference_data1']+varinfo['inference_data2']+varinfo['instability']>0) or (not issues_only):
                if output_notebook:
                    # header
                    display(Markdown('### {}'.format(varinfo['predictor'])))
                if tables:
                    # table output: complete groupby + aggregation
                    if output_folder is not None:
                        varinfo['grp'].to_csv(output_folder+'/'+varinfo['predictor']+'_inference.csv')
                    if output_notebook:
                        display(varinfo['grp'])
                if charts:
                    # graphical output: charts with WoEs
                    fig, ax1 = plt.subplots(figsize=(10,5))
                    plt.plot(varinfo['grp'].index.values,
                            varinfo['grp'][(self.target+self.woe_postfix,'data1')].values,
                            linestyle='solid',
                            color='gray',
                            label=self.target+' WoE, data1 sample')
                    plt.plot(varinfo['grp'].index.values,
                            varinfo['grp'][(self.reject+self.woe_postfix,'data1')].values,
                            linestyle='solid',
                            color='indianred',
                            label=self.reject+' WoE, data1 sample')
                    if self.data2exist:
                        plt.plot(varinfo['grp'].index.values,
                                varinfo['grp'][(self.target+self.woe_postfix,'data2')].values,
                                linestyle='dashed',
                                color='gray',
                                label=self.target+' WoE, data2ation sample')
                        plt.plot(varinfo['grp'].index.values,
                                varinfo['grp'][(self.reject+self.woe_postfix,'data2')].values,
                                linestyle='dashed',
                                color='indianred',
                                label=self.reject+' WoE, data2ation sample')
                    # dotted zero line
                    plt.axhline(y=0, color='black', linestyle='dotted')
                    # legend to the right side, outside chart
                    plt.legend(loc='center left', bbox_to_anchor=(1,0.5), frameon=False)
                    plt.xticks(rotation=45)
                    if output_folder is not None:
                        plt.savefig(output_folder+'/'+varinfo['predictor']+'.png', bbox_inches='tight', dpi = 72)
                    if output_notebook:
                        plt.show()
                if warnings:
                    # text warnings saying what was found out during fit()
                    warntext = self.__textOutput(varinfo['predictor'],varinfo['inference_data1'],varinfo['inference_data2'],varinfo['instability'])
                    if output_folder is not None:
                        with open(output_folder+'/inference_warnings.txt','a') as f:
                            f.write(warntext)
                    if output_notebook:
                        print(warntext)
        if output_folder is not None:
            print('Output saved to folder '+output_folder)
        if (not output_notebook) and charts:
            # we call plt.close() in case that there was no notebook output
            # this is necessary because otherwise a broken chart would display in the notebook at the end
            plt.close()
        return


class TargetImputer():
    """
    Class for target imputation based on pre-calculated target probabilities. Typical usage is if we want to assign a proxy target to rows where target is unobserved.
    There are three imputation types that can be used:
    - randomized: for each observation where we want to impute the target, an random number between 0 and 1 is generated. Then this ranom number is compared with the pre-calculated probability and target is assigned based on this comparison.
    - cutoff: for each observation where we want to impute the target, the pre-caluclated probability is compared with this cutoff value and target is assigned based on this comparison.
    - weighted: for each observation where we want to impute the target, two observations are generated. Each of them has a different value of the target, one of the has weight p and the other has weight 1-p, where p is the pre-calculated target probability.
        
    Args:
        imputation_type (str, optional): Imputation which we want to use, can be 'randomized', 'cutoff' or 'weighted' (default: 'randomized')
        cutoff (float, optional): Cutoff value which is used if 'cutoff' imputation_type is selected (default: 0.5)
        random_seed (int, optional): Random seed which is used if 'randomized' imputation_type is selected (default: 987)
    
    Returns:
        pd.DataFrame: data set with imputed target (and weight)
    """
    
    def __init__(self, imputation_type='randomized', cutoff=0.5, random_seed=987):
        """
        Initialization method.
        """
        
        if imputation_type not in {'weighted', 'cutoff', 'randomized'}:
            self.imputation_type = 'randomized'
            print('Invalid imputation type. Must be "weighted", "cutoff" or "randomized". "randomized" is set by default.')
        else:
            self.imputation_type = imputation_type
            
        self.cutoff = cutoff
        self.random_seed = random_seed
    
    def fit(self, data, col_probs, col_reject, col_weight=None, prob_of=1):
        """
        A dataset where the target should be imputed is served to the fit method.
        This dataset must contain a column with pre-caluclated target probabilities (floats between 0 and 1) and a column with indicator whether target should be imputed for given row (integers with values 0 or 1).
        If the dataset is already weighted, the weight column should be specified as well. In such case, if imputation_type='weighted', the weights which are assigned to the rows are p*original weight ((1-p)*original weight respectively).
        
        Args:
            data (pd.DataFrame): the whole dataframe with scorecard development data (can be pre-filtered row- or column-wise, or can be the full data set)
            col_probs (str): name of column in data, where the probabilities of target are. This column must be float with values between 0 and 1.
            col_reject (str): name of column in data, where the indicator of unobserved target is (for scorecard development purpose, this is usually indicator of rejected loan application). 
                Value of the column should be 1 for rows where the target is unobserved and should be imputed, 0 for rows where the target should not be imputed.
            col_weight (str, optional): name of column in data, where weights are. Applies for weighted datasets only. (default: None)
            prob_of (int, optional): 0 or 1, based on whether col_probs contains probability of target being 0 or 1 (default: 1)
        """
        
        prob_array = data[data[col_reject]==1][col_probs]
        if col_weight is not None:
            weight_array = data[data[col_reject]==1][col_weight]
        else:
            weight_array = data[data[col_reject]==1][col_probs].apply(lambda x: 1)
            weight_array.name = '_IMP_ONES'
            
        self.imputed_data = pd.concat([prob_array,weight_array], axis=1)
        
        if self.imputation_type == 'weighted':
            imputed_data_0 = self.imputed_data.copy()
            self.imputed_data['_IMP_TARGET'] = prob_of
            imputed_data_0['_IMP_TARGET'] = 1-prob_of
            self.imputed_data['_IMP_WEIGHT'] = self.imputed_data[weight_array.name]*self.imputed_data[prob_array.name]
            imputed_data_0['_IMP_WEIGHT'] = imputed_data_0[weight_array.name]*(1-imputed_data_0[prob_array.name])
            self.imputed_data = pd.concat([self.imputed_data, imputed_data_0], axis=0)
            
        elif self.imputation_type == 'randomized':
            random.seed(self.random_seed)
            self.imputed_data['_RANDOM_NUMBER'] = 1
            self.imputed_data['_RANDOM_NUMBER'] = self.imputed_data['_RANDOM_NUMBER'].apply(lambda x: random.uniform(0, 1)) 
            self.imputed_data['_IMP_TARGET'] = self.imputed_data.apply(
                lambda x: prob_of if x[prob_array.name] > x['_RANDOM_NUMBER'] else 1-prob_of, axis=1)
            self.imputed_data['_IMP_WEIGHT'] = self.imputed_data[weight_array.name]
            
        elif self.imputation_type == 'cutoff':
            self.imputed_data['_IMP_TARGET'] = self.imputed_data.apply(
                lambda x: prob_of if x[prob_array.name] > self.cutoff else 1-prob_of, axis=1)
            self.imputed_data['_IMP_WEIGHT'] = self.imputed_data[weight_array.name]
            
        self.imputed_data = self.imputed_data[['_IMP_TARGET','_IMP_WEIGHT']]
        
        imputation_stats = self.imputed_data.groupby(['_IMP_TARGET'])['_IMP_WEIGHT'].agg({'count','sum'})
        imputation_stats.rename(columns={'count':'Count of rows', 'sum':'Sum of weights'},inplace=True)
        print(imputation_stats)
            
    def transform(self, data, col_target, col_weight=None, as_new_columns=False, reset_index=False):
        """
        Imputes target in the dataset with targets calculated in fit mehod.
        
        Args:
            data (pd.DataFrame): dataframe where the targets should be imputed
            col_target (str): name of column in data, where the target is and where the target should be imputed to
            col_weight (str, optional): name of column in data, where weights are. Applies for weighted datasets only. (default: {None})
            as_new_columns (bool, optional): boolean whether the imputed target (and also imputed weight) should be added to the dataframe as new columns or if they should rewrite the original values (default: {False})
            reset_index (bool, optional): boolean whether the row index of imputed dataset should be reset. Makes sense for imputation_type='weighted' where two rows are in the imputed dataset insted of every imputed one, so these two rows would have the same index. (default: {False})
        
        Returns:
            pd.DataFrame: data set with imputed target (and weight)
        """
        
        if not hasattr(self, 'imputed_data'):
            print('Run fit method first!')
            return
        
        new_data = data.join(self.imputed_data, how='left', rsuffix='_imputed')
        
        if as_new_columns:
            new_data[col_target+'_IMPUTED'] = new_data[col_target]
            col_target = col_target+'_IMPUTED'
            if col_weight is not None:
                new_data[col_weight+'_IMPUTED'] = new_data[col_weight]
                col_weight = col_weight+'_IMPUTED'
                
        if col_weight is None:
            col_weight = 'WEIGHT_IMPUTED'
            new_data[col_weight] = 1
        
        new_data.loc[new_data['_IMP_TARGET'].notnull(), col_target] = new_data.loc[new_data['_IMP_TARGET'].notnull(), '_IMP_TARGET']
        new_data.loc[new_data['_IMP_WEIGHT'].notnull(), col_weight] = new_data.loc[new_data['_IMP_WEIGHT'].notnull(), '_IMP_WEIGHT']
        new_data.drop(['_IMP_TARGET', '_IMP_WEIGHT'], axis=1, inplace=True)
        
        if reset_index:
            new_data.reset_index(inplace=True)
        
        return new_data