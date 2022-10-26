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
import numpy as np
import pandas as pd
import random
import sys
try:
    from sklearn.model_selection import train_test_split
except ModuleNotFoundError:
    from sklearn.cross_validation import train_test_split
from pandas.api.types import is_numeric_dtype
from .db import get_optimal_numerical_type

def split_predictors_bytype(data, pred_list = None, non_pred_list = None, num_pred_list = None, cat_pred_list = None, optimize_types = True, convert_bool2int = True, print_lists = True):
    """This function creates list of all, numerical and categorical predictors based on dtypes of pandas DataFrame with the underlying data, or based on optional lists of predictors defined by the user.
    If user passes this function data only (the only mandatory argument), all columns of data will be considered to be predictors and will be categorized either as numerical or as categorical predictors.
    If user passes this function also pred_list (list of all predictors), only columns included in this list will be considered to be predictors.
    If user passes this function also non_pred_list (list of non-predictive columns), only columns not included in this list will be considered to be predictors. (Based on whether pred_list is filled, this means that either pred_list-non_pred_list or data.columns-non_pred_list are considered.)
    If user passes this function at least one of num_pred_list and cat_pred_list, this will be primarily used to determine predictor categories (i.e. function will try to re-cast data columns to match these lists). If both of them are filled, their union is used as all predictors list instead of pred_list.
    
    Args:
        data (pd.DataFrame): data set with predictors
        pred_list (list of str, optional): user pre-defined list of all predictors. if not specified, all columns of data will be used. (default: [])
        non_pred_list (list of str, optional): user pre-defined list of non-predictive variables. if specified, these predictor names will be removed from all predictors list. (default: [])
        num_pred_list (list of str, optional): user pre-defined list of numerical predictors. if both this and cat_pred_list are not specified, the predictors will be automatically assigned their category based on data types in data frame 'data' (default: [])
        cat_pred_list (list of str, optional): user pre-defined list of numerical predictors. if both this and num_pred_list are not specified, the predictors will be automatically assigned their category based on data types in data frame 'data' (default: [])
        optimize_types (bool, optional): if True and some data columns need to be re-casted (assigned different data type to match the pre-defined list), new datatypes will be memory-optimized (default: True)
        convert_bool2int (bool, optional):- if True, all predictors that have bool type in data will be re-casted as integer and used as numerical predictors (default: True)
        print_lists (bool, optional): turns text output on/off (default: True)
    
    Returns:
        list, list, list: list of all predictors, list of numerical predictors, list of categorical predictors
    """

    col_list = list(data.columns)
    
    # ERROR if some predictors from pred_list not in data
    for user_defined_list in [pred_list, num_pred_list, cat_pred_list]:
        if user_defined_list:
            err_pred_list = [pred for pred in user_defined_list if pred not in col_list]
            if len(err_pred_list) > 0:
                raise ValueError(f'Predictors {err_pred_list} not found in data.')
    
    # if list of all predictors is not given, we will determine it from all columns of data
    if pred_list is None:
        pred_list = col_list
        
    # if there is a specific list of columns that should not be considered predictors, we will remove them from pred_list
    if non_pred_list:
        if pred_list:
            pred_list = [pred for pred in pred_list if pred not in set(non_pred_list)]
        if num_pred_list:
            num_pred_list = [pred for pred in num_pred_list if pred not in set(non_pred_list)]
        if cat_pred_list:
            cat_pred_list = [pred for pred in cat_pred_list if pred not in set(non_pred_list)]
    
    # NO TYPE of predictors predefined - determine by column type in data
    if (num_pred_list is None) and (cat_pred_list is None):
        num_pred_list = [pred for pred in pred_list if is_numeric_dtype(data[pred])]
        cat_pred_list = [pred for pred in pred_list if pred not in num_pred_list]
    # NUMERICAL predictors predefined
    elif (cat_pred_list is None) and (len(num_pred_list) > 0):
        cat_pred_list = [pred for pred in pred_list if pred not in num_pred_list]
    # CATEGORICAL predictors predefined
    elif (num_pred_list is None) and (len(cat_pred_list) > 0):
        num_pred_list = [pred for pred in pred_list if pred not in cat_pred_list]
    # BOTH TYPES of predictors predefined
    else:
        pred_list = num_pred_list + cat_pred_list
    
    # Check if datatypes in data correctly match those defined in the num and cat pred lists. Try to fix, if they don't.
    for column_name, column in data.iteritems():
        # loop all predictors listed as numerical
        if column_name in num_pred_list:
            # if dtype of numerical predictor is boolean, convert it to integer
            if (column.dtype.name == 'bool') and convert_bool2int:
                column.astype(np.int)
                if optimize_types:
                    data[column_name] = column.astype(get_optimal_numerical_type(column))
            # if dtype of numerical predictor is not numerical (and was not boolean), try to convert it to numerical dtype. if impossible, move it to categorical predictor list.
            elif not is_numeric_dtype(column.values.dtype):
                try:
                    column.astype(np.number)
                    if optimize_types:
                        data[column_name] = column.astype(get_optimal_numerical_type(column))
                except:
                    print(f'Column {column_name} couldn\'t be converted to numerical. Will be used as categorical.')
                    num_pred_list.remove(column_name)
                    cat_pred_list.append(column_name)
    
        # loop all predictors listed as categorical
        if column_name in cat_pred_list:
            # if dtype of categorical predictor is boolean, convert it to integer and move it to numerical predictor list.
            if (column.dtype.name == 'bool') and convert_bool2int:
                column.astype(np.int)
                if optimize_types:
                    data[column_name] = column.astype(get_optimal_numerical_type(column))
                print(f'Boolean {column_name} will be converted to integer. Will be used as numerical.')
                cat_pred_list.remove(column_name)
                num_pred_list.append(column_name)
            # if dtype of categorical predictor is not in typical categorical types, try to convert it to category, or string (based on input parameter)
            elif column.dtype.name not in {'object', 'string', 'category'}:
                if optimize_types:
                    try:
                        data[column_name] = column.astype('category')
                    except:
                        data[column_name] = column.astype(str)
                else:
                    data[column_name] = column.astype(str)
            elif column.dtype.name in {'object', 'string'}:
                if optimize_types:
                    try:
                        data[column_name] = column.astype('category')
                    except:
                        pass
                  
    # print output (formatted lists of numerical and categorical predictors along with their dtypes)
    if print_lists:
        print(f'List of numerical predictors: [{len(num_pred_list)}]\n')
        for pred in num_pred_list:
            print(str.ljust(pred, 35), data[pred].dtype.name)
        print()
        print(f'List of categorical predictors: [{len(cat_pred_list)}]\n')
        for pred in cat_pred_list:
            print(str.ljust(pred, 35), data[pred].dtype.name)
    
    # put together final predictor list
    pred_list = num_pred_list + cat_pred_list
    
    return pred_list, num_pred_list, cat_pred_list

def data_sample_split(data,
                      sample_sizes = [0.6,0.2,0.2],
                      sample_names = ['train','valid','test'],
                      stratify_by_columns = [],
                      time_column = None,
                      time_from = None, 
                      time_to = None,
                      random_seed = 12345):
    """Function to split a data set into multiple samples randomly, with option to stratify the split by one or more variables.

    Args:
        data (pandas.DataFrame): the data frame to be split.
        sample_sizes (list of float, optional): relative sizes of the samples, should add up to 1. If there add up to number between 0 and 1, one more sample for the rest will be created. (default: [0.6,0.2,0.2])
        sample_names (list of str, optional): strings with the names of the samples which will be created. The list should have the same length as sample_sizes list. (default: ['train','valid','test'])
        stratify_by_columns (list of str, optional): names of the columns of data, which should be used for stratification. There should be reasonably small number of unique values of these columns. (default: [])
        time_column (str, optional): name of column of data with time of observation. Used along with time_from and time_to in cases when only certain time period should be splitted and the rest of the data should be ignored. (default: None)
        time_from (data.time_column.dtype, optional): minimal value of data.time_column for row of data to be used in data set splitting. Rows with exactly this value will be used. (default: None)
        time_to (data.time_column.dtype, optional): maximal value of data.time_column for row of data to be used in data set splitting. Rows with exactly this value will NOT be used. (default: None)
        random_seed (int, optional): random seed for the random values which will be used for splitting. (default: 12345)

    Returns:
        pandas.DataFrame: a data frame with the same length and indexes as data, containing one column 'sample_name' with name of sample for the corresponding row in data
    """

    # check index uniqueness
    if not data.index.is_unique:
        raise IndexError('Data don\'t have unique index!')

    if any([(ss > 1) or (ss < 0) for ss in sample_sizes]):
        raise ValueError('Sample size must be between 0 and 1')

    if len(sample_sizes) != len(sample_names):
        raise ValueError('Lengths of sample_sizes and sample_names lists are different!')

    # if user gave us sizes of sample which are lower than 1, one more sample is added so the sum of sizes is 1

    if (not np.isclose(sum(sample_sizes), 1)) and (sum(sample_sizes) < 1):
        print('Sum of sample_sizes is smaller than 1. Additional sample will be added.')
        sample_sizes.append(1-sum(sample_sizes))
        sample_names.append('unknown')
    elif (not np.isclose(sum(sample_sizes), 1)) and (sum(sample_sizes) > 1):
        print('Sum of sample_sizes is larger than 1. The sample sizes will be adjusted.')
        sample_sizes = [ss/sum(sample_sizes) for ss in sample_sizes]

    sample_names = [sn for i, sn in enumerate(sample_names) if sample_sizes[i] > 0]
    sample_sizes = [ss for i, ss in enumerate(sample_sizes) if sample_sizes[i] > 0]
        
    # create new pd series with concatenated stratification columns
    stratification = None
    if len(stratify_by_columns) > 0:
        if isinstance(stratify_by_columns, str):
            if stratify_by_columns not in data.columns:
                raise ValueError('Stratification column '+stratify_by_columns+' is not in dataset.')
            stratification = data[[stratify_by_columns]].copy()
        else:
            for stratcol in stratify_by_columns:
                if stratcol not in data.columns:
                    raise ValueError('Stratification column '+stratcol+' is not in dataset.')
                if stratification is None:
                    stratification = data[stratcol].copy().astype(str)
                else:
                    stratification = stratification + '_' + data[stratcol].copy().astype(str)
        
    #time condition
    time_condition = pd.Series(True,index=data.index)
    if time_column is not None:
        if time_column not in data.columns:
            raise ValueError('Time column '+time_column+' is not in dataset.')
        if time_from is not None:
            if time_to is not None:
                time_condition = (data[time_column] >= time_from) & (data[time_column] < time_to)
            else:
                time_condition = (data[time_column] >= time_from)
        elif time_to is not None:
            time_condition = (data[time_column] < time_to)
    if time_condition.astype(int).sum() == 0:
        sample_name = pd.DataFrame(columns = ['data_type'])
    
    # create data set with one column
    ix = pd.DataFrame(data[time_condition].index)
    ix.index = data[time_condition].index
    if stratification is not None:
        stratification = stratification[time_condition]

    # random seed for sampling
    np.random.seed(random_seed)
    random_numbers = np.random.randint(0, 99999, size=len(sample_sizes)-1)
        
    # sampling 
    sample_name = ix.astype(str)
    sample_name.columns = ['data_type']
    new_sample_sizes = [ss for ss in sample_sizes]
    if len(sample_sizes) == 1:
        sample_name['data_type'] = sample_names[0]
    else:
        for i in range(len(sample_sizes)-1):
            try:
                ix1, ix2 = train_test_split(ix, test_size = 1-new_sample_sizes[i], stratify = stratification, random_state = random_numbers[i])
            except ValueError as err:
                if str(err).startswith('The least populated class in y has only 1 member'):
                    data_check = (
                        data[stratify_by_columns].fillna('nan')
                        .reset_index()
                        .groupby(stratify_by_columns)
                        .count()
                        )
                    raise ValueError(f'Stratification can\'t be done since these classes have too few observations:\n {data_check[data_check[data.index.name] == 1]}').with_traceback(sys.exc_info()[2])

            sample_name.loc[ix1.index,'data_type'] = sample_names[i]
            if i == len(sample_sizes)-2:
                sample_name.loc[ix2.index,'data_type'] = sample_names[i+1]
            else:
                new_sample_sizes = [ss/sum(sample_sizes[i+1:]) for ss in sample_sizes]
                ix = ix2
                if stratification is not None:
                    stratification = stratification.loc[ix2.index]
    
    # print number of rows that were included in each of the samples
    for n in sample_names:
        print(f"{n:<8}{len(sample_name[sample_name['data_type']==n]):,} rows")
    
    return sample_name   

def data_sample_time_split(data, 
                           time_column,
                           splitting_points = [],
                           sample_sizes = [],
                           sample_names = [],
                           stratify_by_columns = [],
                           random_seed = 12345):
    """Function to split a data set into multiple time periods and in each period into multiple samples randomly, with option to stratify the split by one or more variables.

    Args:
        data (pandas.DataFrame): the data frame to be split.
        time_column (str): name of column of data with time of observation.
        splitting points (list of data.time_column.dtype, optional): values of time column where the time splits should be done. First time period is period from -infinity to the value of first splitting point (not including it). Each splitting points is minimum of a time period. For n time periods to be created, n-1 time split values are needed. (default: [])
        sample_sizes (list of list of float, optional): for each time period, a list of relative sizes of the samples, should add up to 1. It is a list of list, the outer list should have length of len(splitting_points)+1. (default: [])
        sample_names (list of list of str, optional): strings with the names of the samples which will be created. The list should have the same structure as sample_sizes list. (default: [])
        stratify_by_columns (list of str, optional): names of the columns of data, which should be used for stratification. There should be reasonably small number of unique values of these columns. (default: [])
        random_seed (int, optional): random seed for the random values which will be used for splitting. (default: 12345)

    Returns:
        pandas.DataFrame: a data frame with the same length and indexes as data, containing one column 'sample_name' with name of sample for the corresponding row in data
    """
    
    # if there is no time split, call the function for random split just once for the whole data set
    if len(splitting_points)==0:
        sample_name = data_sample_split(data,
                                        sample_sizes = sample_sizes[0],
                                        sample_names = sample_names[0],
                                        stratify_by_columns = stratify_by_columns,
                                        random_seed = random_seed)
    
    # for each splitting points, set the correct time period for the data_sample_split function
    else:
        for i in range(len(splitting_points)+1):
            
            if i == 0:
                # first interval goes from -inf to first splitting point
                time_from = None
                time_to = splitting_points[i]
            elif i == len(splitting_points):
                # last interval goes from the last splitting point to inf
                time_from = splitting_points[i-1]
                time_to = None
            else:
                time_from = splitting_points[i-1]
                time_to = splitting_points[i]
            
            # call the function for random split for this time period
            tmp = data_sample_split(data,
                                        sample_sizes = sample_sizes[i],
                                        sample_names = sample_names[i],
                                        stratify_by_columns = stratify_by_columns,
                                        time_column = time_column,
                                        time_from = time_from,
                                        time_to = time_to,
                                        random_seed = random_seed)

            # otuput data frame
            if i == 0:
                sample_name = tmp
            else:
                sample_name = sample_name.append(tmp)
                
    return sample_name