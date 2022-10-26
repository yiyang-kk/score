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
from pandas.api.types import is_numeric_dtype

def merge_rare(t, columns, k=5000, rare_value=-10000):
    """In given dataframe, it merges all values that have smaller occurence than parametrized value k
    into one category (its value is also given by parameter).

    Args:
        t (pd.DataFrame): data
        columns (list of str): Names of columns of t where rare value should be replaced.
        k (int, optional): Maximal occurence (count) of a single value to be considered rare. Defaults to 5000.
        rare_value (int or float or str, optional): Replacement value. Defaults to -10000.
    """
    for c in columns:
        vc=t[c].value_counts()
        for i in vc[vc<k].index:
            t.loc[t[c]==i, c]=rare_value

class Discretizer:
    """Discretizes columns of a matrix to a given number of categories.

        Args:
            bin_count (int, optional): Number of categories. Defaults to 20.
    """
    def __init__(self, bin_count=20):
        """
        """
        self.bin_count=bin_count
    
    def fit(self, X):
        """Calculates replacement values (i.e. discretization)

        Args:
            X (np.matrix): matrix to be discretized
        """
        self.bins=np.nanpercentile(X, np.linspace(0,100,self.bin_count+1), axis=0)
        self.bins[0, :]=-np.inf
        self.bins[-1, :]=np.inf
    
    def transform(self, X):
        """Replaces continous values by discretization

        Args:
            X (np.matrix): matrix to be discretized

        Returns:
            np.matrix: matrix with discrete values
        """
        res = np.empty(shape=X.shape)
        for j in range(res.shape[1]):
            res[:, j] = np.digitize(X[:, j], self.bins[:, j])
            res[np.isnan(X)] = np.nan
        return res

def fake_binning(X, bin_count=5):
    """Creates binning dictionary which can be then loaded by grouping.Grouping object.
    The binning is not fitted by machine learning technique, but is equifreqent (in case of numerical columns)
    or just calculates WOE for each distinct value (in case of categorical columns).

    Args:
        X (pd.DataFrame): dataframe with columns to be binned
        bin_count (int, optional): Max number of bins for numerical columns. Defaults to 5.

    Returns:
        dict: dictionary with binning compatible with Grouping object
    """
    bin_dict = dict()
    for column in [column for column in X.columns if is_numeric_dtype(X[column])]:
        bins = np.nanpercentile(X[column], np.linspace(0,100,bin_count+1), axis=0)
        bins[0] = -np.inf
        bins[-1] = np.inf
        woes = list(range(len(bins)))
        bin_dict[column] = {
            "bins": bins,
            "woes": woes,
            "nan_woe": -1,
            "dtype": str(X[column].dtype)
        }
    for column in [column for column in X.columns if not is_numeric_dtype(X[column])]:
        cat_bins = list([list (z) for z in zip(X[column].unique(), range(X[column].nunique()))])
        woes = list(range(X[column].nunique()))
        bin_dict[column] = {
            "cat_bins": cat_bins,
            "woes": woes,
            "unknown_woe": -1,
            "dtype": str(X[column].dtype)
        }
    return bin_dict




'''
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, TransformerMixin

class KNNShaper(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, k=10):
        self.k=k
        self.columns=columns
    def fit(self, X, y):
        self.knns_={}
        for idx in self.columns:
            knn = KNeighborsClassifier(n_neighbors=self.k, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', 
                             metric_params=None)
            knn.fit(np.expand_dims(X[:, idx], axis=1), y)
            self.knns_[idx]= knn
        return self
    def transform(self, X):
        X2=np.copy(X)
        for idx, knn in self.knns_.items():
            #print(idx)
            #dist, ind=knn.kneighbors(X)
            pred=knn.predict_proba(np.expand_dims(X[:, idx], axis=1))[:, 0]
            X2[:, idx]=pred
        return X2
'''
        

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.utils import check_array
from sklearn.utils import as_float_array
from sklearn.utils.validation import check_is_fitted

import numpy as np

class WOE(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, copy=True):
        self.copy=copy
        self.columns=columns
        
    def fit(self, X, y):
        #print('fit Begin')
        X = check_array(X, accept_sparse='csc', dtype=np.float64,
                            force_all_finite=True)       
        self.woe_={}

        if self.columns:   
            for j in self.columns:
                self.woe_[j]={}
                x=X[:, j]
                print(x)
                for v in np.unique(x):
                    print((1.*(len(x[(x==v)&(y==0)])+1)/(len(x[y==0])+1))/(1.*(len(x[(x==v)&(y==1)])+1)/(len(x[y==1])+1)))
                    self.woe_[j][v]=(1.*(len(x[(x==v)&(y==0)])+1)/(len(x[y==0])+1))/(1.*(len(x[(x==v)&(y==1)])+1)/(len(x[y==1])+1))
        return self
    
    def transform(self, X):
        check_is_fitted(self, 'woe_')
        X = as_float_array(X, copy=self.copy, force_all_finite=True)
        
        for c,_ in self.woe_.items():
            for v, woe in _.items():
                mask=X[:, c]==v
                X[mask, c]=woe
        return X    
