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

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.utils import check_array
from sklearn.utils import as_float_array
from sklearn.utils.validation import check_is_fitted

import numpy as np
import scipy
import warnings

class ScoreImputer(BaseEstimator, TransformerMixin):
    """
    1) Calculation of event rate for NAN value
    2) Calculation of event rate in n quantiles of attribute values
    3) Find quantile with event rate closest to event rate on NANs
    4) Impute NANs fith mean value of the quantile found
    """
    
    def __init__(self,n=100, columns=None, extra_nans={}, copy=True):
        self.copy=copy
        self.n = n
        self.columns = columns
        self.extra_nans = extra_nans
        
    def fit(self, X, y):
        #print('fit Begin')
        X = check_array(X, accept_sparse='csc', dtype=np.float64,
                            force_all_finite=False)
        
        if self.columns is None:
            self.columns = range(X.shape[1])
        self.imp_={}

        
        #np.isnan(x) | x.isin([extra_nans])

        #print(self.columns)
        for j in self.columns: # iterate over columns
            self.imp_[j] = {}
            extra_nans = self.extra_nans[j] if j in self.extra_nans else []
            #print j, extra_nans, self.extra_nans
            for nan_value in [np.nan] + extra_nans:
                #print nan_value
                eq_nan = (lambda x: np.isnan(x))  if np.isnan(nan_value) else (lambda x: x == nan_value)
                if eq_nan(X[:, j]).sum()<100:
                    if eq_nan(X[:, j]).sum()==0:
                        continue
                    raise ValueError('Not enough NaN samples to impute {} for column {} {}'.format(nan_value, j, X[:, j]))
                if np.logical_not(eq_nan(X[:, j])).sum()==0: # all NANs
                    raise ValueError('All values are NAN in columns {} {}'.format(j, X[:, j]))
                
                mask=np.logical_not(np.isnan(X[:, j]) | np.isin(X[:, j], extra_nans))
                perc=np.nanpercentile(X[mask, j], np.arange(100.+100./self.n, step=100./self.n))
                values=np.array([(l+r)/2 for l,r in zip(perc[:-1], perc[1:])])
                perc[0]=-np.inf
                perc[-1]=np.inf
                
                
                xx=X[:, j][mask]
                yy=y[mask]
                
                brs=np.array([yy[(xx>=l)&(xx<r)].mean() for l,r in zip(perc[:-1], perc[1:])])
                br=y[eq_nan(X[:, j])].mean()
                #print nan_value

                self.imp_[j][nan_value] = values[np.abs(brs-br).argmin()]
        return self

    def transform(self, X):
        #print('transform begin')
        check_is_fitted(self, 'imp_')
        #print(self.imp_)
        # Copy just once
        X = as_float_array(X, copy=self.copy, force_all_finite=False)
        for j, v in self.imp_.items():
            for nan_value, val in v.items():
                eq_nan = (lambda x: np.isnan(x))  if np.isnan(nan_value) else (lambda x: x == nan_value)
                X[eq_nan(X[:, j]), j] = val
        return X

class Logit(BaseEstimator, TransformerMixin):
    def __init__(self, columns=[], copy=True):
        self.copy=copy
        self.columns=columns
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        X = as_float_array(X, copy=self.copy, force_all_finite=False)
        for j in self.columns:
            x=X[:, j]
            if x[(x<=0)|(x>=1)].any():
                #print(x.min(), x.max())
                raise ValueError('Attribute {} {} containes values not between (0;1)'.format(j, x))
        for j in self.columns:
            X[:, j]=scipy.special.logit(X[:, j])
            #X[:, j]=np.log(X[:, j]/(1-X[:, j]))
        return X    

class Range(BaseEstimator, TransformerMixin):
    def __init__(self, n = 25, copy = True):
        self.copy=copy
        self.n = n
        
    def fit(self, X, y = None):
        X = check_array(X, accept_sparse='csc', dtype=np.float64,
                            force_all_finite=False)
        
        self.imp_={}
        
        for j in range(X.shape[1]): # iterate over columns
            u = np.unique(X[:, j])
            mask=np.logical_not(np.isnan(u))
            u = u[mask]
            self.imp_[j] = u
        return self

    def transform(self, X):
        #print('transform begin')
        check_is_fitted(self, 'imp_')

        # Copy just once
        X = as_float_array(X, copy=self.copy, force_all_finite=False)
        
        for k,v in self.imp_.items():
            vfunc = np.vectorize(lambda x: np.abs(x - v).argmin())
            mask = np.logical_not(np.isnan(X[:, k]))
            X[mask, k] = 1.* vfunc(X[mask, k]) / (v.shape[0] - 1) # what if 0 length???
            X[mask, k] = 1.* (vfunc(X[mask, k])+1) / v.shape[0] # what if 0 length???
        return X
