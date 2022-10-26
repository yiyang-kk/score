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



class WeightOfEvidenceEncoder():
    """Calculates WOE encoding.
  
    Args:
        smooth_coef (float, optional): smoothing coefficient, the larger it is, the closer to zero the WoE value will be
            (default: 0.001)
        max_woe (float, optional): censoring threshold, WoE which is in absolute value larger than this are censored to this
            number (with corresponding +/- signature) (default: 10)
        unknown_fill_value (float, optional): value that will be filled during transformation to observations
            with value of predictor which was not known during fitting. (default: 0.0)
    """

    def __init__(self, smoothing_coefficient=0.001, max_abs_woe=10, unknown_fill_value=0.0):
        """Initialization
        """

        self.smoothing_coefficient = smoothing_coefficient
        self.max_abs_woe = max_abs_woe
        self.unknown_fill_value = unknown_fill_value

    def fit(self, predictor, target, weight=None):
        """
        Calculates Wight of Evidence.

        The formula is:
        .. math::
                                     ( weighted good rate in category + smoothing coef )      ( weighted good rate in all obs + smoothing coef )
             weight of evidence = ln ( ----------------------------------------------- ) - ln ( ---------------------------------------------- )
                                     ( weighted bad rate in category  + smoothing coef )      ( weighted bad rate in all obs  + smoothing coef )

        Args:
            predictor (np.array or pd.Series): array with predictor values (can be numerical or categorical)
            target (np.array or pd.Series): array with target values (should be 0 or 1)
            weight (np.array or pd.Series): array with observation weights (must be numeric) (default: None)
        """

        self.codelist = {}

        # impute weights if missing
        if weight is None:
            weight = np.ones(len(target))

        # not use observations with missing targets:
        predictor = predictor[~np.isnan(target)]
        weight = weight[~np.isnan(target)]
        target = target[~np.isnan(target)]

        ln_odds_overall = np.log( 
            (
            sum(np.multiply(1-target, weight)) / sum(weight) + self.smoothing_coefficient
            ) / (
            sum(np.multiply(target, weight)) / sum(weight) + self.smoothing_coefficient
            ) 
        )

        for predictor_value in set(pd.Series(predictor).unique()):

            if predictor_value != predictor_value: # solving np.nan
                obs_mask = pd.isnull(predictor)
            else:
                obs_mask = predictor == predictor_value

            ln_odds_segment = np.log( 
                (
                sum(np.multiply(1-target[obs_mask], weight[obs_mask])) / sum(weight[obs_mask]) + self.smoothing_coefficient
                ) / (
                sum(np.multiply(target[obs_mask], weight[obs_mask])) / sum(weight[obs_mask]) + self.smoothing_coefficient
                ) 
            )

            weight_of_evidence = ln_odds_segment - ln_odds_overall

            if np.absolute(weight_of_evidence) > self.max_abs_woe:
                weight_of_evidence = np.sign(weight_of_evidence) * self.max_abs_woe
            
            self.codelist[predictor_value] = weight_of_evidence

        self.codelist["__unknown"] = self.unknown_fill_value

    def transform(self, predictor):
        """Transforms a predictor using the codelist created in fit method
        
        Args:
            predictor (np.array or pd.Series): array with predictor values (can be numerical or categorical)
        
        Returns:
            np.array: encoded (transformed) predictor
        """

        encoded_predictor = np.zeros(len(predictor))
        encoded_predictor[:] = np.nan

        for predictor_value, encoded_value in self.codelist.items():
            encoded_predictor[predictor == predictor_value] = encoded_value

        if np.nan in self.codelist.keys():
            encoded_predictor[pd.isnull(predictor)] = self.codelist[np.nan]
        encoded_predictor[pd.isnull(encoded_predictor)] = self.codelist["__unknown"]
            
        return encoded_predictor

    def get_encoded_value(self, predictor_value):
        """For a given predictor value, it returns WOE-encoded value.
        If such value does not exist, it return default value (unknown_fill_value)
        
        Args:
            predictor_value (str or int or float): Predictor value that should be looked up in the internal dictionary
        
        Returns:
            float: lookup result
        """

        if predictor_value in self.codelist.keys():
            return self.codelist[predictor_value]
        else:
            return self.codelist["__unknown"]



class MeanTargetEncoder():
    """Encoder of categorical or non-linear predictors. For each distinct value of such predictor, mean target 
    (with optional regularization) can be calculated and during transformation assigned on place of the original 
    predictor value. The regularization can be adjusted by regularization_parameter which is set during 
    the initialization.
        
    Args:
        regularization_parameter (float, optional): importance of mean target of the whole data set (default: 0)
        unknown_fill_value ('mean' or float, optional): value that will be filled during transformation to observations
        with value of predictor which was not known during fitting. Can be either 'mean' (i.e. overall mean
        will be assigned) or float (i.e. that number will be assigned). (default: 'mean')
    
    Returns:
        np.array: encoded (transformed) predictor

    Properties:
        codelist: dictionary created in fit() method
    """

    def __init__(self, regularization_parameter=0, unknown_fill_value="mean"):
        """Initialization
        """

        self.regularization_parameter = regularization_parameter
        self.unknown_fill_value = unknown_fill_value

    def fit(self, predictor, target, weight=None):
        """Calculates regularized mean target for each category (distinct value) of predictor.

        The formula is:
        .. math::

                                    sum weight of obs in category * mean target in category + regularization parameter * sum weight of all obs * overall mean target
            regularized mean target = --------------------------------------------------------------------------------------------------------------------------------
                                               sum weight of obs in category +  regularization parameter * sum weight of all obs


        Args:
            predictor (np.array or pd.Series): array with predictor values (can be numerical or categorical)
            target (np.array or pd.Series): array with target values (should be 0 or 1)
            weight (np.array or pd.Series, optional): array with observation weights (must be numeric) (default: None)
        """

        self.codelist = {}

        # impute weights if missing
        if weight is None:
            weight = np.ones(len(target))

        # not use observations with missing targets:
        predictor = predictor[~np.isnan(target)]
        weight = weight[~np.isnan(target)]
        target = target[~np.isnan(target)]

        weighted_bad_overall = sum(np.multiply(target, weight))
        mean_target_overall = weighted_bad_overall / sum(weight)
        regularization_constant = self.regularization_parameter * sum(weight)

        for predictor_value in set(pd.Series(predictor).unique()):

            if predictor_value != predictor_value: # solving np.nan
                obs_mask = pd.isnull(predictor)
            else:
                obs_mask = predictor == predictor_value

            weighted_bad_segment = sum(
                np.multiply(
                    target[obs_mask],
                    weight[obs_mask],
                )
            )
            weight_segment = sum(weight[obs_mask])
            mean_target_segment = weighted_bad_segment / weight_segment
            mean_target_regularized = (
                (mean_target_segment * weight_segment)
                + (mean_target_overall * regularization_constant)
            ) / (weight_segment + regularization_constant)
            self.codelist[predictor_value] = mean_target_regularized

        if self.unknown_fill_value == "mean":
            self.codelist["__unknown"] = mean_target_overall
        else:
            self.codelist["__unknown"] = self.unknown_fill_value

        self.mean_target_overall = mean_target_overall

    def transform(self, predictor, return_type = 'mean'):
        """Transforms a predictor using the codelist created in fit method
        
        Args:
            predictor (np.array or pd.Series): array with predictor values (can be numerical or categorical)
            return_type (str, optional): Whether mean target ('mean') or logistic transformation of mean target ('logit')
                should be returned (default: 'mean')
        
        Returns:
            np.array: encoded (transformed) predictor
        """

        if return_type not in ['mean', 'logit']:
            raise ValueError('return_type must be \'mean\' or \'logit\'.')

        encoded_predictor = np.zeros(len(predictor))
        encoded_predictor[:] = np.nan

        for predictor_value, encoded_value in self.codelist.items():
            encoded_predictor[predictor == predictor_value] = encoded_value

        if np.nan in self.codelist.keys():
            encoded_predictor[pd.isnull(predictor)] = self.codelist[np.nan]
        encoded_predictor[pd.isnull(encoded_predictor)] = self.codelist["__unknown"]

        if return_type == 'logit':
            logit_predictor = np.zeros(len(predictor))
            logit_predictor[(0 < encoded_predictor) & (encoded_predictor < 1)] = np.log(
                encoded_predictor[(0 < encoded_predictor) & (encoded_predictor < 1)]
                / (1 - encoded_predictor[(0 < encoded_predictor) & (encoded_predictor < 1)])
            )
            abs_max_value = max(abs(logit_predictor))
            logit_predictor[encoded_predictor <= 0] = -abs_max_value * 1.1
            logit_predictor[encoded_predictor >= 1] = abs_max_value * 1.1
            return logit_predictor
        elif return_type == 'mean':
            return encoded_predictor

    def get_encoded_value(self, predictor_value, return_type = 'mean'):
        """For a given predictor value, it returns WOE-encoded value.
        If such value does not exist, it return default value (unknown_fill_value)
        
        Args:
            predictor_value (str or int or float): Predictor value that should be looked up in the internal dictionary
            return_type (str): Whether mean target ('mean') or logistic transformation of mean target ('logit')
                should be returned (default: {'mean'})
        
        Returns:
            float: lookup result
        """

        if return_type not in ['mean', 'logit']:
            raise ValueError('return_type must be \'mean\' or \'logit\'.')

        if predictor_value in self.codelist.keys():
            encoded_predictor = self.codelist[predictor_value]
        else:
            encoded_predictor = self.codelist["__unknown"]

        if return_type == 'logit':
            if 0 < encoded_predictor < 1:
                return_value = np.log(
                    encoded_predictor / (1-encoded_predictor)
                )
            else:
                raise ValueError('Logit cannot be returned, encoded value is not strictly between 0 and 1.')
        elif return_type == 'mean':
            return_value = encoded_predictor

        return return_value