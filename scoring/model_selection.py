
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


import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.svm import l1_min_c
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from IPython.display import display
import sklearn.linear_model
import sklearn.metrics
import math
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures.thread
from .grouping import Grouping, InteractiveGrouping

class GiniStepwiseLogit():
    """Stepwise logistic regression. We start with set of initial predictors (can be empty) and in each step we try to add (if selection method is 'forward'), remove (if selection method is 'backward') or both (if seletion method is 'stepwise') a predictor.
    Model with the best Gini is then passed to a next iteration. After convergence criteria are met, the iterations stop.

    Args:
        initial_predictors (list of str, optional): list of predictors the stepwise should start with. if not filled, it will start with an empty set (default: [])
        all_predictors (list of str, optional): list of predictors that would be considered to enter the model. if not filled, all columns in the data served to model in fit() method will be considered. (default: [])
        dummy_regression (bool, optional): whether dummy variables should be used instead of Weight of Evidence variables (default: False)
        dummy_bindings (Grouping.get_dummies() or InteractiveGrouping.get_dummies() - dictionary, optional): if dummy_regression: dummy variable dictionary from Grouping or InteractiveGrouping (default: None)
        selection_method (str, optional):  'forward', 'backward' or 'stepwise' (default: 'stepwise')
        max_iter (int, optional): end after 1000 iterations, if no other stopping criteria fulfilled sooner (default: 1000)
        min_increase (float, optional): min gini increase in forward regression step (default: 0.5)
        max_decrease (float, optional): max gini decrease in backward regression step (default: 0.25)
        max_predictors (int, optional): end when >= max_predictors are in model, if no other stopping criteria fulfilled sooner. If 0, no such criteria applies. (default: 0)
        max_correlation (float, optional): if dummy_regression==False and selection_method=='forward': max possbile absolute correlation between two predictors. If 1, no such criteria applies. (default: 1)
        beta_sgn_criterion (bool, optional): if dummy_regression==False and selection_method=='forward': restriction that all predictors should have the same signature of their coefficients in the model. (default: False)
        penalty (str, optional): "l1" or "l2" to choose type of regularization applied (default: "l2")
        C (float, optional): regularization, higher means smaller regularization (very high means practically no regularization) (default: 1000)
        correlation_sample (int, optional): size of sample from the data that is used to calculate correlation matrix (default: 10000)
        use_cv (bool, optional): should cross validation be used instead of train/validate split? (default: False)
        cv_folds (int, optional): if use_cv: nr of folds in cv (default: 5)
        cv_seed (int, optional): if use_cv: random seed for cv (default: 98765)
        cv_stratify_by_target (bool, optional): if use_cv: whether the cross validation samples should by stratified by the target variable (default: False)
        n_jobs (int, optional): number of threads for parallelization (default: 1)
    """

    def __init__(
        self, 
        initial_predictors = [],
        all_predictors = [],
        dummy_regression = False,
        dummy_bindings = None,
        selection_method="stepwise",
        max_iter=1000,
        min_increase=0.5,
        max_decrease=0.25,
        max_predictors=0,
        max_correlation=1,
        beta_sgn_criterion=False,
        penalty="l2",
        C=1000,
        correlation_sample=10000,
        use_cv=False,
        cv_folds=5,
        cv_seed=98765,
        cv_stratify_by_target=True,
        n_jobs=1,
    ):
        """Constructor.
        """
        self.initial_predictors = list(initial_predictors)
        if (len(all_predictors) > 0) and (not set(initial_predictors).issubset(set(all_predictors))):
            raise ValueError('all_predictors should be either empty (so all columns will be considered predictors) or initial_predictors must be subset of all_predictors')
        else:
            self.all_predictors = list(all_predictors)

        self.dummy_regression = dummy_regression
        if self.dummy_regression:
            if dummy_bindings is None:
                raise ValueError('No full logit constraints received.')
            # test whether we received grouping as the entity with contraints. in such case, they will be in property called dummies
            elif type(dummy_bindings) in (Grouping, InteractiveGrouping):
                self.dummy_bindings = dummy_bindings.get_dummies()
            else:
                self.dummy_bindings = dummy_bindings
            # check whether all predictors given by all_predictors and initial_predictors exist also in dummy_bindings
            for predictor in self.all_predictors + self.initial_predictors:
                if predictor not in self.dummy_bindings.keys():
                    raise ValueError('Predictor '+predictor+' not specified in dummy_bindings')
        else:
            self.dummy_bindings = None
                
        self.selection_method = selection_method
        self.max_iter = max_iter
        self.min_increase = min_increase
        self.max_decrease = max_decrease
        self.max_predictors = max_predictors

        if self.dummy_regression:
            self.max_correlation = 1
            self.beta_sgn_criterion = False
        else:
            self.max_correlation = max_correlation
            self.beta_sgn_criterion = beta_sgn_criterion
        
        self.penalty = penalty
        self.C = C
        
        self.correlation_sample = correlation_sample
        
        self.use_cv = use_cv
        self.cv_folds = cv_folds
        self.cv_seed = cv_seed
        self.cv_stratify_by_target = cv_stratify_by_target
        
        self.n_jobs = n_jobs

    def _cros_val_auc(self, X, y, weights=None):
        """Performs crossvalidation training and calculates average (cross)validation Gini

        Args:
            X (pd.DataFrame): predictors
            y (ps.Series): target
            weights (pd.Series, optional): Observation weights. Defaults to None.

        Returns:
            float: cross validated AUC
        """
        # convert pandas structures to numpy structures
        X = X.values
        y = y.values
        if weights is not None:
            weights = weights.values

        # stratified k-fold
        if self.cv_stratify_by_target:
            kf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.cv_seed)
            kf.get_n_splits(X, y)
            split_indexes = kf.split(X, y)
        else:
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.cv_seed)
            kf.get_n_splits(X)
            split_indexes = kf.split(X)

        aucs = []

        for train_index, test_index in split_indexes:

            newModel = LogisticRegression(
                penalty=self.penalty, C=self.C, solver="liblinear"
            )

            # train/test split
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if weights is not None:
                weights_train, weights_test = weights[train_index], weights[test_index]
                newModel.fit(X_train, y_train, sample_weight=weights_train)
                y_pred = newModel.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_pred, sample_weight=weights_test)

            else:
                newModel.fit(X_train, y_train)
                y_pred = newModel.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_pred)

            # evaluate score metric
            aucs.append(auc)

        avg_auc = sum(aucs) / len(aucs)

        return avg_auc

    def _one_model(
        self, X, y, X_valid, y_valid, sample_weight=None, sample_weight_valid=None
    ):
        """Calculates a regression model on the given set and with given parameters returns its gini and indicator whether the betas have the same signum.

        Args:
            X (pd.DataFrame): training sample predictors
            y (ps.Series):  training sample target
            X_valid (pd.DataFrame): validation sample predictors
            y_valid (ps.Series): validation sample target
            sample_weight (pd.Series, optional): training sample observation weights. Defaults to None.
            sample_weight_valid (pd.Series, optional): validation sample observation weights. Defaults to None.

        Returns:
            float: Gini (caclulated on validation sample)
            bool: boolean indicator of same signum of all betas in model
        """

        # train model using training sample
        # this must be done even for Cross Validation !!!
        # the reason is: criteria as same beta signum must be evaluated on one firm set of coefficients
        newModel = LogisticRegression(
            penalty=self.penalty, C=self.C, solver="liblinear"
        )
        # if there is a variable with weights, use weighted regression
        if sample_weight is None:
            newModel.fit(X, y)
        else:
            newModel.fit(X, y, sample_weight)

        # without Cross Validation
        if not (self.use_cv):
            # measure Gini of such model
            predictions = newModel.predict_proba(X_valid)[:, 1]
            if sample_weight_valid is None:
                gini_result = 200 * roc_auc_score(y_valid, predictions) - 100
            else:
                gini_result = 200 * roc_auc_score(y_valid, predictions, sample_weight=sample_weight_valid) - 100

        # with Cross Validation
        else:
            # the weight is None codition is resolved inside the called function
            cv_auc = self._cros_val_auc(X, y, sample_weight)
            gini_result = 200 * cv_auc - 100

        # sum of absolute values of beta is the same as absolute value of sum of beta
        # which occurs if and only if all the betas have the same signum
        if self.dummy_regression:
            same_beta_sgn = None
        elif sum(sum(abs(newModel.coef_))) == abs(sum(sum(newModel.coef_))):
            same_beta_sgn = True
        else:
            same_beta_sgn = False

        return gini_result, same_beta_sgn

    def _max_abs_corr(self, cormat, predictors):
        """returns maximum absolute correlation in the given set of predictors

        Args:
            cormat (pd.DataFrame): Correlation matrix of all preidctors
            predictors (list of str): list of predictors whose correlation should be evaluated
        Returns:
            float: max abs correlation within predictors
        """

        if not self.dummy_regression:

            # correlation submatrix for predictor set given by the model
            subcormat = cormat[predictors].loc[predictors]

            # set diagonal values to 0 (as the original 1 should not enter the maximum calculation)
            np.fill_diagonal(subcormat.values, 0)

            # calculates absolute value of the correlations (we are interested in both negative and positive correlations)
            # then finds the greatest of these values
            max_corr = abs(subcormat).values.max()

            return max_corr

        else:

            # for dummies, correlation does not make much sense, return NaN
            return np.nan

    def _find_same_model(self, predictors, allModels):
        """determine whether there is already model with given predictors set in allModels

        Args:
            predictors (list of str): list of predictors in model we are searching for
            allModels (list of dict): list of metadata dicts of each model that has been already considered

        Returns:
            int: number of identical models that have been already considered
            float or None: Gini of the identical model (if such model exists)
            bool or None: value of used_before of the identical model (if such model exists)
            float or None: max corr of the identical model (if such model exists)
            bool or None: same_beta_sgn of the identical model (if such model exists)
        """

        # if such model was calculated before, use numbers from the previous calculation
        same_model_list = list(
            filter(lambda model: set(model["predictors"]) == set(predictors), allModels)
        )

        # number of same models already calculated
        same_model_count = len(same_model_list)

        # assign values from the first same model or empty values
        if same_model_count > 0:
            gini_result = same_model_list[0]["Gini"]
            used_before = same_model_list[0]["used"]
            max_corr = same_model_list[0]["maxCorr"]
            same_beta_sgn = same_model_list[0]["betaSgn"]
        else:
            gini_result = None
            used_before = None
            max_corr = None
            same_beta_sgn = None

        return same_model_count, gini_result, used_before, max_corr, same_beta_sgn

    def _filter_possible_models(self, iterNum, allModels, incdec):
        """filter models which are feasible based on given criteria

        Args:
            iterNum (int): iteration number
            allModels (list of dict): list of metadata dicts of each model that has been already considered
            incdec (str): 'inc' or 'dec'. Whether we are are looking for models where Gini increment >= min_increase or where Gini decrease <= max_decrease

        Returns:
            list of dict: list of models that fulfill criteria specified by incdec and were not used before
        """

        if incdec == "inc":
            # filter models that meet criteria: was not used before, Gini increment >= min_increase
            possibleModels = list(
                filter(
                    lambda model: model["iteration"] == iterNum
                    and model["addrm"] == 1
                    and model["used"] == 0
                    and model["diff"] >= self.min_increase,
                    allModels,
                )
            )

        elif incdec == "dec":
            # filter models that meet criteria: was not used before, Gini decrease <= max_decrease
            possibleModels = list(
                filter(
                    lambda model: model["iteration"] == iterNum
                    and model["addrm"] == -1
                    and model["used"] == 0
                    and model["diff"] >= -self.max_decrease,
                    allModels,
                )
            )

        else:
            possibleModels = allModels

        # additional filtering in case we want all betas to have the same signum
        if self.beta_sgn_criterion:
            possibleModels = list(
                filter(lambda model: model["betaSgn"] == True, possibleModels)
            )

        # additional filtering in case we have a condition for maximum correlation
        if self.max_correlation < 1 and self.max_correlation >= 0:
            possibleModels = list(
                filter(
                    lambda model: model["maxCorr"] <= self.max_correlation,
                    possibleModels,
                )
            )

        # order such models by added Gini desc
        possibleModels = sorted(possibleModels, key=lambda d: d["diff"], reverse=True)

        # if there is such model used flag must be updated
        if len(possibleModels) > 0:
            toUpdateModels = list(
                filter(
                    lambda model: model["predictors"]
                    == possibleModels[0]["predictors"],
                    allModels,
                )
            )
            toUpdateModels = [m.update({"used": 1}) for m in toUpdateModels]

        return possibleModels, allModels

    def _dummy_variables(self, predictors, columns, nuniques, X):
        """for given list of predictors, this finds dummy variables in columns related to these predictors, using relationships given by self.dummy_bindings

        Args:
            predictors (list of str): predictors we are looking for dummy variable for
            columns (list of str): columns we are looking in
            nuniques (dict): number of unique values in each column we are looking in
            X (pd.DataFrame): data

        Returns:
            list of str: list of variables (sublist of columns) which are dummy variables for predictors specified in parameter predictors
        """

        variables_tmp = []

        # for each predictor, add dummies related to it into a temporary dummy set.
        # the task is to add only such dummies that are linearly independent, i.e. we have to omit one of them (as together, they add up to vector of ones) and we also have to omit trivial dummies (vectors of zeros) if they exist
        for predictor in predictors:
            first = True
            for variable in self.dummy_bindings[predictor]:
                if variable in columns:
                    # first of the dummies for each original predictor will NOT enter the column set to solve linear dependency
                    if first:
                        first = False
                    # dummy which has only 1 or 0 unique value will NOT enter the column set as it is useless and linearly dependent with everything
                    elif nuniques[variable] <= 1:
                        pass
                    # if both previous conditions are false, the dummy WILL enter the column set
                    else:
                        variables_tmp.append(variable)

        variables = []

        # check whether the dummy set does not include two variables which have identical values (might hapen mainly with dummies for NaN values for predictors that came from the same source)
        for variable in variables_tmp:
            identity_found = False
            # search all dummies that already have been added into the final set
            for previous_variable in variables:
                # if identical variable was no found yet, check identity
                if not identity_found:
                    identity_found = X[previous_variable].equals(X[variable])
            # if and identical variable was found, we will not add "variable" to the final variable set (as the identical is already there). otherwise, we add it
            if not identity_found:
                variables.append(variable)

        return variables

    

    def _remove_predictor(
        self,
        currentModel,
        remPredictor,
        allModels,
        X,
        y,
        X_valid,
        y_valid,
        sample_weight,
        sample_weight_valid,
        cormat,
        modelID,
        iterNum,
        newPredSet,
        newGini,
        currentGini,
        betaSgn,
        maxCorr,
        nuniques,
    ):
        """Tries to remove one predictor from the model (part of iteration).

        Args:
            currentModel (dict): metadat of model we are removing predictor from
            remPredictor (str): predictor to remove
            allModels (list of dict): metadata of all models so far
            X (pd.DataFrame): training sample predictors
            y (ps.Series):  training sample target
            X_valid (pd.DataFrame): validation sample predictors
            y_valid (ps.Series): validation sample target
            sample_weight (pd.Series): training sample observation weights
            sample_weight_valid (pd.Series): validation sample observation weights
            cormat (pd.DataFrame): correlation matrix of X
            modelID (int): ID of model to be constructed by this method
            iterNum (int): current iteration number
            newPredSet (list of str): will be overwritten
            newGini (float): will be overwritten
            currentGini (float): gini of current model
            betaSgn (bool): will be overwritten
            maxCorr (float): will be overwritten
            nuniques (dict): dictionary of number of unique values in each column, used if dummy regression is performed

        Returns:
            dict: metadata of model with the removed predictor
        """
        # try model with removal of one predictor
        newPredSet = [pred for pred in currentModel["predictors"] if pred != remPredictor]
        if self.dummy_regression:
            newDummySet = self._dummy_variables(newPredSet, X.columns, nuniques, X)

        # if such model was calculated before, use numbers from the previous calculation
        sameModelCnt, newGini, usedBefore, maxCorr, betaSgn = self._find_same_model(
            newPredSet, allModels
        )

        # else calculate the model now
        if sameModelCnt < 1:
            if self.dummy_regression:
                # logit regression - use special function
                newGini, betaSgn = self._one_model(
                    X[newDummySet],
                    y,
                    X_valid[newDummySet],
                    y_valid,
                    sample_weight,
                    sample_weight_valid,
                )
            else:
                # logit regression - use special function
                newGini, betaSgn = self._one_model(
                    X[newPredSet],
                    y,
                    X_valid[newPredSet],
                    y_valid,
                    sample_weight,
                    sample_weight_valid,
                )
            # maximum correlation between the predictors
            maxCorr = self._max_abs_corr(cormat, newPredSet)
            usedBefore = 0

        # add to data structure
        # modelID = modelID + 1
        result = {
            "ID": modelID,
            "iteration": iterNum,
            "addrm": -1,
            "predictors": newPredSet,
            "prednum": len(newPredSet),
            "Gini": newGini,
            "diff": newGini - currentGini,
            "used": usedBefore,
            "betaSgn": betaSgn,
            "maxCorr": maxCorr,
        }
        return result


    def _add_predictors(
        self,
        currentModel,
        newPredictor,
        allModels,
        X,
        y,
        X_valid,
        y_valid,
        sample_weight,
        sample_weight_valid,
        cormat,
        modelID,
        iterNum,
        newPredSet,
        newGini,
        currentGini,
        betaSgn,
        maxCorr,
        nuniques,
    ):
        """Tries to add one predictor from the model (part of iteration).

        Args:
            currentModel (dict): metadat of model we are removing predictor from
            newPredictor (str): predictor to add
            allModels (list of dict): metadata of all models so far
            X (pd.DataFrame): training sample predictors
            y (ps.Series):  training sample target
            X_valid (pd.DataFrame): validation sample predictors
            y_valid (ps.Series): validation sample target
            sample_weight (pd.Series): training sample observation weights
            sample_weight_valid (pd.Series): validation sample observation weights
            cormat (pd.DataFrame): correlation matrix of X
            modelID (int): ID of model to be constructed by this method
            iterNum (int): current iteration number
            newPredSet (list of str): will be overwritten
            newGini (float): will be overwritten
            currentGini (float): gini of current model
            betaSgn (bool): will be overwritten
            maxCorr (float): will be overwritten
            nuniques (dict): dictionary of number of unique values in each column, used if dummy regression is performed

        Returns:
            dict: metadata of model with the added predictor
        """
        # try model with addition of new predictor
        newPredSet = currentModel["predictors"] + [newPredictor]
        if self.dummy_regression:
            newDummySet = self._dummy_variables(newPredSet, X.columns, nuniques, X)

        # if such model was calculated before, use numbers from the previous calculation
        sameModelCnt, newGini, usedBefore, maxCorr, betaSgn = self._find_same_model(
            newPredSet, allModels
        )

        # else calculate the model now
        if sameModelCnt < 1:
            if self.dummy_regression:
                # logit regression - use special function
                newGini, betaSgn = self._one_model(
                    X[newDummySet],
                    y,
                    X_valid[newDummySet],
                    y_valid,
                    sample_weight,
                    sample_weight_valid,
                )
            else:
                # logit regression - use special function
                newGini, betaSgn = self._one_model(
                    X[newPredSet],
                    y,
                    X_valid[newPredSet],
                    y_valid,
                    sample_weight,
                    sample_weight_valid,
                )
            # maximum correlation between the predictors
            maxCorr = self._max_abs_corr(cormat, newPredSet)
            usedBefore = 0

        # add to data structure
        modelID = modelID + 1
        result = {
            "ID": modelID,
            "iteration": iterNum,
            "addrm": 1,
            "predictors": newPredSet,
            "prednum": len(newPredSet),
            "Gini": newGini,
            "diff": newGini - currentGini,
            "used": usedBefore,
            "betaSgn": betaSgn,
            "maxCorr": maxCorr,
        }
        return result


    def fit(
        self,
        X,
        y,
        X_valid=None,
        y_valid=None,
        sample_weight=None,
        sample_weight_valid=None,
        silent=False,
    ):
        """fit regression model while iterating in stepwise/forward/backward way.

        The fit method can be called with two arguments fit(X,y) or with four agruments fit(X_train,y_train,X_valid,y_valid). When called with four arguments, the Gini is measured on the validation sample (i.e. validation sample is used for decisions about what steps to be done in stepwise).

        There are another optional arguments, sample_weight and sample_weight_valid where you can put the vector (data set column) with weights of the observations for the train and validation samples.
            
        Args:
            X (pd.DataFrame): df with predictors - training sample
            y (pd.Series): target - training sample
            X_valid (pd.DataFrame, optional): df with predictors - validation sample. if unspecified, training sample is used for validation. if use_cv = True in object initialization, both train and validation sample are used for cv folds. (default: None)
            y_valid (pd.Series, optional): target - validation sample. if unspecified, training sample is used for validation. if use_cv = True in object initialization, both train and validation sample are used for cv folds. (default: None)
            sample_weight (pd.Series, optional): obervarion weights - training sample. if unspecified, weights = 1 for each observation are used. (default: None)
            sample_weight_valid (pd.Series, optional): obervarion weights. if unspecified, weights = 1 for each observation are used. (default: None)
            silent (bool, optional): no output? (default: False)
        """

        if self.use_cv:
            if (X_valid is not None) or (y_valid is not None):
                if not silent:
                    print("Cross validation will be used for the union of training and validation sample.")
                    print("If you want to use cross validation for training sample only, do not submit any validation sample.")
                X = pd.concat([X, X_valid])
                y = pd.concat([y, y_valid])
                X_valid = X
                y_valid = y
                if sample_weight is not None:
                    sample_weight = pd.concat([sample_weight, sample_weight_valid])
                else:
                    sample_weight = None
                sample_weight_valid = sample_weight
            else:
                if not silent:
                    print("Cross validation will be used for the training sample.")
                X_valid = X
                y_valid = y
                sample_weight_valid = sample_weight
        else:
            if (X_valid is not None) and (y_valid is not None):
                if not silent:
                    print("Regression will be trained using training sample, Gini will be evaluated using validation sample.")
            else:
                X_valid = X
                y_valid = y
                sample_weight_valid = sample_weight
                if not silent:
                    print("No validation sample submitted, Gini will be evaluated using training sample.")

        nuniques = X.nunique()
        
        # if all_predictors is empty, it means that we want use all variables of X as potential predictive columns, so we add all predictors we find in X to all_predictors
        if len(self.all_predictors) == 0:
            for column in X.columns:
                if self.dummy_regression:
                    for predictor_name, predictor_dummies in self.dummy_bindings:
                        if column in predictor_dummies:
                            self.all_predictors.append(predictor_name)
                else:
                    self.all_predictors.append(column)

        #make sure that min_increase > max_decrease >= 0 - otherwise there might be an infinite loop
        if self.max_decrease < 0:
            self.max_decrease = 0
            if not silent:
                print('max_decrease parameter was invalid, it is set to 0 now.')
        if self.min_increase <= self.max_decrease:
            self.min_increase = self.max_decrease+0.01
            if not silent:
                print('min_increase parameter was <= max_decrease, it is set to max_decrease+0.01 now')

        #make sure we are in such settings that beta sgn criterion and max corr param make sense
        if (
            (self.selection_method != "forward") or (self.dummy_regression)
        ) and (
            (self.max_correlation < 1) or (self.beta_sgn_criterion)
        ):
            self.beta_sgn_criterion = False
            self.max_correlation = 1
            if not silent:
                print('Beta signum criterion and max correlation will not be used as the selection method is not "forward" or dummy regression is used.')

        # correlation matrix of all the predictors, calculated on sample with size defined by a parameter
        if not self.dummy_regression:
            rowsCount = len(X.index)
            if rowsCount <= self.correlation_sample:
                cormat = X.corr()
            else:
                cormat = X.sample(self.correlation_sample).corr()
        else:
            cormat = None

        stopNow = 0 #variable consolidating stopping criteria
        iterNum = 0 #current step number
        if not silent:
            print("Iteration ",iterNum)
        modelID = 0 #unique model identifier
        
        #number of predictors in the initial set
        newPredNum = len(self.initial_predictors)
        newPredSet = self.initial_predictors
        
        if self.dummy_regression:
            newDummySet = self._dummy_variables(newPredSet, list(X.columns), nuniques, X)

        # there are some predictors in the inital set, the create the initial model of it
        if newPredNum > 0:
            if self.dummy_regression:
            # logit regression
                newGini, betaSgn = self._one_model(X[newDummySet], y, X_valid[newDummySet], y_valid, sample_weight, sample_weight_valid)
            else:
            # logit regression
                newGini, betaSgn = self._one_model(X[newPredSet], y, X_valid[newPredSet], y_valid, sample_weight, sample_weight_valid)
            maxCorr = self._max_abs_corr(cormat, newPredSet)
        # the inital set of predictors is empty, then the model has 0 Gini
        else:
            newGini, betaSgn, maxCorr = 0, True, 0

        # init data structure with all models so far
        # iteration number, add or remove, set of predictors, number of predictors, Gini, Gini difference, model used
        allModels = [
            {
                "ID": modelID,
                "iteration": 0,
                "addrm": 0,
                "predictors": newPredSet,
                "prednum": newPredNum,
                "Gini": newGini,
                "diff": 0,
                "used": 1,
                "betaSgn": betaSgn,
                "maxCorr": maxCorr,
            }
        ]
        
        #output to screen
        if not silent:
            print()
            print(f"Iter    Gini    GiniΔ  #Pred   AddedPred                                RemovedPred")
            print(f"[{iterNum:>2}]   {newGini:>5.2f}              {newPredNum:<5}{newPredSet}")

        # go to step 1
        iterNum = iterNum + 1
        if iterNum > self.max_iter:
            stopNow = 1
            currentModel = allModels[0]

        #iterate until stopping criteria met
        while stopNow == 0:

            # find the model we want to tune
            originalModel = (
                list(
                    filter(
                        lambda model: model["iteration"] == iterNum - 1
                        and model["addrm"] == 0,
                        allModels,
                    )
                )
            )[0]
            
            #get the Gini of this model from the previous step
            currentModel = originalModel
            currentGini = currentModel['Gini']
            addedPredictor = None
            removedPredictor = None

            # check whether current number of predictors == max_predictors parameter value
            if (
                (self.selection_method == "stepwise")
                or (self.selection_method == "forward")
            ) and (
                (currentModel["prednum"] < self.max_predictors)
                or (self.max_predictors <= 0)
            ):
                if self.n_jobs <= 1:
                    # we still can add more predictors
                    # iterate through all unused predictors
                    for newPredictor in [pred for pred in self.all_predictors if pred not in currentModel["predictors"]]:

                        # try model with addition of new predictor
                        result = self._add_predictors(
                            currentModel,
                            newPredictor,
                            allModels,
                            X,
                            y,
                            X_valid,
                            y_valid,
                            sample_weight,
                            sample_weight_valid,
                            cormat,
                            modelID,
                            iterNum,
                            newPredSet,
                            newGini,
                            currentGini,
                            betaSgn,
                            maxCorr,
                            nuniques,
                        )

                        allModels.append(result)

                else:
                    def add_predictors(newPredictor):
                        return self._add_predictors(
                            currentModel,
                            newPredictor,
                            allModels,
                            X,
                            y,
                            X_valid,
                            y_valid,
                            sample_weight,
                            sample_weight_valid,
                            cormat,
                            modelID,
                            iterNum,
                            newPredSet,
                            newGini,
                            currentGini,
                            betaSgn,
                            maxCorr,
                            nuniques,
                        )

                    with ThreadPoolExecutor(self.n_jobs) as p:
                        allModels += list(
                            p.map(
                                add_predictors,
                                [pred for pred in self.all_predictors if pred not in currentModel["predictors"]],
                            )
                        )

                # filter only models fulfilling criteria to be used
                possibleModels, allModels = self._filter_possible_models(
                    iterNum, allModels, incdec="inc"
                )

                # if there is such model, set that the selected model is the current one
                if len(possibleModels) > 0:
                    currentModel = possibleModels[0]
                    currentGini = currentModel["Gini"]
                    addedPredictor = (
                        [pred for pred in currentModel["predictors"] if pred not in originalModel["predictors"]]
                    ).pop()

                #else the original model proceeds to the next step   

            # if more than one predictor, try to remove one
            if (
                (self.selection_method == "stepwise")
                or (self.selection_method == "backward")
            ) and currentModel["prednum"] > 1:

                # iterate through all used predictors
                if self.n_jobs <= 1:
                    for remPredictor in currentModel["predictors"]:

                        # try model with removal of one predictor
                        result = self._remove_predictor(
                            currentModel,
                            remPredictor,
                            allModels,
                            X,
                            y,
                            X_valid,
                            y_valid,
                            sample_weight,
                            sample_weight_valid,
                            cormat,
                            modelID,
                            iterNum,
                            newPredSet,
                            newGini,
                            currentGini,
                            betaSgn,
                            maxCorr,
                            nuniques,
                        )
                        allModels.append(result)
                else:
                    with ThreadPoolExecutor(self.n_jobs) as p:
                        allModels += list(
                            p.map(
                                lambda remPredictor: self._remove_predictor(
                                    currentModel,
                                    remPredictor,
                                    allModels,
                                    X,
                                    y,
                                    X_valid,
                                    y_valid,
                                    sample_weight,
                                    sample_weight_valid,
                                    cormat,
                                    modelID,
                                    iterNum,
                                    newPredSet,
                                    newGini,
                                    currentGini,
                                    betaSgn,
                                    maxCorr,
                                    nuniques,
                                ),
                                currentModel["predictors"],
                            )
                        )

                # filter only models fulfilling criteria to be used
                possibleModels, allModels = self._filter_possible_models(
                    iterNum, allModels, incdec="dec"
                )

                # if there is such model, set that the selected model is the current one
                if len(possibleModels) > 0:
                    currentModel = possibleModels[0]
                    currentGini = currentModel["Gini"]
                    removedPredictor = (
                        [pred for pred in originalModel["predictors"] if pred not in currentModel["predictors"]]
                    ).pop()

                # else the original model proceeds to the next step

            # add the basic model for the next iteration to the data set
            modelID = modelID + 1
            allModels.append(
                {
                    "ID": modelID,
                    "iteration": iterNum,
                    "addrm": 0,
                    "predictors": currentModel["predictors"],
                    "prednum": currentModel["prednum"],
                    "Gini": currentModel["Gini"],
                    "diff": 0,
                    "used": 1,
                    "betaSgn": currentModel["betaSgn"],
                    "maxCorr": currentModel["maxCorr"],
                }
            )
            # output to screen
            changedGini = currentGini - originalModel["Gini"]
            if not silent:
                print(
                    f"[{iterNum:>2}]   {currentModel['Gini']:>5.2f}   {changedGini:>+6.2f}    {currentModel['prednum']:>2}    {addedPredictor if addedPredictor else '':<40} {removedPredictor if removedPredictor else ''}"
                )

            # new iteration number
            iterNum = iterNum + 1

            # if the model proceeding to the next iteration is the original model or maximal interation number achived, stop
            if (currentModel == originalModel) or (iterNum > self.max_iter):
                stopNow = 1

        # output attributes creation
        # set data frame with all iterations
        self.progress = pd.DataFrame.from_records(allModels)

        # set final model description
        #set final model description
        if self.dummy_regression:
            self.predictors_orig = currentModel["predictors"]
            self.predictors = self._dummy_variables(self.predictors_orig, X.columns, nuniques, X)
        else:
            self.predictors_orig = currentModel["predictors"]
            self.predictors = currentModel["predictors"]

        # fits the final model
        self.model = LogisticRegression(
            penalty=self.penalty, C=self.C, solver="liblinear"
        )
        if sample_weight is None:
            self.model.fit(X[self.predictors], y)
        else:
            self.model.fit(X[self.predictors], y, sample_weight)

        self.coef = self.model.coef_[0]
        self.intercept = self.model.intercept_[0]

        # variables for backward compatibility
        self.coef_, self.intercept_, self.final_model_, self.final_predictors_, self.final_predictors_orig_, self.model_progress_ =\
            [self.coef], [self.intercept], self.model, self.predictors, self.predictors_orig, self.progress

        return

    def predict(self, X):
        """uses the fitted final model to calculate predictions

        Args:
            X (pd.DataFrame): data, must include all predictors that are part of the fitted model

        Returns:
            np.array: array with predictions for data X
        """
        check_is_fitted(self, ["progress", "predictors", "model"])

        # calculates the predicted probabilities
        y = self.model.predict_proba(X[self.predictors])[:, 1]

        return y


    def draw_gini_progression(self, output_file=None):
        """Draws progression plot of gini values during step-wise fitting.
        Plots values of gini based on step during step-wise fitting of model.
        Draws a plot with mathplotlib to notebook. 

        Args:
            output_file (str, optional): relative path to file to export
        """
        it = range(
            0, len(self.progress[self.progress["addrm"] == 0]["prednum"])
        )
        _ = self.progress[self.progress["addrm"] == 0]["prednum"]
        ginis = self.progress[self.progress["addrm"] == 0]["Gini"]
        plt.figure(figsize=(7, 7))
        plt.plot(it, ginis)
        _, _ = plt.ylim()
        plt.xlabel("Iteration")
        plt.ylabel("Gini")
        plt.title("Stepwise model selection")
        plt.axis("tight")
        if output_file is not None:
            plt.savefig(output_file, bbox_inches="tight", dpi=72)
        plt.show()
        plt.clf()
        plt.close()


    def print_final_model(self):
        """Exports final model to pd.Dataframe.

        Multiindex is used for original and dummy predictors,
        coefficiets are in the only value column.

        Returns:
            pd.DataFrame: model table
        """
        preds = self.predictors
        coefs = self.coef
        intercept = self.intercept
        
        if self.dummy_regression:

            inverted_dummy = {}
            for orig, dummies in self.dummy_bindings.items():
                for dummy in dummies:
                    inverted_dummy[dummy] = orig
            
            coefs =[coef for pred,coef in sorted(zip(preds,coefs))]
            preds = sorted(preds)
            orig_preds = [inverted_dummy[pred] for pred in preds]
            
            m_index = pd.MultiIndex.from_arrays([["", *orig_preds], ["Intercept", *preds]], names=["Original", "Dummy"])
            
            df_out = pd.DataFrame(data=[intercept, *coefs], index=m_index, columns=["Coefficient"])

        else:

            df_out = pd.DataFrame.from_dict({"Variable": ["Intercept"] + preds, "Coefficient": [intercept] + list(coefs)})
            df_out.index = df_out["Variable"]
            df_out = df_out[["Coefficient"]]

        return df_out


    def marginal_contribution(
        self,
        X,
        y,
        X_valid=None,
        y_valid=None,
        sample_weight=None,
        sample_weight_valid=None,
        predictors_to_add=[],
        output_path=None,
        silent=False,
    ):
        """Calculates marginal contribution of each predictor in the fitted model.
        Also can calculate marginal contribution of additional predictors on top of the fitted model.

        Args:
            X (pd.DataFrame): df with predictors - training sample
            y (pd.Series): target - training sample
            X_valid (pd.DataFrame, optional): df with predictors - validation sample. if unspecified, training sample is used for validation. if use_cv = True in object initialization, both train and validation sample are used for cv folds. (default: None)
            y_valid (pd.Series, optional): target - validation sample. if unspecified, training sample is used for validation. if use_cv = True in object initialization, both train and validation sample are used for cv folds. (default: None)
            sample_weight (pd.Series, optional): obervarion weights - training sample. if unspecified, weights = 1 for each observation are used. (default: None)
            sample_weight_valid (pd.Series, optional): obervarion weights. if unspecified, weights = 1 for each observation are used. (default: None)
            predictors_to_add (list, optional): List of predictors which should be tried to be added to the fitted model, so their marginal contribution on top of it can be calculated. Defaults to [].
            output_path (str, optional): path for csv (including file name) where the results should be outputted. Defaults to None.
            silent (bool, optional): no on screen output? Defaults to False.

        Returns:
            pd.DataFrame: table with marginal contribution value for each predictor and each variable from predictors_to_add list.
        """

        current_predictors = self.predictors
        predictors_to_add = [pred for pred in predictors_to_add if pred not in current_predictors]

        # adding predictors
        temp_model = GiniStepwiseLogit(
            initial_predictors=current_predictors,
            all_predictors=current_predictors+predictors_to_add,
            dummy_regression=self.dummy_regression,
            dummy_bindings=self.dummy_bindings,
            selection_method="forward",
            max_iter=1,
            min_increase=self.min_increase,
            max_decrease=self.max_decrease,
            max_predictors=0,
            max_correlation=self.max_correlation,
            beta_sgn_criterion=self.beta_sgn_criterion,
            penalty=self.penalty,
            C=self.C,
            correlation_sample=self.correlation_sample,
            use_cv=self.use_cv,
            cv_folds=self.cv_folds,
            cv_seed=self.cv_seed,
            cv_stratify_by_target=self.cv_stratify_by_target,
            n_jobs=self.n_jobs,
        )

        temp_model.fit(X, y, X_valid, y_valid, sample_weight, sample_weight_valid, silent=silent)
        mcadd = temp_model.progress[temp_model.progress["addrm"]==1].copy()
        mcadd.loc[:, "predictor"] = mcadd.loc[:, "predictors"].apply(lambda x: list(set(x)-set(current_predictors))[0])
        mcadd = mcadd[["addrm", "predictor", "Gini", "diff"]].sort_values("diff", ascending=False)
        mcadd["addrm"] = "added"

        # removing predictors
        temp_model = GiniStepwiseLogit(
            initial_predictors=current_predictors,
            all_predictors=current_predictors,
            dummy_regression=self.dummy_regression,
            dummy_bindings=self.dummy_bindings,
            selection_method="backward",
            max_iter=1,
            min_increase=self.min_increase,
            max_decrease=self.max_decrease,
            max_predictors=0,
            max_correlation=self.max_correlation,
            beta_sgn_criterion=self.beta_sgn_criterion,
            penalty=self.penalty,
            C=self.C,
            correlation_sample=self.correlation_sample,
            use_cv=self.use_cv,
            cv_folds=self.cv_folds,
            cv_seed=self.cv_seed,
            cv_stratify_by_target=self.cv_stratify_by_target,
            n_jobs=self.n_jobs,
        )

        temp_model.fit(X, y, X_valid, y_valid, sample_weight, sample_weight_valid, silent=silent)
        mcrem = temp_model.progress[temp_model.progress["addrm"]==-1].copy()
        mcrem.loc[:, "predictor"] = mcrem.loc[:, "predictors"].apply(lambda x: list(set(current_predictors)-set(x))[0])
        mcrem = mcrem[["addrm", "predictor", "Gini", "diff"]].sort_values("diff", ascending=False)
        mcrem["addrm"] = "removed"

        self.mc_table = pd.concat([mcadd, mcrem], axis=0)

        if output_path is not None:
            self.mc_table.to_csv(output_path)

        return self.mc_table






class L1GiniModelSelection:
    """Trains model using various values of C parameter of L1 regularization which leads to consequent adding of predictors.
    We start with no predictor in the model and try to add predictors.
    Optimal model is based on convergence criteria which are set by the parameters and validation Gini.

    Args:
        steps (int, optional): number of steps of grid search. Defaults to 50.
        grid_length (float, optional): length of the grid for grid search for logarithm of C (L1 regularization parameter). Defaults to 5.
        log_C_init (float, optional): initial value of log10 of C parameter for grid search (L1 regularization parameter). Defaults to None.
        max_predictors (int, optional): maximal number of predictors to enter the model. Ignored if set to 0. Defaults to 20.
        max_correlation (float, optional): maximal absolute value of correlation of predictors in the model (variable with larger correlation with existing predictors will not be added to the model). Defaults to 1.
        beta_sgn_criterion (bool, optional): if this is set to True, all the betas in the model must have the same signature (all positive or all negative). Defaults to False.
        stop_when_decr (bool, optional): if this is set to True, the Gini must increase in each iteration. Models with more predictor having lower Gini than the previous will be considered invalid.. Defaults to False.
        stop_immediately (bool, optional): the iteration process will be stopped immediately after a model which is not fulfilling the criteria (max_predictors, max_correlation, beta_sgn_criterion, stop_when_decr) is found. No further models are searched for.. Defaults to False.
        correlation_sample (int, optional): for better performance, correlation matrix is calculated just on a sample of data. The size of the sample is set in this parameter. Defaults to 10000.
        penalty (str, optional): Whether to use L1 ('l1') penalty or L2 ('l2'). Defaults to "l1".
        use_cv (bool, optional): boolean if Cross Validation should be used instead of train/validation split. In this case, if both training and validation samples are presented to the fit method, they are concatenated together and are used for CV. Gini is evaluated as average of all CV's folds' validation Gini. Please, be aware that using CV after automatic grouping which was trained using train/validate split might lead to overfitted model. Defaults to False.
        cv_folds (int, optional): parameter for Cross Validation - number of folds. Defaults to 5.
        cv_seed (int, optional): parameter for Cross Validation - random seed used to split the folds. Defaults to 98765.
        cv_stratify_by_target (bool, optional): parameter for Cross Validation - whether the samples should be stratified so all of them the same mean target value. Defaults to True.
        n_jobs (int, optional): number of parallel jobs (for multi core computing). Defaults to 1.
    """

    def __init__(
        self,
        steps=50,
        grid_length=5,
        log_C_init=None,
        max_predictors=20,
        max_correlation=1,
        beta_sgn_criterion=False,
        stop_when_decr=False,
        stop_immediately=False,
        correlation_sample=10000,
        penalty="l1",
        use_cv=False,
        cv_folds=5,
        cv_seed=98765,
        cv_stratify_by_target=True,
        n_jobs=1,
    ):
        """Constructor
        """

        # number of iterations for L1 C parameter grid search
        self.steps = steps
        self.max_predictors = max_predictors
        self.max_correlation = max_correlation
        self.beta_sgn_criterion = beta_sgn_criterion
        self.correlation_sample = correlation_sample
        self.grid_length = grid_length
        if log_C_init is not None:
            self.C_init = 10 ** log_C_init
        else:
            self.C_init = None
        self.stop_when_decr = stop_when_decr
        self.stop_immediately = stop_immediately
        self.penalty = penalty
        # cross validation - boolean if to use it
        self.use_cv = use_cv
        # cross validation folds - number
        self.cv_folds = cv_folds
        # cross validation folds - random seed for shuffling
        self.cv_seed = cv_seed
        # cross validation folds - stratify by
        self.cv_stratify_by_target = cv_stratify_by_target
        # parallelization
        self.n_jobs = n_jobs

    def _cros_val_auc(self, X, y, weights=None, C=1):
        """Performs crossvalidation training and calculates average (cross)validation Gini

        Args:
            X (pd.DataFrame): predictors
            y (ps.Series): target
            weights (pd.Series, optional): Observation weights. Defaults to None.
            C (float, optional): C parameter of regularization

        Returns:
            float: cross validated AUC
        """
        # performs crossvalidation training and calculates average (cross)validation Gini

        # convert pandas structures to numpy structures
        X = X.values
        y = y.values
        if weights is not None:
            weights = weights.values

        # stratified k-fold
        if self.cv_stratify_by_target:
            kf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.cv_seed)
            kf.get_n_splits(X, y)
            split_indexes = kf.split(X, y)
        else:
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.cv_seed)
            kf.get_n_splits(X)
            split_indexes = kf.split(X)

        aucs = []

        for train_index, test_index in split_indexes:

            newModel = LogisticRegression(penalty=self.penalty, C=C, solver="liblinear")

            # train/test split
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if weights is not None:
                weights_train, weights_test = weights[train_index], weights[test_index]
                newModel.fit(X_train, y_train, sample_weight=weights_train)
                y_pred = newModel.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_pred, sample_weight=weights_test)

            else:
                newModel.fit(X_train, y_train)
                y_pred = newModel.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_pred)

            # evaluate score metric
            aucs.append(auc)

        avg_auc = sum(aucs) / len(aucs)

        return avg_auc

    def _max_abs_corr(self, cormat, predictors):
        """returns maximum absolute correlation in the given set of predictors

        Args:
            cormat (pd.DataFrame): Correlation matrix of all preidctors
            predictors (list of str): list of predictors whose correlation should be evaluated
        Returns:
            float: max abs correlation within predictors
        """

        # correlation submatrix for predictor set given by the model
        subcormat = cormat[list(predictors)].loc[list(predictors)]

        # set diagonal values to 0 (as the original 1 should not enter the maximum calculation)
        np.fill_diagonal(subcormat.values, 0)

        # calculates absolute value of the correlations (we are interested in both negative and positive correlations)
        # then finds the greatest of these values
        max_corr = abs(subcormat).values.max()

        return max_corr


    def _model_with_c(
        self,
        c,
        X,
        y,
        X_valid,
        y_valid,
        sample_weight,
        sample_weight_valid,
        cormat,
    ):
        """Train model with given value c of C parameter of regularization

        Args:
            c (float): value of C parameter
            X (pd.DataFrame): training sample predictors
            y (ps.Series):  training sample target
            X_valid (pd.DataFrame): validation sample predictors
            y_valid (ps.Series): validation sample target
            sample_weight (pd.Series, optional): training sample observation weights. Defaults to None.
            sample_weight_valid (pd.Series, optional): validation sample observation weights. Defaults to None.
            cormat (pd.DataFrame): Correlation matrix of all preidctors

        Returns:
            bool: whether the resulting model is fulfilling all criteria set by classes instance parameters
            dict: dictionary with resulting model metadata
        """

        # fit regression in this iteration
        # this must be done even for Cross Valiation !!!
        # some criteria as same beta sign must be evaluated on one firm (final) set of coefficients which will be the set trained on full sample
        lr = LogisticRegression(C=1, penalty=self.penalty, solver="liblinear")
        lr.set_params(C=c)
        # if there is a variable with weights, use weighted regression
        if sample_weight is None:
            lr.fit(X, y)
        else:
            lr.fit(X, y, sample_weight)

        # evaluate same-sign-beta and corr criteria
        coefs = lr.coef_.ravel().copy()
        intercepts = lr.intercept_
        coefs_nonzero = np.count_nonzero(coefs)
        coefs_samesign = abs(np.sum(coefs)) - np.sum(abs(coefs)) == 0
        predictors = set(X.columns[coefs != 0])
        # if intercepts != 0:
        #     predictors_w_intercept = predictors | {"Intercept"}
        # else:
        #     predictors_w_intercept = predictors
        if coefs_nonzero > 0:
            max_corr = self._max_abs_corr(cormat, predictors)
        else:
            max_corr = 0

        # Gini caluclation is done separately for CV and nonCV case:

        # without Cross Validation
        if not (self.use_cv):

            # calculate predictions
            pred_train = lr.predict_proba(X)[:, 1]
            pred_valid = lr.predict_proba(X_valid)[:, 1]

            # calculate Gini coefficients
            if sample_weight is None:
                gini_train = 200 * roc_auc_score(y, pred_train) - 100
            else:
                gini_train = (
                    200
                    * roc_auc_score(y, pred_train, sample_weight=sample_weight)
                    - 100
                )
            if sample_weight_valid is None:
                gini_valid = 200 * roc_auc_score(y_valid, pred_valid) - 100
            else:
                gini_valid = (
                    200
                    * roc_auc_score(
                        y_valid, pred_valid, sample_weight=sample_weight_valid
                    )
                    - 100
                )

        # with Cross Validation
        else:
            # measure Gini of such model
            # the weight is None codition is resolved inside the called function
            cv_auc = self._cros_val_auc(X, y, sample_weight, C=c)
            gini_valid = 200 * cv_auc - 100
            gini_train = np.nan

        # evaluate meeting of stopping criteria

        if len(self.allModels) > 0:
            gini_before = np.max([m["gini validate"] for m in self.allModels])
        else:
            gini_before = 0

        if (
            (
                (coefs_nonzero <= self.max_predictors)
                and (self.max_predictors > 0)
            )
            and ((self.beta_sgn_criterion == False) or (coefs_samesign == True))
            and (max_corr <= self.max_correlation)
            and ((self.stop_when_decr == False) or (gini_before <= gini_valid))
        ):
            criteria_passed = True
            criteria_failed = False
        else:
            criteria_passed = False
            criteria_failed = True

        output_dict = {
            "C": c,
            "predictors": predictors,
            "non-zero betas": coefs_nonzero,
            "same signature of betas": coefs_samesign,
            "max corr": max_corr,
            "gini train": gini_train,
            "gini validate": gini_valid,
            "criteria passed": criteria_passed,
            "coefs": coefs,
            "intercept": intercepts,
        }                    

        return criteria_failed, output_dict


    def fit(
        self,
        X,
        y,
        X_valid=None,
        y_valid=None,
        sample_weight=None,
        sample_weight_valid=None,
        progress_bar=False,
    ):
        """Fits model by iterating over values of C parameter of regularization.

        The fit method can be called with two arguments fit(X,y) or with four agruments fit(X_train,y_train,X_valid,y_valid). When called with four arguments, the Gini is measured on the validation sample (i.e. validation sample is used for decisions about what steps to be done in stepwise).

        There are another optional arguments, sample_weight and sample_weight_valid where you can put the vector (data set column) with weights of the observations for the train and validation samples.

        Note: The drawback of regularized model is that it is not calibrated to mean target.

        Args:
            X (pd.DataFrame): df with predictors - training sample
            y (pd.Series): target - training sample
            X_valid (pd.DataFrame, optional): df with predictors - validation sample. if unspecified, training sample is used for validation. if use_cv = True in object initialization, both train and validation sample are used for cv folds. (default: None)
            y_valid (pd.Series, optional): target - validation sample. if unspecified, training sample is used for validation. if use_cv = True in object initialization, both train and validation sample are used for cv folds. (default: None)
            sample_weight (pd.Series, optional): obervarion weights - training sample. if unspecified, weights = 1 for each observation are used. (default: None)
            sample_weight_valid (pd.Series, optional): obervarion weights. if unspecified, weights = 1 for each observation are used. (default: None)
            progress_bar (bool, optional): show progress bar? (default: False)

        Returns:
            [type]: [description]
        """

        if self.n_jobs > 1 and self.stop_immediately:
            print('Parallelization is being used. Immediate early stopping was turned off (algorithm will behave as with stop_immediately=False).')

        if self.use_cv:
            if (X_valid is not None) or (y_valid is not None):
                print(
                    "Cross validation will be used for the union of training and validation sample."
                )
                print(
                    "If you want to use cross validation for training sample only, do not submit any validation sample."
                )
                X = pd.concat([X, X_valid])
                y = pd.concat([y, y_valid])
                X_valid = X
                y_valid = y
                if sample_weight is not None:
                    sample_weight = pd.concat([sample_weight, sample_weight_valid])
                else:
                    sample_weight = None
                sample_weight_valid = sample_weight
            else:
                print("Cross validation will be used for the training sample.")
                X_valid = X
                y_valid = y
                sample_weight_valid = sample_weight
        else:
            if (X_valid is not None) and (y_valid is not None):
                print(
                    "Regression will be trained using training sample, Gini will be evaluated using validation sample."
                )
            else:
                X_valid = X
                y_valid = y
                sample_weight_valid = sample_weight
                print(
                    "No validation sample submitted, Gini will be evaluated using training sample."
                )

        # correlation matrix of all the predictors, calculated on sample with size defined by a parameter
        rowsCount = len(X.index)
        # the correlation is calculated on a sample, however for small data use the whole dataset
        if rowsCount <= self.correlation_sample:
            cormat = X.corr()
        else:
            cormat = X.sample(self.correlation_sample).corr()
        # initial C parameter value
        if self.C_init is None:
            if self.penalty == "l2":
                loss_init = "squared_hinge"
            else:
                loss_init = "log"
            C_init = l1_min_c(X, y, loss=loss_init)
        else:
            C_init = self.C_init

        # grid for C parameter values
        cs = C_init * np.logspace(0, self.grid_length, num=self.steps)

        self.allModels = []

        if progress_bar and self.n_jobs==1:
            iterator = tqdm(cs)
        else:
            iterator = cs

        if self.n_jobs <= 1:

            criteria_failed = False

            for c in iterator:

                if (not self.stop_immediately) or (not criteria_failed):
                    criteria_failed, result = self._model_with_c(
                        c=c,
                        X=X,
                        y=y,
                        X_valid=X_valid,
                        y_valid=y_valid,
                        sample_weight=sample_weight,
                        sample_weight_valid=sample_weight_valid,
                        cormat=cormat,
                    )

                    self.allModels.append(result)

                else:
                    pass

        else:

            def model_with_c(c):
                return self._model_with_c(
                    c=c,
                    X=X,
                    y=y,
                    X_valid=X_valid,
                    y_valid=y_valid,
                    sample_weight=sample_weight,
                    sample_weight_valid=sample_weight_valid,
                        cormat=cormat,
                )

            with ThreadPoolExecutor(self.n_jobs) as p:
                if progress_bar:
                    result = list(tqdm(p.map(
                            model_with_c,
                            iterator,
                        ), total=len(iterator))
                    )
                else:
                    result = list(p.map(
                            model_with_c,
                            iterator,
                        )
                    )
                criteria_failed = max([r[0] for r in result])

                result = [r[1] for r in result]
                self.allModels += result

        # order models by added Gini desc
        possibleModels = list(
            filter(
                lambda model: model["non-zero betas"] > 0
                and model["criteria passed"] == True,
                self.allModels,
            )
        )
        sortedModels = sorted(
            possibleModels, key=lambda d: d["gini validate"], reverse=True
        )
        self.progress = pd.DataFrame.from_records(sorted(self.allModels, key=lambda d: d["C"]))
        self.all_coefs = self.progress["coefs"]
        self.all_intercepts = self.progress["intercept"]

        if len(possibleModels) > 0:
            self.C = sortedModels[0]["C"]
            self.predictors = list(sortedModels[0]["predictors"])

            self.model = LogisticRegression(
                penalty=self.penalty, C=self.C, solver="liblinear"
            )

            if sample_weight is None:
                self.model.fit(X[self.predictors], y)
            else:
                self.model.fit(X[self.predictors], y, sample_weight)

            self.coef = self.model.coef_[0]
            self.intercept = self.model.intercept_[0]

            # variables for backward compatibility
            self.C_, self.coef_, self.intercept_, self.final_model_, self.final_predictors_, self.model_progress_ =\
                self.C, [self.coef], [self.intercept], self.model, self.predictors, self.progress

            print(f'Best model: C {self.C}, Gini {sortedModels[0]["gini validate"]}, # Predictors {len(self.predictors)}')
            print(f'Predictors: {self.predictors}')

    def predict(self, X):
        """uses the fitted final model to calculate predictions

        Args:
            X (pd.DataFrame): data, must include all predictors that are part of the fitted model

        Returns:
            np.array: array with predictions for data X
        """
        # uses the fitted final model to calculate predictions

        check_is_fitted(self, ["progress", "predictors", "model"])

        # calculates the predicted probabilities
        y = self.model.predict_proba(X[self.predictors])[:, 1]

        return y
        

    def draw_coeff_progression(self, predictor_names, output_file=None):
        """Draws progression plot of coefficient values during L1 fitting.
        Plots values of ceofficients based on value of regularization parameter C.
        Draws a plot with mathplotlib to notebook. 

        Args:
            predictors_names (list): names of predictors in order in which they were used during fitting
            output_file (str, optional): relative path to file to export

        """
        coefs = [
            [CC for CC in C] + [II for II in I]
            for C, I in zip(np.array(self.all_coefs), np.array(self.all_intercepts))
        ]
        cs = self.progress["C"]
        plt.figure(figsize=(7, 7))
        plt.plot(np.log10(cs), coefs)
        _, _ = plt.ylim()
        plt.xlabel("log10(C)")
        plt.ylabel("Coefficients")
        plt.title("Logistic Regression Path")
        plt.axis("tight")
        plt.legend(
            predictor_names + ["Intercept"],
            loc="upper center",
            bbox_to_anchor=(1.4, 1.0),
        )
        if output_file is not None:
            plt.savefig(output_file, bbox_inches="tight", dpi=72)
        plt.show()
        plt.clf()
        plt.close()

    def draw_gini_progression(self, output_file=None):
        """Draws progression plot of gini values during L1 fitting.
        Plots values of gini based on value of regularization parameter C.
        Draws a plot with mathplotlib to notebook. 

        Args:
            output_file (str, optional): relative path to file to export

        """
        plt.figure(figsize=(7, 7))
        ginis = self.progress[["gini train", "gini validate"]]
        cs = self.progress["C"]
        plt.plot(np.log10(cs), ginis)
        _, _ = plt.ylim()
        plt.xlabel("log10(C)")
        plt.ylabel("Ginis")
        plt.title("Logistic Regression Path")
        plt.axis("tight")
        plt.legend(
            ["Train", "Validate"], loc="upper center", bbox_to_anchor=(1.20, 1.0)
        )
        if output_file is not None:
            plt.savefig(output_file, bbox_inches="tight", dpi=72)
        plt.show()
        plt.clf()
        plt.close()






class VarClusSurrogates():
    """
    Class for searching for surrogate variables for a finished regression model. When you have a model and you ran
    clustering analysis (from scoring.varclus module) on a superset of the model's predictors, you can try to swap each
    predictor with other variables from its cluster to see how the Gini would change.
    
    Args:
        variables (list of str): names of varibles from clustering output
        clusters (list of int): cluster numbers of the varibles (list must have same length as variables list) from the clustering output
        predictors (list of str): list of predictors of the original model
        penalty (str, optional): 'l1' or 'l2' - penalization type of the regularized regression (default: 'l2)
        C (float, optional): the larger C is, the weaker is the penalization of the regularized regression (default: 1000)
        
    Properties:
        model_progress\_ (pandas data frame): Created by fit() method. Table with all surrogates and information how usage
        of each surrogate would affect gini of the final model
        
    Example:
        >>> vcs = VarClusSurrogates(km.variables_, km.labels_, modelSW.final_predictors_)
        >>> vcs.fit(data[train_mask], data[train_mask][col_target], data[valid_mask], data[valid_mask][col_target])
        >>> vcs.displaySurrogates(output_file = output_folder+'/predictors/predictor_surrogates.csv')
    """

    def __init__(self, variables, clusters, predictors, penalty="l2", C=1000):
        """
        Initialization method.
        """
        self.variables = variables
        self.clusters = clusters
        self.predictors = predictors
        self.penalty = penalty
        self.C = C
        return

    def __one_model(
        self, X, y, X_valid, y_valid, sample_weight=None, sample_weight_valid=None
    ):
        """
        Calculates a regression model on the given set and with given parameters. Returns its gini and indicator whether the betas have the same signum.
        
        Args:
            X (pd.DataFrame): training set of predictors
            y (pd.Series): training array of target
            X_valid (pd.DataFrame): validation set of predictors (used for Gini measurement)
            y_valid (pd.Series): validation array of target (used for Gini measurement)
            sample_weight (pd.Series, optional): weights of training observations. Defaults to None.
            sample_weight_valid (pd.Series, optional): weights of validation observations. Defaults to None.
        """

        newModel = LogisticRegression(
            penalty=self.penalty, C=self.C, solver="liblinear"
        )

        # if there is a variable with weights, use weighted regression
        if sample_weight is None:
            newModel.fit(X, y)
        else:
            newModel.fit(X, y, sample_weight)

        # measure Gini of such model
        predictions = newModel.predict_proba(X_valid)[:, 1]
        if sample_weight_valid is None:
            gini_result = 200 * roc_auc_score(y_valid, predictions) - 100
        else:
            gini_result = (
                200
                * roc_auc_score(y_valid, predictions, sample_weight=sample_weight_valid)
                - 100
            )

        return gini_result

    def fit(
        self,
        X,
        y,
        X_valid=None,
        y_valid=None,
        sample_weight=None,
        sample_weight_valid=None,
    ):
        """
        Calculates all possible regression models. The algorithm works as follows:
            * For each predictor, find out in which cluster it is
            * For each variable from the same cluster, fit model and calculate its Gini
            * Sort the models by Gini
            * Create table with all the steps, where in each row user can see how the Gini would change if one of the predictors is changed to one of its surrogates
        
        Args:
            X (pd.DataFrame): training set of predictors
            y (pd.Series): training array of target
            X_valid (pd.DataFrame, optional): validation set of predictors (used for Gini measurement). Defaults to None.
            y_valid (pd.Series, optional): validation array of target (used for Gini measurement). Defaults to None.
            sample_weight (pd.Series, optional): weights of training observations. Defaults to None.
            sample_weight_valid (pd.Series, optional): weights of validation observations. Defaults to None.

        """

        variables_count = len(self.variables)
        allModels = []

        # if there is no validation sample, then Gini will be calculated on the training sample
        if (X_valid is None) or (y_valid is None):
            X_valid = X
            y_valid = y
            sample_weight_valid = sample_weight

        # for each predictor, we will try to replace it with its surrogates
        for pred in self.predictors:
            if pred in self.variables:

                # Find surrogates in variables list
                print("Finding surrogates for predictor", pred, "...")
                idx = self.variables.index(pred)
                cl = self.clusters[idx]
                indices = [i for i, e in enumerate(self.clusters) if e == cl]
                surrogates = [self.variables[i] for i in indices]

                # for each surrogate, estimate a model, measure its gini
                for s in surrogates:
                    new_predictors = [p for p in self.predictors if p != pred]
                    new_predictors.append(s)
                    gini = self.__one_model(
                        X[new_predictors],
                        y,
                        X_valid[new_predictors],
                        y_valid,
                        sample_weight,
                        sample_weight_valid,
                    )

                    if pred == s:
                        flag_original = 1
                    else:
                        flag_original = 0

                    # append measured data into internal data structure
                    allModels.append(
                        {
                            "Original predictor": pred,
                            "Surrogate predictor": s,
                            "Gini": gini,
                            "Flag original": flag_original,
                        }
                    )

            else:
                print("Predictor", pred, "was not found in variables list.")

        # create data frame from internal data structure
        self.progress = pd.DataFrame.from_records(allModels)
        reference = self.progress[self.progress["Flag original"] == 1][
            ["Original predictor", "Gini"]
        ]
        self.progress = pd.merge(
            self.progress,
            reference,
            left_on="Original predictor",
            right_on="Original predictor",
            suffixes=("", " Reference"),
        )
        self.progress["Gini Difference"] = (
            self.progress["Gini"] - self.progress["Gini Reference"]
        )
        self.progress = self.progress.sort_values(
            ["Original predictor", "Gini Difference"], ascending=[True, False]
        )[["Original predictor", "Surrogate predictor", "Gini", "Gini Difference"]]

        self.model_progress_ = self.progress

        print("Finished. Use method displaySurrogates() to view the results.")
        return

    def displaySurrogates(self, output_file=None):
        """
        Displays table (data frame) with surrogates: In each row, there is name of original predictor, name of its surrogate and Gini delta. Gini delta = how Gini would change if instead of the predictor, surrogate variable is used.
        
        Args:
            output_file (str, optional): filename where dataframe is saved to as csv file. Defaults to None.
        """

        check_is_fitted(self, ["progress"])
        display(self.progress)
        if output_file is not None:
            self.progress.to_csv(output_file, index=False)
        return




class BootstrapLogit():
    """
    Model selection using bootstrapping. Wraps GiniStepwiseLogit or L1GiniModelSelection object and uses it to train multiple models on samples generated by bootstrapping. Averages the models at the end.
        
    Args:
        base_model (GiniStepwiseLogit or L1GiniModelSelection): existing instance of model selection class with chosen parameters
        bootstrap_samples (int, optional): How many time bootstrapping sample selection should be performed, i.e. how many models should be trained and averaged (default: 10)
        bootstrap_sample_size (float, optional): Sample size in each bootstrap iteration (1=full data) (default: 1)
        random_seed (int, optional): Random seed for replicable results (default: 98765)
    """

    def __init__(
        self,
        base_model,
        bootstrap_samples=10,
        bootstrap_sample_size=1.0,
        random_seed=98765,
    ):
        """
        Initializes BootstrapLogit instance. Must be given existing instance of GiniStepwiseLogit or L1GiniModelSelection.
        """

        self.base_model = base_model
        self.bootstrap_samples = bootstrap_samples
        self.bootstrap_sample_size = bootstrap_sample_size
        self.seeds = np.arange(10000 + 2 * self.bootstrap_samples)
        np.random.seed(random_seed)
        np.random.shuffle(self.seeds)

    def __avg_coefs(self, models):
        """
        Averages coefficients from models trained in all the bootstrap iterations.
        
        Args:
            models (list of dict): list created in self.fit() method
        
        Returns:
            list, list, float: ordered list of predictors in final (average) model, ordered list of corresponding final regression coefficients, intercept of final regression model
        """

        coef_df = pd.DataFrame(models)
        coef_df.fillna(0)
        final_predictors_ = []
        coef_ = []
        for col in coef_df.columns:
            if col != "INTERCEPT_":
                final_predictors_.append(col)
                coef_.append(coef_df[col].mean())
            else:
                intercept_ = coef_df[col].mean()
        return final_predictors_, coef_, intercept_

    def fit(
        self,
        X,
        y,
        X_valid=None,
        y_valid=None,
        sample_weight=None,
        sample_weight_valid=None,
    ):
        """
        Iterates bootstrapping. Trains a model in each iteration. Averages the models at the end.
        
        Args:
            X (pd.DataFrame): training data set with named predictors. This set will be used to estimate the regression coefficients.
            y (pd.Series): training series of target (indexes should correspond to X)
            X_valid (pd.DataFrame, optional): validation data set with named predictors. This set will be used to measure Gini select the variables into models. If None, training data set will be used. (default: None)
            y_valid (pd.Series, optional): validation series of target (indexes should correspond to X_valid) (default: None)
            sample_weight (pd.Series, optional): weights of observations in X (if None, all observations have weight 1) (default: None)
            sample_weight_valid (pd.Series, optional): weights of observations in X_valid (if None, all observations have weight 1) (default: None)
        """

        models = []

        for i in range(self.bootstrap_samples):

            print("Bootstrap round", i)

            X_sampled = X.sample(
                frac=self.bootstrap_sample_size,
                random_state=self.seeds[i],
                replace=True,
            )
            y_sampled = y.sample(
                frac=self.bootstrap_sample_size,
                random_state=self.seeds[i],
                replace=True,
            )
            if sample_weight is not None:
                sample_weight_sampled = sample_weight.sample(
                    frac=self.bootstrap_sample_size, random_state=self.seeds[i]
                )
            else:
                sample_weight_sampled = None
            if X_valid is not None:
                X_valid_sampled = X_valid.sample(
                    frac=self.bootstrap_sample_size,
                    random_state=self.seeds[self.bootstrap_samples + i],
                )
            else:
                X_valid_sampled = None
            if y_valid is not None:
                y_valid_sampled = y_valid.sample(
                    frac=self.bootstrap_sample_size,
                    random_state=self.seeds[self.bootstrap_samples + i],
                )
            else:
                y_valid_sampled = None
            if sample_weight_valid is not None:
                sample_weight_valid_sampled = sample_weight_valid.sample(
                    frac=self.bootstrap_sample_size,
                    random_state=self.seeds[self.bootstrap_samples + i],
                )
            else:
                sample_weight_valid_sampled = None

            self.base_model.fit(
                X_sampled,
                y_sampled,
                X_valid_sampled,
                y_valid_sampled,
                sample_weight_sampled,
                sample_weight_valid_sampled,
            )

            coef_dict = dict(
                zip(
                    self.base_model.predictors, self.base_model.coef
                )
            )
            coef_dict["INTERCEPT_"] = self.base_model.intercept_[0]
            models.append(coef_dict)

        self.predictors, self.coef, self.intercept = self.__avg_coefs(models)


        self.model = LogisticRegression(
            penalty=self.base_model.penalty, C=self.base_model.C, solver="liblinear"
        )
        self.model.fit(X[self.predictors], y)
        self.model.coef_ = np.array([self.coef])
        self.model.intercept_ = np.array([self.intercept])

        self.final_predictors_, self.coef_, self.intercept_, self.final_model_ = self.predictors, [self.coef], [self.intercept], self.model

    def predict(self, X):
        """
        Based on values of predictors, it calculates prediction (score).
        
        Args:
            X (pd.DataFrame): dataset the prediction should be performed on. Must contain all the predictors which are contained in the fitted model.

        Returns:
            pd.Series: prediction... probablity(target==1)
        """

        check_is_fitted(self, ["predictors", "model"])
        y = self.model.predict_proba(X[self.predictors])[:, 1]

        return y
