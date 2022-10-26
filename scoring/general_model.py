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

import numpy as np
import pandas as pd
import ipywidgets as widgets
import qgrid
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from .score_imputation import missing_value, quantile_imputer
from .metrics import gini, bootstrap_gini
from scipy.special import logit, expit
from tqdm.notebook import tqdm
from sklearn.exceptions import NotFittedError

class CrossValidatedGeneralModel():
    """Cross Validated General Model: Creates a General Model (model combining various subscores in logistic regression), optionally using
    cross validation to fit the model and bootstrapping to evaluate model performance.
    Subscores/predictors given to this class can be transformed inside class - if the subscores are in form of probability, logit transformation
    can be applied to them. If the subscores have some missing values, logic for automatic missing value imputation can be specified and the
    missing values are imputed by calculated value based on the rest of the observations.
        
    Args:
        predictors (list of str, optional): list with names of predictors (subscores) the final score should be based on (default: None)
        predictors_pd_form (list of str, optional): list with names of predictos (subscores) which are in the probability form (i.e. expit form) (default: None)
        imputation_type (str, optional): 'linear' if missing values should be imputed using simple proportion or 'quantile' if missing values should be imputed by mean score from the closest quantile of observations with non-missing score (default: 'linear')
        imputation_dicts (list of dict, optional): list of dictionaries with imputation logic. Each dictionary must have three keys:
            - fill_variable: name of the variable to be imputed
            - hit_condition: condition (in form of pandas query) for rows the imputation should be based for (imputed score values is calculated from score of these rows)
            - nohit_condition: condition (in form of pandas query) for the rows where the score should be imputed
            (default: None)
        cv (bool, optional):- if cross-validation should be used for fitting imputation values and regression coefficients (default: True)
        cv_folds (int, optional): number of cross-validation folds if cv==True (default: 5)
        cv_seed (int, optional): reandom seed for cross-validation if cv==True (default: 975)
        bootstrapped_gini (bool, optional): if model performance should be evaluated based on bootstrapping, so also confidence intervals for Gini are calculated (default: True)
        bootstrap_iters (int, optional): number of samples for bootstrap if bootstrapped_gini==True (default: 100)
        bootstrap_seed (int, optional): reandom seed for bootstrapping if bootstrapped_gini==True (default: 1850)
    """
    
    _empty_imputation_dict = {
        'fill_variable': '',
        'hit_condition': '',
        'nohit_condition': '',
        'manual_value': None,
    }
    
    _imputation_col_order = ['fill_variable', 
                             'hit_condition', 
                             'nohit_condition',
                             'manual_value']
    
    def __init__(self, predictors=None, predictors_pd_form = None, imputation_type='linear', imputation_dicts=None, cv=True, cv_folds=5, cv_seed=975, bootstrapped_gini=True, bootstrap_iters = 100, bootstrap_seed = 1850):
        """Initialization of instance.
        """
        
        self.set_imputations(imputation_dicts)
        if imputation_type not in ['linear', 'quantile']:
            raise ValueError("imputation_type must be either 'linear' or 'quantile'")
        self.imputation_type = imputation_type
        self.bootstrapped_gini = bootstrapped_gini
        self.cv = cv
        self.cv_folds = cv_folds
        self.cv_seed = cv_seed
        self.set_predictors(predictors)
        self.set_pdcols(predictors_pd_form)
        self.intercept_value = None
        self.coef_values = None
        self.imputations = None
        self.fitted = False
        self.mc_results = None
        self.bootstrap_iters = bootstrap_iters
        self.bootstrap_seed = bootstrap_seed
        self.validation_set = None
        
        
    def set_predictors(self, predictors):
        """Changes the list of predictors (subscores) the final score should be based on.
        
        Args:
            predictors (list of str): list with names of predictors (subscores)
        """
        
        if predictors is not None:
            self.all_predictors = list(predictors)
        elif isinstance(predictors, str):
            self.all_predictors = [predictors]
        else:
            self.all_predictors = None
        
        
    def set_pdcols(self, cols_pd):
        """Changes the list of predictos (subscores) which are in the probability form (i.e. expit form)
        
        Args:
            cols_pd (list of str): list with names of predictos (subscores) in the probability form
        """
        
        if cols_pd is not None:
            self.cols_pd = list(cols_pd)
        elif isinstance(cols_pd, str):
            self.cols_pd = [cols_pd]
        else:
            self.cols_pd = None
    
    
    def set_imputations(self, imputation_dicts):
        """Changes the list of dictionaries with imputation logic.
        
        Args:
            imputation_dicts (list of dict, optional): list of dictionaries with imputation logic. Each dictionary must have three keys:
                - fill_variable: name of the variable to be imputed
                - hit_condition: condition (in form of pandas query) for rows the imputation should be based for (imputed score values is calculated from score of these rows)
                - nohit_condition: condition (in form of pandas query) for the rows where the score should be imputed
                (default: None)
        """
        
        if imputation_dicts is not None:
            self.imputation_dicts = imputation_dicts
        elif isinstance(imputation_dicts, dict):
            self.imputation_dicts = [imputation_dicts]
        else:
            self.imputation_dicts = [self._empty_imputation_dict]

        for imputation_dict in self.imputation_dicts:
            if "manual_value" not in imputation_dict:
                imputation_dict["manual_value"] = None
            
            
    def _on_imputation_change(self, event, current_widget):
        """Event handler for set_imputations interactive.
        Handles change of value in qgrid table.
        
        Args:
            event (str):
            current_widget (str):
        """
        
        self.imputation_df = self.imputation_widget.get_changed_df()
        self.imputation_dicts = self.imputation_df[self.imputation_df['fill_variable']!=''].to_dict('records')
        if len(self.imputation_dicts) == 0:
            self.imputation_dicts = [self._empty_imputation_dict]
        for imputation_dict in self.imputation_dicts:
            if pd.isnull(imputation_dict['manual_value']):
                imputation_dict["manual_value"] = None
           
            
    def _on_imputation_add(self, c):
        """Event handler for set_imputations interactive.
        Handles adding of a new row to qgrid table by clicking a button below the table.
        
        Args:
            c (str):
        """
        
        self.imputation_df = self.imputation_df.append(pd.DataFrame([self._empty_imputation_dict]), ignore_index=True, sort=False)
        self.imputation_widget.df = self.imputation_df
        self.imputation_dicts = self.imputation_df[self.imputation_df['fill_variable']!=''].to_dict('records')
        if len(self.imputation_dicts) == 0:
            self.imputation_dicts = [self._empty_imputation_dict]
        for imputation_dict in self.imputation_dicts:
            if pd.isnull(imputation_dict['manual_value']):
                imputation_dict["manual_value"] = None
            
            
    def _on_colspd_change(self, event, current_widget):
        """Event handler for set_colspd_interactive interactive.
        Handles change of value in qgrid table.
        
        Args:
            event (str):
            current_widget (str):
        """

        self.colspd_df = self.colspd_widget.get_changed_df()
        self.set_pdcols(list(self.colspd_df[self.colspd_df['ispd']==True]['variable']))


    def close_interactive(self):
        self.imputation_widget = None
        self.colspd_widget = None
        
    
    def set_imputations_interactive(self):
        """Interactive qgrid table where the imputation logic can be edited by the user. A new row with imputation logic can be added by using a button.
        If fill_variable is not filled in for certain row, this row will be deleted.
        
        Returns:
            Jupyter widget
        """
        
        self.imputation_df = pd.DataFrame(self.imputation_dicts)[self._imputation_col_order]
        self.imputation_df = self.imputation_df[self.imputation_df['fill_variable']!='']
        self.imputation_widget = qgrid.show_grid(self.imputation_df, show_toolbar=True)
        self.imputation_widget.on('cell_edited', self._on_imputation_change)
        add_row_btn = widgets.Button(description='New empty row')
        add_row_btn.on_click(self._on_imputation_add)
        
        return widgets.VBox([self.imputation_widget, add_row_btn])
        
    
    def set_colspd_interactive(self, variables=None):
        """Interactive qgrid table where for each predictor, boolean value can be set whether the predictor is in probability (i.e. expit) form and should
        be logistically transformed before being used inside the general model.
        
        Args:
            variables (list of str, optional): list of predictor the probability form predictors will be chosen from. If None, self.all_predictors (set by cols_pred argument in initialization or by self.set_predictors() method) is used (default: {None})
        
        Returns:
            Jupyter widget
        """
        
        if variables is None:
            if self.all_predictors is None:
                print("Specify from which variables you want to chose in parameter 'variables'.")
            else:
                variables = self.all_predictors

        self.colspd_df = pd.DataFrame({'variable':variables, 'ispd':[False]*len(variables)})[['variable','ispd']]
        if self.cols_pd is not None:
            self.colspd_df.loc[self.colspd_df['variable'].isin(self.cols_pd), 'ispd'] = True
        self.colspd_widget = qgrid.show_grid(self.colspd_df, show_toolbar=False,
                             column_options={'editable':False}, 
                             column_definitions={'ispd':{'editable':True}},)
        self.colspd_widget.on('cell_edited', self._on_colspd_change)
        
        return self.colspd_widget
        
        
    @property
    def imputation_dictionaries(self):
        """Returns imputation logic in form of list of dictionaries which can be imported to another CVGM instance
        
        Returns:
            list of dict: imputation logic
        """
        
        return self.imputation_dicts


    @property
    def predictors(self):
        """Returns list of predictors
        
        Returns:
            list of str: list of names of predictors
        """

        return self.all_predictors


    @property
    def predictors_pd_form(self):
        """Returns list of predictors which are expected to be in probability (expit) form
        
        Returns:
            list of str: list of names of predictors in probability form
        """

        return self.cols_pd
        
        
    def fit(self, X, y, X_valid=None, y_valid=None, w=None, w_valid=None, predictors=None):
        """Fits the general model
        
        Args:
            X (pd.DataFrame): training predictor dataset. Should contain all the predictors given by predictors argument or by self.predictors if predictors argument is None
            y (pd.Series): training target. Series should have the same indexes as X. Values of target should be 0 or 1
            X_valid (pd.DataFrame, optional): validation predictor dataset. Should have the same columns as X (default: {None})
            y_valid (pd.Series, optional): validation target. Series should have the same indexes as X_valid. Values of target should be 0 or 1 (default: {None})
            w (pd.Series, optional):- weight of each observation. Series should have the same indexes as X. Values should be positive floats or positive integers (default: {None})
            w_valid (pd.Series, optional): weight of each validation observation. Series should have the same indexes as X_valid. Values should be positive floats or positive integers (default: {None})
            predictors (list of str, optional):- list of predictors. If specified, it will rewrite self.all_predictors. If not specified self.all_predictors is used. If neither is specified, all columns of X are considered as predictors. (default: {None})
        """
        
        # SET WHAT PREDICTORS WILL BE CONSIDERED
        # if predictors argument is specified, list from this argument is used as predictor list
        # else, if self.all_predictors is also not specified, all X.columns are used as predictor list
        # else, self.all_predictors is specified is used as predictor list
        # In case that this is called just as a part of marginal_contribution, self.all_predictors are left untouched, otherwise it is rewritten by what will be used in the actual model
        if predictors is not None:
            if type(predictors)==str:
                predictors = [predictors]
            model_predictors = predictors
            self.set_predictors(predictors)
        elif self.all_predictors is not None:
            model_predictors = self.all_predictors.copy()
        else:
            raise ValueError('List of predictors must be specified as "predictors" parameter either during initialization or when calling fit() method.')
        
        print(f'Model predictors: {model_predictors}')
        
        # save the data internally to be able to load them when we will train more models during marginal contribution
        # self.X, self.X_valid, self.y, self.y_valid, self.w, self.w_valid = X.copy(), X_valid.copy(), y.copy(), y_valid.copy(), w.copy(), w_valid.copy()

        # CROSS VALIDATION
        if self.cv:
            
            # PREPARE DATASET FOR CROSS VALIDATION
            # In case both X and X_valid datasets were specified, they are concatenated and used for K-fold CV sample generation as there is no use for separate validation dataset
            if (X_valid is not None):
                print('Cross validation will be used for the union of training and validation sample.')
                print('If you want to use cross validation for training sample only, do not submit any validation sample.')
                X = pd.concat([X, X_valid])
                y = pd.concat([y, y_valid])
                if w is not None:
                    w = pd.concat([w, w_valid])
                else:
                    w = y.copy()
                    w[:] = 1.0
                    w = w.rename('_weight')
            else:
                print('Cross validation will be used for the training sample.')
                if w is None:
                    w = y.copy()
                    w[:] = 1.0
                    w = w.rename('_weight')
                    
            kf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.cv_seed)
            
            validation_score_iters = []
            
            # Loop through all folds generated by cross-validation. The splits are stratified by target value.
            for train_rownum, test_rownum in kf.split(X.values, y.values):
                
                train_idx, test_idx = X.index.isin(X.iloc[train_rownum].index), X.index.isin(X.iloc[test_rownum].index)
                
                # Fitting of model on one fold. Gini will be taken from the validation part.
                _, _, _, validation_score = self._fit_fold(
                    X[train_idx], y[train_idx], w[train_idx],
                    X[test_idx], y[test_idx], w[test_idx],
                    model_predictors)
                
                validation_score_iters.append(validation_score)
                
            # After Gini is evaluated using cross-validation, final model coefficients are fitted on the full sample.
            intercept, coefs, imputations, _ = self._fit_fold(
                X, y, w, 
                X, y, w,
                model_predictors)
            
            intercept_iters = [intercept]
            coefs_iters = [coefs]
            imputations_iters = [imputations]
            
            # Save validation predictions to be able to caluclate Gini from them
            self._save_validation_dataset(
                validation_score_iters, y, w
            )
            
            
        # WITHOUT CROSS VALIDATION
        else:
            
            # If X_valid is not specified, X is copied to its place, meaning that performance evaluation will be technically done on the training sample.
            if (X_valid is not None):
                print('Model will be trained using training sample, Gini will be evaluated using validation sample.')
                if w is None:
                    w = y.copy()
                    w[:] = 1.0
                    w = w.rename('_weight')
                if w_valid is None:
                    w_valid = y_valid.copy()
                    w_valid[:] = 1.0
                    w_valid = w_valid.rename('_weight')
            else:
                X_valid = X
                y_valid = y
                if w is None:
                    w = y.copy()
                    w[:] = 1.0
                    w_valid = w
                    w = w.rename('_weight')
                print('No validation sample submitted, Gini will be evaluated using training sample.')
                
            # Fitting the model coefficients
            intercept, coefs, imputations, validation_score = self._fit_fold(
                X, y, w, 
                X_valid, y_valid, w_valid,
                model_predictors)
            
            intercept_iters = [intercept]
            coefs_iters = [coefs]
            imputations_iters = [imputations]
            validation_score_iters = [validation_score]
            
            # Save validation predictions to be able to caluclate Gini from them
            self._save_validation_dataset(
                validation_score_iters, y_valid, w_valid
            )
        
        # Logic to arrange the model coefs, imputation values etc. to a proper form. It also calculates mean validation gini in case we used cross validation and we have gini numbers from each fold separately.
        self._save_model_results(
            intercept_iters, coefs_iters, imputations_iters
        )
        
    
    def _fit_fold(self, X_train, y_train, w_train, X_test, y_test, w_test, predictors):
        """Logic of model fitting. Always called by fit() method.
        
        Args:
            X_train (pd.DataFrame): dataset with predictors
            y_train (pd.Series): series with target (0/1)
            w_train (pd.Series): series with weights
            X_test (pd.DataFrame): dataset with predictors
            y_test (pd.Series): series with target (0/1)
            w_test (pd.Series): series with weights
            predictors (list of str): list of predictors (all of them should be contained as columns in both X and X_test)
        
        Returns:
            float, dict, [dict], float, (float, float): fitted intercept, 
                fitted regression coefficients for each variable as a key, 
                imputation dictionary with values of fitted imputations, 
                validation gini, 
                confidence interval for validation gini
        """
        
        # fits the imputaion coefficients based on the imputation dictionary (self.impuation_dicts)
        X_train, X_test, imputation_dicts = self._fit_imputation(
            X_train, y_train, w_train, X_test, y_test, w_test, predictors
        )
        
        # fits model coefficients (intercept and betas)
        logreg_cv = LogisticRegression(penalty = 'l2', C = 1000, solver='liblinear')
        logreg_cv.fit(X=X_train[predictors],
                      y=y_train,
                      sample_weight=w_train)
        prediction_test = pd.Series(logreg_cv.predict_proba(X_test[predictors])[:,1])
        prediction_test = prediction_test.rename('_prediction_fold')
        prediction_test.index = y_test.index
        
        coefs = dict(zip(predictors,logreg_cv.coef_[0]))
        intercept = logreg_cv.intercept_[0]
        
        return intercept, coefs, imputation_dicts, prediction_test
    
    
    def _fit_imputation(self, X_train, y_train, w_train, X_test, y_test, w_test, predictors):
        """Logic of fitting imputation values. Always called by _fit_fold() method.
        
        Args:
            X_train (pd.DataFrame): dataset with predictors
            y_train (pd.Series): series with target (0/1)
            w_train (pd.Series): series with weights
            X_test (pd.DataFrame):- dataset with predictors
            y_test (pd.Series): series with target (0/1)
            w_test (pd.Series):- series with weights
            predictors (list of str):- list of predictors (all of them should be contained as columns in both X and X_test)
        
        Returns:
            pd.DataFrame, pd.DataFrame, [dict]: X_train with imputed values, 
                X_test with imputed values, 
                imputation dictionary with values of fitted imputations
        """
        
        # we create copy of our data as we will change values of some columns. this changed copy will be output of this method
        X_train = X_train.copy()
        X_test = X_test.copy()

        # LOGIT TRANSFORMATION
        # for those predictors which are in the probability (expit) form we do logistic transformation
        for col in self.cols_pd:
            if col in predictors:
                X_train[col] = logit(X_train[col])
                X_test[col] = logit(X_test[col])
        
        imputation_dicts_out = []
        
        # for each imputation dictionary, we fit imputation value
        for imp in self.imputation_dicts:
            if imp['fill_variable'] in predictors:
                
                # creating masks for rows imputation is based on and rows imputation is applied to
                train_hit_mask = X_train.eval(imp["hit_condition"],engine='python')
                train_nohit_mask = X_train.eval(imp["nohit_condition"],engine='python')
                test_nohit_mask = X_test.eval(imp["nohit_condition"],engine='python')

                # forming the data to be compatible with imputers from .score_imputation module
                sample_hit = X_train[train_hit_mask]
                sample_nohit = X_train[train_nohit_mask]
                if y_train.name not in list(sample_hit.columns):
                    sample_hit = pd.concat([sample_hit, y_train[train_hit_mask]], axis=1)
                    sample_nohit = pd.concat([sample_nohit, y_train[train_nohit_mask]], axis=1)
                if w_train.name not in list(sample_hit.columns):
                    sample_hit = pd.concat([sample_hit, w_train[train_hit_mask]], axis=1)
                    sample_nohit = pd.concat([sample_nohit, w_train[train_nohit_mask]], axis=1)

                # imputation logic from scoring.score_imputation module
                if imp['manual_value'] is not None:
                    fill_value = imp['manual_value']
                elif self.imputation_type == 'linear':
                    fill_value = missing_value(sample_hit = sample_hit,
                                               sample_nohit = sample_nohit,
                                               target = y_train.name,
                                               weight = w_train.name,
                                               score = imp['fill_variable'],
                                               ispd = False)
                elif self.imputation_type == 'quantile':
                    fill_value = quantile_imputer(sample_hit = sample_hit,
                                                  sample_nohit = sample_nohit,
                                                  target = y_train.name,
                                                  weight = w_train.name,
                                                  score = imp['fill_variable'],
                                                  quantiles = 25)
                
                # filling the imputation values into the copy of our data and filling the outputs
                X_train.loc[train_nohit_mask, imp['fill_variable']] = fill_value
                X_test.loc[test_nohit_mask, imp['fill_variable']] = fill_value
                
                imp_out = imp.copy()                    
                imp_out['fill_value'] = fill_value
                imputation_dicts_out.append(imp_out)  
                                       
        return X_train, X_test, imputation_dicts_out
                                       
    
    def marginal_contribution(self, X, y, X_valid=None, y_valid=None, w=None, w_valid=None, predictors_to_add=None):
        """Calculation of marginal contribution of adding certain predictors or removing one of the predictors.

        Args:
            X (pd.DataFrame): training predictor dataset. Should contain all the predictors given by self.predictors and predictors_to_add argument
            y (pd.Series): training target. Series should have the same indexes as X. Values of target should be 0 or 1
            X_valid (pd.DataFrame, optional): validation predictor dataset. Should have the same columns as X (default: None)
            y_valid (pd.Series, optional): validation target. Series should have the same indexes as X_valid. Values of target should be 0 or 1 (default: None)
            w (pd.Series, optional): weight of each observation. Series should have the same indexes as X. Values should be positive floats or positive integers (default: None)
            w_valid (pd.Series, optional): weight of each validation observation. Series should have the same indexes as X_valid. Values should be positive floats or positive integers (default: None)
            predictors_to_add (list of str, optional): list of predictors we should try to add to the base model (default: None)
        
        Returns:
            pd.DataFrame: table with gini differences for addition or removal of predictors
        """
        
        if self.fitted == True:
                                       
            # once model is fitted, its predictors is always saved in self.all_predictors, so we can safely use this list as a base for adding or removal of predictors
            predictors_base = self.all_predictors.copy()
                                        
            # creating the list of predictors we will try to add - it will be either those specified in predictors_to_add, or (if this argument is kept as None) all of X.columns - minus those which are already in base
            if predictors_to_add is not None:
                if type(predictors_to_add)=='str':
                    predictors_to_add = [predictors_to_add]
                predictors_to_add = [pred for pred in predictors_to_add if pred not in set(predictors_base)]
            else:
                predictors_to_add = []
            
            gini_value_ref, gini_ci_ref = self.gini_result
            mc_dicts = [{'add/remove':'', 'predictor':'base_model', 'gini':gini_value_ref, 'diff':0, 'gini_5%': gini_ci_ref[0], 'gini_95%': gini_ci_ref[1],
                'hit_gini_base_model':None, 'hit_gini':None, 'hit_diff':None}]

            mc_cvgm = CrossValidatedGeneralModel(
                predictors=predictors_base,
                predictors_pd_form=self.cols_pd,
                imputation_type=self.imputation_type,
                imputation_dicts=self.imputation_dicts,
                cv=self.cv,
                cv_folds=self.cv_folds,
                cv_seed=self.cv_seed,
                bootstrapped_gini=self.bootstrapped_gini,
                bootstrap_iters=self.bootstrap_iters,
                bootstrap_seed=self.bootstrap_seed
                )
            
            # loop through all predictors we want to try to add
            if len(predictors_to_add) > 0:
                print('Adding predictors...')
            for pred in tqdm(predictors_to_add):
                mc_cvgm.fit(X=X, y=y, X_valid=X_valid, y_valid=y_valid, w=w, w_valid=w_valid, predictors=predictors_base+[pred])
                gini_value_mc, gini_ci_mc = mc_cvgm.gini_result

                hit_indexes = list(X[X[pred].notnull()].index)
                if X_valid is not None:
                    hit_indexes += list(X_valid[X_valid[pred].notnull()].index)

                gini_value_ref_hit, _ = self.calculate_gini(hit_indexes)
                gini_value_mc_hit, _ = mc_cvgm.calculate_gini(hit_indexes)

                mc_dicts.append({'add/remove':'+', 'predictor':pred, 'gini':gini_value_mc, 'diff':gini_value_mc-gini_value_ref, 'gini_5%': gini_ci_mc[0], 'gini_95%': gini_ci_mc[1],
                    'hit_gini_base_model':gini_value_ref_hit, 'hit_gini':gini_value_mc_hit, 'hit_diff':gini_value_mc_hit-gini_value_ref_hit})
                
            # loop through all predictors we want to try to remove (which means each one from base)
            print('Removing predictors...')
            for pred in tqdm(predictors_base):
                mc_cvgm.fit(X=X, y=y, X_valid=X_valid, y_valid=y_valid, w=w, w_valid=w_valid, predictors=[p for p in predictors_base if p!=pred])
                gini_value_mc, gini_ci_mc = mc_cvgm.gini_result

                hit_indexes = list(X[X[pred].notnull()].index)
                if X_valid is not None:
                    hit_indexes += list(X_valid[X_valid[pred].notnull()].index)

                gini_value_ref_hit, _ = self.calculate_gini(hit_indexes)
                gini_value_mc_hit, _ = mc_cvgm.calculate_gini(hit_indexes)

                mc_dicts.append({'add/remove':'-', 'predictor':pred, 'gini':gini_value_mc, 'diff':gini_value_mc-gini_value_ref, 'gini_5%': gini_ci_mc[0], 'gini_95%': gini_ci_mc[1],
                    'hit_gini_base_model':gini_value_ref_hit, 'hit_gini':gini_value_mc_hit, 'hit_diff':gini_value_mc_hit-gini_value_ref_hit})
                
            self.mc_results = mc_dicts
            mc_results_df = pd.DataFrame(mc_dicts)[['predictor','add/remove','gini','diff','gini_5%','gini_95%','hit_gini_base_model','hit_gini','hit_diff']]
            mc_results_df.set_index('predictor', inplace=True)
                                        
            return mc_results_df
        
        else:
            raise NotFittedError("Model is not fitted yet.")
    
    
    @property
    def imputation_table(self):
        """Returns tabular output with imputation value for each imputation defined in imputation dicitonaries.
        
        Returns:
            pd.DataFrame: table with fitted imputation values
        """
        
        if self.fitted:
            imputation_table = pd.DataFrame(self.imputations)[self._imputation_col_order + ['fill_value']]
            return imputation_table
        else:
            raise NotFittedError("Model is not fitted yet.")

    
    @property
    def scorecard_table(self):
        """Returns tabular output of model coefficients.
        
        Returns:
            pd.DataFrame: table with fitted coefficients values
        """
        
        if self.fitted:
            scorecard_table = pd.DataFrame.from_dict(self.coef_values, orient='index')
            scorecard_table.columns = ['Coefficient']
            scorecard_table.loc['Intercept','Coefficient'] = self.intercept_value
            scorecard_table['Logit'] = ''
            scorecard_table.loc[scorecard_table.index.isin(self.cols_pd), 'Logit'] = 'Yes'
            return scorecard_table
        else:
            raise NotFittedError("Model is not fitted yet.")
                                                                               
        
    def _save_model_results(self, intercept_list, coefs_list, mv_imp_list):
        """This method will take the values of coefficients, imputations and Ginis from fit() method as arguments, clean them;
        if there are multiple gini values (from CV), calculates averages;
        if specified in rewrite parameter, saves them into instance attributes (self. ...);
        and always returns them in specific order.
        Originally, I thought that also the fitted coefficients from CV will be averaged here, but then we decided to rather fit them once again on the full dataset,
        so the original purpose of this method is reduced now. The original logic is commented if we change our minds in the future.
        
        Args:
            intercept_list (list of float): list of intercepts (should have length 1 in current implementation)
            coefs_list (list of dict): list of beta coefficient dictionaries (should have length 1 in current implementation)
            gini_list (list of float): list of gini values
        """

        intercept_value = intercept_list[0]
        coef_values = coefs_list[0]
        imputation_values = mv_imp_list[0]
            
        self.intercept_value = intercept_value
        self.coef_values = coef_values
        self.imputations = imputation_values
        self.fitted = True


    def _save_validation_dataset(self, prediction_list, y, w):
        """Creates dictionary with 3 Series with prediction, target and weight of validation (or union of test parts of cross validation folds) dataset
        
        Args:
            prediction_list (list of pd.Series): list of prediction on test folds
            y (pd.Series): targets (index should be superset of union of prediction_list indexes)
            w (pd.Series): weights (index should be superset of union of prediction_list indexes)
        """

        validation_score = pd.concat(
            objs=prediction_list,
            axis=0,
        )

        self.validation_set = {
            'score': validation_score,
            'target': y[validation_score.index].copy(),
            'weight': w[validation_score.index].copy(),
        }

        
    @property
    def imputation_values(self):
        """Returns raw form of imputation dictionaries including the fitted imputation values
        
        Returns:
            list of dict: imputation logic with fitted imputation values
        """
        
        if self.fitted:
            return self.imputations
        else:
            raise NotFittedError("Model is not fitted yet.")
    
    
    @property
    def coefficients(self):
        """Returns dictionary with fitted regression coefficients
        
        Returns:
            dict: key = predictor name, value = beta coef
        """
        
        if self.fitted:
            return self.coef_values
        else:
            raise NotFittedError("Model is not fitted yet.")
    
    
    @property
    def intercept(self):
        """Returns fitted regression intercept
        
        Returns:
            float: intercept
        """
        
        if self.fitted:
            return self.intercept_value
        else:
            raise NotFittedError("Model is not fitted yet.")

    
    @property
    def validation_prediction(self):
        """Returns prediction that was calculated during training on validation (or cross validation) sample(s)
        
        Returns:
            pd.Series: predictions
        """
        
        if self.fitted:
            return self.validation_set['score']
        else:
            raise NotFittedError("Model is not fitted yet.")
    
    
    def calculate_gini(self, indexes=None):
        """Returns Gini of the model. If the Gini was calculated by boostrapping, the confidence interval is returned as well (otherwise it is None).

        Args:
            indexes (index): which rows should be used for Gini calculation
        
        Returns:
            float, [float, float]: model gini, [5% conf. interval border, 95% conf. interval border]
        """
        
        if self.fitted:

            if indexes is None:
                indexes = list(self.validation_set['score'].index)
            else:
                # indexes = [i for i in self.validation_set['score'].index if i in indexes]
                indexes = np.intersect1d(self.validation_set['score'].index, indexes)

            if self.bootstrapped_gini:
                gini_value, _, gini_ci = bootstrap_gini(data=pd.concat([
                                                            self.validation_set['target'].loc[indexes],
                                                            self.validation_set['score'].loc[indexes],
                                                            self.validation_set['weight'].loc[indexes]
                                                        ],axis=1),
                                                        col_target = self.validation_set['target'].name,
                                                        col_score = self.validation_set['score'].name,
                                                        col_weight = self.validation_set['weight'].name,
                                                        n_iter = self.bootstrap_iters,
                                                        ci_range = 5,
                                                        use_tqdm = False,
                                                        random_seed = self.bootstrap_seed)

            else:
                gini_value = gini(
                    self.validation_set['target'].loc[indexes],
                    self.validation_set['score'].loc[indexes],
                    self.validation_set['weight'].loc[indexes]
                )
                gini_ci = [None, None]

            return gini_value, gini_ci

        else:
            raise NotFittedError("Model is not fitted yet.")


    @property
    def gini_result(self):
        """Returns Gini of the model. If the Gini was calculated by boostrapping, the confidence interval is returned as well (otherwise it is None).
        
        Returns:
            float, [float, float]: model gini, [5% conf. interval border, 95% conf. interval border]
        """
        
        if self.fitted:
            return self.calculate_gini()
        else:
            raise NotFittedError("Model is not fitted yet.")
    
    
    def transformation_code(self, dataset_name, result_name="score"):
        """Generates Python/pandas code to calculate the GM score.
        
        Args:
            dataset_name (str): how the input dataset is called (name of dataset from which the predictors are taken)
            result_name (str, optional): how the series with the score should be named (default: {"score"})
        
        Returns:
            pd.Series: series with GM score values
        """
        
        if self.fitted:
            score_str = f'{self.intercept_value}'
            pred_rows = []
            
            for var, coef in self.coef_values.items():
                score_str += f' + ({coef} * pred_{var})'
                if var in self.cols_pd:
                    pred_rows.append(f'pred_{var} = logit({dataset_name}["{var}"].copy())')
                else:
                    pred_rows.append(f'pred_{var} = {dataset_name}["{var}"].copy()')
                for imputation in self.imputations:
                    if imputation["fill_variable"] == var:
                        pred_rows.append(f'pred_{var}[{dataset_name}.eval(\'{imputation["nohit_condition"]}\',engine=\'python\')] = {imputation["fill_value"]}')

            result = 'import pandas as pd\nfrom scipy.special import expit\n\n'
            result += '\n'.join(pred_rows) 
            result += f'\n\n{result_name}_log = {score_str}'
            result += f'\n\n{result_name} = expit({result_name}_log)'
            
            return result

        else:
            raise NotFittedError("Model is not fitted yet.")
    
    
    def transform(self, X):
        """Transforms dataframe by calculating the score using fitted model.
        
        Args:
            X (pd.DataFrame): dataframe with predictors
        
        Returns:
            pd.Series: series with resulting GM score
        """
        
        if self.fitted:
            y = pd.Series(index=X.index)
            y = y.fillna(self.intercept_value)
            
            for var, coef in self.coef_values.items():
                x = X[var].copy()
                if var in self.cols_pd:
                    x = logit(x)
                for imputation in self.imputations:
                    if imputation["fill_variable"] == var:
                        x[X.eval(imputation["nohit_condition"],engine='python')] = imputation["fill_value"]
                y = y + coef * x
            
            y = expit(y)
            
            return y

        else:
            raise NotFittedError("Model is not fitted yet.")


    def impute(self, X):
        """Transforms dataframe by imputing the subscores
        
        Args:
            X (pd.DataFrame): dataframe with predictors
        
        Returns:
            pd.dataFrame: dataframe with imputed subscores
        """

        X2 = pd.DataFrame(index=X.index)
        
        if self.fitted:
            y = pd.Series(index=X.index)
            y = y.fillna(self.intercept_value)
            
        for var in self.all_predictors:
            x = X[var].copy()
            if var in self.cols_pd:
                x = logit(x)
            for imputation in self.imputations:
                if imputation["fill_variable"] == var:
                    x[X.eval(imputation["nohit_condition"], engine='python')] = imputation["fill_value"]
            X2 = pd.concat([X2, pd.DataFrame(x)], axis=1)
            
        return X2




def expected_ar(data, col_score, query_subset, col_weight=None, reference_ar=0.50, def_by_score_ascending=False):
    """On dataset data, this function calculates cutoff for given approval rate (reference_ar) for rejections based on col_score.
    Then, it uses this cutoff on subset of data defined by data.query(query_subset) and calculates what will be the approval rate in this specific subset.
    
    Args:
        data (pd.DataFrame): data the cutoff will be calculated on
        col_score (str): name of score column (must be in data.columns) the cutoff value will be applied to
        query_subset (str): pandas query of subset of data where approval rate of the calculated cutoff will be calculated
        col_weight (str, optional): name of weight column (if defined, it must be in data.columns) (default: {None})
        reference_ar (float, optional): expected approval rate the cutoff is derived from, must be number between 0 and 1 (default: {0.50})
        def_by_score_ascending (boolean, optional): True if the score grows with probability default. False if the score decreases with PD. (default: {True})
    
    Returns:
        float: approval rate on subset of data defined by query_subset if approval rate on whole data is given by reference_ar
    """
    
    if col_weight is not None:
        
        data_ref = data[[col_score, col_weight]].sort_values(col_score).copy()
        data_ref['cum_weight'] = data_ref[col_weight].cumsum() / data_ref[col_weight].sum()
        if def_by_score_ascending:
            cutoff = data_ref[data_ref['cum_weight'] > 1-reference_ar][col_score].min()
        else:
            cutoff = data_ref[data_ref['cum_weight'] <= reference_ar][col_score].max()
    
        data_subset = data.query(query_subset, engine="python")[[col_score, col_weight]].copy()
        if def_by_score_ascending:
            approval_rate = data_subset[data_subset[col_score] > cutoff][col_weight].sum() / data_subset[col_weight].sum()
        else:
            approval_rate = data_subset[data_subset[col_score] <= cutoff][col_weight].sum() / data_subset[col_weight].sum()
        
    else:
        
        if def_by_score_ascending:
            cutoff = np.percentile(data[col_score], (1-reference_ar)*100)
        else:
            cutoff = np.percentile(data[col_score], reference_ar*100)
    
        data_subset = data.query(query_subset, engine="python")[[col_score]].copy()
        if def_by_score_ascending:
            approval_rate = len(data_subset[data_subset[col_score] > cutoff]) / len(data_subset)
        else:
            approval_rate = len(data_subset[data_subset[col_score] <= cutoff]) / len(data_subset)
    
    return approval_rate