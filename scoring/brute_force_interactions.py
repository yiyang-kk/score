
# Home Credit Python Scoring Library and Workflow
# Copyright © 2017-2019, Pavel Sůva, Marek Teller, Martin Kotek, Jan Zeller,
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
import itertools
from .grouping import Grouping
from .metrics import gini
from .variable_encoding import MeanTargetEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from scipy.stats import chi2



def logreg_lr_test(
    data,
    predictors_model,
    predictors_submodel,
    target,
    weight=None,
    penalty="l2",
    C=10e3,
    print_output=False,
):
    """Performs Likelihood Ratio test of significance of a submodel for logistic regression estimator.
    Two models are estimated in this function, one must be submodel of the other for the test to work properly.
    
    Args:
        data (pd.DataFrame): data frame with predictors, target and optionally weights
        predictors_model (list of str): list of names of predictors (columns of data) in the main model 
        predictors_submodel (list of str): list of names of predictors (columns of data) in the submodel, must be subset of predictors_model
        target (str): name of target column in data
        weight (str, optional): name of weight column in data. Defaults to None.
        penalty (str, optional): regularization penalty type in LogisticRegression estimator. Defaults to 'l2'.
        C (float, optional): regularization penalty parameter in LogisticRegression estimator. Defaults to 10e3.
        print_output (bool, optional): whether text output of statistical likelihood ratio test parameters should be printed. Defaults to `False`.
    
    Returns:
        float: p-value of likelihood ratio test
    """    

    if weight:
        weight_array = data[weight]
    else:
        weight_array = None

    if not set(predictors_submodel) <= set(predictors_model):
        raise ValueError("predictors_submodel must be a subset of predictors_model")

    logreg = LogisticRegression(penalty=penalty, C=C, solver="lbfgs")
    # fit the model with full set of predictors
    logreg.fit(X=data[predictors_model], y=data[target], sample_weight=weight_array)
    model_prediction = logreg.predict_proba(data[predictors_model])[:, 1]
    model_logloss = log_loss(
        y_true=data[target],
        y_pred=model_prediction,
        normalize=False,
        sample_weight=weight_array,
    )

    if predictors_submodel is not None and len(predictors_submodel) > 0:
        # we have non-trivial submodel - fit the model with subset of predictors
        logreg.fit(
            X=data[predictors_submodel], y=data[target], sample_weight=weight_array
        )
        submodel_prediction = logreg.predict_proba(data[predictors_submodel])[:, 1]
        degrees_of_freedom = len(predictors_model) - len(predictors_submodel)
    elif weight_array is None:
        # submodel is just intercept and there are no weights - prediction is just average target
        submodel_prediction = np.full(
            shape=data[target].shape[0], fill_value=data[target].mean()
        )
        degrees_of_freedom = len(predictors_model)
    else:
        # submodel is just intercept and there are weights - prediction is weighted average of targets
        submodel_prediction = np.full(
            shape=data[target].shape[0],
            fill_value=(data[target] * data[weight]).sum() / data[weight].sum(),
        )
        degrees_of_freedom = len(predictors_model)
    submodel_logloss = log_loss(
        y_true=data[target],
        y_pred=submodel_prediction,
        normalize=False,
        sample_weight=weight_array,
    )

    lr_test_statistics = -2 * ((-1 * submodel_logloss) - (-1 * model_logloss))
    p_value = chi2.sf(lr_test_statistics, degrees_of_freedom)

    if print_output:
        print(
            f"Model LogLoss: {model_logloss}\nSubmodel LogLoss {submodel_logloss}\nTest Statistics: {lr_test_statistics}\nDegrees of freedom: {degrees_of_freedom}\np-value: {p_value}"
        )

    return p_value



def gini_difference(
    data,
    predictors_model1,
    predictors_model2,
    target,
    weight=None,
    penalty="l2",
    C=10e3,
    print_output=False,
):
    """Calculates Gini difference of two models on the same dataset which are using different predictor sets.
    Unlike logreg_lr_test(), the two models which are estimated in this function dont have to relate to each other 
    
    Args:
        data (pd.DataFrame): data frame with predictors, target and optionally weights
        predictors_model1 (list of str): list of names of predictors (columns of data) in the first model 
        predictors_model2 (list of str): list of names of predictors (columns of data) in the second model 
        target (str): name of target column in data
        weight (str, optional): name of weight column in data. Defaults to None.
        penalty (str, optional): regularization penalty type in LogisticRegression estimator. Defaults to 'l2'.
        C (float, optional): regularization penalty parameter in LogisticRegression estimator. Defaults to 10e3.
        print_output (bool, optional): whether text output of Ginis should be printed. Defaults to `False`.
    
    Returns:
        float: Gini of first model minus Gini of second model (scale of Ginis is 0 (eventually -1 for inverted models) to 1)
    """

    if weight:
        weight_array = data[weight]
    else:
        weight_array = None

    logreg = LogisticRegression(penalty=penalty, C=C, solver="lbfgs")
    # fit the model with full set of predictors
    logreg.fit(X=data[predictors_model1], y=data[target], sample_weight=weight_array)
    prediction1 = logreg.predict_proba(data[predictors_model1])[:, 1]
    logreg.fit(X=data[predictors_model2], y=data[target], sample_weight=weight_array)
    prediction2 = logreg.predict_proba(data[predictors_model2])[:, 1]
    gini1 = gini(
            y_true=data[target], y_pred=prediction1, sample_weight=weight_array
    )
    gini2 = gini(
            y_true=data[target], y_pred=prediction2, sample_weight=weight_array
    )
    gini_diff = gini1 - gini2

    if print_output:
        print(
            f"Gini of 1st model: {gini1}\nGini of 2nd model: {gini2}\nDifference: {gini_diff}"
        )

    return gini_diff



class BruteForceInteractions():
    """Class for testing interactions of predictors using brute force. It tests each predictor with each predictor
    and can calculate whether the interaction is a significant benefit for the model. Various ways how to perform
    the test are set in the initialization.
        
    Args:
        pred_num (list of str): List of numerical predictor names
        pred_cat (list of str): List of categorical predictor names
        target (str): Name of target column
        base (str, optional): Name of base column (where the target is observable). 
            If not filled, it is consider to be identically 1 (default: None)
        weight (str, optional): Name of weight column (importance of observations). 
            If not filled, it is consider to be identically 1 (default: None)
        test_method (str, optional): `gini` or `lr`. Either Gini difference of models with and without interactions
            should be calculated or Likelihood Ratio chi-squared test should be performed. default: `gini`
        use_grouping (bool, optional): Whether variables should be grouped by Grouping class before being analyzed. 
            If not, categorical variables will have as many categories as in original data, and numerical 
            variables will be categorized using through quantiles. Mean target will be assigned to them then.
            (default: `True`)
        use_grouping_interactions (bool, optional): Whether interaction variables should be grouped by Grouping class
            before being analyzed. If not, cartesian products of interacting variables will be created
            and assigned mean target. default: `True`
        groups_per_predictor (int, optional): For Grouping and for quantiles (if use_grouping=`False` and numerical
            predictors exist) - max number of groups that should be crated (default: 5)
        min_group_size (int, optional): For Grouping minimal size of each group (default: 100)
        mean_target_regularization_weight (float, optional): For Mean Target Encoding regularization strength parameter 
            (default: 0.1)
        grouping_category_limit (int, optional): max number of distinct categories for Grouping - if there is a 
            categorical variable with a larger nubmer of categories, error is raised (default: 100)
        
    Returns:
        pd.DataFrame: all couples of predictors with result of interaction tests. sorted by significance
    """

    def __init__(
        self,
        pred_num,
        pred_cat,
        target,
        base=None,
        weight=None,
        test_method="gini",
        use_grouping=True,
        use_grouping_interactions=True,
        groups_per_predictor=5,
        min_group_size=100,
        mean_target_regularization_weight=0.1,
        grouping_category_limit=100,
    ):
        """Initalization
        """

        self.pred_num = pred_num
        self.pred_cat = pred_cat
        self.target = target
        self.base = base
        self.weight = weight
        self.use_grouping = use_grouping
        self.use_grouping_interactions = use_grouping_interactions
        self.groups_per_predictor = groups_per_predictor
        self.min_group_size = min_group_size
        self.mean_target_regularization_weight = mean_target_regularization_weight
        self.grouping_category_limit = grouping_category_limit
        if test_method not in ["gini", "lr"]:
            raise ValueError("parameter test_method must be either 'gini' or 'lr'")
        else:
            self.test_method = test_method

    def _prepare_final_columns(self, data):
        """Prepare target, base and weight columns
        
        Args:
            data (pd.DataFrame): dataframe with all the predictors, target and other variables specified in init
        
        Returns:
            pd.DataFrame: dataframe with target, base and weight column
        """

        data_transformed = data[[self.target]].copy()
        if self.base:
            data_transformed[self.base] = data[self.base]
        else:
            self.base = "__base"
            data_transformed[self.base] = 1
        if self.weight:
            data_transformed[self.weight] = data[self.weight]
        else:
            self.weight = "__weight"
            data_transformed[self.weight] = 1

        return data_transformed

    def _transform_predictors(self, data, data_transformed):
        """Transform predictor columns from data and add the transformed results to data_transformed
        
        Args:
            data (pd.DataFrame): dataframe with all the predictors
            data_transformed (pd.DataFrame): dataframe with target, base and weight column
        
        Returns:
            list of str, pd.DataFrame: list of predictors and updated data_transformed
        """

        base_mask = data_transformed[self.base] == 1

        # a) using Grouping from scoring library
        if self.use_grouping:
            grouping = Grouping(
                columns=self.pred_num,
                cat_columns=self.pred_cat,
                group_count=self.groups_per_predictor,
                min_samples=self.min_group_size,
                min_samples_cat=self.min_group_size,
            )
            grouping.fit(
                X=data[base_mask][self.pred_num + self.pred_cat],
                y=data_transformed[base_mask][self.target],
                w=data_transformed[base_mask][self.weight],
                category_limit=self.grouping_category_limit,
            )
            data_transformed_pred = grouping.transform(
                data=data[self.pred_num + self.pred_cat], transform_to="woe"
            )
            data_transformed = pd.concat(
                [data_transformed, data_transformed_pred], axis=1
            )
            predictors = [pred + "_WOE" for pred in self.pred_num + self.pred_cat]

        # b) let categorical as they are, use quantiles for numerical, then apply mean target encoding
        else:
            data_transformed[self.pred_cat] = data[self.pred_cat]

            for col in self.pred_num:
                data_transformed[col] = pd.qcut(
                    data[col],
                    q=self.groups_per_predictor,
                    labels=False,
                    duplicates="drop",
                )
            
            mean_target_encode = MeanTargetEncoder(
                regularization_parameter=self.mean_target_regularization_weight
            )

            for col in self.pred_num + self.pred_cat:
                mean_target_encode.fit(
                    predictor=data_transformed[base_mask][col],
                    target=data_transformed[base_mask][self.target],
                    weight=data_transformed[base_mask][self.weight],
                )
                data_transformed[col] = mean_target_encode.transform(
                    predictor=data_transformed[col], return_type='logit'
                )
            predictors = self.pred_num + self.pred_cat

        return predictors, data_transformed

    def _create_interaction(self, data_transformed, pred1, pred2):
        """Creates new columns in dataframe as interaction of two of its given columns
        
        Args:
            data_transformed (pd.DataFrame): name of dataframe
            pred1 (str): name of column with first predictor to make interaction from
            pred2 (str): name of column with second predictor to make interaction from
        
        Returns:
            pd.DataFrame: updated data_transformed 
                (columns "interaction" and "interaction_num" are result of what was done in this method)
        """
        data_transformed["interaction"] = (
            data_transformed[pred1].astype(str)
            + "_"
            + data_transformed[pred2].astype(str)
        )

        base_mask = data_transformed[self.base] == 1

        # make a numerical variable from interaction - a) create a grouping variable
        if self.use_grouping_interactions:
            interaction_grouping = Grouping(
                columns=[],
                cat_columns=["interaction"],
                group_count=self.groups_per_predictor,
                min_samples=self.min_group_size,
                min_samples_cat=self.min_group_size,
            )
            interaction_grouping.fit(
                X=data_transformed[base_mask][["interaction"]],
                y=data_transformed[base_mask][self.target],
                w=data_transformed[base_mask][self.weight],
                category_limit=self.grouping_category_limit,
            )
            data_transformed["interaction_num"] = interaction_grouping.transform(
                data=data_transformed[["interaction"]], transform_to="woe"
            )

        # b) use mean target logit encoding
        else:
            mean_target_encode = MeanTargetEncoder(
                regularization_parameter=self.mean_target_regularization_weight
            )
            mean_target_encode.fit(
                predictor=data_transformed[base_mask]["interaction"],
                target=data_transformed[base_mask][self.target],
                weight=data_transformed[base_mask][self.weight],
            )
            data_transformed["interaction_num"] = mean_target_encode.transform(
                predictor=data_transformed["interaction"], return_type='logit'
            )

        return data_transformed

    def test_interactions(self, data):
        """Iteratively test each predictor with each predictor and test the couple whether there is a significant 
        interaction between them. The paramteres of such test are set in initialization method.
        
        Args:
            data (pd.DataFrame): dataframe with all the predictors, target and other variables specified in init
        
        Returns:
            pd.DataFrame: all couples of predictors with result of interaction tests. sorted by significance
        """

        # create dataset with target, base and weight
        data_transformed = self._prepare_final_columns(data)

        # transform predictors so the interactions can be created from tehem
        predictors, data_transformed = self._transform_predictors(data, data_transformed)

        # test each with each for interaction
        self.results = []
        iterations_total = int((len(predictors) * (len(predictors) - 1)) / 2)

        for iteration_num, (pred1, pred2) in enumerate(itertools.combinations(predictors, 2)):
            print(
                f"Iteration {iteration_num+1}/{iterations_total}\nPredictors: {pred1}, {pred2}"
            )

            data_transformed = self._create_interaction(data_transformed, pred1, pred2)
            
            # test interaction by statistical test
            if self.test_method == "lr":
                p_value = logreg_lr_test(
                    data=data_transformed,
                    predictors_model=[pred1, pred2, "interaction_num"],
                    predictors_submodel=[pred1, pred2],
                    target=self.target,
                    weight=self.weight,
                )
                self.results.append(
                    {"Predictors": sorted([pred1, pred2]), "p-value": p_value}
                )
                print(f"p-value: {p_value}\n")

            # test interaction by gini difference
            elif self.test_method == "gini":
                gini_diff = gini_difference(
                    data=data_transformed,
                    predictors_model1=["interaction_num"],
                    predictors_model2=[pred1, pred2],
                    target=self.target,
                    weight=self.weight,
                )
                self.results.append(
                    {"Predictors": sorted([pred1, pred2]), "Gini diff": gini_diff}
                )
                print(f"Gini diff: {gini_diff}\n")

        # output
        results_df = pd.DataFrame(self.results)
        results_df.set_index("Predictors", inplace=True)
        if self.test_method == "lr":
            results_df.sort_values("p-value", inplace=True, ascending=True)
            results_df["p-value"] = results_df["p-value"].apply(
                lambda x: round(x, 6)
            )
        elif self.test_method == "gini":
            results_df.sort_values("Gini diff", inplace=True, ascending=False)
            results_df["Gini diff"] = results_df["Gini diff"].apply(
                lambda x: round(100 * x, 4)
            )

        return results_df
