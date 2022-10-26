
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


import configparser
import json
import math
import os
import time
import ast

import matplotlib as mpl
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .._utils import create_logger, kill_logger
from .gui import InteractionsGUI
from .sql_creation import create_sql as _create_sql

CONFIG_FILE_PATH = "config.ini"


class Interactions:
    """
    Class for potential estimation

    Args::
        config_file_path (os.path) where the configuration file is located
        logger (logging.Logger, optional) logger instance. Created if None. (default: None)
    """

    def __init__(self, config_file_path=None, config_dict=None, logger=None):
        """
        Interaction class initialisation
        """
        self._data = (
            None
        )  # The data will be initialised inside functions - so they can be modified.
        # TkInter does not allow for data frame return
        self._config = configparser.ConfigParser()
        if config_file_path:
            self._config.read(config_file_path)
        elif config_dict:
            assert type(config_dict) is dict
            self._config.read_dict(config_dict)
        else:
            raise ValueError("Missing either configuration file")

        if not logger:
            self._logger = create_logger(
                filename=self._config.get("common", "log_filename") + ".log",
                loglevel=int(self._config.get("common", "log_level")),
            )
        else:
            self._logger = logger
        self._read_config()
        self._metadata = {}
        self._sql_info = {}
        self._logger.info("Interaction module successfuly initialized")

    def _read_config(self):
        """
        Configuration reading function.
        Reads needed variables from config and stores them in `self`
        """
        # ast.literal_eval is safer than normal eval
        # allows for strings, bytes, numbers, tuples, lists, dicts, sets, booleans, None, bytes and sets.
        # thus no code can be passed
        self._categorical_cols = ast.literal_eval(
            self._config.get("data_specification", "categorical_cols")
        )
        self._logger.debug(f"Categorical cols: {self._categorical_cols}")
        self._numerical_cols = ast.literal_eval(
            self._config.get("data_specification", "numerical_cols")
        )
        self._logger.debug(f"Numerical cols: {self._numerical_cols}")
        self._target_col = self._config.get("data_specification", "target_col")
        self._logger.debug(f"Target col: {self._target_col}")

    def _potential_estimation_input_check(self):
        """
        Function to check data validity
        """

        try:
            assert set(self._numerical_cols).issubset(
                set(self._data.columns)
            ), "Numerical column names are not in the provided data sample. Function is interupted!!!"
            assert set(self._categorical_cols).issubset(
                set(self._data.columns)
            ), "Categorical column names are not in the provided data sample. Function is interupted!!!"
            assert (
                self._target_col in self._data.columns
            ), "Specified target is not in the provided data sample. Function is interupted!!!"
            assert (
                self._data[self._target_col].isnull().sum() == 0
            ), "Target column cannot include null values. Function is interupted!!!"
            self._logger.debug("Input checks passed.")
        except AssertionError as err:
            self._logger.error(err)
            raise

    def _interaction_calculation(self, tested_data_clean, first_var, second_var, n_bin):
        """
        Function to calculate interaction potential of two functions

        Args:
            tested_data_clean (pandas.DataFrame): dataframe with dropped NAs, with given columns for testing
            first_var (str): name of first column
            second_var (str): name of second column to be tested
            n_bins (integer): number of bins to split each variable for interaction testing

        Returns:
            tuple of floats -- pvalue of interaction term and frequency of the moderator.
        """
        self._potential_estimation_input_check()

        try:
            tested_data_clean["moder"] = np.where(tested_data_clean[second_var] > tested_data_clean[second_var].mean(), 1, 0)
        except ValueError as err:
            raise ValueError('You might have incorrect columns in your data. Check for "TARGET" column. ').with_traceback(err)

        # frequency of values above average. if it is below 0.5, we take 1-this value
        moderator_freq = (
            tested_data_clean["moder"].sum() / tested_data_clean["moder"].count()
        )
        if moderator_freq < 0.5:
            moderator_freq = 1 - moderator_freq
        try:
            x1_perc = np.unique(
                np.percentile(
                    tested_data_clean[first_var], np.arange(0, 100.001, round(100 / n_bin))
                )
            )
        except IndexError:
            self._logger.warn(
                f'Variable {first_var} is causing problems. Check its values. \n')
            self._logger.debug(
                'This warning is issued when clean data are having no values (or small amount to make percentiles from them) \n'
                'and therefore percentiles cannot be calculated. Therefore interaction p-value "nan" is returned.'
                )
            
            pval_inter = float('nan')
            return pval_inter, moderator_freq
        tested_data_clean[first_var + "_BIN"] = pd.cut(
            x=tested_data_clean[first_var],
            bins=x1_perc,
            labels=np.arange(1, len(x1_perc)),
        )

        aggregated_data = (
            tested_data_clean.groupby([first_var + "_BIN", "moder"])[
                [self._target_col, first_var]
            ]
            .mean()
            .reset_index()
        )

        aggregated_data["inter"] = aggregated_data[first_var] * aggregated_data["moder"]

        if len(aggregated_data) > 0:
            results = smf.ols(
                f"{self._target_col} ~  {first_var} + moder + inter",
                data=aggregated_data,
            ).fit()
            pval_inter = results.pvalues[3]
        else:
            self._logger.warn(f'Variable {first_var} is causing problems in probability value calculations. Check its values.')
            pval_inter = float('nan')

        return pval_inter, moderator_freq

    def interaction_potential(self, data, n_bin=10):
        """ Compute interaction potential between predictors.

        Args:
            n_bins (integer): number of bins to split each variable for interaction testing

        Warnings:
            Function can be applyed only on numerical predictors.
            Results will be biased if used on categorical (string values) predictor encoded as numerical.
        """
        self._data = data
        to_test = self._numerical_cols[:]
        i = 1

        file_address = os.path.join(
            self._config.get("common", "work_dir"),
            self._config.get("common", "potential_output_filename") + ".txt",
        )

        # hacky: there is need to check whether file exists before `with open(...) as file - otherwise it is created`
        file_exists = os.path.isfile(file_address)
        with open(file_address, "a") as file:
            if not file_exists:
                file.write(
                    "X1;X2;p_val_x1;p_val_x2;obs_cnt;moder_freq1;moder_freq2;p_val_min\n"
                )
            else:
                self._logger.warn(
                    "File already exists. Results will be appended to an existing file!!!"
                )

            for x1 in self._numerical_cols:
                time_start = time.time()
                to_test.remove(x1)

                for x2 in to_test:
                    # cleaned subset of tested columns - ommitting rows where there are any NAs
                    tested_clean = self._data[[self._target_col, x1, x2]].dropna(
                        how="any"
                    )

                    p_val_x1, mod_freq1 = self._interaction_calculation(
                        tested_clean, x1, x2, n_bin=n_bin
                    )
                    p_val_x2, mod_freq2 = self._interaction_calculation(
                        tested_clean, x2, x1, n_bin=n_bin
                    )
                    file.write(
                        "{};{};{};{};{};{};{};{}\n".format(
                            x1,
                            x2,
                            p_val_x1,
                            p_val_x2,
                            len(tested_clean),
                            mod_freq1,
                            mod_freq2,
                            min([p_val_x1, p_val_x2]),
                        )
                    )

                self._logger.info(
                    "{} / {}; Round duration: {:.2f} sec.".format(
                        i, str(len(self._numerical_cols)), time.time() - time_start
                    )
                )
                i += 1

    def _interactions_assertions(self, row_var, column_var, n_bins):
        """
        Assertions oof the interactions - checks the correct input for interactions

        Args:
            row_var (str): variable from which interaction rows will be created
            column_var (str): variable from which interaction columns will be created
            n_bins (int): number of bins
        """

        try:
            assert row_var != column_var, "Row and column variable are the same!"
            assert (
                row_var in self._data.columns
            ), "Column {} is not in provided data sample!".format(row_var)
            assert (
                column_var in self._data.columns
            ), "Column {} is not in provided data sample!".format(column_var)
            assert (
                self._target_col in self._data.columns
            ), "Provided target is not in the data sample!"

            # for next checks, see https://docs.scipy.org/doc/numpy/reference/generated/numpy.dtype.kind.html
            # and https://stackoverflow.com/questions/19900202/how-to-determine-whether-a-column-variable-is-numeric-or-not-in-pandas-numpy
            assert self._data[row_var].dtype.kind in "biufcO", (
                "Only numeric or string data types are allowed. "
                "Column {} has type {}"
            ).format(row_var, self._data[row_var].dtype)
            assert self._data[column_var].dtype.kind in "biufcO", (
                "Only numeric or string data types are allowed. "
                "Column {} has type {}"
            ).format(column_var, self._data[column_var].dtype)

            assert (
                self._data[self._target_col].isnull().sum() == 0
            ), "Target variable cannot include null values!"
            assert n_bins > 0, "Parameter n_bins must be positive (greater than zero)!"

        except AssertionError as err:
            self._logger.error(err)
            raise

    def _interactions_bins_validation(
        self, mask, row_bins, col_bins, row_var, column_var
    ):
        """
        Validation of bin format.

        Args:
            mask (bool): filtering mask
            row_bins (numeric or list of sets): bins for row variable
            col_bins (numeric or list of sets): bins for column variable
            row_var (str): variable from which interaction rows will be created
            column_var (str): variable from which interaction columns will be created

        Returns:
            mask, row_bins, col_bins: if they don't pass the check, they are returned as None
        """

        if mask:
            if mask.dtype != bool:
                self._logger.warn("Provided mask has incorrect type and is ignored!")
                mask = None

        if row_bins and not self._check_bins(row_bins, row_var, "row"):
            row_bins = None

        if col_bins and not self._check_bins(col_bins, column_var, "col"):
            col_bins = None

        return mask, row_bins, col_bins

    def _check_bins(self, bins, variable, rows_or_cols):
        """
        Main function for bin checking. Checks the format of bins/

        Args:
            bins (list): bins which should be used for binning the variable
            variable (str): which variable should be used
            rows_or_col (str): whether used variable is row or column

        Returns:
            boolean: did the given bin passed the check?
        """

        result = True
        if type(bins) != list:
            self._logger.warn(
                (
                    "Provided bin borders for row variable have invalid format"
                    f' (parameter type should be "list"). Parameter {rows_or_cols}_bins is ignored!'
                )
            )
            result = False

        column_cond = np.issubdtype(self._data[variable].dtype, np.number)
        bin_cond = isinstance(bins[0], (float, int))
        if column_cond and not bin_cond:
            self._logger.warn(
                (
                    "Provided bin borders for row variable contain non numerical values. "
                    f"Parameter {rows_or_cols}_bins is ignored!"
                )
            )
            result = False

        if (self._data[variable].dtype == "O") & (type(bins[0]) != set):
            self._logger.warn(
                (
                    "Provided bin borders for row variable should be list of sets. "
                    f"Parameter {rows_or_cols}_bins is ignored!"
                )
            )
            result = False

        return result

    def _numerical_bin_processing(self, data, bins, variable, n_bins):
        """
        Processing of the bins and creating interaction fro given bin
        if the chosen variable is numerical.
        Creates bins, if they do not exist.

        Args:
            data (pandas.DataFrame): data from which bins should be calcualted
            bins (list): list of bins to separate the variable
            variable (str): which variable should be used for percentile creation
            n_bins (int): number of bins

        Returns:
            interactions, bin_names, empty dict: calculated interactions, names of the individual bins, placeholder for 'category_alias'
        """

        if not bins:
            percentiles = np.unique(
                np.percentile(
                    data[variable][data[variable].notnull()],
                    np.arange(0, 100.001, 100 / n_bins),
                ).round(2)
            )
            percentiles[0] = -np.inf
            percentiles[len(percentiles) - 1] = np.inf
        else:
            percentiles = bins[:]
            percentiles = [-np.inf] + percentiles + [np.inf]
        bins_numeric = list(
            zip(
                percentiles[0 : len(percentiles) - 1], percentiles[1 : len(percentiles)]
            )
        )

        bin_names = self._create_bin_names(bins_numeric)

        interactions = pd.cut(
            self._data[variable],
            percentiles,
            labels=np.arange(1, len(percentiles)),
            include_lowest=True,
        )

        if np.abs(np.nan_to_num((interactions)).min()) == 0:
            interactions.cat.add_categories([0], inplace=True)
            interactions.fillna(0, inplace=True)
            bin_names += ["null"]
            bins_numeric += [(None, None)]

        return interactions, bin_names, bins_numeric

    @staticmethod
    def _create_bin_names(bins_numeric):
        """Create bin names from numeric bins
        
        Args:
            bins_numeric (zip): zipped lower, upper bounds
        
        Returns:
            list: same as bins numeric, but in str
        """

        bin_names = ["({};{}]".format(low, upp) for low, upp in bins_numeric]
        bin_names[len(bin_names) - 1] = bin_names[len(bin_names) - 1][:-1] + ")"
        return bin_names

    def _categorical_bin_processing(self, data, bins, variable):
        """
        Processing of the bins and creating interaction fro given bin
        if the chosen variable is categorical.
        Creates bins, if they do not exist.

        Args:
            data (pandas.DataFrame): data from which bins should be calcualted
            bins (list): list of bins to separate the variable
            variable (str): which variable should be used for percentile creation

        Returns:
            interactions, bin_names, category_alias: calculated interactions, names of the individual bins, alias for individual bin categories
        """

        category_alias = {}
        if bins:
            for i in range(len(bins)):
                for cat in bins[i]:
                    category_alias[cat] = "cat_{}".format(i + 1)
            unassigned_cats = [
                cat if str(cat) != "nan" else "null"
                for cat in data[variable].unique()
                if cat not in category_alias.keys()
            ]
            if unassigned_cats:
                for cat in unassigned_cats:
                    category_alias[cat] = "cat_{}".format(len(bins) + 1)

            data[variable] = data[variable].replace(to_replace=category_alias)

            self._data[variable] = self._data[variable].replace(
                to_replace=category_alias
            )

        dt_grp = data.groupby(variable).agg({self._target_col: ["sum", "count"]})
        dt_grp.columns = ["bad_cnt", "tot_cnt"]
        dt_grp["def_rx"] = dt_grp["bad_cnt"] / dt_grp["tot_cnt"]
        dt_grp.sort_values("def_rx", inplace=True)
        dt_grp["categ"] = range(1, len(dt_grp) + 1)
        interactions = self._data[[variable]].replace(
            to_replace=dt_grp[["categ"]].to_dict()["categ"]
        )[variable]
        bin_names = list(dt_grp.index)
        interactions = interactions.astype("category")
        if interactions.isnull().sum() > 0:
            interactions.cat.add_categories(
                [0], inplace=True
            )  # TODO: according to pandas docs,
            interactions.fillna(
                0, inplace=True
            )  # using inplace is discouraged practice and will be deprecated
            bin_names += ["null"]

        return interactions, bin_names, category_alias

    def interactions(
        self,
        row_var,
        column_var,
        data,
        n_bins=10,
        mask=None,
        row_bins=None,
        col_bins=None,
    ):
        """
        Returns count / bad count / default rate matrices created for two variables in data set.

        Args:
            row_var (str): variable from which interaction rows will be created
            column_var (str): variable from which interaction columns will be created
            n_bins (int, optional): In case of processing numerical variable, n_bins defines into how many bins
                will be variable divided (percentiles are used for defining bin borders).
                This parameter is ignored in case that row_bins / col_bins are provided.
            mask (pandas.core.series.Series, optional): Defines which rows will be used to compute outputs.
            row_bins (list, optional): Defines bin borders for row variable. In case of numerical variable it will
                be list of numbers, in case of categorical variables row_bins will be list
                of sets (each set contains categories that will be grouped together).
            col_bins (list, optional): Defines bin borders for column variable. In case of numerical variable it
                will be list of numbers, in case of categorical variables row_bins will be
                list of sets (each set contains categories that will be grouped together).

        Returns:
            bad_mtrx (???):
            cnt_mtrx (???):
            def_mtrx (???):
            inter_coord (???):
            row_cat_alias (???):
            col_cat_alias (???):
        """

        self._interactions_assertions(
            row_var=row_var, column_var=column_var, n_bins=n_bins
        )
        self._logger.debug("Interaction assertions passed")
        mask, row_bins, col_bins = self._interactions_bins_validation(
            mask=mask,
            row_bins=row_bins,
            col_bins=col_bins,
            row_var=row_var,
            column_var=column_var,
        )
        self._logger.debug("Interaction bins validated")

        interaction_data = data[[self._target_col, row_var, column_var]]
        if mask:
            interaction_data = interaction_data[mask]
        else:
            mask = data[self._target_col] >= 0
        row_bins_numeric = None
        col_bins_numeric = None
        row_cat_alias = {}
        col_cat_alias = {}
        # Processing rows - numerical
        row_isnumeric = interaction_data[row_var].dtype.kind in "biufc"
        column_isnumeric = interaction_data[column_var].dtype.kind in "biufc"
        if row_isnumeric:
            self._logger.debug(f"Variable {row_var} is processed as numerical.")
            row_inter, row_bin_names, row_bins_numeric = self._numerical_bin_processing(
                data=interaction_data, bins=row_bins, variable=row_var, n_bins=n_bins
            )
        # Processing rows - categorical
        else:
            self._logger.debug(f"Variable {row_var} is processed as categorical.")
            row_inter, row_bin_names, row_cat_alias = self._categorical_bin_processing(
                data=interaction_data, bins=row_bins, variable=row_var
            )

        # Processing columns - numerical
        if column_isnumeric:
            self._logger.debug(f"Variable {column_var} is processed as numerical.")
            col_inter, col_bin_names, col_bins_numeric = self._numerical_bin_processing(
                data=interaction_data, bins=col_bins, variable=column_var, n_bins=n_bins
            )

        # Processing columns - categorical
        else:
            self._logger.debug(f"Variable {column_var} is processed as categorical.")
            col_inter, col_bin_names, col_cat_alias = self._categorical_bin_processing(
                data=interaction_data, bins=col_bins, variable=column_var
            )

        cnt_mtrx = pd.crosstab(
            row_inter[mask],
            col_inter[mask],
            values=interaction_data[self._target_col],
            aggfunc=[len],
            dropna=False,
        )
        self._logger.debug("Calculated count matrix")

        row_ids = [
            int(idx) - 1 if idx > 0 else len(row_bin_names) - 1
            for idx in cnt_mtrx.index
        ]

        col_ids = [
            int(idx[1]) - 1 if idx[1] > 0 else len(col_bin_names) - 1
            for idx in cnt_mtrx.columns.values
        ]


        cnt_mtrx.index = [row_bin_names[idx] for idx in row_ids]
        cnt_mtrx.columns = [col_bin_names[idx] for idx in col_ids]

        bad_mtrx = pd.crosstab(
            row_inter[mask],
            col_inter[mask],
            values=interaction_data[self._target_col],
            aggfunc=[sum],
            dropna=False,
        )
        self._logger.info("Calculated bad_mtrx")
        bad_mtrx.index = [row_bin_names[idx] for idx in row_ids]
        bad_mtrx.columns = [col_bin_names[idx] for idx in col_ids]

        def_mtrx = bad_mtrx / cnt_mtrx

        return dict(
            bad_mtrx=bad_mtrx.fillna(0),
            cnt_mtrx=cnt_mtrx.fillna(0),
            def_mtrx=def_mtrx.fillna(0),
            inter_coord=pd.DataFrame(row_inter.astype(int)).join(col_inter.astype(int)),
            row_cat_alias=row_cat_alias,
            col_cat_alias=col_cat_alias,
            row_bins_numeric=row_bins_numeric,
            col_bins_numeric=col_bins_numeric,
            row_isnumeric=row_isnumeric,
            column_isnumeric=column_isnumeric,
        )

    def fit(
        self,
        row_var,
        column_var,
        data,
        interaction_variable_name=None,
        rewrite=True,
        sql=False,
        n_bins=10,
        mask=None,
        row_bins=None,
        col_bins=None,
    ):
        """
        Calculate interaction metadata altogether with Gui start up.
        If this method is called multiple times, all metadata for multiple variables are saved.
        
        Args:
            row_var (str): variable from which interaction rows will be created
            column_var (str): variable from which interaction columns will be created
            data (pandas.DataFrame): data which should be processed.
            interaction_variable_name (str): how should the interaction variable be named.
                                               This is useful when we would like to create multiple interaction bins
                                               with the same pair of variables
            rewrite (bool): Whether existing metadata should be rewritten. 
            interaction_kwargs: arguments to be passed to interaction function which calculates the interactions itself.
        
        Returns:
            pandas.DataFrame: Data with new column with created interaction variables
        """

        self._data = data
        # self._data = self._data[~self._data[self._target_col].isna()]
        interactions_result = self.interactions(
            row_var=row_var,
            column_var=column_var,
            data=self._data,
            row_bins=row_bins,
            col_bins=col_bins,
            mask=mask,
            n_bins=n_bins,
        )

        app = InteractionsGUI(
            interactions_result=interactions_result,
            config=self._config,
            row_var=row_var,
            column_var=column_var,
            data=data,
        )
        app.mainloop()

        # We go through this part only if app created metadata.
        if app.metadata:
            # when the app is closed, we go through this part.
            if interaction_variable_name is None:
                interaction_variable_name = app.inter_name

            self._sql_info[interaction_variable_name] = dict(
                row_variable=row_var,
                column_variable=column_var,
                row_isnumeric=interactions_result["row_isnumeric"],
                column_isnumeric=interactions_result["column_isnumeric"],
                categories=app.categories,
                row_cat_alias=interactions_result["row_cat_alias"],
                col_cat_alias=interactions_result["col_cat_alias"],
            )
            self._metadata[interaction_variable_name] = dict(
                row_variable=row_var, column_variable=column_var, metadata=app.metadata
            )

            file_address = os.path.join(
                self._config.get("common", "work_dir"),
                self._config.get("interactions", "metadata_file") + ".json",
            )
            if rewrite:
                self._logger.info("Existing metadata will be deleted.")
                try:
                    os.remove(file_address)
                except FileNotFoundError:
                    self._logger.warning(f"{file_address} does not exist")
            else:
                new_file_address = file_address
                counter = 1
                while True:
                    if os.path.isfile(new_file_address):
                        new_file_address = os.path.join(
                            self._config.get("common", "work_dir"),
                            f"{self._config.get('interactions', 'metadata_file')}_{counter}.json",
                        )
                        counter += 1
                    else:
                        file_address = new_file_address
                        break

            with open(file_address, "w") as fp:
                json.dump(self._metadata, fp)
            self._logger.info(f"Metadata for {interaction_variable_name} saved.")
        else:
            self._logger.info("No metadata saved - have you clicked on Save Grouping?")

        # kill_logger(self._logger)
        # return app._inter_coord

    def transform(self, data, metadata=None):
        """ Apply all interactions created by .fit (or by loading metadata)
        This method depends on te metadata structure.
        That currently is as following:
        {
            "interaction_variable_1": {
                "row_variable": "variable_1",
                "column_variable": "variable_2".
                "metadata": {
                    "1": [                    <-- first group
                        [
                            "AAA",            <-- indicates categorical variable
                            [                 <-- as the second variable is enclosed as another list
                                0.83,             this is being taken as interval
                                1.01              thus being cared of as numerical variable
                            ]
                        ],
                        [
                            "AAA",
                            [
                                1.5,
                                1.92
                            ]
                        ]
                    ],
                    "2": [                    <-- second group
                        [
                            "BBB",
                            [-Infinity,       <-- This is how `json.dump` saves `np.inf`
                                0.37
                            ]
                        ],
                        [
                            "AAA",
                            [-Infinity,
                                0.37
                            ]
                        ],
                        ...                   <-- indicating that there could be more
                    ]
                }
            }
        }
        

        
        Args:
            data (pandas.DataFrame): data, which have columns on which interactions were calculated
            metadata (dict): metadata which will be applied on provided data. 
                               If None - we check whether Interactions class have some metadata inside calculated
        """

        def _get_selector(variable_values, compare_with):
            """Helper function to obtain selector (bool vector) satisfying given conditions.
                If the `compare_with` is tuple - we are assuming that given variable is numerical, and using interval conditions
                If the `compare_with` is something else - we assume that given variable is categorical
            
            
            Args:
                variable_values (pandas.Series): values, which are being compared. Either numerical or categorical
                compare_with (str or tuple): values which are being compared
            
            Returns:
                pandas.Series: Series with boolean values indicating satisfied condition
            """
            # tuple assumes interval values - getting interval
            if type(compare_with) in (tuple, list):
                if compare_with[0] not in (None, "null") and compare_with[1] not in (
                    None,
                    "null",
                ):
                    _selector = (variable_values > compare_with[0]) & (
                        variable_values <= compare_with[1]
                    )
                else:
                    _selector = variable_values.isna()
            # otherwise categorical
            else:
                if compare_with not in (None, "null"):
                    _selector = variable_values == compare_with
                else:
                    _selector = variable_values.isna()
            return _selector

        if metadata is None:
            metadata = self._metadata
        try:
            assert metadata, "No metadata created - have you called .fit method?"
            assert type(metadata) == dict, "Metadata must be dictionary"
        except AssertionError as err:
            self._logger.error(err)

        data = data.copy()

        # We can have multiple interaction variables in the metadata:
        for interaction_var, interaction_data in metadata.items():
            # Creating empty vector
            data[interaction_var] = np.zeros(data.shape[0])
            # getting relevant variables. For both row and columns
            try:
                row_values = data[interaction_data["row_variable"]]
                column_values = data[interaction_data["column_variable"]]
            except KeyError as err:
                self._logger.error(
                    "Variable from which interactions are created is not in the data."
                )
                raise
            # getting relevant metadata
            relevant_metadata = interaction_data["metadata"]
            for group in relevant_metadata:
                group_metadata = relevant_metadata[group]

                selector = [False] * data.shape[0]
                for column, row in group_metadata:
                    selector = selector | (
                        _get_selector(column_values, column)
                        & _get_selector(row_values, row)
                    )
                data[interaction_var][selector] = int(group)

        return data

    def list_interaction_names(self, metadata=None):

        if metadata is None:
            metadata = self._metadata
        try:
            assert metadata, "No metadata created - have you called .fit method?"
            assert type(metadata) == dict, "Metadata must be dictionary"
        except AssertionError as err:
            self._logger.error(err)

        interaction_var_list = []
        for interaction_var, _ in metadata.items():
            interaction_var_list.append(interaction_var)

        return interaction_var_list

    def create_sql_from_metadata(self, metadata=None, print_output=True):
        def _condition(variable_name, variable_value):
            if type(variable_value) in (tuple, list):
                if variable_value[0] == -np.inf:
                    result = f"{variable_name} <= {variable_value[1]}"
                elif variable_value[1] == np.inf:
                    result = f"{variable_name} > {variable_value[0]}"
                elif (variable_value[0] is None) and variable_value[1] is None:
                    result = f"{variable_name} is null"
                else:
                    result = f"{variable_name} > {variable_value[0]} and {variable_name} <= {variable_value[1]}"
            # otherwise categorical
            else:
                if variable_value is None or variable_value == "null":
                    result = f"{variable_name} is null"
                else:
                    result = f"{variable_name} = '{variable_value}'"
            return result

        if metadata is None:
            metadata = self._metadata
        try:
            assert metadata, "No metadata created - have you called .fit method?"
            assert type(metadata) == dict, "Metadata must be dictionary"
        except AssertionError as err:
            self._logger.error(err)

        # We can have multiple interaction variables in the metadata:
        for interaction_var, interaction_data in metadata.items():
            # getting relevant variables. For both row and columns
            row_name = interaction_data["row_variable"]
            column_name = interaction_data["column_variable"]
            # getting relevant metadata
            relevant_metadata = interaction_data["metadata"]
            all_conditions = ""
            max_loops = len(relevant_metadata)
            counter = 1
            for group, group_metadata in relevant_metadata.items():
                if counter == max_loops and max_loops != 1:
                    all_conditions += f"\telse {group}\n"
                    break
                for column, row in group_metadata:
                    all_conditions += f"\twhen {_condition(column_name, column)} and {_condition(row_name, row)} then {group}\n"
                counter += 1

            all_conditions = f"case\n{all_conditions}end as {interaction_var}"
            file_address = os.path.join(
                self._config.get("common", "work_dir"),
                f"{self._config.get('interactions', 'sql_file_prefix')}{interaction_var}.sql",
            )

            with open(file_address, "w") as file:
                file.write(all_conditions)
            self._logger.info(f"Sql for {interaction_var} saved.")
            if print_output:
                print(all_conditions)

    def create_sql(self, metadata, print_output=True):
        final_sql = ""
        for interaction_var, sql_info in self._sql_info.items():
            final_sql += _create_sql(interaction_variable=interaction_var, **sql_info)
        file_address = os.path.join(
            self._config.get("common", "work_dir"),
            self._config.get("interactions", "sql_file") + ".sql",
        )

        with open(file_address, "w") as file:
            file.write(final_sql)
        self._logger.info(f"Sql for {interaction_var} saved.")
        if print_output:
            print(final_sql)


def main():
    """
    Main function to be run if the file is executed by itself.
    """
    data = pd.read_csv(
        "C:/Users/jan.hynek/Documents/HCI/research/Interactions/ExampleData6.csv"
    )
    ints = Interactions(config_file_path=CONFIG_FILE_PATH)
    print(ints.fit("Numerical_1", "Numerical_2", data=data))
    # kill_logger(ints._logger)


if __name__ == "__main__":
    main()
