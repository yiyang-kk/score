import itertools
from typing import Dict, List, Tuple
from textwrap import dedent
import pandas as pd
import numpy as np
from ..front_end.config_handler import (
    RATIO_CONFIG_KEY,
    SIMPLE_CONFIG_KEY,
    TIME_SINCE_CONFIG_KEY,
    DEFAULT_TIME_ORDER,
)
from ..utils.logger import create_logger  # pylint: disable=relative-beyond-top-level
from .config_processors import (
    ConfigProcessor,
)  # pylint: disable=relative-beyond-top-level
from .time_order import (
    TimeOrder,
    GRANULARITY_ALIASES,
    ORDER_GRANULARITY,
    MONTH_GRANULARITY,
    YEAR_GRANULARITY,
    DAY_GRANULARITY,
    MINUTE_GRANULARITY,
    SECONDS_GRANULARITY,
    HOUR_GRANULARITY,
)
from pandas.core.groupby.groupby import DataError


class MetaAggregation:
    """
    Base MetaAggregation class which contains functions
    which are used in specialised classes
    Namely:
    - obtain queries from segmentations (one less loop!)
    - get feature names
    - divide ratio settings into simple settings

    Args:
        config (dict): configuration file describing aggregation to be performed.
            Has to contain keys `meta`, `simple`, `time_since`, `ratio`.
        shortening_dictionary (dict, optional): key: word in feature name; value: word to replace this word with (default: None)
        raise_on_error (bool, optional): whether class should raise error in case it encounters some problem (default: False)
        logger (logging.Logger, optional): logging instance. If None, logger is created. (default: None)
        logger_kwargs {dict, optional): keyword arguments to be passed to logger (default: None)
        max_nan_share (float, optional): (default: None)

    """

    def __init__(
        self,
        config,
        shortening_dictionary=None,
        raise_on_error=False,
        logger=None,
        logger_kwargs=None,
        max_nan_share=None,
    ):
        """
        Initialisation which is used among all consecutive classes
        """
        self.raise_on_error = raise_on_error
        if logger_kwargs:
            self.logger = create_logger(**logger_kwargs)
        else:
            self.logger = logger if logger else create_logger()
        self.config = ConfigProcessor(
            config, raise_on_error=raise_on_error, logger=self.logger
        )
        (
            self.index,
            self.transaction_time,
            self.target_time,
            self.order,
            granularity,
            self.nan_value,
            self.inf_value,
        ) = self.config.get_metadata()
        self.order_assigner = TimeOrder()
        self.check_granularity = self.order_assigner.check_granularity
        if self.order == DEFAULT_TIME_ORDER:
            self.granularity = self.check_granularity(granularity)
        else:
            self.granularity = ORDER_GRANULARITY
        self.variable_list_sql = {SIMPLE_CONFIG_KEY: [], RATIO_CONFIG_KEY: []}
        self.variable_list_pandas = {SIMPLE_CONFIG_KEY: [], RATIO_CONFIG_KEY: []}
        self.shortening_dictionary = shortening_dictionary
        self.columns_with_inf = None
        self.columns_with_nan = None
        self.max_nan_share = max_nan_share

    @staticmethod
    def _query_to_sql_query(query):
        """Helper method to transform pandas query statements to sql like statements

        Args:
            query (str): pandas query

        Returns:
            str: sql-like query
        """
        return query.replace("==", "=").replace(".isna()", " is null")

    @staticmethod
    def _segmentations_to_queries(data, segmentations):
        """Create list of pandas queries from selected segmentations
        This method is used for transformation from variable, which contains list of
        unique values to pandas queries

        Args:
            data (pd.DataFrame): data which contain given segmentation columns
            segmentations (list[str]): list of column names

        Returns:
            list[str]: list of pandas-like queries
        """

        queries = []
        if (
            (None in segmentations)
            or
            # fix of bug - front end does not support None object,
            # so it is casted to string
            ("None" in segmentations)
        ):
            queries += [""]
            segmentations = [sgm for sgm in segmentations if sgm]
        for segmentation in segmentations:
            unique_values = data[segmentation].unique()
            for value in unique_values:
                if value == value:  # check for np.nan, float('nan')
                    queries += [f"{segmentation}=='{value}'"]
                else:
                    queries += [f"{segmentation}.isna()"]

        return queries

    def _clean_feature_name(self, feature_name_parts):
        """From individual feature name parts create variable name, which is cleaned.
        If name_shortening dictionary is specified, replace given words by other words.

        Args:
            feature_name_parts (list of str): individual parts from which final name is created

        Returns:
            str: cleaned string where individual parts are concatenated and invalid symbols are replaced
        """
        dirty_feature_name = "_".join([feat for feat in feature_name_parts])
        clean_feature_name = (
            dirty_feature_name.replace(".isna()", "=null")
            .replace("==", "=")
            .replace("!=", "_neq_")
            .replace(">=", "_gteq_")
            .replace(">", "_gt_")
            .replace("<=", "_lteq_")
            .replace("-", "_")
            .replace("=", "_eq_")
            .replace("<", "_lt_")
            .replace("&", "_and_")
            .replace("|", "_or_")
            .replace("'", "")
            .replace("/", "_div_")
            .replace('"', "")
            .replace(" ", "")
            .upper()
        )
        if self.shortening_dictionary:
            for old_word, new_word in self.shortening_dictionary.items():
                clean_feature_name = clean_feature_name.replace(old_word, new_word)
        return clean_feature_name

    def _get_time_since_feature_name(
        self,
        original_variable,
        time_since_function,
        unit,
        query=None,
    ):
        """Create name for time-since feature

        Args:
            original_variable (str): name of the original variable
            time_since_function (str): time since function used
            unit (str): what granularity was used
            query (str, optional): exact query used. Defaults to None.

        Returns:
            str: time since variable name
        """
        resulting_parts = [
            time_since_function,
            original_variable,
            unit,
        ]
        if query:
            resulting_parts += [query]
        return self._clean_feature_name(resulting_parts)

    def _get_ratio_feature_name(self, numerator, denominator):
        """

        Args:
            numerator (str): numerator variable name
            denominator (str): denominator variable name

        Returns:
            str: final feature name
        """
        return f"{numerator}_DIV_{denominator}"

    def _get_feature_name(self, original_variable, function, time_range, query=None):
        """Get simple feature name, given original functions

        Args:
            original_variable (str): original variable name, on which given transformations were applied
            function (str): name of function applied
            time_range (tuple[str]): what time range was used
            query (str, optional): query used. Defaults to None.

        Returns:
            str: final feature name
        """
        feature_name_parts = self._get_feature_name_parts(
            original_variable, function, time_range, query
        )
        return self._clean_feature_name(feature_name_parts)

    def _get_feature_name_parts(self, original_variable, function, time_range, query):
        """Transform name parts, such as time range tuple into list of strings

        Args:
            original_variable (str): name of the variable
            function (str): function applied
            time_range (tuple[str]):
            query (str): query applied

        Returns:
            list[str]: list of individual parts
        """

        time_range_format = (
            f"{time_range[0]}{self.granularity}-{time_range[1]}{self.granularity}"
        )

        resulting_parts = [function, original_variable, time_range_format]
        if query:
            resulting_parts += [query]

        return resulting_parts

    def _handle_query(self, data, query):

        try:
            subset = data.query(query, engine="python") if query else data
            return subset
        except ValueError as err:
            err_message = f"""
            Incorrectly specified query.

            Possible issues:
            - Missing apostrophes in values
                HOME_PLACE==Value -> HOME_PLACE=='Value'
            - Comparison should be done using '==', not '='
                HOME_PLACE='Value' -> HOME_PLACE=='Value'
                
            Query:
            {query}

            Original error:
            {err}
            """
            self.logger.error(err_message)
            if self.raise_on_error:
                raise ValueError(err_message).with_traceback(err)
            return data

    def _get_existing_ratio_variables(
        self,
        data,
        numerator,
        denominator,
        time_ranges,
        functions,
        segmentations,
        queries,
    ):
        """
        UGH... i forgot that the variable relationships (like max-max) are important.
        So I am hacking them back
        I will try to do that in the best way possible, but the code is not written around it
        bear that in mind
        however - all calculated features had to be calculated anyway.
        there probably even isnt better way to get the requested features?
        """
        queries += self._segmentations_to_queries(data, segmentations)
        existing_ratio_features = []

        for function, time_range, query in itertools.product(
            functions, time_ranges, queries
        ):
            numerator_name = self._get_feature_name(
                numerator, function[0], time_range[0], query
            )
            denominator_name = self._get_feature_name(
                denominator, function[1], time_range[1], query
            )
            existing_ratio_features += [
                self._get_ratio_feature_name(numerator_name, denominator_name)
            ]

        return existing_ratio_features


class SimpleAggregationPandas(MetaAggregation):
    """
    Class for simple pandas aggregations.

    """

    def _simple_nan_imputing(self, data):
        if self.nan_value is not None:
            data = data.fillna(self.nan_value)
        return data

    def _get_time_range_subset(self, data, time_range):
        """Helper function to get simple timeranges

        Args:
            data (pd.DataFrame): input data
            time_range (tuple of int): time range of days

        Returns:
            pd.DataFrame: data where given time range applies
        """
        return data[
            (time_range[0] <= data[self.order]) & (data[self.order] <= time_range[1])
        ]

    def _apply_function(self, grouped, function):
        """Helper function to help me with modes and some other obscure functions

        Args:
            grouped (Grouped pd.DataFrame): DataFrame before application of some functin
            function (str): name of the applied function

        Returns:
            pd.Series: calculated variable
            or pd.DataFrame in case of mode_multicolumn
        """

        # i just dont like mode. please, dont use it.
        if function == "mode":

            def _special_mode(x):
                res = x.mode()
                return res[0] if len(res) > 0 else None

            feature = grouped.agg(_special_mode)

        elif function == "mode_multicolumn":
            feature = (
                grouped.apply(lambda x: x.mode())
                .reset_index()
                .pivot(index=self.index, columns="level_1")
                .T.reset_index(drop=True)
                .T
            )
        elif function == "is_monotonic":
            feature = grouped.apply(lambda x: x.is_monotonic)
        elif function == "is_monotonic_increasing":
            feature = grouped.apply(lambda x: x.is_monotonic_increasing)
        elif function == "is_monotonic_decreasing":
            feature = grouped.apply(lambda x: x.is_monotonic_decreasing)
        else:
            feature = getattr(grouped, function)()

        return feature

    def _calculate_function(
        self,
        grouped,
        variable,
        feature_name,
        function,
    ):
        """Function for error handling, usually with incorrectly specified datasets

        Args:
            grouped (Grouped pd.DataFrame): DataFrame before application of some function
            variable (str): name of the variable
            feature_name (str): name of the calculated feature
            function (str): name of the applied function

        Returns:
            pd.Series: calculated variable
            or pd.DataFrame in case of mode_multicolumn
        """
        feature = None
        try:
            feature = self._apply_function(grouped=grouped, function=function)

        except AttributeError as err:
            if function == "mode":
                self.logger.error(
                    dedent(
                        f"""
                        Unsupported function "mode" for variable {variable}.
                        {feature_name} not calculated.
                        """
                    )
                )

            else:
                self.logger.fatal(err)
                raise
        except DataError as err:
            self.logger.error(
                dedent(
                    f"""
                    Unsupported aggregation {function} for variable {variable}.
                    Possibly using numeric aggregation on categorical variable.
                    {feature_name} not calculated.
                    """
                )
            )

        except ValueError as err:
            if "not 1-dimensional" in str(err):
                message = """
                        Improperly set index. Check whether you set the correct index column.
                        Should be SKP_CREDIT_CASE or similar.
                        """
                self.logger.fatal(dedent(message))
                raise ValueError(message).with_traceback(None)
        except TypeError as err:
            if "not supported between instances of 'float' and 'str'" in str(err):
                message = f"""
                Unsupported function '{function}' for variable '{variable}'. Omitting.
                """
                self.logger.warning(dedent(message))

            raise
        return feature

    def _get_grouped_dataset(self, subset_time_range, variable):
        """Helper & catching function for creation of the pre-grouped dataset

        Args:
            subset_time_range (pd.DataFrame): already pre-limited dataset
            variable (str): name of the variable

        Raises:
            ValueError: when variable is incorrectly specified for aggregation
        Returns:
            Grouped pd.DataFrame
        """
        try:
            grouped = subset_time_range.groupby(self.index)[variable]
        except ValueError as err:
            # bugfix
            if "1" in str(err):
                raise ValueError(
                    "Variable cannot be used for given aggregation"
                ) from err
            raise
        return grouped

    def _get_simple_pandas_feature_for_variable(
        self, data, variable, functions, time_ranges, queries, segmentations
    ) -> Dict:
        """
        Main feature calculation function for simple & ratio variables.
        For single variable (as settings can be created for multiple variables!),
        create all pre-specified features


        Args:
            data (pandas.DataFrame): data on which this shoudl be calculated
            variable (str): Name of variable for which features should be calculated
            functions (list of str): functions to be applied
            time_ranges (list of tuples of ints): list of time ranges which are stored as tuples (from, to)
            queries (list of str): list of strings to be fed into pd.DataFrame.query()
            segmentations (list of str): list of variable names to be used for segmentations

        Returns:
            dict {str: pd.Series}:
                dictionary which contains all calculated features.
                Ready to plug into pd.DataFrame,
                but still allows fast calculation of ratio features
        """
        # TODO: if the queries are empty, take data subset with segmentations only (to save memory)
        queries += self._segmentations_to_queries(data, segmentations)
        features = {}
        feature = None

        # for computational performance - instead of itertools - nested for loop
        self.logger.debug(
            ("_get_simple_pandas_feature_for_variable: " "data memory size - %.2f"),
            sum(data.memory_usage()) / (1024 ** 2),
        )
        # for computational performance - instead of itertools - nested for loop
        for query in queries:
            # we create first subset, which stays in the memory - no recalculation
            # _handle_query returns pd.DataFrame where relevant columns only are taken afterwards
            subset_query = self._handle_query(data, query)[
                [self.index, self.order, variable]
            ]
            self.logger.debug(
                (
                    "_get_simple_pandas_feature_for_variable: "
                    "subset_query memory size - %.2f"
                ),
                sum(subset_query.memory_usage()) / (1024 ** 2),
            )
            for time_range in time_ranges:
                # second sub-subset
                subset_time_range = self._get_time_range_subset(
                    subset_query, time_range
                )[[self.index, variable]]
                # on which all functions are applied
                for function in functions:
                    # The `feature` is pd.Series with index (!!!)
                    # The index comes from our self.index
                    # - on which whole data are grouped by

                    # The index ensures
                    # that afterwards the dataframe is cast
                    # for the correct customer.
                    try:
                        grouped = subset_time_range.groupby(self.index)[variable]
                    except ValueError as err:
                        # bugfix
                        if "1" in str(err):
                            raise ValueError(
                                "Variable cannot be used for given aggregation"
                            ) from err
                        raise

                    feature_name = self._get_feature_name(
                        variable, function, time_range, query
                    )
                    feature = self._calculate_function(
                        grouped=grouped,
                        variable=variable,
                        feature_name=feature_name,
                        function=function,
                    )
                    if feature is not None:
                        features[feature_name] = feature
        self.logger.debug(f"Pandas features for {variable} done")

        return features

    def get_simple_features_pandas(
        self, data, simple_settings, max_nan_share
    ) -> Dict[str, pd.Series]:
        """
        Obtain dictionary of features for all simple variables
        Simple settings are iterated over variable, and then final dictionary is created
        This dictionary contains all features for given simple settings

        Args:
            data (pd.DataFrame): data to be aggregated
            simple_settings (dict): configuration of simple variable.
            max_nan_share ():

        Returns:
            dict {str: pd.Series}: dictionary of features
        """
        self.logger.debug(f"Calculating simple features")
        if self.order == DEFAULT_TIME_ORDER:
            data, _ = self.order_assigner.transform(
                data=data, config=self.config.config
            )
        features_pandas: Dict[str, pd.Series] = {}

        # simple settings is an iterator.
        # from this iterator, we get single set of config to be applied for given variable
        for settings_for_variable in simple_settings:
            features_pandas = {
                **features_pandas,
                **self._get_simple_pandas_feature_for_variable(
                    data=data, **settings_for_variable
                ),
            }

        self.logger.info(
            f"Number of simple features {'(before pruning)' if max_nan_share else ''}: {len(features_pandas)}"
        )
        series_features, dataframe_features = self._prune_features(
            features_pandas, data[self.index].unique(), max_nan_share
        )
        return series_features, dataframe_features

    def _prune_features(self, features, index, max_nan_share):
        """Function to omit variables which are not populated enough.
            Also sorts variables by their type - pd.Series & pd.DataFrames

        Args:
            features (dict[str: pd.Series or pd.DataFrame]): calculated features,
            stored in dictionary with feature names as their key
            index (str): name of the index column
            max_nan_share (float): maximal share of rows with NaN

        Returns:
            tuple[dict[str:pd.Series], dict[str:pd.DataFrame]]: filtered variables.
            DataFrame variables are in one dictionar, Series variables are in second dictionary
        """

        dataframe_features = {}
        to_omit = []
        # this is hell of an ugly hack, isnt it?
        # why the fuck am I  checking the instance of the features here?
        # why am I doing the renaming here as well?
        # AAA
        for feature_name, feature_value in features.items():
            if isinstance(feature_value, pd.DataFrame):
                dataframe_features[feature_name] = feature_value.rename(
                    lambda colname: f"{feature_name}_{colname}", axis=1
                )
                to_omit += [feature_name]

            else:
                if max_nan_share:
                    nan_share = 1 - (len(feature_value) / len(index))
                    self.logger.debug(f"NaN share: {nan_share: .3F}")
                    if nan_share > max_nan_share:
                        to_omit += [feature_name]
                        self.logger.debug(
                            f"Removing {feature_name} ({nan_share: .3F}) using max_nan_share"
                        )

        for feature in to_omit:
            features.pop(feature)
        if max_nan_share:
            self.logger.info(f"# of omitted features: {len(to_omit)}")
            self.logger.info(
                f"Number of simple features (after pruning): {len(features) + len(list(dataframe_features))}"
            )
        return features, dataframe_features

    def dataframe_simple(self, data, max_nan_share) -> pd.DataFrame:
        """Function to calculate all simple features from given data, config & max_nan_share

        Returns:
            pd.DataFrame: all simple variables
        """
        series_features, dataframe_features = self.get_simple_features_pandas(
            data=data,
            simple_settings=self.config.yield_simple_variable_settings(),
            max_nan_share=max_nan_share,
        )

        # self.variable_list_pandas[SIMPLE_CONFIG_KEY] = [key for key in features_all]

        # 1) I think that up to this point,
        #    lazy evaluation is applied and nothing is computed yet.
        #    All calculations are made when casting to DataFrame.
        # 2) the data are reindexed so we are sure
        #    that we have all original customers in our new data

        result = pd.DataFrame(series_features).reindex(index=data[self.index].unique())
        result = pd.concat(
            [result] + [df for df in dataframe_features.values()], axis=1
        )

        result = self._simple_nan_imputing(result)
        return result


class RatioAggregationPandas(SimpleAggregationPandas):
    """
    Ratio calculation of pandas features
    """

    def get_ratio_features_pandas(self, data, ratio_settings, max_nan_share) -> Dict:
        """
        Calculate ratio features in pandas and return dictionary

        Args:
            data (pd.DataFrame): data to be processed using prespecified config
            ratio_settings (): iterator which returns settings
                for all specified combinations
                of numerator/denominator
            max_nan_share ():

        Returns:
            dict {str: pd.Series}: dict of features
        """

        # TODO: rozdelit tuto funkci
        ratio_features = {}

        # from iterator we get settings
        # for given numerator/denominator combination
        for settings in ratio_settings:
            # as it would be really long to name all arguments,
            # we break them down in this function
            (
                numerator_settings,
                denominator_settings,
            ) = self.config.get_numerator_denominator_settings(*settings)

            existing_ratio_features = self._get_existing_ratio_variables(
                data, *settings
            )
            self.logger.info(
                (
                    f"Calculating {numerator_settings[0]['variable']} /"
                    f" {denominator_settings[0]['variable']}"
                )
            )
            self.logger.debug(
                f"Number of existing variables: {len(existing_ratio_features)}"
            )
            # and numerator_settings in this case is list of single dictionary
            # thus iterable with single item
            # and thus we can use it as simple calculation
            numerators, _ = self.get_simple_features_pandas(
                data=data,
                simple_settings=numerator_settings,
                max_nan_share=max_nan_share,
            )
            # similarily for denominator
            denominators, _ = self.get_simple_features_pandas(
                data=data,
                simple_settings=denominator_settings,
                max_nan_share=max_nan_share,
            )

            # and now we have all building blocks
            # from which final features will be calculated
            # we iterate over all of them,
            # and if the feature exists in `existing_ratio_features`,
            # we create it and add it to final dictionary
            # `ratio_features`
            for numerator_name, numerator_values in numerators.items():
                for denominator_name, denominator_values in denominators.items():

                    name = self._get_ratio_feature_name(
                        numerator_name, denominator_name
                    )
                    if name not in existing_ratio_features:
                        continue
                    feature = numerator_values / denominator_values
                    feature = self._ratio_nan_inf_imputting(feature)
                    ratio_features[name] = feature
            self.logger.debug(
                f"IN PROGRESS: Number of ratio features: {len(ratio_features)}"
            )
        self.variable_list_pandas[RATIO_CONFIG_KEY] = [key for key in ratio_features]
        self.logger.info(f"FINAL: Number of ratio features: {len(ratio_features)}")
        return ratio_features

    def _ratio_nan_inf_imputting(self, feature):
        if self.inf_value is not None:

            positive_inf = self.inf_value
            if isinstance(self.inf_value, str):
                negative_inf = self.inf_value
            else:
                negative_inf = -self.inf_value
            feature = feature.replace([np.inf, float("inf")], positive_inf)
            feature = feature.replace([-np.inf, -float("inf")], negative_inf)
        feature = self._simple_nan_imputing(feature)

        return feature

    def dataframe_ratio(self, data, max_nan_share) -> pd.DataFrame:
        """
        Calculate ratio features in pandas and return dictionary.
        Reindex the final dataset with original index,
        so the final data have all customers in the original dataset

        Args:
            data (pd.DataFrame): data to be processed using prespecified config
            max_nan_share ():

        Returns:
            pd.DataFrame: dataframe with all calculated features
        """

        return pd.DataFrame(
            self.get_ratio_features_pandas(
                data=data,
                ratio_settings=self.config.yield_ratio_variable_settings(),
                max_nan_share=max_nan_share,
            )
        ).reindex(index=data[self.index].unique())


class TimeSincePandas(MetaAggregation):
    @staticmethod
    def _months_between(d1, d2):
        """
        Method which calculates number of months between two columns
        consistent with Oracle SQL

        """
        # same day in month = _months_between = 0 (d1.dt.day<=d2.dt.day);
        # count from 1 (the +1)
        return (
            (d1.dt.year - d2.dt.year) * 12
            + d1.dt.month
            - d2.dt.month
            - (d1.dt.day <= d2.dt.day) * 1
            + 1
        )

    def _get_time_since_features_pandas(self, data):
        features_pandas = {}
        for time_since_settings in self.config.yield_time_since_variable_settings():
            features_pandas = {
                **features_pandas,
                **self._get_time_since_features_for_variable(
                    data, **time_since_settings
                ),
            }
        self.logger.info(f"Number of Time Since features: {len(features_pandas)}")
        return features_pandas

    def _get_time_since_features_for_variable(
        self, data, variable, time_since_functions, segmentations, queries, units
    ):
        queries += self._segmentations_to_queries(data, segmentations)
        features = {}

        from_dates = data.groupby(self.index)[self.target_time].max()
        for query in queries:
            subset_query = self._handle_query(data, query)
            for unit in units:
                unit = self.check_granularity(unit)
                for function in time_since_functions:
                    grouped = subset_query.groupby(self.index)[variable]
                    time_since_dates = getattr(grouped, function)()
                    if unit != ORDER_GRANULARITY:
                        feature = self._column_difference(
                            from_dates, time_since_dates, unit
                        )
                    else:
                        feature = grouped.rank(ascending=False, method="first")
                    feature_name = self._get_time_since_feature_name(
                        variable, function, unit, query
                    )
                    if self.nan_value is not None:
                        feature = feature.fillna(self.nan_value)
                    features[feature_name] = feature

        return features

    def dataframe_time_since(self, data) -> pd.DataFrame:
        features_pandas = self._get_time_since_features_pandas(data=data)
        self.variable_list_pandas[TIME_SINCE_CONFIG_KEY] = [
            key for key in features_pandas
        ]
        return pd.DataFrame(features_pandas).reindex(index=data[self.index].unique())

    def _column_difference(self, date_from, date_to, unit):
        try:
            if unit != MONTH_GRANULARITY:
                result_column = (date_from - date_to) / np.timedelta64(1, unit)
            else:

                result_column = self._months_between(date_from, date_to)
        except TypeError as err:
            raise TypeError(
                f"""
            You probably forgot to cast your datetimes. Please cast them to the correct format.
            Original error:
            {err}
            """
            ).with_traceback(None)

        return result_column


class SimpleAggregationSQL(MetaAggregation):
    """
    Class for sql generation for simple aggregations

    """

    def _create_simple_sql_feature(self, variable, function, time_range, query) -> str:
        """
        SQL generation for single feature

        Args:
            variable (): name of the original variable from which the feature will be created
            function (): function to be applied as aggregation
            time_range (): time period from - to to agggregate the individual data
            query (): string which subselects the data. Used for segmentations

        Returns:
            str:
        """
        if function == "mean":
            sql_function = "avg"
        else:
            sql_function = function

        outer_time_range_sql = time_range[1]
        if time_range[1] in (float("inf"), np.inf):
            if self.inf_value:
                outer_time_range_sql = self.inf_value
            else:
                outer_time_range_sql = 999_999

        sql_variable = variable
        if self.nan_value:
            sql_variable = f"nvl({variable}, {self.nan_value})"

        sql_query = ""
        if query:
            sql_query = f"and {self._query_to_sql_query(query)}"

        sql_tmp = f"""
            {sql_function}(case when {self.order} >= {time_range[0]}
                         and {self.order} <= {outer_time_range_sql}
                         {sql_query}
                        then {sql_variable}
                        end)
            """
        return sql_tmp

    def _get_simple_sql_feature_for_variable(
        self, data, variable, functions, time_ranges, queries, segmentations
    ) -> Dict:
        """
        Obtain sql features for single variable in a dictionary

        Args:
            data (pd.DataFrame): data from which the aggreations are calculated
            variable (str): name of the original variable from which the feature will be created
            functions (list of str): functions to be applied as aggregation
            time_ranges (list of tuples of ints): time periods from - to to agggregate the individual data
            queries (list of str): strings which subselects the data. Used for segmentations
            segmentations (list of str): name of variabls by which subselections should be created

        Returns:
            dict: name of feature: sql string of the feature
        """
        features_sql = {}
        queries += self._segmentations_to_queries(data, segmentations)
        for settings in itertools.product(functions, time_ranges, queries):
            feature_name = self._get_feature_name(variable, *settings)
            features_sql[feature_name] = self._create_simple_sql_feature(
                variable, *settings
            )
        self.logger.debug(f"SQL for {variable} done")

        return features_sql

    def get_simple_features_sql(self, data, simple_settings) -> Dict:
        """Obtain sql features for all variables, in a dictionary

        This function is used as simple feature generator
        and also inside ratio feature generation inside, for
        code reusability.

        Args:
            data (pd.DataFrame): original data
            simple_settings (iterable): iterable producing settings for single variables

        Returns:
            dict: feature name: sql feature string
        """
        self.logger.info("Creating features in sql")
        features_sql: Dict[str, str] = {}
        for settings_for_variable in simple_settings:
            features_sql = {
                **features_sql,
                **self._get_simple_sql_feature_for_variable(
                    data=data, **settings_for_variable
                ),
            }

        self.logger.debug(f"Number of simple sql features: {len(features_sql)}")
        return features_sql

    def sql_simple(self, data) -> str:
        """
        Generate sql for simple aggregation

        Args:
            data (pd.DataFrame): data from which given string should be calculated

        Returns:
            str: sql with all simple aggregations calculated
        """
        features_sql = self.get_simple_features_sql(
            data=data, simple_settings=self.config.yield_simple_variable_settings()
        )
        return features_sql


class RatioAggregationSQL(SimpleAggregationSQL):
    def sql_ratio(self, data) -> str:
        """
        Create SQL ratio features from given config and data.

        Args:
            data (pd.DataFrame):

        Returns:
            str:
        """
        ratio_features = {}
        for settings in self.config.yield_ratio_variable_settings():
            (
                numerator_settings,
                denominator_settings,
            ) = self.config.get_numerator_denominator_settings(*settings)
            existing_ratio_features = self._get_existing_ratio_variables(
                data, *settings
            )
            numerator_sqls = self.get_simple_features_sql(
                data=data, simple_settings=numerator_settings
            )
            denominator_sqls = self.get_simple_features_sql(
                data=data, simple_settings=denominator_settings
            )

            for numerator_name, numerator_sql in numerator_sqls.items():
                for denominator_name, denominator_sql in denominator_sqls.items():
                    name = self._get_ratio_feature_name(
                        numerator_name, denominator_name
                    )
                    if name not in existing_ratio_features:
                        continue

                    raw_sql = f"{numerator_sql} / {denominator_sql}"

                    # boilerplate sql code for managing inf values
                    if self.inf_value:
                        sql_inf = dedent(
                            f"""
                            case 
                                when 
                                    {denominator_sql} = 0 
                                and 
                                    {numerator_sql} > 0 
                                then {self.inf_value}
                                when 
                                    {denominator_sql} = 0 
                                and 
                                    {numerator_sql} < 0 
                                then 
                                    -{self.inf_value}
                                else {raw_sql}
                            end

                        """
                        )
                    else:
                        sql_inf = raw_sql

                    # boilerplate sql code for managing nan values
                    if self.nan_value:
                        sql = f"nvl({sql_inf}, {self.nan_value})"
                    else:
                        sql = sql_inf

                    ratio_features[name] = sql

        self.variable_list_sql[RATIO_CONFIG_KEY] = [key for key in ratio_features]
        self.logger.info(f"Number of created ratio variables: {len(ratio_features)}")
        return ratio_features


class TimeSinceSQL(MetaAggregation):
    """
    Class for generation of TimeSince SQLs

    Methods:
        sql_time_since: processes given data and time_since config and generates sql

    """

    def _get_time_since_sql_feature_for_variable(
        self, data, variable, time_since_functions, queries, segmentations, units
    ) -> Dict:
        features_sql = {}
        queries += self._segmentations_to_queries(data, segmentations)
        units = [self.check_granularity(unit) for unit in units]
        for settings in itertools.product(time_since_functions, units, queries):
            feature_name = self._get_time_since_feature_name(variable, *settings)
            features_sql[feature_name] = self._create_time_since_sql_feature(
                variable, feature_name, *settings
            )
        self.logger.debug(f"SQL for {variable} done")

        return features_sql

    def _get_time_since_sql(self, data, time_since_settings):
        self.logger.info("Creating features in sql")
        features_sql: Dict[str, str] = {}
        for settings_for_variable in time_since_settings:
            features_sql = {
                **features_sql,
                **self._get_time_since_sql_feature_for_variable(
                    data=data, **settings_for_variable
                ),
            }

        self.logger.debug(f"Number of simple sql features: {len(features_sql)}")
        return features_sql

    def _order_granularity_sql(
        self, variable, feature_name, time_since_function, unit, query
    ):
        sql_fun = "desc" if time_since_function == "first" else "asc"
        sql = f"""

            -- {feature_name}
            select * 
              from (select t.*, 
                           row_number() 
                                over (
                                    partition by {self.index}
                                        order by {variable}
                                        {sql_fun}
                                    ) as {feature_name}
                      where 1=1
               from _TABLENAME_ t
              where 1=1
             {f"and {query}" if query else ""}
            """

        return sql

    def _time_granularity_sql(
        self, variable, feature_name, time_since_function, unit, query
    ):
        sql_fun = "max" if time_since_function == "first" else "min"
        time_variable_sql = {
            YEAR_GRANULARITY: f"ceil(_months_between({sql_fun}({variable}), {self.target_time}) / 12)",
            MONTH_GRANULARITY: f"ceil(_months_between({sql_fun}({variable}), {self.target_time}))",
            DAY_GRANULARITY: f"ceil({sql_fun}({variable}) - {self.target_time})",
            HOUR_GRANULARITY: f"ceil(({sql_fun}({variable}) - {self.target_time}) * 24)",
            MINUTE_GRANULARITY: f"ceil(({sql_fun}({variable}) - {self.target_time}) * 1440)",
            SECONDS_GRANULARITY: f"ceil(({sql_fun}({variable}) - {self.target_time}) * 86400)",
        }

        sql_query = f"and {self._query_to_sql_query(query)}" if query else ""
        sql = f"""

            -- {feature_name}
            select {time_variable_sql[unit]} as {feature_name},
              from _TABLENAME_ t
             group by {self.index}
             where 1=1
             {sql_query}

            """
        return sql

    def _create_time_since_sql_feature(self, *args):
        if args[-1] == ORDER_GRANULARITY:
            return self._order_granularity_sql(*args)
        return self._time_granularity_sql(*args)

    def sql_time_since(self, data) -> str:
        features_sql = self._get_time_since_sql(
            data=data,
            time_since_settings=self.config.yield_time_since_variable_settings(),
        )

        return features_sql
