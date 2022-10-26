import ast

from typing import List, Tuple, Dict

from ..front_end.config_handler import (  # pylint: disable=relative-beyond-top-level
    FUNCTION_CONFIG_KEY,
    META_CONFIG_KEY,
    QUERY_CONFIG_KEY,
    RATIO_CONFIG_KEY,
    SEGMENTATION_CONFIG_KEY,
    SIMPLE_CONFIG_KEY,
    TIME_RANGE_CONFIG_KEY,
    TIME_SINCE_CONFIG_KEY,
    TIME_SINCE_FROM_CONFIG_KEY,
    TIME_SINCE_UNIT_CONFIG_KEY,
    META_SUBKEYS,
)
from ..utils.logger import create_logger  # pylint: disable=relative-beyond-top-level


class ConfigProcessor:
    """
    aggregate all config manipulations in a single place.
    in case config would change - all changes would need to be made here.
    """

    def __init__(self, config, raise_on_error, logger=None, logger_kwargs=None):
        if logger_kwargs:
            self.logger = logger if logger else create_logger(**logger_kwargs)
        else:
            self.logger = logger if logger else create_logger()

        self.raise_on_error = raise_on_error
        self.config = config

    def _get_config_key(self, aggregations, config_key, default):
        """
        Provides error handling for cases when given config part is not available in the config

        Args:
            aggregations (dict): the whole config
            config_key (str): name of the key which should be found in the aggregation
            default (str): if raise_on_error is False, what should be replacing given part

        Raises:
            KeyError: if raise_on_error is True

        Returns:
            dict: specified config part
        """
        try:
            return aggregations[config_key]
        except KeyError as err:
            message = f"Missing config part: {config_key}"
            if self.raise_on_error:
                self.logger.error(message)
                raise KeyError(message).with_traceback(err)
            else:
                self.logger.warning(f"{message}. Using default value: {default}")
                return default

    def get_metadata(self):
        """
        Return individual metadata values from config/meta

        Returns:
            Tuple[str] --  index, transaction_time, target_time, time_order, granularity
                           in this order
        """
        return (self.config[META_CONFIG_KEY][key] for key in META_SUBKEYS)

    def _get_clean_time_ranges(self, aggregations, default) -> List[Tuple]:
        """
        Clean time ranges.
        Cleans both simple and ratio time ranges.

        Args:
            aggregations (dict): aggregations specififed for given variable (simple) or pair of variables (ratio)
            default (list of str or str): which value should be processed, when given variable is not present in the original config

        Returns:
            list of tuple: list of cleaned time ranges.
                If simple: [(a, b), ...]
                If ratio: [((a,b), (c,d)), ...] where b, d can be `float("inf")`
        """

        def clean_time_ranges(time_range):
            """
            When the time range is made from integers, `ast.literal_eval` is used.
            When infinity is present, we have to use eval.

            Args:
                time_range (str): time range like:
                    - "(a, b)"
                    - "((a,b), (c,d))"
                    where b, d can be inf, indicating `float("inf")`

            Returns:
                tuple of ints or tuple of tuples of ints: same as before, however different format
            """

            try:
                return ast.literal_eval(time_range)
            except ValueError:
                if "inf" in time_range:
                    inf = float("inf")  # pylint: disable=unused-variable

                    # i am so sorry, code.
                    # man, I really did not want to use this eval here.
                    # however in order to be able to parse infinities,
                    # which are represented as 'inf'
                    # ast.literal_eval does not work for that.
                    # and i am lazy to parse the string exactly
                    return eval(time_range)  # pylint: disable=eval-used
                raise

        # it is possible that this key does not exist in the config
        time_ranges = self._get_config_key(aggregations, TIME_RANGE_CONFIG_KEY, default)
        # and it is possible that if the key exist, it will be just empty string
        if not time_ranges:
            time_ranges = default
        # so we normalize output
        return [clean_time_ranges(time_range) for time_range in time_ranges]

    def _get_clean_functions(self, aggregations, default):

        # principle is same as in _get_clean_time_ranges
        functions = self._get_config_key(aggregations, FUNCTION_CONFIG_KEY, default)
        if not functions:
            functions = default
        return [ast.literal_eval(val) if "(" in val else val for val in functions]

    def _get_clean_segmentations(self, aggregations, default):

        # principle is same as in _get_clean_time_ranges
        segmentations = self._get_config_key(
            aggregations, SEGMENTATION_CONFIG_KEY, default
        )
        if not segmentations:
            segmentations = default
        return [var if var != "None" else None for var in segmentations]

    def _get_clean_queries(self, aggregations, default):
        queries = self._get_config_key(aggregations, QUERY_CONFIG_KEY, default)
        return [agg for agg in queries.split(";") if agg]

    def get_time_since_aggregations(self, variable):
        aggregations = self.config[TIME_SINCE_CONFIG_KEY][variable]
        time_since_functions = self._get_config_key(
            aggregations, TIME_SINCE_FROM_CONFIG_KEY, ["first"]
        )
        units = self._get_config_key(
            aggregations, TIME_SINCE_UNIT_CONFIG_KEY, ["days"]
        )
        segmentations = self._get_clean_segmentations(aggregations, default=["None"])
        queries = self._get_clean_queries(aggregations, default="")

        return dict(
            variable=variable,
            time_since_functions=time_since_functions,
            units=units,
            segmentations=segmentations,
            queries=queries,
        )

    def get_simple_variable_aggregations(
        self, variable
    ):  # pylint: disable=missing-docstring
        """
        Useful as it returns variables in pre-specified order
        (unlike in original config - can have different order)

        Args:
            variable (str): variable name

        Returns:
            dict(str, list(str), list(str), list(None or str), list):
                variable, time_ranges, functions, segmentations, queries
        """
        aggregations = self.config[SIMPLE_CONFIG_KEY][variable]

        time_ranges = self._get_clean_time_ranges(aggregations, default=["(0, inf)"])
        functions = self._get_clean_functions(aggregations, default=["mean"])
        segmentations = self._get_clean_segmentations(aggregations, default=["None"])
        queries = self._get_clean_queries(aggregations, default="")

        return dict(
            variable=variable,
            time_ranges=time_ranges,
            functions=functions,
            segmentations=segmentations,
            queries=queries,
        )

    def get_ratio_variable_aggregations(self, numerator, denominator):
        """
        Get specified aggregations for pair of numerator and denominator

        Args:
            numerator (str): numerator variable
            denominator (str): denominator variable

        Returns:
            tuple (str, str, List[Tuple], List[str], List[str], List[str]):
                numerator, denominator, time_ranges, functions, segmentations, queries
        """
        aggregations = self.config[RATIO_CONFIG_KEY][numerator][denominator]

        time_ranges = self._get_clean_time_ranges(
            aggregations, default=["((0, inf), (0, inf))"]
        )

        functions = self._get_clean_functions(aggregations, default=["(mean, mean)"])

        segmentations = self._get_clean_segmentations(aggregations, default=["None"])
        queries = self._get_clean_queries(aggregations, default="")

        return (numerator, denominator, time_ranges, functions, segmentations, queries)

    def get_numerator_denominator_settings(
        self, numerator, denominator, time_ranges, functions, segmentations, queries
    ):

        time_ranges_numerator = [tr[0] for tr in time_ranges]
        time_ranges_denominator = [tr[1] for tr in time_ranges]

        functions_numerator = [fun[0] for fun in functions]
        functions_denominator = [fun[1] for fun in functions]

        numerator_settings = [
            dict(
                variable=numerator,
                time_ranges=time_ranges_numerator,
                functions=functions_numerator,
                segmentations=segmentations,
                queries=queries,
            )
        ]
        denominator_settings = [
            dict(
                variable=denominator,
                time_ranges=time_ranges_denominator,
                functions=functions_denominator,
                segmentations=segmentations,
                queries=queries,
            )
        ]

        return (numerator_settings, denominator_settings)

    # endpoints
    def yield_time_since_variable_settings(self):
        """
        Config endpoint for time since feature aggregations.

        """
        for variable in self.config[TIME_SINCE_CONFIG_KEY].keys():
            yield self.get_time_since_aggregations(variable)

    def yield_simple_variable_settings(self):
        """
        Config endpoint for simple features
        """
        for variable in self.config[SIMPLE_CONFIG_KEY].keys():
            yield self.get_simple_variable_aggregations(variable)

    def yield_ratio_variable_settings(self):
        """
        Config endpoint for ratio features.
        Yields ratio variable aggregations for all numerator/denominator combinations in config
        """
        for numerator in self.config[RATIO_CONFIG_KEY].keys():
            for denominator in self.config[RATIO_CONFIG_KEY][numerator].keys():
                yield self.get_ratio_variable_aggregations(numerator, denominator)
