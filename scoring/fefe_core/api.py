"""
    Key concepts:
        - consistency:
            - use black autoformatter for formatting
            - naming should be consistent
                - use suffixes for sql and dataframe
        - modularity: every logical part should have its own function
        - least interdependence:
            - classes should not be dependent on each other.
            - function chains should be like trees
        - fail fast:
            - raise error as soon as possible
        - as shallow as possible
            -  do not nest the functions too deep.
        - be clever, but don't try to be more clever than the user:
            make the tool as easy to use as possible,
            but always do only what the user specifically asks for.

    Inheritance:
                           /->------->---> TimeSincePandas >--------->---->\ 
                          /                                                 \ 
                         /-SimpleAggregationPandas -> RatioAggregationPandas-\ 
     MetaAggregation   -<                                                     >-   FeatureEngineeringAPI
                         \--- SimpleAggregationSQL -> RatioAggregationSQL ---/
                          \                                                 /
                           \->------->--->  TimeSinceSQL   >--------->---->/


"""

from typing import Dict

from .jupyter_api.aggregators import (
    RatioAggregationPandas,
    RatioAggregationSQL,
    TimeSincePandas,
    TimeSinceSQL,
)
import pandas as pd


class FeatureEngineeringAPI(
    RatioAggregationSQL, RatioAggregationPandas, TimeSincePandas, TimeSinceSQL
):
    """
    Calculate your features easily.

    Methods:
        sql(data): returns sql for given config
        dataframe(data): returns dataframe for given config
    
    Args:
        config: output from FEFE - desired variables
        raise_on_error: should the functions raise errors, or just log them?
        logger: logging instance (for logger inheritance, if needed). Defaults to None.
        logger_kwargs:
            -- key-word arguments passed to logger
            -- possible arguments:
                -- 'log_name' - name of the logger in log
                -- 'log_folder' - address of folder where logs are stored.
                                 If not set, folder 'log' is created
                -- 'log_filename' - name of the logging file
                -- 'log_level' - level of logging - 10 - DEBUG ... to 50 - CRITICAL
                -- 'log_format' - formatting of log messages
            Defaults to None.
    """

    def _process_simple_ratio_sql(self, sql_features, table_name):

        result_sql = {
            name: f"""

            -- {name}
            select
            {self.index},
            {feature} as {name}
            from {table_name}
            group by {self.index}
            order by {self.index};
            """
            for name, feature in sql_features.items()
        }
        return result_sql

    def sql(
        self, data: pd.DataFrame, feature_subset=None, table_name=None
    ) -> Dict[str, str]:
        """
        Get SQL string for given instance config.

        Arguments:
            data - dataset for which given sql is created.
                 - internally the data are used only
                   for calculating unique values for segmentations
        """
        if table_name is None:
            table_name = "_TABLENAME_"

        _, time_order_sql = self.order_assigner.transform(config=self.config.config)
        self.logger.info(
            f"""
            Creating SQL for SIMPLE features
            {'_'*25}
            """
        )

        sql_simple = self.sql_simple(data)
        self.logger.info(
            f"""
            Creating SQL for RATIO features
            {'_'*25}
            """
        )
        sql_ratio = self.sql_ratio(data)
        self.logger.info(
            f"""
            Creating SQL for TIME SINCE features
            {'_'*25}
            """
        )
        sql_time_since = self.sql_time_since(data)

        features_sql = {
            "TIME_ORDER": time_order_sql,
            **self._process_simple_ratio_sql(sql_simple, table_name=table_name),
            **self._process_simple_ratio_sql(sql_ratio, table_name=table_name),
            **sql_time_since,
        }

        if feature_subset:
            not_in_data = set(feature_subset) - set(features_sql)
            if not_in_data:
                self.logger.warning(
                    f"Following variables are not in feature data: {not_in_data}"
                )
            features = list(set(feature_subset).intersection(set(features_sql)))
            final_features = {
                name: sql_feature
                for name, sql_feature in features_sql.items()
                if name in features
            }
        else:
            final_features = features_sql

        result = "".join(final_features.values())
        return result

    def dataframe(self, data: pd.DataFrame, max_nan_share: float = 0.0) -> pd.DataFrame:
        """
        Transform and aggregate input data
        from transaction level to application level

        Args:
            data: dataset to be aggregated
            max_nan_share: share of the data that can be nans
                -> nanshare 0.95 = maximum of 95% of the data can be NaN 
                -> only 5% of the observations do have value

        """
        self.logger.info(
            f"""
            Calculating SIMPLE features
            {'_'*25}
            """
        )
        df_simple = self.dataframe_simple(data, max_nan_share)

        self.logger.info(
            f"""
            Calculating RATIO features
            {'_'*25}
            """
        )
        df_ratio = self.dataframe_ratio(data, max_nan_share)
        self.logger.info(
            f"""
            Calculating TIME_SINCE features
            {'_'*25}
            """
        )
        df_time_since = self.dataframe_time_since(data)

        final_data = pd.concat([df_simple, df_ratio, df_time_since], axis=1)

        if not final_data.index.name:
            final_data.index.name = self.index
        self.logger.info(f"FINAL NUMBER OF FEATURES: {final_data.shape[1]}")
        return final_data
