import itertools

# import dask.dataframe as ddf
import numpy as np
import pandas as pd

# from dask import compute

from ..front_end.config_handler import META_CONFIG_KEY

# https://docs.scipy.org/doc/numpy/reference/arrays.datetime.html#datetime-units
YEAR_GRANULARITY = "Y"
MONTH_GRANULARITY = "M"
DAY_GRANULARITY = "D"
HOUR_GRANULARITY = "h"
MINUTE_GRANULARITY = "m"
SECONDS_GRANULARITY = "s"
ORDER_GRANULARITY = "order"

# first alias is used for naming variables
GRANULARITY_ALIASES = {
    YEAR_GRANULARITY: ["years", "y", "year", "yr"],
    MONTH_GRANULARITY: ["months", "mt", "month", "mth", "months"],
    DAY_GRANULARITY: ["days", "d", "day"],
    HOUR_GRANULARITY: ["hours", "h", "hour", "hr"],
    MINUTE_GRANULARITY: ["minutes", "mi", "minutes", "min", "mins", "minute"],
    SECONDS_GRANULARITY: ["seconds", "s", "secs", "sec", "second"],
    ORDER_GRANULARITY: [ORDER_GRANULARITY],
}


class TimeOrder:
    """
    Class to assign granularity to the original dataset.
    
    """

    def __init__(
        self,
        index=None,
        transaction_time=None,
        target_time=None,
        future_filter=False,
        n_cores=1,
        time_format="%Y-%m-%d",
        cast_datetimes=False,
        time_order_colname=None,
    ):
        self.index = index
        self.transaction_time = transaction_time
        self.target_time = target_time

        self._future_filter = future_filter
        self._n_cores = n_cores
        self._time_format = time_format
        self._time_order_colname = (
            time_order_colname if time_order_colname else "TIME_ORDER"
        )

        self.cast_datetimes = cast_datetimes

    def _populate_init_from_config(self, meta):
        self.index = meta["index"]
        self.transaction_time = meta["transaction_time"]
        self.target_time = meta["target_time"]

    def _process_config(self, config):
        if config is None:
            if None in (self.index, self.transaction_time, self.target_time):
                raise ValueError(
                    "When config is not specified, "
                    "you have to populate 'index', 'transaction_time', 'target_time' "
                    "in class initialisation"
                )
        else:
            self._populate_init_from_config(config["meta"])

    def transform(
        self,
        data=None,
        config=None,
        history_length=None,
        granularity=None,
        return_sql=True,
        partition=None,
    ):
        self._process_config(config)
        if granularity:
            granularity = self.check_granularity(granularity)
            self._check_order_granularity(granularity, partition)
        else:
            granularity = self.check_granularity(config[META_CONFIG_KEY]["granularity"])

        result_data, sql_result = None, None

        if data is not None:
            self._data_check(data)
            result_data = data.copy()

            if self.cast_datetimes:
                result_data = self._cast_datetimes(result_data)

            result_data = self._time_order_data_transform(
                result_data, granularity, partition
            )
            result_data = self._time_order_data_filter(result_data, history_length)

        if return_sql:
            sql_result = self._time_order_sql_transform(
                granularity, history_length, partition
            )

        return result_data, sql_result

    def _data_check(self, data):
        for column in [self.index, self.transaction_time, self.target_time]:
            if column not in data.columns:
                raise ValueError(f"Missing column {column} in the data.")

    def _cast_datetimes(self, data):
        for time_column in [self.transaction_time, self.target_time]:
            data[time_column] = pd.to_datetime(data[time_column])
        return data

    @staticmethod
    def _check_order_granularity(granularity, partition):
        if granularity == ORDER_GRANULARITY:
            if not partition:
                raise ValueError(
                    "When granularity is set to 'order', argument 'partition' has to be set"
                )

    def _time_order_data_filter(self, data, history_length):
        if self._future_filter:
            past_data_selector = data[self.target_time] > data[self.transaction_time]
            data = data[past_data_selector]

        if history_length:
            max_history_selector = data[self._time_order_colname] > history_length
            data = data[max_history_selector]

        return data

    def _time_order_data_transform(self, data, granularity, partition):
        if granularity != ORDER_GRANULARITY:
            try:
                if granularity != MONTH_GRANULARITY:
                    result_column = (
                        data[self.target_time] - data[self.transaction_time]
                    ) / np.timedelta64(1, granularity)
                else:

                    def months_between(d1, d2):
                        # same day in month = months_between = 0 (d1.dt.day<=d2.dt.day); count from 1 (the +1)
                        return (
                            (d1.dt.year - d2.dt.year) * 12
                            + d1.dt.month
                            - d2.dt.month
                            - (d1.dt.day <= d2.dt.day) * 1
                            + 1
                        )

                    result_column = months_between(
                        data[self.target_time], data[self.transaction_time]
                    )
            except TypeError as err:
                raise TypeError(
                    f"""
                You probably forgot to cast your datetimes. Please cast them to the correct format.
                Original error:
                {err}
                """
                ).with_traceback(None)

        else:
            result_column = data.groupby(self.index)[self.transaction_time].rank(
                ascending=False, method="first"
            )
        data[self._time_order_colname] = result_column
        return data

    def _order_granularity_sql(self, partition, history_length):
        sql = f"""
            select * 
              from (select t.*, 
                           row_number() 
                                over (
                                    partition by {self.index}
                                        order by {self.transaction_time}
                                        desc
                                    ) as {self._time_order_colname}
                      where 1=1
                     {f"and {self.transaction_time} < {self.target_time}" if self._future_filter else ''})
               from _TABLENAME_ t
              where 1=1
             {f"and {self._time_order_colname} < {history_length}" if history_length else ''}
            """

        return sql

    def _time_granularity_sql(self, granularity, history_length):
        time_variable_sql = {
            YEAR_GRANULARITY: f"ceil(months_between({self.target_time}, {self.transaction_time}) / 12)",
            MONTH_GRANULARITY: f"ceil(months_between({self.target_time}, {self.transaction_time}))",
            DAY_GRANULARITY: f"ceil({self.target_time} - {self.transaction_time})",
            HOUR_GRANULARITY: f"ceil(({self.target_time} - {self.transaction_time}) * 24)",
            MINUTE_GRANULARITY: f"ceil(({self.target_time} - {self.transaction_time}) * 1440)",
            SECONDS_GRANULARITY: f"ceil(({self.target_time} - {self.transaction_time}) * 86400)",
        }
        sql = f"""
            select t.*, 
                   {time_variable_sql[granularity]} as {self._time_order_colname},
              from _TABLENAME_ t
             where 1=1
            {f"and {self._time_order_colname} < {history_length}" if history_length else ''}
            {f"and {self._time_order_colname} >= 1" if self._future_filter else ''}
            """
        return sql

    def _time_order_sql_transform(self, granularity, history_length, partition):
        if granularity == ORDER_GRANULARITY:
            return self._order_granularity_sql(partition, history_length)
        return self._time_granularity_sql(granularity, history_length)

    @staticmethod
    def check_granularity(granularity):
        if granularity == "m":
            raise ValueError(
                "Improper granularity. Did you mean 'months' or 'minutes'?"
            )
        for clean_granularity, aliases in GRANULARITY_ALIASES.items():
            if granularity.lower() in aliases:
                return clean_granularity
        else:
            raise ValueError(f"Incorrect time granularity '{granularity}'")
