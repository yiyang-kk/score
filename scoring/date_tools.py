import numpy as np
from datetime import datetime, timedelta


def fix_y2k_errors(datetime_series, reference_date=None, years=None):
    """Subtract 100 years from given datetime series,
    for dates which are over specified date (default: now)

    This function is intended to be used on datetime series
    coming from Y2K-problem-prone datetime format like "21-08-68".
    If we cast these dates using `pandas.to_datetime`,
    they get represented like "2068-08-21", however the correct date is "1968-08-21".
    To fix this, every date over reference date (default: now) is set 100 years backwards.
    

    Args:
        datetime_series (pd.Series): pandas.Series
        reference_date (datetime-like, optional): datetime to be used as reference for the dates which should be fixed.
            Can be anything, what pandas accept for comparison with datetimes, i.e. `'2000-01-01', `datetime.now()``
            (default: None - datetime.now() is used afterwards)
        years (int, optional): convenience function to subtract `n` years from datetime.now() to fix this problem.
            could be i.e. 18 - if calulating age, we are suggesting that anyone younger than 18 would not be in the data
            (default: None)

    """

    if reference_date is None:
        reference_date = datetime.now()
        if years:
            reference_date = reference_date - timedelta(days=365.25 * years)

    datetime_series[datetime_series > reference_date] = datetime_series[
        datetime_series > reference_date
    ] - timedelta(days=365.25 * 100)
    return datetime_series


def datetime_difference(
    date_series1,
    date_series2,
    unit="days",
    rounding=None,
    show_warnings=True,
    fix_y2k=False,
):
    """Funtion to calculate time difference of two columns.

    Args:
        date_series1 (str): first datetime column (e.g. application date)
        date_series2 (str): second datetime column (e.g. client's birth date)
        unit (str, optional): Unit in which the result should be caluclated.
            Possible units are "years", "months", "weeks", "days", "hours", "minutes", "seconds".
            In case of years and months, precision of the caluclation is days.
            In case of the other, precision of the calculation is seconds.
            (default: "days")
        rounding (str, optional): Type of rounding that should be applied on the float resulting difference.
            Possible roundings are None, "round", "floor" and "ceil".
            If None, then series of floats caluclated with abovementioned precision is returned.
            In other cases, series of integer values (in float format) is returned
            (with the float either rounded or rounded down ("floor") or rounded up ("ceil"))
            (default: None)
        show_warnings (bool, optional): Show warnings whether date_series contains values over `datetime.now()` (default: True)
        fix_y2k (bool, optional): Use scoring.date_tools.fix_y2k_errors function to fix date_series containing values over `datetime.now()`

    
    Returns:
        pd.Series: series with the datetime differences
    """

    def _check_date_series(date_series, show_warnings, fix_y2k):
        if max(date_series) > datetime.now() and show_warnings:
            print(
                "'WARNING: date_series' contains values over current date. "
                "Consider using `scoring.date_tools.fix_y2k_errors` to fix this problem."
            )
        if fix_y2k:
            print(
                "Fixing y2k errors using reference date `datetime.now()`"
                "(=> any date over `datetime.now()` will have 100 years subtracted.)"
            )
            date_series = fix_y2k_errors(date_series)
        return date_series

    # dictionaries to be used for validation of inputs and as replacements for if/elif/else cases
    date_difference_units = {
        "weeks": "W",
        "days": "D",
        "hours": "h",
        "minutes": "m",
        "seconds": "s",
        "years": "Y",
        "months": None,
    }
    rounding_functions = {
        "floor": np.floor,
        "ceil": np.ceil,
        "round": np.round,
        None: lambda x: x,
    }

    ####################
    # INPUT VALIDATION #
    ####################

    if unit not in date_difference_units:
        raise ValueError(
            f"{unit} is unsupported time unit. Supported time units are:  {', '.join(date_difference_units)}"
        )

    if rounding not in rounding_functions:
        raise ValueError(
            f"{rounding} is unsupported rounding type. Supported rounding types are: {', '.join(rounding_functions)}"
        )

    date_series1 = _check_date_series(date_series1, show_warnings, fix_y2k)
    date_series2 = _check_date_series(date_series2, show_warnings, fix_y2k)

    ###############
    # CALCULATION #
    ###############

    if unit == "months":

        def months_between(d1, d2):
            # Credit for this function goes to Martin Kotek (CN)
            # This function should be consistent with Oracle SQL datediff
            # same day in month = months_between = 0 (d1.dt.day<=d2.dt.day); count from 1 (the +1)
            return (
                (d1.dt.year - d2.dt.year) * 12
                + d1.dt.month
                - d2.dt.month
                - (d1.dt.day <= d2.dt.day) * 1
                + 1
            )

        unrounded_result = months_between(date_series1, date_series2)
    else:
        unrounded_result = (date_series1 - date_series2) / np.timedelta64(
            1, date_difference_units[unit]
        )

    rounded_result = rounding_functions[rounding](unrounded_result)

    return rounded_result
