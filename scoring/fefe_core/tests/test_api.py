
import pandas as pd
from fefe.nfe.api import FeatureEngineeringAPI

# CONFIG={
#     'meta': {
#         'granularity': 'days',
#         'index': 'CUSTOMER_ID',
#         'target_time': 'TIME',
#         'time_order': 'TIME_ORDER',
#         'transaction_time': 'TRANSACTION_TIME',
#     },
#     'ratio': {},
#     'simple': {},
#     'time_since': {'TRANSACTION_TIME': {'from': ['first', 'last'], 'segmentations': ['TRANSACTION_PLACE']}},
# }


# CONFIG = {
#     "meta": {
#         "granularity": "days",
#         "index": "CUSTOMER_ID",
#         "target_time": "TIME",
#         "transaction_time": "TRANSACTION_TIME",
#         "inf_value": None,
#         "nan_value": None,
#         "order": "TIME_ORDER",
#     },
#     "ratio": {},
#     "simple": {
#         "TRANSACTION_AMOUNT": {
#             "functions": ["max", "mode", "mode_multicolumn"],
#             "segmentations": ["None"],
#             "time_ranges": ["(0, 360)"],
#         },
#         "TRANSACTION_PURPOSE": {
#             "functions": ["mode", "mode_multicolumn", "nunique"],
#             "segmentations": ["None"],
#             "time_ranges": ["(0, 360)"],
#         }
#     },
#     "time_since": {},
# }


def test_dataframe_integration():

    CONFIG = {
        "meta": {
            "granularity": "days",
            "index": "CUSTOMER_ID",
            "inf_value": None,
            "nan_value": None,
            "order": "TIME_ORDER",
            "target_time": "TIME",
            "transaction_time": "TRANSACTION_TIME",
        },
        "ratio": {
            "TRANSACTION_AMOUNT": {
                "TRANSACTION_AMOUNT": {
                    "functions": [
                        "('min', 'min')",
                        "('max', 'max')",
                        "('mean', 'mean')",
                        "('sum', 'sum')",
                    ],
                    "segmentations": ["None", "TRANSACTION_PURPOSE"],
                    "time_ranges": ["((0, 30), (30, 180))", "((0, 180), (180, 360))"],
                }
            }
        },
        "simple": {
            "TRANSACTION_AMOUNT": {
                "functions": ["min", "max", "sum", "mean", "mode"],
                "queries": "TRANSACTION_FEE < 100;TRANSACTION_CLASS == 'ATM'",
                "segmentations": ["None", "TRANSACTION_PURPOSE", "TRANSACTION_TYPE"],
                "time_ranges": ["(0, 360)", "(0, inf)"],
            },
            "TRANSACTION_FEE": {
                "functions": ["min", "max", "sum", "mean"],
                "queries": "TRANSACTION_FEE < 100;TRANSACTION_CLASS == 'ATM'",
                "segmentations": ["None", "TRANSACTION_PURPOSE", "TRANSACTION_TYPE"],
                "time_ranges": ["(0, 360)", "(0, inf)"],
            },
        },
        "time_since": {
            "TRANSACTION_TIME": {
                "from": ["first", "last"],
                "queries": "TRANSACTION_PURPOSE=='hazard'",
            }
        },
    }

    data = pd.read_csv("../fefe/tests/toy_data.csv", index_col=0)

    # cast your datetimes, please!
    data["TIME"] = pd.to_datetime(data["TIME"])
    data["TRANSACTION_TIME"] = pd.to_datetime(data["TRANSACTION_TIME"])
    fe = FeatureEngineeringAPI(
        CONFIG, raise_on_error=False, logger_kwargs={"log_level": 10}
    )

    data = fe.dataframe(data, max_nan_share=0.2)
    assert isinstance(data, pd.DataFrame)


# data.to_csv('data_output.csv')

