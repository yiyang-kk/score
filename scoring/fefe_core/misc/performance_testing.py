#%%

import contextlib
import os
from datetime import datetime

import pandas as pd
import plotly_express as px
from nfe import FeatureEngineeringAPI
from tqdm import tqdm, trange

#%%
CONFIG = {
    "meta": {
        "granularity": "D",
        "index": "CUSTOMER_ID",
        "target_time": "TIME",
        "time_order": "TIME_ORDER",
        "transaction_time": "TRANSACTION_TIME",
    },
    "ratio": {
        "TRANSACTION_AMOUNT": {
            "TRANSACTION_AMOUNT": {
                "functions": [
                    "('min', 'min')",
                    "('mean', 'mean')",
                    "('sum', 'sum')",
                    "('std', 'std')",
                    "('max', 'mean')",
                ],
                "queries": 'TRANSACTION_PURPOSE=="charity";TRANSACTION_PURPOSE=="hazard";',
                "segmentations": ["TRANSACTION_PLACE", "None"],
                "time_ranges": [
                    "((0, 60), (60, 180))",
                    "((0, 180), (0, 180))",
                    "((0, 180), (180, inf))",
                ],
            },
            "TRANSACTION_FEE": {
                "functions": [
                    "('min', 'min')",
                    "('mean', 'mean')",
                    "('sum', 'sum')",
                    "('std', 'std')",
                    "('max', 'mean')",
                ],
                "time_ranges": [
                    "((0, 60), (60, 180))",
                    "((0, 180), (0, 180))",
                    "((0, 180), (180, inf))",
                ],
            },
        },
        "TRANSACTION_FEE": {
            "TRANSACTION_AMOUNT": {
                "functions": [
                    "('min', 'min')",
                    "('mean', 'mean')",
                    "('sum', 'sum')",
                    "('std', 'std')",
                    "('max', 'mean')",
                ],
                "time_ranges": [
                    "((0, 60), (60, 180))",
                    "((0, 180), (0, 180))",
                    "((0, 180), (180, inf))",
                ],
            },
            "TRANSACTION_FEE": {
                "functions": [
                    "('min', 'min')",
                    "('mean', 'mean')",
                    "('sum', 'sum')",
                    "('std', 'std')",
                    "('max', 'mean')",
                ],
                "time_ranges": [
                    "((0, 60), (60, 180))",
                    "((0, 180), (0, 180))",
                    "((0, 180), (180, inf))",
                ],
            },
        },
    },
    "time_since": {
        "TRANSACTION_TIME": {
            "from": ["first", "last"],
            "segmentations": ["TRANSACTION_PLACE", "None"],
        }
    },
    "simple": {
        "TRANSACTION_AMOUNT": {
            "functions": ["min", "mean", "std", "median"],
            "queries": 'TRANSACTION_PURPOSE=="charity";TRANSACTION_PURPOSE=="hazard";',
            "segmentations": ["TRANSACTION_PLACE", "None"],
            "time_ranges": ["(0, 360)", "(0, 180)", "(0, 720)"],
        },
        "TRANSACTION_FEE": {
            "functions": ["min", "mean", "std", "median"],
            "queries": 'TRANSACTION_PURPOSE=="charity";TRANSACTION_PURPOSE=="hazard";',
            "segmentations": ["TRANSACTION_PLACE", "None"],
            "time_ranges": ["(0, 360)", "(0, 180)", "(0, 720)"],
        },
    },
}


TIME_COLUMN = "TRANSACTION_TIME"
TIME_APPLICATION_COLUMN = "TIME"

rawdata = pd.read_csv(
    r"C:\Users\jan.hynek\Documents\HCI\scoring-sim\fe\TRANSACTIONS_filtered.csv",
    index_col=False,
)
# rawdata[ID_TRANSACTIONS_COLUMN] = pd.Series(range(rawdata.shape[0])).astype(int)
OOT_WT = pd.read_csv(r"C:\Users\jan.hynek\Documents\HCI\scoring-sim\fe\OOT_WT.csv")
IT = pd.read_csv(r"C:\Users\jan.hynek\Documents\HCI\scoring-sim\fe\IT.csv")
all_data = pd.concat([IT.drop(columns=["TARGET"]), OOT_WT], ignore_index=True)
merged_data = all_data.merge(rawdata, on="CUSTOMER_ID", how="right")
merged_data[TIME_COLUMN] = pd.to_datetime(merged_data[TIME_COLUMN])
merged_data[TIME_APPLICATION_COLUMN] = pd.to_datetime(
    merged_data[TIME_APPLICATION_COLUMN]
)
fe = FeatureEngineeringAPI(
    CONFIG, raise_on_error=False, logger_kwargs={"log_level": 10, "handlers": "file"}
)

from tqdm import trange, tqdm
from datetime import datetime
import os
import contextlib

def test_method(method, dataset_sizes, n_subruns=3, run_name='test'):
    stats = {}
    try:
        for idx, multiplicator in tqdm(enumerate(dataset_sizes)):
            print(f"----------- MULTIPLICATOR: {multiplicator} -----------")
            dataframes = []
            for mpl in range(1, multiplicator + 1):
                df = merged_data.copy()
                # df["CUSTOMER_ID"] = df["CUSTOMER_ID"] * mpl
                df["CUSTOMER_ID"] = df["CUSTOMER_ID"] + (1 + max(df["CUSTOMER_ID"].unique()) * mpl)
                df["ID_TRANSACTION"] = df["ID_TRANSACTION"] + ((1 + max(df["ID_TRANSACTION"].unique())) * (mpl))
                dataframes += [df]
            new_data = pd.concat(dataframes, axis="index")
            subtimes = []

            now = datetime.now()
            for _ in trange(n_subruns):
                subtime = datetime.now()
                with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                    _ = method(new_data)
                subtimes += [datetime.now() - subtime]
            total_time = (datetime.now() - now) / n_subruns

            memory_usage = sum(new_data.memory_usage()) / (1024 * 1024)
            print(f"min {min(subtimes)} | avg {total_time} | max {max(subtimes)}")
            print(new_data.shape)
            print(_.shape)
            print(memory_usage)
            stats[f"run_{idx}"] = dict(
                multiplicator=multiplicator,
                total_time=total_time,
                shape=new_data.shape,
                memory_usage_mb=memory_usage,
                n_subruns=n_subruns,
                best=min(subtimes),
                worst=max(subtimes),
            )

    except MemoryError:
        print(f"MEMORY ERROR FOR MULTIPLICATOR {multiplicator}")
        pass
    except KeyboardInterrupt:
        pass
    finally:
        data_stats = pd.DataFrame(stats).T
        data_stats["shape"] = [a[0] for a in data_stats["shape"]]
        data_stats["avg"] = [a.total_seconds() for a in data_stats["total_time"]]
        data_stats["best"] = [a.total_seconds() for a in data_stats["best"]]
        data_stats["worst"] = [a.total_seconds() for a in data_stats["worst"]]
        data_stats.to_csv(f'{run_name}.csv')
    return data_stats

data_stats = test_method(fe.dataframe, dataset_sizes=[2 ** i for i in range(7)], run_name='test_fefe')

#%%

data_stats.to_csv("data_stats.csv")
#%%
graph_data = data_stats.melt(id_vars="shape", value_vars=["avg", "best", "worst"])
px.line(
    data_frame=graph_data,
    y="value",
    x="shape",
    color="variable",
    labels={
        "shape": "Dataset size (number of transactions)",
        "variable": "case",
        "value": "Computation time (in seconds)",
    },
)
#%%
