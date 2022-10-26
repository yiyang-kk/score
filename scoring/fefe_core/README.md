# *NFE:* New Feature Engineering

_Feature Engineering made easy._



Transform your transaction-level data to the application-level.


## Quick start:

       
Open terminal inside folder `/fefe` and install the package itself using `pip`:

        pip install .

## Usage:

1) Inside Jupyter notebook, `nfe.notebook_front_end(data)` will **start interactive front-end** (see Overview for more):
<details><summary> Code example  - click to expand </summary>

```
import pandas as pd
from nfe import notebook_front_end, FeatureEngineeringAPI

data = pd.read_csv('my/imported/data.csv')

# cast your datetimes, please!
data['TIME'] = pd.to_datetime(data['TIME'])
data['TRANSACTION_TIME'] = pd.to_datetime(data['TRANSACTION_TIME'])

notebook_front_end(data)
```
</details>
<br><br>

2) When your __config__ will be done, __copy and paste it back to Jupyter Notebook.__ Like this example:

<details><summary> Example of config - click to expand </summary>

```
CONFIG={
    'meta': {
        'granularity': 'days',
        'index': 'CUSTOMER_ID',
        'inf_value': 99999999,
        'nan_value': 0,
        'target_time': 'TIME',
        'time_order': 'TIME_ORDER',
        'transaction_time': 'TRANSACTION_TIME',
    },
    'ratio': {
        'TRANSACTION_AMOUNT': {
            'TRANSACTION_AMOUNT': {
                'functions': ["('min', 'min')", "('mean', 'mean')", "('max', 'mean')"],
                'segmentations': ['None', 'TRANSACTION_PLACE'],
                'time_ranges': ['((0, 360), (360, 720))', '((0, 180), (180, inf))'],
            },
        },
    },
    'simple': {
        'TRANSACTION_AMOUNT': {
            'functions': ['max', 'sum', 'count'],
            'segmentations': ['None', 'TRANSACTION_PURPOSE'],
            'time_ranges': ['(0, 360)', '(0, inf)'],
        },
        'TRANSACTION_FEE': {
            'functions': ['max', 'sum', 'count'],
            'segmentations': ['None', 'TRANSACTION_PURPOSE'],
            'time_ranges': ['(0, 360)', '(0, inf)'],
        },
    },
    'time_since': {
        'TRANSACTION_TIME': {
            'from': ['first', 'last'],
            'segmentations': ['None', 'TRANSACTION_PLACE', 'TRANSACTION_PURPOSE'],
        },
    },
}
```

</details>
<br><br>

3) __Afterwards, just run:__

```
fe = FeatureEngineeringAPI(config=CONFIG)
new_features = fe.dataframe(data)
feature_sql = fe.sql(data)
```
<br>
In the last snippet, you can see the only two methods of `FeatureEngineeringAPI` class

- `dataframe` - Only input is `data` - dataframe, on which the pre-specified aggregations should be applied.
- `sql` - Input is `data`, and afterwards dictionary with all sqls is generated. 
    - you can also define `feature_subset` to generate sqls only for subset of the features
    - other input can be `table_name` so your sqls are ready to be inputted.
    
<br>

__And that is pretty much it.__  
You can find more in documentation:

- [**front-end documentation**](../info/docs/front_end.md) - defining the front-end and use in python
- [**config documentation**](../info/docs/config.md) - specifying the exact config structure and use
<br>



<!-- <img src="https://media.giphy.com/media/upg0i1m4DLe5q/giphy.gif" width="200"> -->

## Overview:

![overview](../info/docs/pics/extended_use.gif) 

## Performance:

Try it yourself.

![performance](../info/docs/pics/performance.png)

You can find code for this graph in [`misc/performance_testing.py`](misc/performance_testing.py).  
Original data are Scoring Simulator Feature Engineering data (10k applications, ~330k transactions).  
Every run transactions were duplicated and new 'applications' were created.  
-> so the data were aggregated for more and more clients.

__Endline:__ You can easily create __450 features__ for __450k clients__ from __19M transactions__ in __six minutes.__

### Comparison:

- (Old) Feature Enineering Module: 10x time improvement, while twice as many features were calculated
- FeatureTools: ~ 100x time improvement (single core performance), while four times as many features were created.

## TODO:

- SQLs are missing header/footer
- Better documentation




