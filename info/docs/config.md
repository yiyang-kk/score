# Config description


## Why?

- Config is heart of the New Feature Engineering. It is computing all specified aggregations.
- It has several desgin features:
    - All feature specifications should be in one place
    - It should be possible to create it in front-end
    - Small changes should be possible to edit in the config

## Example Config

<details>
<details><summary> Details - click to expand </summary>

```

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
```

## Explanation

- meta
    - ...
- simple
    - ...
- ratio
    - ...
- time_since
    - ...