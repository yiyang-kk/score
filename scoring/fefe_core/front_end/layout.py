import dash_core_components as dcc
import dash_html_components as html
from textwrap import dedent


def get_layout(metadata):
    header = html.Div(
        [
            html.H3("HomeCredit", style={"color": "#FF4136"}),
            html.H2("FEFE: Front End for Feature Engineering"),
        ]
    )
    top_dropdowns = html.Div(
        [
            html.Div(
                [
                    html.Div("Select index:"),
                    dcc.Dropdown(
                        id="index_var",
                        options=[
                            {"label": name, "value": name}
                            for name in sorted(
                                [variable.name for variable in metadata.all_variables]
                            )
                        ],
                    ),
                ],
                style={"display": "inline-block", "width": "24%"},
            ),
            html.Div(
                [
                    html.Div("Transaction time:"),
                    dcc.Dropdown(
                        id="transaction_time",
                        options=[
                            {"label": variable.name, "value": variable.name}
                            for variable in metadata.times
                        ],
                    ),
                ],
                style={"display": "inline-block", "width": "24%"},
            ),
            html.Div(
                [
                    html.Div("Target time:"),
                    dcc.Dropdown(
                        id="target_time",
                        options=[
                            {"label": variable.name, "value": variable.name}
                            for variable in metadata.times
                        ],
                    ),
                ],
                style={"display": "inline-block", "width": "24%"},
            ),
            html.Div(
                [
                    html.Div("Time Granularity:"),
                    dcc.Dropdown(
                        id="granularity",
                        options=[
                            {"label": option, "value": option}
                            for option in ["days", "weeks", "months", "years", "order"]
                        ],
                        clearable=False,
                        value="order",
                    ),
                ],
                style={"display": "inline-block", "width": "24%"},
            ),
        ],
        style={
            "width": "50%",
            "border-style": "solid",
            "border-width": "20px",
            "border-color": "#fefefe",
        },
    )

    second_row = html.Div(
        [
            html.Div(
                [
                    html.Div("Order column"),
                    dcc.Dropdown(
                        id="time_order",
                        options=[
                            {"label": name, "value": name}
                            for name in sorted(
                                [variable.name for variable in metadata.all_variables]
                                + ["TIME_ORDER"]
                            )
                        ],
                        value="TIME_ORDER",
                    ),
                    # dcc.Input(
                    #     id="time_order",
                    #     type="text",
                    #     placeholder="TIME_ORDER",
                    #     value=None,
                    # ),
                    html.Div(
                        [
                            html.Abbr(
                                "󠀠󠀠__❓__",
                                title=dedent(
                                    """
                            TIME_ORDER indicates column created from 'Transaction time' and 'Target time'.
                            If you have a column already created, select it here.
                            
                            """
                                ),
                            )
                        ],
                        # style={"text-align": "center"},
                    ),
                ],
                style={"display": "inline-block", "width": "24%"},
            ),
            # html.Div([],style={'width': '1%'}),
            html.Div(
                [
                    html.Div([html.Div("NAN replacement")]),
                    dcc.Input(
                        id="nan_value",
                        type="text",
                        placeholder="",
                        value=None,
                        style={"width": "100%"},
                    ),
                    html.Div(
                        [
                            html.Abbr(
                                "󠀠󠀠__❓__",
                                title=dedent(
                                    """
                            Value which will be used for replacing missing values.
                            Can be string or integer\n
                            """
                                ),
                            ),
                        ]
                    ),
                ],
                style={"display": "inline-block", "width": "24%"},
            ),
            html.Div(
                [
                    html.Div("INF replacement"),
                    dcc.Input(
                        id="inf_value",
                        type="text",
                        placeholder="",
                        value=None,
                        style={"width": "100%"},
                    ),
                    html.Div(
                        [
                            html.Abbr(
                                "󠀠󠀠__❓__",
                                title=dedent(
                                    """
                        Value which will be used for replacing missing values.
                        Can emerge from ratio variables. Can be string or integer\n
                        """
                                ),
                            ),
                        ]
                    ),
                ],
                style={"display": "inline-block", "width": "24%"},
            ),
        ],
        style={
            "width": "50%",
            "border-style": "solid",
            "border-width": "20px",
            "border-color": "#fefefe",
        },
    )
    simple_tab = dcc.Tab(
        label="Simple",
        value="simple",
        # style={"width": "20%"},
        children=html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3("Variable to aggregate:"),
                                html.Div(
                                    dcc.Dropdown(
                                        id="simple-variables",
                                        options=[
                                            {"label": name, "value": name}
                                            for name in sorted(
                                                [
                                                    variable.name
                                                    for variable in metadata.all_variables
                                                ]
                                            )
                                        ],
                                    ),
                                    style={
                                        "border-style": "solid",
                                        "border-width": "10px",
                                        "border-color": "#dddddd",
                                    },
                                ),
                            ],
                            className="five columns",
                        )
                    ],
                    className="row",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Br(),
                                html.Div("Select time ranges"),
                                html.Div(id="simple-time-ranges-container"),
                                html.Div("Select functions"),
                                html.Div(id="simple-functions-container"),
                                html.Div("Select segmentations"),
                                html.Div(id="simple-segmentations-container"),
                                html.Div(
                                    "Input segmentation queries (divided with ';')"
                                ),
                                html.Div(id="simple-query-container"),
                            ],
                            className="four columns",
                        ),
                        html.Div(
                            [
                                html.Br(),
                                html.Div("Copy settings to variable:"),
                                html.Div(
                                    dcc.Dropdown(
                                        id="simple-copy-variables-dropdown",
                                        options=[
                                            {"label": name, "value": name}
                                            for name in sorted(
                                                [
                                                    variable.name
                                                    for variable in metadata.all_variables
                                                ]
                                            )
                                        ],
                                    )
                                ),
                                html.Button(
                                    "Copy settings",
                                    id="simple-copy-button",
                                    type="submit",
                                ),
                            ],
                            className="three columns",
                        ),
                        html.Div(
                            [
                                dcc.Markdown(
                                    """

                        - Simple variables are created from carthesian product from
                            - `time ranges`,
                            - `functions`
                            - and `segmentations + queries`
                        - Segmentations show variables with less than 10 unique values.
                        - Segmentations are transformed to queries (in the backend).
                        - Queries are applied to the dataset using `pd.DataFrame.query` method.
                        - Queries select subsets of the data, where chosen aggregation is applied.

                        """
                                )
                            ],
                            className="five columns",
                        ),
                    ],
                    className="row",
                    style={"background-color": "#eeeeee"},
                ),
            ]
        ),
    )

    ratio_tab = dcc.Tab(
        label="Ratio",
        value="ratio",
        children=html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3("Variable for numerator values:"),
                                html.Div(
                                    dcc.Dropdown(
                                        id="ratio-variables-numerator",
                                        options=[
                                            {"label": name, "value": name}
                                            for name in sorted(
                                                [
                                                    variable.name
                                                    for variable in metadata.numerics
                                                ]
                                            )
                                        ],
                                    ),
                                    style={
                                        "border-style": "solid",
                                        "border-width": "10px",
                                        "border-color": "#dddddd",
                                    },
                                ),
                            ],
                            className="five columns",
                        ),
                        html.Div(
                            [
                                html.H3("Variable for denominator values:"),
                                html.Div(
                                    dcc.Dropdown(
                                        id="ratio-variables-denominator",
                                        options=[
                                            {"label": name, "value": name}
                                            for name in sorted(
                                                [
                                                    variable.name
                                                    for variable in metadata.numerics
                                                ]
                                            )
                                        ],
                                    ),
                                    style={
                                        "border-style": "solid",
                                        "border-width": "10px",
                                        "border-color": "#dddddd",
                                    },
                                ),
                            ],
                            className="five columns",
                        ),
                    ],
                    className="row",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Br(),
                                html.Div("Select time ranges"),
                                html.Div(id="ratio-time-ranges-container"),
                                html.Div("Select function pairs for aggregations"),
                                html.Div(id="ratio-functions-container"),
                                html.Div(
                                    "Select segmentations (to be applied on both variables)"
                                ),
                                html.Div(id="ratio-segmentations-container"),
                                html.Div(
                                    "Input segmentation queries (divided with ';', to be applied on both variables)"
                                ),
                                html.Div(id="ratio-query-container"),
                            ],
                            className="four columns",
                        ),
                        html.Div(
                            [
                                html.Br(),
                                html.Div("Copy all settings to variable:"),
                                html.Div(
                                    dcc.Dropdown(
                                        id="ratio-copy-all-variables-dropdown",
                                        options=[
                                            {"label": name, "value": name}
                                            for name in sorted(
                                                [
                                                    variable.name
                                                    for variable in metadata.all_variables
                                                ]
                                            )
                                        ],
                                    )
                                ),
                                html.Button(
                                    "Copy all settings",
                                    id="ratio-copy-all-button",
                                    type="submit",
                                ),
                                html.Div("Copy denominator settings to variable:"),
                                html.Div(
                                    dcc.Dropdown(
                                        id="ratio-copy-denominator-variables-dropdown",
                                        options=[
                                            {"label": name, "value": name}
                                            for name in sorted(
                                                [
                                                    variable.name
                                                    for variable in metadata.all_variables
                                                ]
                                            )
                                        ],
                                    )
                                ),
                                html.Button(
                                    "Copy denominator settings",
                                    id="ratio-copy-denominator-button",
                                    type="submit",
                                ),
                            ],
                            className="three columns",
                        ),
                        html.Div(
                            [
                                dcc.Markdown(
                                    """
                        - Ratio features are calculated as
                        ```
                         simple aggregation of numerator
                        ---------------------------------
                        simple aggregation of denominator
                        ```
                        - Simple variables are created from carthesian product from
                            - `time range` pairs,
                            - `functions` pairs,
                            - and `segmentations + queries`
                        - Segmentations show variables with less than 10 unique values.
                        - Segmentations are transformed to queries (in the backend).
                        - Queries are applied to the dataset using `pd.DataFrame.query` method.
                        - Queries select subsets of the data, where chosen aggregation is applied.


                        """
                                )
                            ],
                            className="five columns",
                        ),
                    ],
                    style={"background-color": "#eeeeee"},
                    className="row",
                ),
            ]
        ),
    )

    time_since_tab = dcc.Tab(
        label="Time Since",
        value="time_since",
        # style={"width": "20%"},
        children=html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3("Name of date variable:"),
                                html.Div(
                                    dcc.Dropdown(
                                        id="time-since-variables",
                                        options=[
                                            {
                                                "label": variable.name,
                                                "value": variable.name,
                                            }
                                            for variable in metadata.times
                                        ],
                                    ),
                                    style={
                                        "border-style": "solid",
                                        "border-width": "10px",
                                        "border-color": "#dddddd",
                                    },
                                ),
                            ],
                            className="five columns",
                        )
                    ],
                    className="row",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Br(),
                                html.Div("Time Since from"),
                                html.Div(id="time-since-from-container"),
                                html.Div("Units"),
                                html.Div(id="time-since-unit-container"),
                                html.Div("Select segmentations"),
                                html.Div(id="time-since-segmentations-container"),
                                html.Div(
                                    "Input segmentation queries (divided with ';')"
                                ),
                                html.Div(id="time-since-query-container"),
                            ],
                            className="four columns",
                        ),
                        html.Div(
                            [
                                html.Br(),
                                html.Div("Copy settings to variable:"),
                                html.Div(
                                    dcc.Dropdown(
                                        id="time-since-copy-variables-dropdown",
                                        options=[
                                            {
                                                "label": variable.name,
                                                "value": variable.name,
                                            }
                                            for variable in metadata.times
                                        ],
                                    )
                                ),
                                html.Button(
                                    "Copy settings",
                                    id="time-since-copy-button",
                                    type="submit",
                                ),
                            ],
                            className="three columns",
                        ),
                        html.Div(
                            [
                                dcc.Markdown(
                                    """
                        - Time Since features are calculated as
                        ```
                        target column - time since column
                        ---------------------------------
                                granularity
                        ```

                        - Resulting variables are created from carthesian product from
                            - `Time Since from` 
                            - and `segmentations + queries`
                        - Segmentations show variables with less than 10 unique values.
                        - Segmentations are transformed to queries (in the backend).
                        - Queries are applied to the dataset using `pd.DataFrame.query` method.
                        - Queries select subsets of the data, where chosen aggregation is applied.

                        """
                                )
                            ],
                            className="five columns",
                        ),
                    ],
                    className="row",
                    style={"background-color": "#eeeeee"},
                ),
            ]
        ),
    )
    tabs = html.Div(
        [
            html.Hr(),
            dcc.Tabs(
                id="tabs",
                value="simple",
                children=[simple_tab, ratio_tab, time_since_tab],
            ),
        ],
        style={
            "border-style": "solid",
            "border-width": "20px",
            "border-color": "#eeeeee",
        },
    )

    config_area = html.Div(
        [
            html.Hr(),
            html.H2("Config"),
            html.Button(
                "Generate Config",
                id="generate",
                n_clicks_timestamp=0,
                className="button-primary",
                type="submit",
            ),
            html.Button(
                "Delete Config", id="delete", n_clicks_timestamp=0, type="reset"
            ),
            html.Button("Import Config", id="import_config"),
            html.Abbr(
                "❓ (beta)",
                title=dedent(
                    """
                        Import config.

                        Make sure that config is valid (no checks are performed). 
                        When config is imported, values for index/transction time/tatget time are not updated
                        but they are still recorded.
                        Function values can be refreshed by clicking to another variable.
                        """
                ),
            ),
            html.Div(
                [
                    dcc.Textarea(
                        id="config-textarea",
                        style={"width": "80%", "height": "50em"},
                    ),
                ],
                id="config",
            ),
        ]
    )

    footer = html.Div(
        [
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Hr(),
            # html.H3("Usage:"),
            # html.Div(
            #     [
            #         dcc.Markdown(
            #             """
            #     This tool is created as front end for internal Home Credit *Feature Engineering* tool.
            #     At this moment, simple creation of features is supported.
            #     Usage:
            #     - select index - unique customer (application) identifier
            #     - transaction time - individual transaction time
            #     - target time - time for aggregations
            #     - variable selection - user action specification:
            #         - select functions - functions which are going to be applied for given variable
            #         - select time range - time ranges, for which are given aggregations going to be calculated
            #     - afterwards:
            #         - generate config - config to be used in Feature Engineering
            #         - delete config - start again
            # """
            #         )
            #     ]
            # ),
        ]
    )
    LAYOUT = html.Div(
        [header, top_dropdowns, second_row, tabs, config_area, footer],
        style={"color": "#333333", "font-family": "Lucida Sans Unicode"},
    )
    return LAYOUT
