import itertools
import dash

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from .app import app
from .config_handler import (
    CONFIG_HANDLER,
    FUNCTION_CONFIG_KEY,
    SEGMENTATION_CONFIG_KEY,
    TIME_RANGE_CONFIG_KEY,
)


@app.callback(
    inputs=[Input("import_config", "n_clicks"), Input("config-textarea", "value")],
    output=Output("import_config", "placeholder"),
)
def import_config(_, imported_cfg):
    CONFIG_HANDLER.import_config(imported_cfg)


@app.callback(
    inputs=[Input("time_order", "value")], output=Output("time_order", "placeholder")
)
def add_time_order(time_order):
    CONFIG_HANDLER.add_time_order(time_order)


@app.callback(
    inputs=[Input("nan_value", "value")], output=Output("nan_value", "placeholder")
)
def input_nans(replacement):
    CONFIG_HANDLER.nan_replacement(replacement)


@app.callback(
    inputs=[Input("inf_value", "value")], output=Output("inf_value", "placeholder")
)
def input_infs(replacement):
    CONFIG_HANDLER.inf_replacement(replacement)


@app.callback(
    inputs=[
        Input("index_var", "value"),
        Input("transaction_time", "value"),
        Input("target_time", "value"),
        Input("granularity", "value"),
    ],
    output=Output("config-textarea", "placeholder"),
)
def disable_index(index, transaction_time, target_time, granularity):
    """
    This function is triggered whenever one of the top dropdowns is changed.
    Therefore it needs to store these variables into config
    and disable given variables in variable list.

    Callback:
        Input:
            - index_var, value
                -- from Index variable dropdown
            - transaction_time, value
                -- From Transaction Time dropdown
            - target_time, value
                -- From Target Time dropdown
        Output:
            - variables, options
                -- disables selected options in variable list
                
    Args:
        index (str): index variable
        transaction_time (str): transaction time variable
        target_time (str): target time variable
    
    Returns:
        list of dict: returns all variables again, but redefines those variables,
            which vere selected.
    """
    CONFIG_HANDLER.process_metadata(index, transaction_time, target_time, granularity)


@app.callback(
    inputs=[
        Input("simple-variables", "value"),
        Input("simple-copy-variables-dropdown", "value"),
        Input("simple-copy-button", "n_clicks"),
    ],
    output=Output("simple-copy-button", "placeholder"),
)
def copy_simple_variable_settings(current_variable, copy_to_variable, _):
    ctx = dash.callback_context
    if ctx.triggered[0]["prop_id"].split(".")[0] == "simple-copy-button":
        CONFIG_HANDLER.copy_simple_settings(current_variable, copy_to_variable)


@app.callback(
    inputs=[
        Input("time-since-variables", "value"),
        Input("time-since-copy-variables-dropdown", "value"),
        Input("time-since-copy-button", "n_clicks"),
    ],
    output=Output("time-since-copy-button", "placeholder"),
)
def copy_time_since_variable_settings(current_variable, copy_to_variable, _):
    ctx = dash.callback_context
    if ctx.triggered[0]["prop_id"].split(".")[0] == "time-since-copy-button":
        CONFIG_HANDLER.copy_time_since_settings(current_variable, copy_to_variable)


@app.callback(
    inputs=[
        Input("ratio-variables-numerator", "value"),
        Input("ratio-copy-all-variables-dropdown", "value"),
        Input("ratio-copy-all-button", "n_clicks"),
    ],
    output=Output("ratio-copy-all-button", "placeholder"),
)
def copy_ratio_variable_all_settings(current_variable, copy_to_variable, _):
    ctx = dash.callback_context
    if ctx.triggered[0]["prop_id"].split(".")[0] == "ratio-copy-all-button":
        CONFIG_HANDLER.copy_ratio_all_settings(current_variable, copy_to_variable)


@app.callback(
    inputs=[
        Input("ratio-variables-numerator", "value"),
        Input("ratio-variables-denominator", "value"),
        Input("ratio-copy-denominator-variables-dropdown", "value"),
        Input("ratio-copy-denominator-button", "n_clicks"),
    ],
    output=Output("ratio-copy-denominator-button", "placeholder"),
)
def copy_ratio_variable_denominator_settings(
    current_variable_numerator, current_variable_denominator, copy_to_variable, _
):
    ctx = dash.callback_context
    if ctx.triggered[0]["prop_id"].split(".")[0] == "ratio-copy-denominator-button":
        CONFIG_HANDLER.copy_ratio_denominator_settings(
            current_variable_numerator, current_variable_denominator, copy_to_variable
        )


@app.callback(
    output=[
        Output("config-textarea", "value"),
        Output("simple-variables", "value"),
        Output("ratio-variables-numerator", "value"),
        Output("ratio-variables-denominator", "value"),
        Output("time-since-variables", "value"),
    ],
    inputs=[
        Input("generate", "n_clicks_timestamp"),
        Input("delete", "n_clicks_timestamp"),
    ],
)
def delete_or_display(generate, delete):
    """Delete or display config
    
    Arguments:
        generate {int} -- timestamp when generate button was clicked
        delete {int} -- timestamp when delete button was clicked
    
    Returns:
        str -- current config. Either empty (if deleted) or just as is
    """
    cfg, value = CONFIG_HANDLER.process_delete_or_display(delete, generate)
    return cfg, value, value, value, value
