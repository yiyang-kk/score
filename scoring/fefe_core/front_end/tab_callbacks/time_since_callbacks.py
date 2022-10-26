import itertools
import dash_core_components as dcc
from dash.dependencies import Input, Output, State

from ..app import app
from ..config_handler import (
    CONFIG_HANDLER,
    FUNCTION_CONFIG_KEY,
    SEGMENTATION_CONFIG_KEY,
    TIME_RANGE_CONFIG_KEY,
    QUERY_CONFIG_KEY,
    TIME_SINCE_CONFIG_KEY,
    TIME_SINCE_FROM_CONFIG_KEY,
    TIME_SINCE_UNIT_CONFIG_KEY,
)


def get_int_name(interval, unit):
    if interval[1] == float("inf"):
        return f"[{interval[0]}{unit}-inf)"
    return f"[{interval[0]}{unit}-{interval[1]}{unit}]"


def get_options(METADATA):
    TIME_SINCE_FROM = {fun: fun for fun in ["first", "last"]}
    UNITS = {
        "years": "years",
        "months": "months",
        "days": "days",
        "hours": "hours",
        "minutes": "minutes",
        "seconds": "seconds",
        "order": "order",
    }
    SEGMENTATIONS = {
        "No segmentation": None,
        **{
            variable.name: variable.name
            for variable in METADATA.all_variables
            if variable.n_unique < 10
        },
    }

    return TIME_SINCE_FROM, SEGMENTATIONS, UNITS


def segment_dropdown_id(variable):
    return "TIME SINCE {} SEGMENT".format(variable)


def time_since_from_dropdown_id(variable):
    return "TIME SINCE {} FROM".format(variable)


def time_since_unit_dropdown_id(variable):
    return "TIME SINCE {} UNIT".format(variable)


def query_textarea_id(variable):
    return "TIME SINCE {} QUERY".format(variable)


class TimeSinceInputGenerator:
    def __init__(
        self,
        id_function,
        config_key,
        variable_id,
        dropdown_container,
        input_type="dropdown",
        options=None,
    ):
        self.id_function = id_function
        if options:
            self.options = options
        else:
            self.options = {}
        self.config_key = config_key
        self.variable_id = variable_id
        self.dropdown_container = dropdown_container
        self.input_type = input_type

    def get_callback_update_config(self, variable):
        def update_config(dropdown_variable, value):
            if dropdown_variable == variable:
                CONFIG_HANDLER.process_time_since_config_updates(
                    variable, value, self.config_key
                )

        return update_config

    def get_callback_display_config(self):
        def display_config(variable, options):

            return CONFIG_HANDLER.get_time_since_config_display(
                variable, self.config_key
            )

        return display_config

    def textarea(self, variable):
        return dcc.Textarea(id=self.id_function(variable))

    def dropdown(self, variable):
        options = [
            {"label": label, "value": str(value)}
            for label, value in self.options.items()
        ]

        return dcc.Dropdown(
            id=self.id_function(variable),
            options=options,
            multi=True,
            disabled=True if variable is None else False,
            placeholder="Select variable first." if variable is None else "Select...",
        )

    def register_callbacks(self):
        if self.input_type == "dropdown":
            callback_fun = self.dropdown
            update_trigger = "id"
        elif self.input_type == "textarea":
            callback_fun = self.textarea
            update_trigger = "id"
        app.callback(
            output=Output(self.dropdown_container, "children"),
            inputs=[Input(self.variable_id, "value")],
        )(callback_fun)

        for variable_value in [
            o["value"] for o in app.layout[self.variable_id].options
        ]:

            app.callback(
                output=Output(self.id_function(variable_value), "config_update_dummy"),
                inputs=[
                    Input(self.variable_id, "value"),
                    Input(self.id_function(variable_value), "value"),
                ],
            )(self.get_callback_update_config(variable_value))

            app.callback(
                Output(self.id_function(variable_value), "value"),
                [
                    Input(self.variable_id, "value"),
                    Input(self.id_function(variable_value), update_trigger),
                ],
                # state=[State(self.id_function(variable_value), "value")],
            )(self.get_callback_display_config())


def get_time_since_inputs(METADATA):
    TIME_SINCE_FROM, SEGMENTATIONS, UNITS = get_options(METADATA)

    time_since_input_functions = TimeSinceInputGenerator(
        options=TIME_SINCE_FROM,
        id_function=time_since_from_dropdown_id,
        config_key=TIME_SINCE_FROM_CONFIG_KEY,
        variable_id="time-since-variables",
        dropdown_container="time-since-from-container",
    )
    time_since_input_units = TimeSinceInputGenerator(
        options=UNITS,
        id_function=time_since_unit_dropdown_id,
        config_key=TIME_SINCE_UNIT_CONFIG_KEY,
        variable_id="time-since-variables",
        dropdown_container="time-since-unit-container",
    )

    time_since_input_segmentation = TimeSinceInputGenerator(
        options=SEGMENTATIONS,
        id_function=segment_dropdown_id,
        config_key=SEGMENTATION_CONFIG_KEY,
        variable_id="time-since-variables",
        dropdown_container="time-since-segmentations-container",
    )

    time_since_input_query = TimeSinceInputGenerator(
        id_function=query_textarea_id,
        config_key=QUERY_CONFIG_KEY,
        variable_id="time-since-variables",
        dropdown_container="time-since-query-container",
        input_type="textarea",
    )

    return (
        time_since_input_functions,
        time_since_input_segmentation,
        time_since_input_query,
        time_since_input_units,
    )


def register_time_since_callbacks(METADATA):
    (
        time_since_input_functions,
        time_since_input_segmentation,
        time_since_input_query,
        time_since_input_units,
    ) = get_time_since_inputs(METADATA)

    time_since_input_functions.register_callbacks()

    time_since_input_segmentation.register_callbacks()

    time_since_input_query.register_callbacks()

    time_since_input_units.register_callbacks()
