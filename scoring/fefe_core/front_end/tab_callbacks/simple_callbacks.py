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
)


def get_int_name(interval, unit):
    if interval[1] == float("inf"):
        return f"[{interval[0]}{unit}-inf)"
    return f"[{interval[0]}{unit}-{interval[1]}{unit})"


def get_options(METADATA):
    FUNCTIONS = {
        fun: fun
        for fun in [
            "min",
            "max",
            "mean",
            "sum",
            "std",
            "count",
            "mode",
            "nunique",
            "var",
            "skew",
            "median",
            "mad",
            "mode_multicolumn",
            "is_monotonic",
            "is_monotonic_increasing",
            "is_monotonic_decreasing",
        ]
    }

    SEGMENTATIONS = {
        "No segmentation": None,
        **{
            variable.name: variable.name
            for variable in METADATA.all_variables
            if variable.n_unique < 10
        },
    }

    TIME_RANGES = {
        "order": {
            get_int_name(interval, ""): interval
            for interval in itertools.combinations(
                [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50, float("inf")], 2
            )
        },
        "years": {
            get_int_name(interval, "y"): interval
            for interval in itertools.combinations([0, 1, 2, 3, float("inf")], 2)
        },
        "months": {
            get_int_name(interval, "m"): interval
            for interval in itertools.combinations(
                [1, 4, 7, 10, 13, 25, float("inf")], 2
            )
        },
        "weeks": {
            get_int_name(interval, "w"): interval
            for interval in itertools.combinations(
                [0, 1, 2, 4, 6, 8, 12, 16, 20, 26, 39, 52, 78, 104, float("inf")], 2
            )
        },
        "days": {
            get_int_name(interval, "d"): interval
            for interval in itertools.combinations(
                [
                    0,
                    1,
                    3,
                    7,
                    10,
                    14,
                    21,
                    28,
                    30,
                    60,
                    75,
                    90,
                    120,
                    180,
                    360,
                    540,
                    720,
                    float("inf"),
                ],
                2,
            )
        },
    }

    return TIME_RANGES, FUNCTIONS, SEGMENTATIONS


def segment_dropdown_id(variable):
    return "SIMPLE {} SEGMENT".format(variable)


def time_range_dropdown_id(granularity, variable):
    return "SIMPLE {} {} TIME RANGE".format(granularity, variable)


def function_dropdown_id(variable):
    return "SIMPLE {} FUNCTION".format(variable)


def query_textarea_id(variable):
    return "SIMPLE {} QUERY".format(variable)


class SimpleTimeRangeGenerator:
    def __init__(
        self,
        config_key,
        variable_id,
        dropdown_container,
        input_type="dropdown",
        options=None,
    ):
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
                CONFIG_HANDLER.process_simple_config_updates(
                    variable, value, self.config_key
                )

        return update_config

    def get_callback_display_config(self):
        def display_config(variable, options):

            return CONFIG_HANDLER.get_simple_config_display(variable, self.config_key)

        return display_config

    def dropdown(self, variable, granularity):

        options = [
            {"label": label, "value": str(value)}
            for label, value in self.options[granularity].items()
        ]

        return dcc.Dropdown(
            id=time_range_dropdown_id(variable, granularity),
            options=options,
            multi=True,
            disabled=True if variable is None else False,
            placeholder="Select variable first." if variable is None else "Select...",
        )

    def register_callbacks(self):

        app.callback(
            output=Output(self.dropdown_container, "children"),
            inputs=[Input(self.variable_id, "value"), Input("granularity", "value")],
        )(self.dropdown)

        for granularity in [o["value"] for o in app.layout["granularity"].options]:
            for variable_value in [
                o["value"] for o in app.layout[self.variable_id].options
            ]:

                app.callback(
                    output=Output(
                        time_range_dropdown_id(variable_value, granularity),
                        "config_update_dummy",
                    ),
                    inputs=[
                        Input(self.variable_id, "value"),
                        Input(
                            time_range_dropdown_id(variable_value, granularity), "value"
                        ),
                    ],
                )(self.get_callback_update_config(variable_value))

                app.callback(
                    Output(
                        time_range_dropdown_id(variable_value, granularity), "value"
                    ),
                    [
                        Input(self.variable_id, "value"),
                        Input(
                            time_range_dropdown_id(variable_value, granularity), "id"
                        ),
                    ],
                    # state=[State(time_range_dropdown_id(variable_value), "value")],
                )(self.get_callback_display_config())


class SimpleInputGenerator:
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
                CONFIG_HANDLER.process_simple_config_updates(
                    variable, value, self.config_key
                )

        return update_config

    def get_callback_display_config(self):
        def display_config(variable, options):

            return CONFIG_HANDLER.get_simple_config_display(variable, self.config_key)

        return display_config

    def textarea(self, variable):
        return dcc.Textarea(id=self.id_function(variable))

    def checklist(self, variable):
        options = [
            {
                "label": label,
                "value": str(value),
                "disabled": True if variable is None else False,
            }
            for label, value in self.options.items()
        ]

        return dcc.Checklist(
            id=self.id_function(variable),
            options=options,
            value=[]
            # multi=True,
            # disabled=True if variable is None else False,
            # placeholder="Select variable first." if variable is None else "Select...",
        )

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
        elif self.input_type == "checklist":
            callback_fun = self.checklist
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


def get_simple_inputs(METADATA):
    TIME_RANGES, FUNCTIONS, SEGMENTATIONS = get_options(METADATA)

    simple_input_time_range = SimpleTimeRangeGenerator(
        options=TIME_RANGES,
        config_key=TIME_RANGE_CONFIG_KEY,
        variable_id="simple-variables",
        dropdown_container="simple-time-ranges-container",
    )

    simple_input_functions = SimpleInputGenerator(
        options=FUNCTIONS,
        id_function=function_dropdown_id,
        config_key=FUNCTION_CONFIG_KEY,
        variable_id="simple-variables",
        dropdown_container="simple-functions-container",
        input_type="dropdown",
    )

    simple_input_segmentation = SimpleInputGenerator(
        options=SEGMENTATIONS,
        id_function=segment_dropdown_id,
        config_key=SEGMENTATION_CONFIG_KEY,
        variable_id="simple-variables",
        dropdown_container="simple-segmentations-container",
        input_type="dropdown",
    )

    simple_input_query = SimpleInputGenerator(
        id_function=query_textarea_id,
        config_key=QUERY_CONFIG_KEY,
        variable_id="simple-variables",
        dropdown_container="simple-query-container",
        input_type="textarea",
    )

    return (
        simple_input_time_range,
        simple_input_functions,
        simple_input_segmentation,
        simple_input_query,
    )


def register_simple_callbacks(METADATA, debug=False):
    if debug:
        (
            simple_input_time_range,
            simple_input_functions,
            simple_input_segmentation,
            simple_input_query,
        ) = get_simple_inputs(METADATA)
        simple_input_time_range.register_callbacks()

        simple_input_functions.register_callbacks()

        simple_input_segmentation.register_callbacks()

        simple_input_query.register_callbacks()

    else:
        (
            simple_input_time_range,
            simple_input_functions,
            simple_input_segmentation,
            simple_input_query,
        ) = get_simple_inputs(METADATA)
        simple_input_time_range.register_callbacks()

        simple_input_functions.register_callbacks()

        simple_input_segmentation.register_callbacks()

        simple_input_query.register_callbacks()
