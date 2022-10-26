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
        return f"({interval[0]}{unit}-inf)"
    return f"({interval[0]}{unit}-{interval[1]}{unit}]"


def get_options(metadata):
    FUNCTIONS = [
        "min",
        "max",
        "mean",
        "sum",
        "std",
        "count",
    ]

    ratio_functions = {
        **{f"{fun}-{fun}": (fun, fun) for fun in FUNCTIONS},
        **{
            f"{fun1}-{fun2}": (fun1, fun2)
            for fun1, fun2 in itertools.permutations(FUNCTIONS, 2)
        },
    }

    segmentations = {
        "No segmentation": None,
        **{
            variable.name: variable.name
            for variable in metadata.all_variables
            if variable.n_unique < 10
        },
    }

    TIME_RANGES = {
        "order": {
            get_int_name(interval, ""): interval
            for interval in itertools.combinations(
                [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50], 2
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

    ratio_time_ranges = {
        "order": {
            f"{get_int_name(a, '')} | {get_int_name(b, '')}": (a, b)
            for a, b in itertools.product(TIME_RANGES["order"].values(), repeat=2)
            if ((a[1] == b[0]) or (a == b))
        },
        "years": {
            f"{get_int_name(a, 'y')} | {get_int_name(b, 'y')}": (a, b)
            for a, b in itertools.product(TIME_RANGES["years"].values(), repeat=2)
            if ((a[1] == b[0]) or (a == b))
        },
        "months": {
            f"{get_int_name(a, 'm')} | {get_int_name(b, 'm')}": (a, b)
            for a, b in itertools.product(TIME_RANGES["months"].values(), repeat=2)
            if ((a[1] == b[0]) or (a == b))
        },
        "weeks": {
            f"{get_int_name(a, 'w')} | {get_int_name(b, 'w')}": (a, b)
            for a, b in itertools.product(TIME_RANGES["weeks"].values(), repeat=2)
            if ((a[1] == b[0]) or (a == b))
        },
        "days": {
            f"{get_int_name(a, 'd')} | {get_int_name(b, 'd')}": (a, b)
            for a, b in itertools.product(TIME_RANGES["days"].values(), repeat=2)
            if ((a[1] == b[0]) or (a == b))
        },
    }
    return ratio_time_ranges, ratio_functions, segmentations


def segment_dropdown_id(variable_numerator, variable_denominator):
    return "RATIO {} {} SEGMENT".format(variable_numerator, variable_denominator)


def time_range_dropdown_id(variable_numerator, variable_denominator, granularity):
    return "RATIO {} {} {} TIME RANGE".format(
        granularity, variable_numerator, variable_denominator
    )


def function_dropdown_id(variable_numerator, variable_denominator):
    return "RATIO {} {} FUNCTION".format(variable_numerator, variable_denominator)


def query_textarea_id(variable_numerator, variable_denominator):
    return "RATIO {} {} QUERY".format(variable_numerator, variable_denominator)


class RatioTimeRangeGenerator:
    """
    
    """

    def __init__(
        self,
        config_key,
        variable_numerator_id,
        variable_denominator_id,
        dropdown_container,
        options=None,
    ):
        self.id_function = time_range_dropdown_id
        if options:
            self.options = options
        else:
            self.options = {}
        self.config_key = config_key
        self.variable_numerator_id = variable_numerator_id
        self.variable_denominator_id = variable_denominator_id
        self.dropdown_container = dropdown_container

    def get_callback_update_config(self, variable_numerator, variable_denominator):
        def update_config(
            dropdown_variable_numerator, dropdown_variable_denominator, value
        ):
            if (dropdown_variable_numerator == variable_numerator) & (
                dropdown_variable_denominator == variable_denominator
            ):
                CONFIG_HANDLER.process_ratio_config_updates(
                    variable_numerator, variable_denominator, value, self.config_key
                )

        return update_config

    def get_callback_display_config(self):
        def display_config(variable_numerator, variable_denominator, options):

            return CONFIG_HANDLER.get_ratio_config_display(
                variable_numerator, variable_denominator, self.config_key
            )

        return display_config

    def dropdown(self, variable_numerator, variable_denominator, granularity):

        options = [
            {"label": label, "value": str(value)}
            for label, value in self.options[granularity].items()
        ]

        return dcc.Dropdown(
            id=self.id_function(variable_numerator, variable_denominator, granularity),
            options=options,
            multi=True,
            disabled=True
            if (variable_numerator is None) or (variable_denominator is None)
            else False,
            placeholder="Select variables first."
            if (variable_numerator is None) or (variable_denominator is None)
            else "Select...",
        )

    def register_callbacks(self):

        app.callback(
            output=Output(self.dropdown_container, "children"),
            inputs=[
                Input(self.variable_numerator_id, "value"),
                Input(self.variable_denominator_id, "value"),
                Input("granularity", "value"),
            ],
        )(self.dropdown)
        granularity_options = [o["value"] for o in app.layout["granularity"].options]
        numerator_options = [
            o["value"] for o in app.layout[self.variable_numerator_id].options
        ]
        denominator_options = [
            o["value"] for o in app.layout[self.variable_denominator_id].options
        ]
        for (
            granularity,
            variable_numerator_value,
            variable_denominator_value,
        ) in itertools.product(
            granularity_options, numerator_options, denominator_options
        ):
            app.callback(
                output=Output(
                    self.id_function(
                        variable_numerator_value,
                        variable_denominator_value,
                        granularity,
                    ),
                    "config_update_dummy",
                ),
                inputs=[
                    Input(self.variable_numerator_id, "value"),
                    Input(self.variable_denominator_id, "value"),
                    Input(
                        self.id_function(
                            variable_numerator_value,
                            variable_denominator_value,
                            granularity,
                        ),
                        "value",
                    ),
                ],
            )(
                self.get_callback_update_config(
                    variable_numerator_value, variable_denominator_value
                )
            )

            app.callback(
                output=Output(
                    self.id_function(
                        variable_numerator_value,
                        variable_denominator_value,
                        granularity,
                    ),
                    "value",
                ),
                inputs=[
                    Input(self.variable_numerator_id, "value"),
                    Input(self.variable_denominator_id, "value"),
                    Input(
                        self.id_function(
                            variable_numerator_value,
                            variable_denominator_value,
                            granularity,
                        ),
                        "id",
                    ),
                ],
                # state=[State(self.id_function(variable_value), "value")],
            )(self.get_callback_display_config())


class RatioTimeRangeGenerator:
    def __init__(
        self,
        config_key,
        variable_numerator_id,
        variable_denominator_id,
        dropdown_container,
        options=None,
    ):
        self.id_function = time_range_dropdown_id
        if options:
            self.options = options
        else:
            self.options = {}
        self.config_key = config_key
        self.variable_numerator_id = variable_numerator_id
        self.variable_denominator_id = variable_denominator_id
        self.dropdown_container = dropdown_container

    def get_callback_update_config(self, variable_numerator, variable_denominator):
        def update_config(
            dropdown_variable_numerator, dropdown_variable_denominator, value
        ):
            if (dropdown_variable_numerator == variable_numerator) & (
                dropdown_variable_denominator == variable_denominator
            ):
                CONFIG_HANDLER.process_ratio_config_updates(
                    variable_numerator, variable_denominator, value, self.config_key
                )

        return update_config

    def get_callback_display_config(self):
        def display_config(variable_numerator, variable_denominator, options):

            return CONFIG_HANDLER.get_ratio_config_display(
                variable_numerator, variable_denominator, self.config_key
            )

        return display_config

    def dropdown(self, variable_numerator, variable_denominator, granularity):

        options = [
            {"label": label, "value": str(value)}
            for label, value in self.options[granularity].items()
        ]

        return dcc.Dropdown(
            id=self.id_function(variable_numerator, variable_denominator, granularity),
            options=options,
            multi=True,
            disabled=True
            if (variable_numerator is None) or (variable_denominator is None)
            else False,
            placeholder="Select variables first."
            if (variable_numerator is None) or (variable_denominator is None)
            else "Select...",
        )

    def register_callbacks(self):

        app.callback(
            output=Output(self.dropdown_container, "children"),
            inputs=[
                Input(self.variable_numerator_id, "value"),
                Input(self.variable_denominator_id, "value"),
                Input("granularity", "value"),
            ],
        )(self.dropdown)
        granularity_options = [o["value"] for o in app.layout["granularity"].options]
        numerator_options = [
            o["value"] for o in app.layout[self.variable_numerator_id].options
        ]
        denominator_options = [
            o["value"] for o in app.layout[self.variable_denominator_id].options
        ]
        for (
            granularity,
            variable_numerator_value,
            variable_denominator_value,
        ) in itertools.product(
            granularity_options, numerator_options, denominator_options
        ):
            app.callback(
                output=Output(
                    self.id_function(
                        variable_numerator_value,
                        variable_denominator_value,
                        granularity,
                    ),
                    "config_update_dummy",
                ),
                inputs=[
                    Input(self.variable_numerator_id, "value"),
                    Input(self.variable_denominator_id, "value"),
                    Input(
                        self.id_function(
                            variable_numerator_value,
                            variable_denominator_value,
                            granularity,
                        ),
                        "value",
                    ),
                ],
            )(
                self.get_callback_update_config(
                    variable_numerator_value, variable_denominator_value
                )
            )

            app.callback(
                output=Output(
                    self.id_function(
                        variable_numerator_value,
                        variable_denominator_value,
                        granularity,
                    ),
                    "value",
                ),
                inputs=[
                    Input(self.variable_numerator_id, "value"),
                    Input(self.variable_denominator_id, "value"),
                    Input(
                        self.id_function(
                            variable_numerator_value,
                            variable_denominator_value,
                            granularity,
                        ),
                        "id",
                    ),
                ],
                # state=[State(self.id_function(variable_value), "value")],
            )(self.get_callback_display_config())


class RatioInputGenerator:
    def __init__(
        self,
        id_function,
        config_key,
        variable_numerator_id,
        variable_denominator_id,
        dropdown_container,
        function="dropdown",
        options=None,
    ):
        self.id_function = id_function
        if options:
            self.options = options
        else:
            self.options = {}
        self.config_key = config_key
        self.variable_numerator_id = variable_numerator_id
        self.variable_denominator_id = variable_denominator_id
        self.dropdown_container = dropdown_container
        self.function = function

    def get_callback_update_config(self, variable_numerator, variable_denominator):
        def update_config(
            dropdown_variable_numerator, dropdown_variable_denominator, value
        ):
            if (dropdown_variable_numerator == variable_numerator) & (
                dropdown_variable_denominator == variable_denominator
            ):
                CONFIG_HANDLER.process_ratio_config_updates(
                    variable_numerator, variable_denominator, value, self.config_key
                )

        return update_config

    def get_callback_display_config(self):
        def display_config(variable_numerator, variable_denominator, options):

            return CONFIG_HANDLER.get_ratio_config_display(
                variable_numerator, variable_denominator, self.config_key
            )

        return display_config

    def textarea(self, variable_numerator, variable_denominator):
        return dcc.Textarea(
            id=self.id_function(variable_numerator, variable_denominator),
            style={"width": "50%"},
        )

    def dropdown(self, variable_numerator, variable_denominator):
        options = [
            {"label": label, "value": str(value)}
            for label, value in self.options.items()
        ]

        return dcc.Dropdown(
            id=self.id_function(variable_numerator, variable_denominator),
            options=options,
            multi=True,
            disabled=True
            if (variable_numerator is None) or (variable_denominator is None)
            else False,
            placeholder="Select variables first."
            if (variable_numerator is None) or (variable_denominator is None)
            else "Select...",
        )

    def register_callbacks(self):
        if self.function == "dropdown":
            callback_fun = self.dropdown
            update_trigger = "options"
        elif self.function == "textarea":
            callback_fun = self.textarea
            update_trigger = "id"
        app.callback(
            output=Output(self.dropdown_container, "children"),
            inputs=[
                Input(self.variable_numerator_id, "value"),
                Input(self.variable_denominator_id, "value"),
            ],
        )(callback_fun)

        for variable_numerator_value in [
            o["value"] for o in app.layout[self.variable_numerator_id].options
        ]:
            for variable_denominator_value in [
                o["value"] for o in app.layout[self.variable_denominator_id].options
            ]:
                app.callback(
                    output=Output(
                        self.id_function(
                            variable_numerator_value, variable_denominator_value
                        ),
                        "config_update_dummy",
                    ),
                    inputs=[
                        Input(self.variable_numerator_id, "value"),
                        Input(self.variable_denominator_id, "value"),
                        Input(
                            self.id_function(
                                variable_numerator_value, variable_denominator_value
                            ),
                            "value",
                        ),
                    ],
                )(
                    self.get_callback_update_config(
                        variable_numerator_value, variable_denominator_value
                    )
                )

                app.callback(
                    Output(
                        self.id_function(
                            variable_numerator_value, variable_denominator_value
                        ),
                        "value",
                    ),
                    [
                        Input(self.variable_numerator_id, "value"),
                        Input(self.variable_denominator_id, "value"),
                        Input(
                            self.id_function(
                                variable_numerator_value, variable_denominator_value
                            ),
                            update_trigger,
                        ),
                    ],
                    # state=[State(self.id_function(variable_value), "value")],
                )(self.get_callback_display_config())


def register_ratio_callbacks(metadata):
    ratio_time_ranges, ratio_functions, segmentations = get_options(metadata)
    ratio_input_time_range = RatioTimeRangeGenerator(
        options=ratio_time_ranges,
        config_key=TIME_RANGE_CONFIG_KEY,
        variable_numerator_id="ratio-variables-numerator",
        variable_denominator_id="ratio-variables-denominator",
        dropdown_container="ratio-time-ranges-container",
    )

    ratio_input_functions = RatioInputGenerator(
        options=ratio_functions,
        id_function=function_dropdown_id,
        config_key=FUNCTION_CONFIG_KEY,
        variable_numerator_id="ratio-variables-numerator",
        variable_denominator_id="ratio-variables-denominator",
        dropdown_container="ratio-functions-container",
    )

    ratio_input_segmentation = RatioInputGenerator(
        options=segmentations,
        id_function=segment_dropdown_id,
        config_key=SEGMENTATION_CONFIG_KEY,
        variable_numerator_id="ratio-variables-numerator",
        variable_denominator_id="ratio-variables-denominator",
        dropdown_container="ratio-segmentations-container",
    )

    ratio_input_query = RatioInputGenerator(
        id_function=query_textarea_id,
        config_key=QUERY_CONFIG_KEY,
        variable_numerator_id="ratio-variables-numerator",
        variable_denominator_id="ratio-variables-denominator",
        dropdown_container="ratio-query-container",
        function="textarea",
    )
    ratio_input_time_range.register_callbacks()

    ratio_input_functions.register_callbacks()

    ratio_input_segmentation.register_callbacks()

    ratio_input_query.register_callbacks()
