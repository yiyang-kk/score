import copy
import ast
from pprintpp import pformat
import dash

FUNCTION_CONFIG_KEY = "functions"
TIME_RANGE_CONFIG_KEY = "time_ranges"
META_CONFIG_KEY = "meta"
SEGMENTATION_CONFIG_KEY = "segmentations"
SIMPLE_CONFIG_KEY = "simple"
RATIO_CONFIG_KEY = "ratio"
TIME_SINCE_CONFIG_KEY = "time_since"
TIME_SINCE_FROM_CONFIG_KEY = "from"
TIME_SINCE_UNIT_CONFIG_KEY = "units"
QUERY_CONFIG_KEY = "queries"
DEFAULT_TIME_ORDER = "TIME_ORDER"


GRANULARITY = "granularity"
INDEX = "index"
TRANSACTION_TIME = "transaction_time"
TARGET_TIME = "target_time"
ORDER = "order"

INF_VALUE = "inf_value"
NAN_VALUE = "nan_value"


CONFIG_CLEAN = {
    META_CONFIG_KEY: {
        GRANULARITY: "days",
        INDEX: None,
        INF_VALUE: None,
        NAN_VALUE: None,
        TARGET_TIME: None,
        ORDER: DEFAULT_TIME_ORDER,
        TRANSACTION_TIME: None,
    },
    SIMPLE_CONFIG_KEY: {},
    RATIO_CONFIG_KEY: {},
    TIME_SINCE_CONFIG_KEY: {},
}


META_SUBKEYS = [
    INDEX,
    TRANSACTION_TIME,
    TARGET_TIME,
    ORDER,
    GRANULARITY,
    NAN_VALUE,
    INF_VALUE,
]


CONFIG = copy.deepcopy(CONFIG_CLEAN)


class ConfigHandler:
    """
    Helper class for manipulation with global config variable.

    I know that it is bad to create global variables.
    Suggested dash solution is to create hidden div, and store all values there.

    But as for given output only one function have this output,
    so for our dropdowns for individual variables
    we would need another helper function to resolve the outputs...
    And I did not manage to make such function work properly.

    Therefore, I decided to use global variables.
    And I decided to create this class to handle all cases when global variables are used.
    (check the status of this issue here: https://github.com/plotly/dash/issues/153)

    """

    def __init__(self):
        # self.meta = metadata
        self.run_stats = dict(clear=True, run=0, current_variable="")

    @staticmethod
    def import_config(import_cfg):
        global CONFIG
        if import_cfg:
            CONFIG = ast.literal_eval(import_cfg)

    @staticmethod
    def process_metadata(index, transaction_time, target_time, granularity):
        global CONFIG
        CONFIG[META_CONFIG_KEY][INDEX] = index
        CONFIG[META_CONFIG_KEY][TRANSACTION_TIME] = transaction_time
        CONFIG[META_CONFIG_KEY][TARGET_TIME] = target_time
        CONFIG[META_CONFIG_KEY][GRANULARITY] = granularity

    @staticmethod
    def add_time_order(time_order):
        global CONFIG
        if (time_order != DEFAULT_TIME_ORDER) or (time_order != ""):
            CONFIG[META_CONFIG_KEY][ORDER] = time_order

    @staticmethod
    def copy_simple_settings(variable_from, variable_to):
        global CONFIG
        try:
            CONFIG[SIMPLE_CONFIG_KEY][variable_to] = copy.deepcopy(
                CONFIG[SIMPLE_CONFIG_KEY][variable_from]
            )
        except KeyError:
            return

    @staticmethod
    def copy_time_since_settings(variable_from, variable_to):
        global CONFIG
        try:
            CONFIG[TIME_SINCE_CONFIG_KEY][variable_to] = copy.deepcopy(
                CONFIG[TIME_SINCE_CONFIG_KEY][variable_from]
            )
        except KeyError:
            return

    @staticmethod
    def copy_ratio_all_settings(variable_from, variable_to):
        global CONFIG
        try:
            CONFIG[RATIO_CONFIG_KEY][variable_to] = copy.deepcopy(
                CONFIG[RATIO_CONFIG_KEY][variable_from]
            )
        except KeyError:
            return

    @staticmethod
    def copy_ratio_denominator_settings(
        numerator_variable, denominator_variable_from, denominator_variable_to
    ):
        global CONFIG
        try:
            CONFIG[RATIO_CONFIG_KEY][numerator_variable][
                denominator_variable_to
            ] = copy.deepcopy(
                CONFIG[RATIO_CONFIG_KEY][numerator_variable][denominator_variable_from]
            )
        except KeyError:
            return

    @staticmethod
    def nan_replacement(replacement):
        global CONFIG
        try:
            replacement = int(replacement)
        except (TypeError, ValueError):
            pass
        if replacement == "":
            replacement = None
        CONFIG[META_CONFIG_KEY]["nan_value"] = replacement

    @staticmethod
    def inf_replacement(replacement):
        global CONFIG
        try:
            replacement = int(replacement)
        except (TypeError, ValueError):
            pass
        if replacement == "":
            replacement = None
        CONFIG[META_CONFIG_KEY]["inf_value"] = replacement

    @staticmethod
    def process_ratio_config_updates(variable1, variable2, value, config_part):
        global CONFIG

        if value is None:
            return
        variable_values = CONFIG[RATIO_CONFIG_KEY].get(variable1, {})
        current_arguments = variable_values.get(variable2, {})
        current_arguments[config_part] = value
        variable_values[variable2] = current_arguments
        CONFIG[RATIO_CONFIG_KEY][variable1] = variable_values

        return

    @staticmethod
    def get_ratio_config_display(variable1, variable2, config_part):
        global CONFIG

        if variable1 in CONFIG[RATIO_CONFIG_KEY]:
            if variable2 in CONFIG[RATIO_CONFIG_KEY][variable1]:
                return CONFIG[RATIO_CONFIG_KEY][variable1][variable2].get(
                    config_part, None
                )
        return None

    @staticmethod
    def process_simple_config_updates(variable, value, config_part):
        global CONFIG

        if value is None:
            return
        current_arguments = CONFIG[SIMPLE_CONFIG_KEY].get(variable, {})

        current_arguments[config_part] = value
        CONFIG[SIMPLE_CONFIG_KEY][variable] = current_arguments

        return

    @staticmethod
    def get_simple_config_display(variable, config_part):
        global CONFIG
        if variable in CONFIG[SIMPLE_CONFIG_KEY]:
            return CONFIG[SIMPLE_CONFIG_KEY][variable].get(config_part, None)
        return None

    @staticmethod
    def process_time_since_config_updates(variable, value, config_part):
        global CONFIG

        if value is None:
            return
        current_arguments = CONFIG[TIME_SINCE_CONFIG_KEY].get(variable, {})

        current_arguments[config_part] = value
        CONFIG[TIME_SINCE_CONFIG_KEY][variable] = current_arguments

        return

    @staticmethod
    def get_time_since_config_display(variable, config_part):
        global CONFIG
        if variable in CONFIG[TIME_SINCE_CONFIG_KEY]:
            return CONFIG[TIME_SINCE_CONFIG_KEY][variable].get(config_part, None)
        return None

    @staticmethod
    def process_delete_or_display(delete, generate):
        global CONFIG

        variable = dash.no_update
        if delete > generate:
            variable = None
            CONFIG = copy.deepcopy(CONFIG_CLEAN)
        return pformat(CONFIG, width=120), variable


CONFIG_HANDLER = ConfigHandler()
