from .constants import NUMERIC_TYPE, CATEGORICAL_TYPE, TIME_TYPE
from dataclasses import dataclass, fields
from .functions import Functions, Types, Tabs
import numpy as np
from typing import Any

FUNCTIONS = Functions()
TYPES = Types()
TABS = Tabs()

# TODO: pridat nanshare, unique everywhere
# TODO: Docstrings
@dataclass
class Variable:
    """
    Base Variable class
    """

    name: str
    datatype: np.dtype
    variable_type: str

    def __str__(self):
        return f"{self.name} - {self.datatype}"

    @classmethod
    def get_stats(cls):
        return [
            method.name
            for method in fields(cls)
            if method.name not in ["name", "datatype"]
        ]

    @classmethod
    def get_functions(cls):
        return TYPES.get_vartype_functions(cls.variable_type)

    @classmethod
    def get_tabs(cls):
        return TYPES.get_vartype_tabs(cls.variable_type)


@dataclass
class NumericVariable(Variable):
    """
    Base Numeric Variable class
    """

    mean: float = float("nan")
    std: float = float("nan")
    minimum: float = float("nan")
    maximum: float = float("nan")
    n_unique: int = 0
    variable_type: str = NUMERIC_TYPE


@dataclass
class CategoricalVariable(Variable):
    """
    Base Categorical Variable Class
    """

    mode: str = ""
    n_unique: int = 0
    variable_type: str = CATEGORICAL_TYPE


@dataclass
class TimeVariable(Variable):
    """
    Base Time Variable Class
    """

    min_time: Any = None
    max_time: Any = None
    n_unique: int = 0
    variable_type: str = TIME_TYPE
