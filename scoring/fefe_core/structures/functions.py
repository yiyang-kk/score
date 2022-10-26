from dataclasses import dataclass, field
from typing import List
from .constants import NUMERIC_TYPE, CATEGORICAL_TYPE, TIME_TYPE
from .constants import SIMPLE_TAB, RATIO_TAB, TIME_SINCE_TAB, IS_SOMETHING_TAB

ALLOWED_TYPES = [NUMERIC_TYPE, CATEGORICAL_TYPE, TIME_TYPE]
ALLOWED_TABS = [SIMPLE_TAB, RATIO_TAB, TIME_SINCE_TAB, IS_SOMETHING_TAB]


@dataclass
class Function:
    name: str
    types: List[str]
    tabs: List[str]

    def __init__(self, name, types, tabs):

        self.name = name

        self._validate(types, tabs)
        self.types = types
        self.tabs = tabs

    def _validate(self, types, tabs):
        for vartype in types:
            if vartype not in ALLOWED_TYPES:
                raise ValueError(
                    f"Incorrect value of variable type {vartype}"
                    f" for function {self.name}"
                )
        for tab in tabs:
            if tab not in ALLOWED_TABS:
                raise ValueError(
                    f"Incorrect value of tab {tab} for function {self.name}"
                )

    def is_vartype(self, function_vartype):
        return function_vartype in self.types

    def is_on_tab(self, tab):
        return tab in self.tabs


def populate_functions():
    result = []
    everywhere = ["n_unique"]
    result += [
        Function(
            name=fun,
            types=[NUMERIC_TYPE, CATEGORICAL_TYPE, TIME_TYPE],
            tabs=[SIMPLE_TAB, RATIO_TAB, TIME_SINCE_TAB, IS_SOMETHING_TAB],
        )
        for fun in everywhere
    ]
    numeric_time = ["max", "min", "first", "last"]
    result += [
        Function(
            name=fun,
            types=[NUMERIC_TYPE, TIME_TYPE],
            tabs=[SIMPLE_TAB, RATIO_TAB, TIME_SINCE_TAB],
        )
        for fun in numeric_time
    ]
    numerics = ["mean", "sum", "std", "skew", "median"]
    result += [
        Function(name=fun, types=[NUMERIC_TYPE], tabs=[SIMPLE_TAB, RATIO_TAB])
        for fun in numerics
    ]
    categoricals = ["mode"]
    result += [
        Function(
            name=fun, types=[CATEGORICAL_TYPE], tabs=[SIMPLE_TAB, IS_SOMETHING_TAB]
        )
        for fun in categoricals
    ]
    return result


@dataclass(eq=True, frozen=True)
class Functions:
    functions: List[Function] = field(default_factory=populate_functions)

    def get_vartype_functions(self, function_vartype):
        return [fun for fun in self.functions if fun.is_vartype(function_vartype)]

    def get_tab_functions(self, tab):
        return [fun for fun in self.functions if fun.is_on_tab(tab)]

    def get_tab_vartype_functions(self, function_vartype, tab):
        return [
            fun
            for fun in self.functions
            if (fun.is_on_tab(tab) and fun.is_vartype(function_vartype))
        ]


@dataclass
class Tab:
    name: str
    display_name: str
    functions: List[Function]


@dataclass
class Type:
    name: str
    display_name: str
    functions: List[Function]
    tabs: List[Tab]


def populate_tabs():
    funs = Functions()
    result = []
    result += [
        Tab(
            name=SIMPLE_TAB,
            display_name="SIMPLE",
            functions=funs.get_tab_functions(SIMPLE_TAB),
        )
    ]
    result += [
        Tab(
            name=RATIO_TAB,
            display_name="RATIO",
            functions=funs.get_tab_functions(RATIO_TAB),
        )
    ]
    result += [
        Tab(
            name=IS_SOMETHING_TAB,
            display_name="IS SOMETHING",
            functions=funs.get_tab_functions(IS_SOMETHING_TAB),
        )
    ]
    result += [
        Tab(
            name=TIME_SINCE_TAB,
            display_name="TIME SINCE",
            functions=funs.get_tab_functions(TIME_SINCE_TAB),
        )
    ]
    return result


@dataclass(eq=True, frozen=True)
class Tabs:
    tabs: List[Tab] = field(default_factory=populate_tabs)


def populate_vartypes():
    tabs = Tabs()
    funs = Functions()
    result = []
    result += [
        Type(
            name=NUMERIC_TYPE,
            display_name="NUMERIC",
            functions=funs.get_vartype_functions(NUMERIC_TYPE),
            tabs=[
                tab
                for tab in tabs.tabs
                if tab.name in [SIMPLE_TAB, RATIO_TAB, IS_SOMETHING_TAB]
            ],
        )
    ]
    result += [
        Type(
            name=CATEGORICAL_TYPE,
            display_name="CATEGORICAL",
            functions=funs.get_vartype_functions(CATEGORICAL_TYPE),
            tabs=[
                tab for tab in tabs.tabs if tab.name in [SIMPLE_TAB, IS_SOMETHING_TAB]
            ],
        )
    ]
    result += [
        Type(
            name=TIME_TYPE,
            display_name="TIME",
            functions=funs.get_vartype_functions(TIME_TYPE),
            tabs=[tab for tab in tabs.tabs if tab.name in [SIMPLE_TAB, TIME_SINCE_TAB]],
        )
    ]
    return result


@dataclass(eq=True, frozen=True)
class Types:
    types: List[Type] = field(default_factory=populate_vartypes)

    def _get_vartype(self, vartype_name):
        return [vartype for vartype in self.types if vartype.name == vartype_name][0]

    def get_vartype_tabs(self, vartype_name):
        return self._get_vartype(vartype_name).tabs

    def get_vartype_functions(self, vartype_name):
        funs = Functions()
        return funs.get_vartype_functions(vartype_name)
