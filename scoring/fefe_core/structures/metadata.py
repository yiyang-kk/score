"""
Home Credit Python Scoring Library and Workflow
Copyright © 2017-2019, Pavel Sůva, Marek Teller, Martin Kotek, Jan Zeller, 
Marek Mukenšnabl, Kirill Odintsov, Elena Kuchina, Jan Coufalík, Jan Hynek and
Home Credit & Finance Bank Limited Liability Company, Moscow, Russia.
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np
import pandas as pd

# from .method_cfg import TYPE_TO_FUNCTION
from .constants import CATEGORICAL_TYPE, NUMERIC_TYPE, TIME_TYPE
from .variable import CategoricalVariable, NumericVariable, TimeVariable


@dataclass()
class Metadata:
    """
    Metadata class which should calculate and contain all basic info for individual variables.
    These information should be the minimal amount of information to be displayed
    No full dataframes should be stored here - only the basic info.
    Apart from that - information about which column is target, which column is time etc. should be in definition

    Args:
        data (pd.DataFrame)
    """

    columns: List[str]
    numerics: List[Any]
    categoricals: List[Any]
    times: List[Any]
    all_variables: List[Any]
    shape: Tuple
    # definition: Dict

    def __init__(self, data: pd.DataFrame):
        self.numerics, self.categoricals, self.times = self._create_metadata(data)
        self.all_variables = self.numerics + self.categoricals + self.times
        self.shape = data.shape
        self.columns = [variable.name for variable in self.all_variables]
        # self.definition = definition

    def __str__(self):
        numerics = "".join([f"  {var.__str__()}\n" for var in self.numerics])
        categoricals = "".join([f"  {var.__str__()}\n" for var in self.categoricals])
        times = "".join([f"  {var.__str__()}\n" for var in self.times])

        return f"NUMERICS:\n{numerics}CATEGORICALS:\n{categoricals}TIME VARIABLES:\n{times}"

    @staticmethod
    def _create_numeric_meta(
        name: str, variable_data: pd.Series, datatype: np.dtype
    ) -> NumericVariable:

        stats = dict(
            mean=None,
            std=None,
            minimum=None,
            maximum=None,
            n_unique=len(variable_data.unique()),
        )
        return NumericVariable(name=name, datatype=datatype, **stats)

    @staticmethod
    def _create_categorical_meta(
        name: str, variable_data: pd.Series, datatype: np.dtype
    ) -> CategoricalVariable:

        stats = dict(mode=None, n_unique=len(variable_data.unique()))
        return CategoricalVariable(name=name, datatype=datatype, **stats)

    @staticmethod
    def _create_time_meta(
        name: str, variable_data: pd.Series, datatype: np.dtype
    ) -> TimeVariable:
        stats = dict(min_time=None, max_time=None, n_unique=len(variable_data.unique()))
        return TimeVariable(name=name, datatype=datatype, **stats)

    def _create_metadata(
        self, data: pd.DataFrame
    ) -> Tuple[List[Any], List[Any], List[Any]]:
        numerics = []  # type: List[Any]
        categoricals = []  # type: List[Any]
        times = []  # type: List[Any]

        for name, datatype in data.dtypes.to_dict().items():

            if datatype.kind in "biufc":
                numerics += [self._create_numeric_meta(name, data[name], datatype)]
            elif datatype.kind == "M":
                times += [self._create_time_meta(name, data[name], datatype)]
            else:
                categoricals += [
                    self._create_categorical_meta(name, data[name], datatype)
                ]
        return (numerics, categoricals, times)

    def get_variable(self, name: str):

        chosen = [variable for variable in self.all_variables if variable.name == name]
        if len(chosen) > 1:
            raise AssertionError(f"Too many variables named {name}.")
        if not chosen:
            raise KeyError(f"No variable named {name} in column names.")
        return chosen[0]

