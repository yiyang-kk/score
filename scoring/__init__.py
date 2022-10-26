# -*- coding: utf-8 -*-


# Home Credit Python Scoring Library and Workflow
# Copyright © 2017-2020, Pavel Sůva, Marek Teller, Martin Kotek, Jan Zeller,
# Marek Mukenšnabl, Kirill Odintsov, Elena Kuchina, Jan Coufalík, Jan Hynek and
# Home Credit & Finance Bank Limited Liability Company, Moscow, Russia.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



__version__ = "0.9.0"
"""Scoring module contains submodules for:
    Data exploration
    Grouping
    Model Building
    etc.
"""
import os
from importlib import import_module

from packaging import version as version_parser

from .tools import DisplayBox

_MINIMUM_VERSIONS = {"qgrid": "1.0.3", "tqdm": "4.40.0", "numpy": "1.16.3"}
_EXACT_VERSIONS = {}
_REQUIREMENTS = [
    "numpy",
    "scipy",
    "pandas",
    "sklearn",
    "statsmodels",
    "matplotlib",
    "seaborn",
    "sqlalchemy",
    "cx_Oracle",
    "ipywidgets",
    "packaging",
    "qgrid",
    "tqdm",
    "xgboost",
    "lightgbm",
    "shap",
    "hyperopt",
    "docx",
]


def check_version(PSW_version, list_versions=False, list_versions_noprint=False):
    """Checks if scoring library is the same version as PSW notebook and for required versions of Pandas and Qgrid

    Compares version passed in argument to value of __version__ set in this module.
    Versions are compared as strings.

    Prints warning if versions don"t match or needed libraries are obsolete.

    Args:
        PSW_version (str): version of PSW
        list_versions (bool, optional): if True, prints module and its version, base on REQUIREMENTS list of modules. Defaults to False.
        list_versions_noprint (bool, optional): if True, will return a list of {module, version} for doctools. Defaults to False.
    """
    if PSW_version != __version__:
        DisplayBox.yellow(
            f"<br />/scoring version: {__version__} <br />PSW version:     {PSW_version} <br /><br /> Scoring was imported from: <br /> {os.path.abspath(__file__)}",
            title="WARNING: Your Notebook and scoring library are different versions!",
        )

    for module, targer_version in _MINIMUM_VERSIONS.items():
        version = import_module(module).__version__
        if version_parser.parse(version) < version_parser.parse(targer_version):
            DisplayBox.red(
                f"<br />Your {module} module is version {version}. PSW requires at least {targer_version}",
                title="WARNING",
            )

    for module, targer_version in _EXACT_VERSIONS.items():
        version = import_module(module).__version__
        if version_parser.parse(version) != version_parser.parse(targer_version):
            DisplayBox.red(
                f"<br />Your {module} module is version {version}. PSW requires exactly {targer_version}",
                title="WARNING",
            )

    if list_versions:
        print(f"Module               Version")
        for module in _REQUIREMENTS:
            try:
                version = import_module(module).__version__
            except:
                version = "Not found"
            print(f"{module:<20} {version}")

    if list_versions_noprint:
        the_list = []
        for module in _REQUIREMENTS:
            try:
                version = import_module(module).__version__
            except:
                version = "Not found"
            the_list.append({'name': module, 'version': version})
        return the_list