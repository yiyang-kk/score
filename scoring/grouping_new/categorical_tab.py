from copy import deepcopy
from itertools import compress

import ipywidgets as widgets
import numpy as np
import pandas as pd
import qgrid
from ipywidgets import (
    Accordion,
    BoundedFloatText,
    Button,
    Checkbox,
    Dropdown,
    GridspecLayout,
    HBox,
    IntSlider,
    IntText,
    Label,
    Layout,
    Output,
    Tab,
    Text,
    VBox,
)

from scoring.grouping import _event_rates, woe_scalar


class CategoricalTab:
    def __init__(self, data, bins_data, weight=None, parent=None):
        self._validate_input(data, weight)
        self.bins_data = deepcopy(bins_data)
        if "manual_woe" not in self.bins_data:
            self.bins_data["manual_woe"] = np.full((len(self.bins_data["woes"])), np.NaN)
        self.parent = parent
        self.create_widget()

    def _validate_input(self, data, weight=None):

        if data.shape[1] != 3:
            raise ValueError("Invalid `data` shape")
        if weight is None:
            self.data = deepcopy(data)
            self.data["WEIGHT"] = 1.0
        else:
            if not isinstance(weight, pd.Series):
                raise ValueError("Invalid `weight` type")

            self.data = pd.concat([data, weight], axis=1)

        self.col_pred = self.data.columns[0]
        self.col_target = self.data.columns[1]
        self.col_time = self.data.columns[2]
        self.col_weight = self.data.columns[3]

    def calculate_table(self):
        """
        Takes pd.df with 3 cols [pred, target, weight]
        """
        data = deepcopy(self.data)
        groups = deepcopy(self.bins_data["bins"])
        # quickfix ... use np.nan instead of ordinary nan. must be fixed systematically later!
        groups = {}
        for k, v in self.bins_data["bins"].items():
            if k != k:
                groups[np.nan] = v
            else:
                groups[k] = v

        col, tar, w = self.col_pred, self.col_target, self.col_weight

        data[col].cat.add_categories("nan", inplace=True)
        data[col].fillna("nan", inplace=True)
        data[tar] = data[tar] * data[w]

        result = data.groupby(col).sum()
        result.index = result.index.astype(object)
        result.rename(index={"nan": np.NaN}, inplace=True)
        result["share"] = result[w] / result[w].sum()
        result["def_rate"] = result[tar] / result[w]
        total_good, total_bad = result[w].sum() - result[tar].sum(), result[tar].sum()
        result["woe"] = result.apply(
            lambda row: woe_scalar(
                row[w] - row[tar], row[tar], total_good, total_bad, smoothing_coef=self.parent.woe_smooth_coef
            ),
            axis=1,
        )
        result["group"] = result.apply(lambda row: groups[row.name], axis=1)
        return result[["WEIGHT", "share", "def_rate", "woe", "group"]].rename(columns={"WEIGHT": "cnt"})

    def calculate_grouped(self):
        """
        Takes pd.df with 3 cols [pred, target, weight]
        """
        data = deepcopy(self.data)
        groups = deepcopy(self.bins_data["bins"])
        col, tar, w = self.col_pred, self.col_target, self.col_weight

        data[col].cat.add_categories("nan", inplace=True)
        data[col] = data[col].replace(groups)
        data[tar] = data[tar] * data[w]
        result = data.groupby(col).sum()
        for _, group in groups.items():
            if group not in result.index:
                result = result.append(pd.Series(name=group, data={"DEF": 0, "WEIGHT": 0}))

        result["share"] = result[w] / result[w].sum()
        result["def_rate"] = result[tar] / result[w]
        total_good, total_bad = result[w].sum() - result[tar].sum(), result[tar].sum()
        result["woe"] = result.apply(
            lambda row: woe_scalar(
                row[w] - row[tar], row[tar], total_good, total_bad, smoothing_coef=self.parent.woe_smooth_coef
            ),
            axis=1,
        )
        result["man_woe"] = self.bins_data["manual_woe"]
        result.index.name = "Group"

        return result[["WEIGHT", "share", "def_rate", "woe", "man_woe"]].rename(columns={"WEIGHT": "cnt"})

    def calculate_histogram(self):
        results = deepcopy(self.t1)
        results.sort_values(by="group", inplace=True)
        results.rename(columns={"cnt": "pop"}, inplace=True)
        results = results[["group", "pop", "def_rate"]]

        return results

    def calculate_histogram_grouped(self):
        results = deepcopy(self.t2)
        results.rename(columns={"cnt": "pop"}, inplace=True)
        results = results[["pop", "def_rate"]]

        return results

    def calculate_stability(self):
        df = self.data[self.col_pred].replace(to_replace=self.bins_data["bins"])

        df.name = "group"
        df = pd.concat([df, self.data[[self.col_target, self.col_time, self.col_weight]]], axis=1)

        grouped = df.groupby(["group", self.col_time])[self.col_target, self.col_weight].sum()
        grouped_sum = grouped.reset_index().groupby(self.col_time)[self.col_weight].sum()
        df = grouped.join(grouped_sum, how="inner", rsuffix="_all").reset_index()
        df["def_rate"] = df[self.col_target] / df[self.col_weight]
        df["pop_share"] = df[self.col_weight] / df[f"{self.col_weight}_all"]
        def_rate = df.pivot(index=self.col_time, columns="group", values="def_rate")
        pop_share = df.pivot(index=self.col_time, columns="group", values="pop_share")

        return def_rate, pop_share

    def create_widget(self):
        t1 = self.calculate_table()
        t2 = self.calculate_grouped()

        self.w1 = qgrid.show_grid(
            data_frame=t1,
            grid_options={"filterable": False, "sortable": True},
            column_options={"editable": False},
            column_definitions={"group": {"editable": True}},
        )
        self.w1.on("cell_edited", self._on_group_change)

        self.w2 = qgrid.show_grid(
            data_frame=t2,
            grid_options={"filterable": False, "sortable": False},
            column_options={"editable": False},
            column_definitions={"man_woe": {"editable": True}},
        )
        self.w2.on("cell_edited", self._on_manual_woe_change)

        style = {"description_width": "initial"}
        self.w3 = widgets.FloatText(value=self.bins_data["unknown_woe"], description="Unknown WOE:", style=style)
        self.w3.observe(self._on_unknown_woe_change)
        self.widget = widgets.VBox([self.w1, self.w2, self.w3])

    @property
    def t1(self):
        return self.w1.get_changed_df()

    @property
    def t2(self):
        return self.w2.get_changed_df()

    def _fix_groups(self):
        groups = self.w1.get_changed_df()["group"]
        old_manual_woe = self.bins_data["manual_woe"]

        unique_groups = sorted(set(groups.values))
        transform_values = {old: new for new, old in enumerate(unique_groups)}

        for index, group_value in groups.items():
            if group_value != transform_values[group_value]:
                self.w1.edit_cell(index, "group", transform_values[group_value])

        new_manual_woe = np.full(len(unique_groups), np.NaN)
        for old, new in transform_values.items():
            if old < len(old_manual_woe):
                new_manual_woe[new] = old_manual_woe[old]

        self.bins_data["manual_woe"] = new_manual_woe

    def _update_bins(self):
        for value, group in self.w1.get_changed_df()["group"].items():
            self.bins_data["bins"][value] = group

    def _update_table2(self):
        self.w2.df = self.calculate_grouped()

    def _update_group_woe(self):
        self.bins_data["woes"] = self.t2["woe"].values
        self.bins_data["bins"] = self.t1["group"].to_dict()

    def update(self):
        self.create_widget()
        self.parent.update()

    def _on_group_change(self, event, current_widget):
        if event["new"] < 0:
            self.w1.edit_cell(event["index"], event["column"], event["old"])
        if event["source"] == "gui":
            self._fix_groups()
            self._update_bins()
            self._update_table2()
            self._update_group_woe()
            self.parent.update()

    def _on_unknown_woe_change(self, event):
        if event["name"] == "value":
            self.bins_data["unknown_woe"] = event["new"]

    def _on_manual_woe_change(self, event, current_widget):
        self.bins_data["manual_woe"] = current_widget.get_changed_df()[event["column"]].values
