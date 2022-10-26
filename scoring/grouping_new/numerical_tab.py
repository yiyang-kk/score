from copy import deepcopy
from itertools import compress
from collections import namedtuple
from sklearn.utils import column_or_1d


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


def weighted_percentile(a, q, w=None):
    """
    Calculates percentiles associated with a (possibly weighted) array

    Args:
        a (array-like):
            The input array from which to calculate percents
        q (array-like):
            The percentiles to calculate (0.0 - 100.0)
        w (array-like, optional):
            The weights to assign to values of a.  Equal weighting if None
            is specified. Defaults to None.

    Returns:
        np.array: The values associated with the specified percentiles.  
    """
    # validate a and w inputs

    # Standardize and sort based on values in a
    a = column_or_1d(a)
    q = np.array(q) / 100.0
    if w is None:
        w = np.ones(a.size)
    else:
        w = column_or_1d(w)

    idx = np.argsort(a)
    a_sort = a[idx]
    w_sort = w[idx]

    # Get the cumulative sum of weights
    ecdf = np.cumsum(w_sort)

    # Find the percentile index positions associated with the percentiles
    p = q * (w.sum() - 1)

    # Find the bounding indices (both low and high)
    idx_low = np.searchsorted(ecdf, p, side="right")
    idx_high = np.searchsorted(ecdf, p + 1, side="right")
    idx_high[idx_high > ecdf.size - 1] = ecdf.size - 1

    # Calculate the weights
    weights_high = p - np.floor(p)
    weights_low = 1.0 - weights_high

    # Extract the low/high indexes and multiply by the corresponding weights
    x1 = np.take(a_sort, idx_low) * weights_low
    x2 = np.take(a_sort, idx_high) * weights_high

    # Return the average
    return np.add(x1, x2)


class NumericalBins:
    def __init__(self, bins, manual_woe=None, parent=None):
        self._bins = [np.float(edge) for edge in bins]
        self.parent = parent
        if manual_woe is None:
            self._manual_woe = [np.NaN for _ in range(len(bins) + 1)]
        else:
            # check length
            if len(manual_woe) - len(bins) != 1:
                raise ValueError("Lenght of `manual_woe` should be 1 more than length of `bins`")
            self._manual_woe = manual_woe
        self._create_widget()
        self._input_validation = True

    def _on_add(self, element):
        index = element.index
        if index == 0:
            value = self._bins[index] - 1
        elif index == len(self._bins):
            value = self._bins[index - 1] + 1
        else:
            value = (self._bins[index - 1] + self._bins[index]) / 2

        self._add(index, value)
        self._bins.insert(index, value)
        self._manual_woe.insert(index, self._manual_woe[index])
        #         print("new bins:", self._bins)
        #         print("man woes:", self._manual_woe)
        self._call_parent()

    def _on_remove(self, element):
        index = element.parent.index
        self._remove(index)
        self._bins.remove(self._bins[index])
        self._manual_woe.pop(index)
        #         print("new bins:", self._bins, index)
        #         print("man woes:", self._manual_woe)
        self._call_parent()

    def _on_change_edge(self, change):
        if change["name"] == "value" and self._input_validation:
            value_new = change["new"]
            value_old = change["old"]
            owner = change["owner"]

            new_bins = self._bins.copy()
            new_bins[owner.parent.index] = value_new

            if self._check_monotony(new_bins):
                self._bins[owner.parent.index] = value_new
                self._call_parent()
            else:
                owner.value = value_old

    def _button_plus(self):

        btn = widgets.Button(
            description="+",
            layout=widgets.Layout(width="30px", min_width="30px"),
            button_style="success",
            tooltip="Add split point",
        )
        btn.on_click(self._on_add)
        return btn

    def _button_remove(self):
        btn = widgets.Button(
            description="-",
            layout=widgets.Layout(width="30px", height="30px", align_self="center"),
            button_style="danger",
            tooltip="Remove split point",
        )
        btn.on_click(self._on_remove)
        return btn

    def _text_edge(self, value):
        floattext = widgets.FloatText(layout=widgets.Layout(width="80px"), value=value)
        floattext.observe(self._on_change_edge)
        return floattext

    def _vbox_edge(self, value):
        vbox = widgets.VBox([self._text_edge(value), self._button_remove()], layout=widgets.Layout(min_width="85px"))
        for child in vbox.children:
            child.parent = vbox
        return vbox

    def _text_inf(self, inf):
        return widgets.Text(inf, layout=widgets.Layout(width="60px"), disabled=True)

    def _reindex(self):
        indexes = [i // 2 for i in range(len(self.widget.children))]
        for i, child in zip(indexes, self.widget.children[1:-1]):
            child.index = i

    def _remove(self, index):
        cut_start = 1 + 2 * index
        cut_end = 3 + 2 * index
        self.widget.children = self.widget.children[:cut_start] + self.widget.children[cut_end:]
        self._reindex()

    def _add(self, index, value):
        insert_index = 1 + 2 * index
        self.widget.children = (
            self.widget.children[:insert_index]
            + tuple([self._button_plus(), self._vbox_edge(value)])
            + self.widget.children[insert_index:]
        )
        self._reindex()

    def _check_monotony(self, bins):
        for i in range(len(bins) - 1):
            if not bins[i] < bins[i + 1]:
                return False
        return True

    def _call_parent(self):
        if self.parent:
            self.parent.update(new_bins=self.get_bins())

    def _create_widget(self):
        elements = [self._text_inf("-inf")]
        for edge in self._bins:
            elements.extend([self._button_plus(), self._vbox_edge(edge)])

        elements.extend([self._button_plus(), self._text_inf("+inf")])

        self.widget = widgets.HBox(elements)
        self._reindex()

    def _set_validation(self, toggle=True):
        self._input_validation = toggle
        # print(f"input validation se to {toggle}")

    def update_bins(self, new_bins):
        new_bins = list(new_bins)
        if not self._check_monotony(new_bins):
            raise ValueError("Bins must be monotonic")
        self._set_validation(False)
        while len(new_bins) > len(self._bins):
            self._add(index=0, value=0)
            self._bins.append(0)
            self._manual_woe.append(np.NaN)

        while len(new_bins) < len(self._bins):
            self._remove(index=0)
            del self._bins[-1]
            del self._manual_woe[-1]

        self._bins = deepcopy(new_bins)
        for w in self.widget.children:
            if isinstance(w, widgets.VBox):
                w.children[0].value = new_bins.pop(0)
        self._set_validation(True)

    def update_manual_woe(self, manual_woe):
        if len(manual_woe) != len(self._manual_woe):
            raise ValueError(f"`manual_woe` should have length {len(self._manual_woe)}")

        self._manual_woe = list(manual_woe)

    def get_bins(self):
        return {"bin_edges": np.array([-np.inf] + self._bins + [np.inf]), "manual_woe": np.array(self._manual_woe)}


class MergeNanSelector:
    def __init__(self, interval_dict):
        self.group_intervals = self._process_intervals(interval_dict)
        self.selected_group = None
        dropdown_list = list(self.group_intervals.keys())

        self.widget = widgets.Dropdown(
            options=dropdown_list,
            description="Merge NaNs with bin number:",
            disabled=False,
            style={"description_width": "initial"},
        )
        self.widget.observe(self._on_change)

    def _process_intervals(self, interval_dict):
        group_intervals = {"None": -1}
        group_intervals.update({f"[{g}]  {i}": g for g, i in interval_dict.items() if i != "NaN"})

        return group_intervals

    def update(self, interval_dict):
        old_index = self.group_intervals[self.widget.value]
        self.group_intervals = self._process_intervals(interval_dict)
        self.widget.options = list(self.group_intervals.keys())
        self.widget.value = self.widget.options[min(old_index + 1, len(self.widget.options) - 1)]

    def _on_change(self, change):
        if change["name"] == "value":
            if change["new"] == "None":
                self.selected_group = None
            else:
                self.selected_group = self.group_intervals[change["new"]]


class RoundEdges:
    """
    Creates widgets that calls `rounding_function` with argument of int `value` when clicked
    """

    def __init__(self, rounding_function):
        self.function = rounding_function
        self.button = Button(description="Round bin edges to:", layout=Layout(width="200px", height="30"))
        self.button.on_click(self._on_button)

        self.value = IntText(value=2, layout=Layout(width="40px", height="30%"))
        self.widget = widgets.HBox([self.button, self.value])

    def _on_button(self, widget):
        self.function(self.value.value)
        return
        # self.function(self.value.value)


Data_hist = namedtuple("data_ed", ["x_midpoints", "x_bin_edges", "x_vlines", "y_counts", "y_badrates", "y_groups"])

HistogramData = namedtuple("HistogramData", ["bins", "counts", "badrates"])


class NumericalTab:
    def __init__(self, data, bins_data, weight=None, parent=None):
        """
        Takes data[["Numerical_1", col_target, col_month, col_weight]]
        """
        self.bins_data = deepcopy(bins_data)
        if "manual_woe" not in self.bins_data:
            self.bins_data["manual_woe"] = np.full(len(self.bins_data["woes"]), np.NaN)
        self._validate_input(data, weight)
        self.parent = parent

        self.values = data[self.col_pred]

        self.calculate_table()
        self._create_widget()

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

    def update(self, bins=None, **kwargs):
        if "new_bins" in kwargs.keys():
            self.bins_data["bins"] = kwargs["new_bins"]["bin_edges"]
            self.bins_data["manual_woe"] = kwargs["new_bins"]["manual_woe"]

        self.calculate_table()
        self.bins_data["woes"] = self.table["woe"].values

        self.w_table.df = self.table
        # self.w_merge_nan_selector.update(self._get_group_intervals())

        # grouped_table = self.calculate_grouped()
        # def_rate, pop_share = self.calculate_stability()
        self.parent.update()

    def calculate_table(self):
        groups = pd.cut(self.values[self.values.notnull()], bins=self.bins_data["bins"], right=False).rename("group")
        groups = pd.concat([groups, self.data], axis=1)
        groups[self.col_target] = groups[self.col_target] * groups[self.col_weight]
        grouped = groups.groupby("group").sum()
        grouped.index = grouped.index.astype("object")
        null_row = groups[groups["group"].isnull()].sum()
        null_row.name = "NaN"
        df = grouped.append(null_row)[[self.col_target, self.col_weight]]
        df.rename(columns={self.col_weight: "cnt"}, inplace=True)
        df["share"] = df["cnt"] / df["cnt"].sum()
        df["def_rate"] = df[self.col_target] / df["cnt"]
        total_bad = df[self.col_target].sum()
        total_good = df["cnt"].sum() - total_bad
        df["woe"] = df.apply(
            lambda row: woe_scalar(
                row["cnt"] - row[self.col_target],
                row[self.col_target],
                total_good,
                total_bad,
                smoothing_coef=self.parent.woe_smooth_coef,
            ),
            axis=1,
        )

        df["man_woe"] = np.concatenate((self.bins_data["manual_woe"], [self.bins_data["nan_woe"]]))
        df = df[["cnt", "share", "def_rate", "woe", "man_woe"]]
        self.table = df.reset_index()

    def calculate_stability(self):
        nan_merge = -1
        df = pd.cut(self.values, bins=self.bins_data["bins"], right=False, labels=False)
        df.name = "group"
        df.replace({np.NaN: nan_merge}, inplace=True)
        df = pd.concat([df, self.data[[self.col_target, self.col_time, self.col_weight]]], axis=1)

        grouped = df.groupby(["group", self.col_time])[self.col_target, self.col_weight].sum()
        grouped_sum = grouped.reset_index().groupby(self.col_time)[self.col_weight].sum()
        df = grouped.join(grouped_sum, how="inner", rsuffix="_all").reset_index()
        df["def_rate"] = df[self.col_target] / df[self.col_weight]
        df["pop_share"] = df[self.col_weight] / df[f"{self.col_weight}_all"]
        df["group"] = df["group"].astype(int)
        def_rate = df.pivot(index=self.col_time, columns="group", values="def_rate")
        pop_share = df.pivot(index=self.col_time, columns="group", values="pop_share")

        return def_rate, pop_share

    def calculate_grouped(self):
        nan_merge = -1
        df = pd.cut(self.values, bins=self.bins_data["bins"], right=False, labels=False)
        df.name = "group"
        df.replace({np.NaN: nan_merge}, inplace=True)
        df = pd.concat([df, self.data[[self.col_target, self.col_time, self.col_weight]]], axis=1)
        df[self.col_target] = df[self.col_target] * df[self.col_weight]
        df = df.groupby(["group"])[self.col_target, self.col_weight].sum()
        df["bad_rate"] = df[self.col_target] / df[self.col_weight]
        df.rename(columns={self.col_weight: "pop"}, inplace=True)

        # if nan group, put it at the end for plots
        if -1 in df.index:
            df = df.iloc[1:].append(df.loc[-1])
        histogram_data = HistogramData(
            bins=self.bins_data["bins"][1:-1], counts=df["pop"].values, badrates=df["bad_rate"].values
        )
        # return df[["bad_rate", "pop"]]
        return histogram_data

    def _on_manual_woe_change(self, event, current_widget):
        self.bins_data["manual_woe"] = current_widget.get_changed_df()["man_woe"].values

    def create_table(self, table):
        table = table.copy()

        col_opts = {"editable": False}
        col_defs = {"man_woe": {"editable": True}}

        qgrid_table = qgrid.show_grid(table, column_options=col_opts, column_definitions=col_defs)
        qgrid_table.on(names="cell_edited", handler=self._on_manual_woe_change)
        return qgrid_table

    def _get_group_intervals(self):
        return {g: str(i) for g, i in self.table["group"].to_dict().items()}

    def _round_edges(self, n):
        self.w_bins._set_validation(False)
        bins = self.w_bins.get_bins()["bin_edges"][1:-1]

        bins = np.round(bins, n)

        if self.w_bins._check_monotony(bins):
            self.w_bins.update_bins(bins)
        else:
            print("Bins are not monotonic")
        self.w_bins._set_validation(True)
        self.w_bins._call_parent()

    def _create_widget(self):
        bins = self.bins_data["bins"][1:-1]
        elements = []

        self.w_rounding = RoundEdges(self._round_edges)
        self.w_bins = NumericalBins(bins, parent=self)
        # self.w_merge_nan_selector = MergeNanSelector(self._get_group_intervals())
        self.w_table = self.create_table(self.table)

        elements.append(self.w_rounding.widget)
        elements.append(self.w_bins.widget)
        elements.append(self.w_table)
        # elements.append(self.w_merge_nan_selector.widget)
        self.widget = VBox(elements)

    def _find_midpoints(self, bins, ed_width):
        midpoints = []
        half_width = ed_width / 2
        # interate over intervals
        for a, b in list(zip(bins, bins[1:])):
            if np.isinf(a):
                midpoints.append(b - half_width)
            elif np.isinf(b):
                midpoints.append(a + half_width)
            else:
                midpoints.append((a + b) / 2)

        return midpoints

    def calculate_ew(self):
        bin_count = 10
        bins = self.bins_data["bins"][1:-1]
        x = self.data[self.col_pred]
        y = self.data[self.col_target]
        w = self.data[self.col_weight]

        ew_bins = np.linspace(x.min(), x.max(), bin_count + 1)

        ew_bin_width = ew_bins[1] - ew_bins[0]

        all_bins = np.sort(np.unique(np.hstack((ew_bins, bins))))
        # x2 for histogram counting, what does this do?
        midpoints = self._find_midpoints(all_bins, ew_bin_width)
        groups = np.digitize(midpoints, bins)

        counts, _ = np.histogram(x, bins=all_bins, weights=w.astype(np.float))

        event_rate = _event_rates(x, y, bins=all_bins, w=w)
        vlines = bins
        data_nan = self.data[x.isnull()]

        if data_nan.shape[0] > 1:
            count_nan = data_nan[self.col_weight].sum()
            event_count_nan = data_nan[data_nan[self.col_target] == 1][self.col_weight].sum()
            event_rate_nan = event_count_nan / count_nan
            # count_nan = data_nan.shape[0]
            # event_rate_nan = data_nan[self.col_target].sum() / count_nan
        else:
            count_nan = None
            event_rate_nan = None

        # return Data_hist(
        #     x_midpoints=midpoints,
        #     x_bin_edges=all_bins,
        #     x_vlines=vlines,
        #     y_counts=counts,
        #     y_badrates=event_rate,
        #     y_groups=groups,
        # )

        return {
            "bins": bins,
            "ew_bins": ew_bins,
            "counts": counts,
            "count_nan": count_nan,
            "bad_rates": event_rate,
            "bad_rate_nan": event_rate_nan,
        }

    def calculate_ed(self):
        bin_count = 10
        bins = self.bins_data["bins"][1:-1]
        x = self.data[self.col_pred]
        y = self.data[self.col_target]
        w = self.data[self.col_weight]

        # ed_bins = np.unique(
        #     np.percentile(x[x.notnull()], np.linspace(0, 100, bin_count + 1), interpolation="lower")
        # )  # equi-depth bins (without NaN bin)
        ed_bins = weighted_percentile(x[x.notnull()], np.linspace(0, 100, bin_count + 1), w=w[x.notnull()])
        ed_bin_width = ed_bins[1] - ed_bins[0]  # equi-depth bin width
        # bins + ed_bins (with NaN bin)
        all_bins = np.sort(np.unique(np.hstack((ed_bins, bins))))
        midpoints = self._find_midpoints(all_bins, ed_bin_width)
        groups = np.digitize(midpoints, bins)

        # x2 for histogram counting
        x2 = x.replace(all_bins[-1], (all_bins[-2] + all_bins[-1]) / 2)
        split_points = bins.copy()  # group borders

        counts, _ = np.histogram(x2, bins=all_bins, weights=w.astype(np.float))
        event_rate = _event_rates(x, y, bins=all_bins, w=w)

        data_nan = self.data[x.isnull()]
        if data_nan.shape[0] > 1:
            count_nan = data_nan[self.col_weight].sum()
            event_count_nan = data_nan[data_nan[self.col_target] == 1][self.col_weight].sum()
            event_rate_nan = event_count_nan / count_nan
        else:
            count_nan = None
            event_rate_nan = None

        #         chart = ChartDistribution(
        #             index=range(len(counts) - 1), bar_values=counts, line_values=event_rate, vlines_values=vlines
        #         )

        #         return chart.widget

        # return Data_hist(
        #     x_midpoints=midpoints,
        #     x_bin_edges=all_bins,
        #     x_vlines=vlines,
        #     y_counts=counts,
        #     y_badrates=event_rate,
        #     y_groups=groups,
        # )

        return {
            "bins": bins,
            "ed_bins": ed_bins,
            "counts": counts,
            "count_nan": count_nan,
            "bad_rates": event_rate,
            "bad_rate_nan": event_rate_nan,
        }

