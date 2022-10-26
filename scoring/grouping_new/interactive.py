from copy import deepcopy
from itertools import compress
import os

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
from IPython.display import display
from tqdm.notebook import tqdm


from scoring.grouping import _event_rates, woe_scalar, Grouping

from .categorical_tab import CategoricalTab
from .charts_plt import CategoricalCharts, NumericalCharts, ChartBox
from .utils import ChartsSelector, ToolsTab, AutoGroupingTab, BottomButtons
from .numerical_tab import NumericalTab


class Dummy:
    def __init__(self):
        pass

    def update(self, **kwargs):
        print(kwargs)


class NewGrouping:
    def __init__(self, group_count=5, min_samples=100, min_samples_cat=100, woe_smooth_coef=0.001):
        self.group_count = group_count
        self.min_samples = min_samples
        self.min_samples_cat = min_samples_cat
        self.woe_smooth_coef = woe_smooth_coef
        self.bins_data = dict()
        self.fitted = False

    # exposing bins_data for legacy methods in PSW
    @property
    def bins_data_(self):
        return self.bins_data
    
    def _split_columns_by_dtype(self, data):
        cols_num, cols_cat = [], []
        for name, col in data.iteritems():
            if pd.api.types.is_numeric_dtype(col.dtype):
                cols_num.append(name)
            else:
                cols_cat.append(name)

        return cols_num, cols_cat

    def init_underlying_grouping(self, data):
        cols_num, cols_cat = self._split_columns_by_dtype(data)

        self.grouping = Grouping(
            columns=cols_num,
            cat_columns=cols_cat,
            group_count=self.group_count,
            min_samples=self.min_samples,
            min_samples_cat=self.min_samples_cat,
            woe_smooth_coef=self.woe_smooth_coef,
        )


    def fit(self, data, target, weight=None, progress_bar=False, category_limit=100000):
        ## add more input validation
        self.init_underlying_grouping(data)

        self.grouping.fit(X=data, y=target, w=weight, progress_bar=progress_bar, category_limit=category_limit)
        self.fitted = True
        self.bins_data = self.grouping.bins_data_

    def _check_fitted(self):
        if not self.fitted:
            raise Exception("Model was not yet fitted.")

    def get_dummy_names(self, *args, **kwargs):
        self._check_fitted()

        return self.grouping.get_dummy_names(*args, **kwargs)

    def export_dictionary(self, *args, **kwargs):
        self._check_fitted()

        return self.grouping.export_dictionary(*args, **kwargs)

    def transform(self, *args, **kwargs):
        self._check_fitted()
        self.grouping.bins_data_ = self.bins_data

        return self.grouping.transform(*args, **kwargs)

    def save(self, filename):
        self._check_fitted()

        return self.grouping.save(filename)

    def load(self, file, show_loaded=False):
        self.grouping.load(filename=file)
        if show_loaded:
            print("Loaded groupings for predictors:")
            for pred in self.grouping.bins_data_:
                print(f"{pred}")
        self.bins_data.update(self.grouping.bins_data_)
        self.fitted = True

    def plot_bins(self, *args, **kwargs):
        self._check_fitted()

        return self.grouping.plot_bins(*args, **kwargs)

    def export_as_sql(self, *args, **kwargs):
        self._check_fitted()

        return self.grouping.export_as_sql(*args, **kwargs)

    def export_pictures(self, data, target, time_column, export_path, weight=None, use_tqdm=True):

        cols_num, cols_cat = self._split_columns_by_dtype(data)
        data = pd.concat([data, target, time_column, weight], axis=1)

        os.makedirs(export_path, exist_ok=True)

        if not hasattr(self, '_categorical_charts'):
            self._categorical_charts = CategoricalCharts()

        if not hasattr(self, '_numerical_charts'):
            self._numerical_charts = NumericalCharts()

        if use_tqdm:
            iterator = tqdm(cols_num + cols_cat, total=len(cols_num + cols_cat), leave=True, unit="cols")
        else:
            iterator = cols_num + cols_cat

        for column in iterator:
            if column in cols_num:
                tmp_tab = NumericalTab(
                    data[[column, target.name, time_column.name]],
                    bins_data=self.bins_data[column],
                    weight=weight,
                    parent=self,
                )
                hist_ew = tmp_tab.calculate_ew()
                hist_ed = tmp_tab.calculate_ed()
                hist_grouped = tmp_tab.calculate_grouped()
                stab_pop, stab_br = tmp_tab.calculate_stability()

                self._numerical_charts.draw_charts(
                    data_stab_pop=stab_pop,
                    data_stab_br=stab_br,
                    data_hist_grouped=hist_grouped,
                    data_hist_ed=hist_ed,
                    data_hist_ew=hist_ew,
                )

                self._numerical_charts.fig.savefig(os.path.join(export_path, f"{column}.png"))

            elif column in cols_cat:
                tmp_tab = CategoricalTab(
                    data[[column, target.name, time_column.name]],
                    bins_data=self.bins_data[column],
                    weight=weight,
                    parent=self,
                )

                hist = tmp_tab.calculate_histogram()
                hist_group = tmp_tab.calculate_histogram_grouped()

                stab_pop, stab_br = tmp_tab.calculate_stability()

                self._categorical_charts.draw_charts(
                    data_histogram=hist,
                    data_histogram_grouped=hist_group,
                    data_stab_pop=stab_pop,
                    data_stab_br=stab_br,
                )
                self._categorical_charts.fig.savefig(os.path.join(export_path, f"{column}.png"))

    def interactive(self, data, target, time_column, weight=None):
        self._check_fitted()

        cols_num, cols_cat = self._split_columns_by_dtype(data)
        data = pd.concat([data, target, time_column, weight], axis=1)

        for predictor, bins_data in self.bins_data.items():
            if "manual_woe" not in bins_data.keys():
                num_woes = len(bins_data["woes"])
                self.bins_data[predictor]["manual_woe"] = np.full(num_woes, np.NaN)
            
            if bins_data["dtype"] == "category":
                if np.nan not in bins_data["bins"].keys():
                    bins_data["bins"][np.nan] = len(bins_data["woes"])
                    bins_data["woes"] = np.concatenate((bins_data["woes"], [0]))
                    bins_data["manual_woe"] = np.concatenate((bins_data["manual_woe"], [np.nan]))  

        self.chartbox = ChartBox()
        self.active_tab = None

        self._categorical_charts = CategoricalCharts()
        self._numerical_charts = NumericalCharts()

        def _export_pictures(btn):

            self.export_pictures(
                data=data[cols_num+cols_cat],
                target=target,
                time_column=time_column,
                export_path=self.tools_tab.export_path.value,
                weight=weight,
                )

        self._export_pictures = _export_pictures

        def _update():

            stability_charts = self.charts_selector.selected_charts[-1]

            if isinstance(self.active_tab, NumericalTab):
                hist_ew = self.active_tab.calculate_ew()
                hist_ed = self.active_tab.calculate_ed()
                hist_grouped = self.active_tab.calculate_grouped()
                if stability_charts:
                    stab_pop, stab_br = self.active_tab.calculate_stability()
                else:
                    stab_pop, stab_br = None, None

                self._numerical_charts.draw_charts(
                    data_stab_pop=stab_pop,
                    data_stab_br=stab_br,
                    data_hist_grouped=hist_grouped,
                    data_hist_ed=hist_ed,
                    data_hist_ew=hist_ew,
                )

                self.chartbox.show(self._numerical_charts.fig)

            elif isinstance(self.active_tab, CategoricalTab):
                hist = self.active_tab.calculate_histogram()
                hist_group = self.active_tab.calculate_histogram_grouped()
                
                if stability_charts:
                    stab_pop, stab_br = self.active_tab.calculate_stability()
                else:
                    stab_pop, stab_br = None, None

                self._categorical_charts.draw_charts(
                    data_histogram=hist, data_histogram_grouped=hist_group, data_stab_pop=stab_pop, data_stab_br=stab_br
                )

                self.chartbox.show(self._categorical_charts.fig)

        self.update = _update

        def _save_tab_changes():
            if self.active_tab:
                self.bins_data[self.active_tab.col_pred] = deepcopy(self.active_tab.bins_data)

        self._save_tab_changes = _save_tab_changes

        def _on_change_tab(change):
            if change["name"] == "value":
                
                if self.active_tab:
                    self._save_tab_changes()

                selected = change["new"]
                if selected in cols_num:
                    self.active_tab = NumericalTab(
                        data[[selected, target.name, time_column.name]],
                        bins_data=self.bins_data[selected],
                        weight=weight,
                        parent=self,
                    )

                elif selected in cols_cat:
                    self.active_tab = CategoricalTab(
                        data[[selected, target.name, time_column.name]],
                        bins_data=self.bins_data[selected],
                        weight=weight,
                        parent=self,
                    )

                _update()
                # self.widget.children = tuple([self.chartbox.widget] + list(self.widget.children[1:]))
                self.tab.children = tuple([self.active_tab.widget] + list(self.tab.children[1:]))

        def _automatic_grouping_one_column_from_gui(btn):
            self.grouping.group_count = self.autogrouping_tab.w_group_count_text.value
            self.grouping.min_samples = self.autogrouping_tab.w_min_samples_num_text.value
            self.grouping.min_samples_cat = self.autogrouping_tab.w_min_samples_num_text.value

            x = data[self.active_tab.col_pred]
            y = data[self.active_tab.col_target]
            if self.active_tab.col_weight:
                w = data[self.active_tab.col_weight]
            else:
                w = None
            
            self.grouping._auto_grouping(x ,y, w=w)
            self.active_tab.bins_data = self.grouping.bins_data_[self.active_tab.col_pred]

            num_woes = len(self.active_tab.bins_data["woes"])
            self.active_tab.bins_data["manual_woe"] = np.full(num_woes, np.NaN)

            self.active_tab.update()

        self._automatic_grouping_one_column_from_gui = _automatic_grouping_one_column_from_gui


        def _save_manual_from_gui(btn):
            self._save_tab_changes()
            self.save(filename=self.tools_tab.save_path.value)

        self._save_manual_from_gui = _save_manual_from_gui

        ##########################
        ## end of function defs ##    
        ##########################
        

        self.var_select = widgets.Dropdown(options=cols_num + cols_cat)
        self.var_select.observe(_on_change_tab)

        self.placeholder = widgets.Label(value="Select predictor to start")

        self.tools_tab = ToolsTab(parent=self)
        self.autogrouping_tab = AutoGroupingTab(parent=self)
        self.charts_selector = ChartsSelector(parent=self)
        self.bottom_buttons = BottomButtons(parent=self)

        self.tab = widgets.Tab(
            [self.placeholder, self.autogrouping_tab.widget, self.charts_selector.widget, self.tools_tab.widget]
        )
        for index, name in enumerate(["Manual", "Auto", "Settings", "Tools"]):
            self.tab.set_title(index, name)

        self.widget = widgets.VBox([self.chartbox.widget, self.var_select, self.tab, self.bottom_buttons.widget])

        display(self.widget)
