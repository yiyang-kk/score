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

import pandas as pd
import numpy as np
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt


def calculate_gini_and_lift(
    indata,
    target_name,
    predict_name,
    base_name=None,
    weight_name=None,
    pct=10,
    sort_desc=True,
):
    """Function to calculate Gini, Cumulative lift and prepare data for drawing precise AUC and Lift curve for given prediction.
    
    Args:
        indata (pd.DataFrame): data frame with targets and predictions
        target_name (str): name of column with target (column of indata)
        predict_name (str): name of column with prediction (column of indata)
        base_name (str, optional): name of column with base (i.e. indicator whether target is observable for given row) (column of indata) (default: None)
        weight_name (str, optional): name of column with base (i.e. importance of given row) (column of indata) (default: None)
        pct (list of int, optional): number or list of numbers - levels for which Lift should be calculated for (default: 10)
        sort_desc (bool, optional): for AUC curve: whether the data should be sorted descending by prediction (otherwise it will be sorted ascending by prediciton) (default: True)
    
    Returns:
        float, [float], pd.DataFrame: gini, lift(s), dataframe which can be used to draw AUC curve and Lift curve
    """

    columns_to_be_copied = [target_name, predict_name]
    if base_name is not None:
        columns_to_be_copied.append(base_name)
    if weight_name is not None:
        columns_to_be_copied.append(weight_name)
    data = indata[columns_to_be_copied].copy()

    # base (telling whether observation should be used)
    if base_name is None:
        base_name = "b"
        data[base_name] = 1

    # weight (observation importance)
    if weight_name is None:
        weight_name = "w"
        data[weight_name] = 1

    # weight * base * target
    data["wbt"] = data[weight_name] * data[base_name] * data[target_name]
    # weight * base
    data["wb"] = data[weight_name] * data[base_name]

    # group data by score
    # TO DO fix this bug described in Issue #3
    # data = data[data[base_name]>0].groupby([predict_name])['wb','wbt'].sum()
    # hotfix below:
    d1 = data[data[base_name] > 0].groupby([predict_name])["wb"].sum()
    d2 = data[data[base_name] > 0].groupby([predict_name])["wbt"].sum()
    data = pd.concat([d1, d2], axis=1)

    # sort data ascending or descending by score
    if sort_desc == False:
        data.sort_index(inplace=True)
    else:
        data.sort_index(inplace=True, ascending=False)

    # weight * base * (1-target)
    data["wbtn"] = data["wb"] - data["wbt"]

    # cumulative characteristics
    data["cum_cnt"] = data["wb"].cumsum()
    data["cum_perc"] = 100 * data["cum_cnt"] / data["wb"].sum()
    data["cum_bad"] = data["wbt"].cumsum() / data["wbt"].sum()
    data["cum_good"] = data["wbtn"].cumsum() / data["wbtn"].sum()

    # calculate gini as integral from series of constant functions (height: cumulative bads, step: delta of cumulative goods)
    data["dx"] = data["cum_good"] - data["cum_good"].shift(+1)
    data.iloc[0, data.columns.get_loc("dx")] = data.iloc[
        0, data.columns.get_loc("cum_good")
    ]
    data["x"] = (data["cum_bad"] + data["cum_bad"].shift(+1)) / 2
    data.iloc[0, data.columns.get_loc("x")] = (
        data.iloc[0, data.columns.get_loc("cum_bad")] / 2
    )
    data["gini"] = data["x"] * data["dx"]
    gini = 2 * abs(data["gini"].sum() - 0.5)

    # calculate lift
    data["cum_bad_cnt"] = data["wbt"].cumsum()
    data["cum_lift"] = (data["cum_bad_cnt"] / data["cum_cnt"]) / (
        data["wbt"].sum() / data["wb"].sum()
    )

    if isinstance(pct, list):
        lift = []
        for p in pct:
            lift_tmp = data[data["cum_perc"] >= p]
            lift_tmp = lift_tmp[lift_tmp["cum_perc"] == lift_tmp["cum_perc"].min()]
            lift.append(lift_tmp["cum_lift"].min())
    else:
        lift_tmp = data[data["cum_perc"] >= pct]
        lift_tmp = lift_tmp[lift_tmp["cum_perc"] == lift_tmp["cum_perc"].min()]
        lift = lift_tmp["cum_lift"].min()

    # return [gini, lift], list(zip(list(data['cum_perc']), list(data['lift']), list(data['cum_good']), list(data['cum_bad'])))
    outdata = data[["cum_perc", "cum_good", "cum_bad", "cum_lift"]]
    return gini, lift, outdata


def curves_wrapper(
    data,
    masks,
    col_target,
    col_score,
    col_weight=None,
    lift_perc=10,
    draw_roc=True,
    draw_lc=True,
    output_folder="./performance/",
    colors=["g", "r", "y", "b", "m", "c"],
):
    """Draws charts with ROC and Cumulative Lift curves of score color coded by sample (samples are defined by masks).

    Args:
        data (pd.DataFrame): data to evaluate the performance on
        masks (dict {str: array}): masks of samples within the data
        col_target (str): name of target variable (column of data)
        col_score (str): name of score varible (column of data)
        col_weight (str, optional): name of weight variable (column of data). Defaults to None.
        lift_perc (int, optional): Level for cumulative lift to be evaluated on. Defaults to 10.
        draw_roc (bool, optional): Draw ROC curve? Defaults to True.
        draw_lc (bool, optional): Draw Cumulative Lift curve? Defaults to True.
        output_folder (str, optional): Where to save the outputs. Defaults to "./performance/".
        colors (list, optional): Colors of the curves (assigned to the samples defined in masks parameter). Defaults to ["g", "r", "y", "b", "m", "c"].

    Raises:
        TypeError: col_score must be either string or list of strings
    """
    if isinstance(col_score, str):
        col_score = [col_score]
    elif isinstance(col_score, list):
        pass
    else:
        raise TypeError("col_score must be either string or list of strings")
    curves = {}
    for score in col_score:
        for mask_name, mask in masks.items():
            curve_name = f"{score} {mask_name}"
            curves[curve_name] = {}
            curves[curve_name]["gini"], curves[curve_name]["lift"], curves[curve_name]["points"] = calculate_gini_and_lift(
                data[mask], col_target, score, weight_name=col_weight, pct=lift_perc
            )
    if draw_roc:
        plt.figure(figsize=(7, 7))
        plt.axis([0, 1, 0, 1])
        color_index = 0
        for curve_name, curve in curves.items():
            plt.plot(
                [0] + list(curve["points"]["cum_good"]),
                [0] + list(curve["points"]["cum_bad"]),
                label=curve_name,
                color=colors[color_index],
            )
            color_index += 1
        plt.plot(list(range(0, 101)), list(range(0, 101)), color="k")
        plt.xlabel("Cumulative good count")
        plt.ylabel("Cumulative bad count")
        plt.legend(loc="lower right")
        plt.savefig(output_folder + "roc.png", bbox_inches="tight", dpi=72)
        plt.show()
        plt.close()
    if draw_lc:
        plt.figure(figsize=(10, 5))
        color_index = 0
        for curve_name, curve in curves.items():
            if color_index == 0:
                plt.axis([0, 100, 0, max(curve["points"]["cum_lift"]) + 0.5])
            plt.plot(
                [0] + list(curve["points"]["cum_perc"]),
                [0] + list(curve["points"]["cum_lift"]),
                label=curve_name,
                color=colors[color_index],
            )
            color_index += 1
        plt.xlabel("Cumulative count [%]")
        plt.ylabel("Lift")
        plt.legend(loc="upper right")
        plt.savefig(output_folder + "lift.png", bbox_inches="tight", dpi=72)
        plt.show()
        plt.close()


class DisplayBox:
    @staticmethod
    def _alert_block_html(text, title=None, style="info"):
        """Generates a HTML div with a predefined CSS style from Jupyter.
        
        Args:
            text (str): text in box
            title (str, optional): Bold title at the start. Defaults to None.
            style (str, optional): One of possible styles - "success", "info", "danger" or  "warning". Defaults to "info".
        
        Raises:
            ValueError: if style is not one of "success", "info", "danger" or  "warning"
        
        Returns:
            str: HTML code for CSS styled div box
        """
        styles = ("success", "info", "danger", "warning")
        if style not in styles:
            raise ValueError(f"Invalid style - {style}. Possible styles: {styles}")

        html_out = (
            f'<div class="alert alert-block alert-{style}">'
            f'    {"<b>" + title + "</b>: " if title else ""}{text}'
            f"</div>"
        )

        return html_out

    @classmethod
    def green(cls, text, title=None):
        """Displays a green box with `text` and optional `title` in bold at the start 

        Example: (title _Tip_ will be bold)
        Success: Use this to display positive messages!
        
        Args:
            text (str): Text to display
            title (str, optional): Text of title. Defaults to None.
        """
        from IPython.display import HTML, display

        html_out = cls._alert_block_html(text, title, "success")

        display(HTML(html_out))

    @classmethod
    def blue(cls, text, title=None):
        """Displays a blue box with `text` and optional `title` in bold at the start 

        Example: (title _Tip_ will be bold)
        Tip: Use this to display tips!
        
        Args:
            text (str): Text to display
            title (str, optional): Text of title. Defaults to None.
        """
        from IPython.display import HTML, display

        html_out = cls._alert_block_html(text, title, "info")

        display(HTML(html_out))

    @classmethod
    def yellow(cls, text, title=None):
        """Displays a yellow box with `text` and optional `title` in bold at the start 

        Example: (title _Tip_ will be bold)
        Warning: Use this to display warnings!
        
        Args:
            text (str): Text to display
            title (str, optional): Text of title. Defaults to None.
        """
        from IPython.display import HTML, display

        html_out = cls._alert_block_html(text, title, "warning")

        display(HTML(html_out))

    @classmethod
    def red(cls, text, title=None):
        """Displays a red box with `text` and optional `title` in bold at the start 

        Example: (title _Tip_ will be bold)
        Error: Use this to display errors!
        
        Args:
            text (str): Text to display
            title (str, optional): Text of title. Defaults to None.
        """
        from IPython.display import HTML, display

        html_out = cls._alert_block_html(text, title, "danger")

        display(HTML(html_out))


def weighted_percentiles(data, wt, percentiles): 
    
    """Compute weighted percentiles. 
    If the weights are equal, this is the same as normal percentiles. 
    Elements of the C{data} and C{wt} arrays correspond to 
    each other and must have equal length (unless C{wt} is C{None}). 

    Args:
        data (L{np.ndarray} array or C{list} of numbers): The data. 
        wt (C{None} or L{np.ndarray} array or C{list} of numbers): How important is a given piece of data. 
            All the weights must be non-negative and the sum must be greater than zero. 
        percentiles (C{list} of numbers between 0 and 1): what percentiles to use.  (Not really percentiles, as the range is 0-1 rather than 0-100.) 

    Returns:
        list of float: the weighted percentiles of the data
    """ 
    assert np.greater_equal(percentiles, 0.0).all(), "Percentiles less than zero" 
    assert np.less_equal(percentiles, 1.0).all(), "Percentiles greater than one" 
    data = np.asarray(data) 
    assert len(data.shape) == 1 
    if wt is None:  
        wt = np.ones(data.shape, np.float) 
    else: 
        wt = np.asarray(wt, np.float) 
        assert wt.shape == data.shape 
        assert np.greater_equal(wt, 0.0).all(), "Not all weights are non-negative." 
    assert len(wt.shape) == 1 
    n = data.shape[0] 
    assert n > 0 
    i = np.argsort(data) 
    sd = np.take(data, i, axis=0) 
    sw = np.take(wt, i, axis=0) 
    aw = np.add.accumulate(sw) 
    if not aw[-1] > 0: 
        raise ValueError("Nonpositive weight sum") 
    w = (aw-0.5*sw)/aw[-1] 
    spots = np.searchsorted(w, percentiles) 
    o = [] 
    for (s, p) in zip(spots, percentiles): 
        if s == 0: 
            o.append(sd[0]) 
        elif s == n: 
            o.append(sd[n-1]) 
        else: 
            f1 = (w[s] - p)/(w[s] - w[s-1]) 
            f2 = (p - w[s-1])/(w[s] - w[s-1]) 
            assert f1>=0 and f2>=0 and f1<=1 and f2<=1 
            assert abs(f1+f2-1.0) < 1e-6 
            o.append(sd[s-1]*f1 + sd[s]*f2) 
    return o 