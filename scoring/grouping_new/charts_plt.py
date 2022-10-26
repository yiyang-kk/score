import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from ipywidgets import widgets
from IPython.display import display, clear_output
from collections import namedtuple


BAR_COLORS = plt.cm.get_cmap("Pastel1").colors
# STABILITY_COLORS = plt.cm.get_cmap("Set1").colors
STABILITY_COLORS = plt.cm.get_cmap("Pastel1").colors
NAN_COLOR = "lightgrey"
PLOT_SIZE = 2.6
NUMERICAL_PLOTS_RATIOS = [1, 1, 1, 1, 1]
CATEGORICAL_PLOTS_RATIOS = [2, 1, 1, 1]
FONT_SIZE_X_AXIS = 9


class ChartBox(object):
    def __init__(self):
        self.widget = widgets.Output()
        plt.ioff()

    def show(self, fig):
        with self.widget:
            clear_output(wait=True)
            display(fig)


HistogramChart = namedtuple("HistogramChart", ["ax", "bx"])
StabilityChart = namedtuple("StabilityChart", ["ax"])


class Charts:
    def _stability_plot(self, ax, data, label):
        # clear axes
        ax.clear()

        months = data.index.astype(str)

        for group_num, col in data.iteritems():
            # if group is NaN
            if group_num == -1:
                ax.plot(months, col, color=NAN_COLOR, linestyle="--", linewidth=3)
            # if not NaN
            else:
                ax.plot(months, col, color=STABILITY_COLORS[group_num], linewidth=3)

        ax.set_xticklabels(months, rotation=315, fontsize=8)

        ax.set_ylabel(label, fontsize=8)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))


class NumericalCharts(Charts):
    def __init__(self):
        self.fig, axes = plt.subplots(
            ncols=5, figsize=(sum(NUMERICAL_PLOTS_RATIOS) * PLOT_SIZE, PLOT_SIZE), constrained_layout=True, dpi=200
        )

        self.hist_ew = HistogramChart(ax=axes[0], bx=axes[0].twinx())
        self.hist_ed = HistogramChart(ax=axes[1], bx=axes[1].twinx())
        self.hist_grouped = HistogramChart(ax=axes[2], bx=axes[2].twinx())

        self.stability_pop = StabilityChart(ax=axes[3])
        self.stability_br = StabilityChart(ax=axes[4])

    def draw_charts(self, data_stab_pop, data_stab_br, data_hist_grouped, data_hist_ed, data_hist_ew):
        self._histogram_equiwidth_plot(bar_ax=self.hist_ew.ax, plot_ax=self.hist_ew.bx, data=data_hist_ew)
        self._histogram_equidepth_plot(bar_ax=self.hist_ed.ax, plot_ax=self.hist_ed.bx, data=data_hist_ed)
        self._histogram_grouped_plot(ax=self.hist_grouped.ax, bx=self.hist_grouped.bx, histogram_data=data_hist_grouped)
        
        if data_stab_br is not None:
            self._stability_plot(ax=self.stability_pop.ax, data=data_stab_pop, label="pop share")
            self._stability_plot(ax=self.stability_br.ax, data=data_stab_br, label="event rate")
        else:
            self.stability_br.ax.clear()
            self.stability_pop.ax.clear()

    def _histogram_grouped_plot(self, ax, bx, histogram_data):
        ax.clear()
        bx.clear()
        allbins = np.hstack([[-np.inf], histogram_data.bins, [np.inf]])
        ax.bar(
            np.arange(len(histogram_data.counts)),
            histogram_data.counts,
            edgecolor="black",
            linewidth=0.5,
            color=BAR_COLORS,
            width=1,
            align="edge",
        )
        bx.plot(
            np.arange(len(histogram_data.counts)) + 0.5,
            histogram_data.badrates,
            marker="o",
            color="orangered",
            linestyle="dotted",
            ms=3,
            linewidth=1.5,
        )
        ax.set_xlim(0, len(histogram_data.counts))
        ax.set_xticks(np.arange(len(allbins)))
        ax.set_xticklabels([f"{x:.3f}" for x in allbins], rotation=90, fontsize=FONT_SIZE_X_AXIS)

        bx.set_ylim(0)

        for p in histogram_data.bins:
            ax.axvline(x=np.where(np.array(allbins) == p)[0], color="black", linewidth=0.5)

        if len(histogram_data.counts) > len(histogram_data.bins) + 1:
            ax.patches[-1].set_hatch("/")
            bx.text(
                x=allbins.shape[0] - 0.5,
                y=bx.get_ylim()[1] * 1.05,
                s="NaN",
                horizontalalignment="center",
                verticalalignment="bottom",
                color="black",
                rotation=90,
                fontdict={"size": FONT_SIZE_X_AXIS},
            )

        bx.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

    def _histogram_equidepth_plot(self, bar_ax, plot_ax, data):
        """ Equi-depth plot"""

        ## need to fix plotting NaNs

        # clear axes
        bar_ax.clear()
        plot_ax.clear()

        # data
        # elems as numpy array (without +-inf)
        bins = data["bins"]
        ed_bins = data["ed_bins"]
        ed_bin_width = ed_bins[1] - ed_bins[0]  # equi-depth bin width
        # bins + ed_bins (with NaN bin)
        all_bins = np.sort(np.unique(np.hstack((ed_bins, bins))))
        # x2 for histogram counting
        split_points = bins.copy()  # group borders

        counts = data["counts"]
        brs = data["bad_rates"]

        if data["count_nan"]:
            counts = np.append(counts, data["count_nan"])
            brs = np.append(brs, data["bad_rate_nan"])
            all_bins = np.append(all_bins, ed_bins[-1] + ed_bin_width)
            split_points = np.append(split_points, ed_bins[-1])

        # bin_groups = []
        bin_colors = []
        c = sp = 0
        for border in all_bins:
            if len(split_points) > 0 and sp < len(split_points) and border == split_points[sp]:
                # g += 1
                c += 1
                sp += 1
            # bin_groups.append(g)
            bin_colors.append(BAR_COLORS[c % len(BAR_COLORS)])

        left = np.arange(all_bins.shape[0] - 1)
        width = 1
        patches = bar_ax.bar(
            x=left,
            height=counts,
            width=width,
            color=bin_colors,
            hatch="",
            edgecolor="black",
            linewidth=0.5,
            align="edge",
        )
        if data["count_nan"]:
            patches[-1].set_hatch("/")

        # plot event rates
        # print(np.arange(all_bins.shape[0]-1)+0.5, brs)

        plot_ax.plot(
            np.arange(all_bins.shape[0] - 1) + 0.5,
            brs,
            marker="o",
            color="orangered",
            linestyle="dotted",
            ms=3,
            linewidth=1.5,
        )

        # split_points - plot vertical lines ( | )
        for p in split_points:
            bar_ax.axvline(x=np.where(all_bins == p)[0], color="black", linewidth=0.5)

        # show group border numbers
        for x1 in bins:
            bar_ax.text(
                np.where(all_bins == x1)[0],
                bar_ax.get_ylim()[1] * 1.05,
                "{:.2f}".format(x1),
                horizontalalignment="center",
                verticalalignment="bottom",
                color="black",
                rotation=90,
                fontdict={"size": 8},
            )

        # xticks по границам бинов
        bar_ax.set_xticks(np.arange(all_bins.shape[0] + (0)))
        # print(np.arange(all_bins.shape[0] + (-1 if self.has_nan else 0)), ['{:.2f}'.format(b) for b in (all_bins[:-1] if self.has_nan else all_bins)])
        if data["count_nan"]:
            bar_ax.text(
                all_bins.shape[0] - 1.5,
                bar_ax.get_ylim()[1] * 1.05,
                "NaN",
                horizontalalignment="center",
                verticalalignment="bottom",
                color="black",
                rotation=90,
                fontdict={"size": 8},
            )
        bar_ax.set_xticklabels(["{:.2f}".format(b) for b in (all_bins)], rotation=90, fontsize=FONT_SIZE_X_AXIS)

        bar_ax.set_xlim(0, all_bins.shape[0] - 1)

        # plot_ax.set_visible(False)

        bar_ax.set_ylabel("count", fontsize=8)
        plot_ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
        plot_ax.set_ylabel("event rate", fontsize=8)

    def _histogram_equiwidth_plot(self, bar_ax, plot_ax, data):
        """ Equi-width plot"""
        # clear axes
        bar_ax.clear()
        plot_ax.clear()

        # data
        # elems as numpy array (without +-inf)
        bins = data["bins"]
        ew_bins = data["ew_bins"]
        ew_bin_width = ew_bins[1] - ew_bins[0]  # equi-depth bin width
        # bins + ed_bins (with NaN bin)
        all_bins = np.sort(np.unique(np.hstack((ew_bins, bins))))
        # x2 for histogram counting
        split_points = bins.copy()  # group borders

        counts = data["counts"]
        brs = data["bad_rates"]

        if data["count_nan"]:
            counts = np.append(counts, data["count_nan"])
            brs = np.append(brs, data["bad_rate_nan"])
            all_bins = np.append(all_bins, ew_bins[-1] + ew_bin_width)
            split_points = np.append(split_points, ew_bins[-1])

        # bin_groups = []
        bin_colors = []
        c = sp = 0
        for border in all_bins:
            if len(split_points) > 0 and sp < len(split_points) and border == split_points[sp]:
                # g += 1
                c += 1
                sp += 1
            # bin_groups.append(g)
            bin_colors.append(BAR_COLORS[c % len(BAR_COLORS)])

        left = all_bins[:-1]
        width = [x1[1] - x1[0] for x1 in zip(all_bins[:-1], all_bins[1:])]
        patches = bar_ax.bar(
            x=left,
            height=counts,
            width=width,
            color=bin_colors,
            hatch="",
            edgecolor="black",
            linewidth=0.5,
            align="edge",
        )

        if data["count_nan"]:
            patches[-1].set_hatch("/")

        # plot event rates

        # print(np.arange(all_bins.shape[0]-1)+0.5, brs)

        plot_ax.plot(
            np.vstack([all_bins[1:], all_bins[:-1]]).mean(axis=0),
            brs,
            marker="o",
            color="orangered",
            linestyle="dotted",
            ms=3,
            linewidth=1.5,
        )
        # split_points - plot vertical lines ( | )
        for p in split_points:
            bar_ax.axvline(x=p, color="black", linewidth=0.5)

            # show group border numbers
        for x1 in bins:
            bar_ax.text(
                x1,
                bar_ax.get_ylim()[1] * 1.05,
                "{:.2f}".format(x1),
                horizontalalignment="center",
                verticalalignment="bottom",
                color="black",
                rotation=90,
                fontdict={"size": 8},
            )

        # xticks по границам бинов
        bar_ax.set_xticks(np.arange(all_bins.shape[0] + (0)))
        # print(np.arange(all_bins.shape[0] + (-1 if self.has_nan else 0)), ['{:.2f}'.format(b) for b in (all_bins[:-1] if self.has_nan else all_bins)])
        if data["count_nan"]:
            bar_ax.text(
                (all_bins[-1] + all_bins[-2]) / 2,
                bar_ax.get_ylim()[1] * 1.05,
                "NaN",
                horizontalalignment="center",
                verticalalignment="bottom",
                color="black",
                rotation=90,
                fontdict={"size": 8},
            )

        bar_ax.set_xticks(ew_bins)
        bar_ax.set_xticklabels(["{:.2f}".format(b) for b in ew_bins], rotation=90, fontsize=FONT_SIZE_X_AXIS)

        bar_ax.set_xlim(all_bins[0], all_bins[-1])

        bar_ax.set_ylabel("count", fontsize=8)
        plot_ax.set_ylabel("event rate, %", fontsize=8)


class CategoricalCharts(Charts):
    def __init__(self):
        self.fig, axes = plt.subplots(
            ncols=4,
            figsize=(sum(CATEGORICAL_PLOTS_RATIOS) * PLOT_SIZE, PLOT_SIZE),
            gridspec_kw={"width_ratios": CATEGORICAL_PLOTS_RATIOS},
            constrained_layout=True,
            dpi=200,
        )
        self.histogram = HistogramChart(ax=axes[0], bx=axes[0].twinx())
        self.histogram_grouped = HistogramChart(ax=axes[1], bx=axes[1].twinx())
        self.stability_pop = StabilityChart(ax=axes[2])
        self.stability_br = StabilityChart(ax=axes[3])

    def draw_charts(self, data_histogram, data_histogram_grouped, data_stab_pop, data_stab_br):
        self._histogram_plot(bar_ax=self.histogram.ax, plot_ax=self.histogram.bx, data=data_histogram)
        self._histogram_grouped_plot(
            bar_ax=self.histogram_grouped.ax, plot_ax=self.histogram_grouped.bx, data=data_histogram_grouped
        )

        if data_stab_br is not None:
            self._stability_plot(ax=self.stability_pop.ax, data=data_stab_pop, label="pop share")
            self._stability_plot(ax=self.stability_br.ax, data=data_stab_br, label="event rate")
        else:
            self.stability_br.ax.clear()
            self.stability_pop.ax.clear()

    def _histogram_plot(self, bar_ax, plot_ax, data, valid=True):
        # Original categories plot
        # clear axes
        bar_ax.clear()
        plot_ax.clear()

        #         chartdata = self.values_df.copy()
        #         chartdata['cats'] = chartdata.index.to_series()
        #         chartdata.sort_values(['group', 'cats'], ascending=['True', 'True'], inplace=True)

        cats = data.index
        bins = data["group"]
        counts = data["pop"]

        bin_colors = []
        cats_pos = []
        cats_pos_line = []
        splitlines = []
        c = 0
        oldbin = 0
        # print(bins)
        for i in range(0, len(cats)):
            if bins.iloc[i] > oldbin:  # sg
                c += 1
                oldbin = bins.iloc[i]  # sg
                splitlines.append(i - 0.5)
            bin_colors.append(BAR_COLORS[c % len(BAR_COLORS)])
            cats_pos.append(i - 0.5)
            cats_pos_line.append(i)
        splitlines.append(len(cats) - 0.5)

        height = counts
        width = 1

        patches = bar_ax.bar(
            cats_pos, height, width, color=bin_colors, hatch="", edgecolor="black", linewidth=0.5, align="edge"
        )

        # plot event rates
        #         brs = chartdata['def_rate'].str.strip('%').str.strip('nan').str.strip()
        brs = data["def_rate"].values

        plot_ax.plot(cats_pos_line, brs, marker="o", color="orangered", linestyle="dotted", ms=3, linewidth=1.5)

        # split_points - plot vertical lines ( | ), and names of the groups
        for i in range(0, len(splitlines)):
            bar_ax.axvline(splitlines[i], color="black", linewidth=0.5)
            if i > 0:
                prev_split = splitlines[i - 1]
            else:
                prev_split = -0.5
            text_pos = (splitlines[i] + prev_split) / 2
            bar_ax.text(
                text_pos,
                bar_ax.get_ylim()[1] * 1.05,
                i,
                horizontalalignment="center",
                verticalalignment="bottom",
                color="black",
                rotation=0,
                fontdict={"size": 8},
            )

        # xticks по границам бинов
        bar_ax.set_xticks(range(0, len(cats)))
        bar_ax.set_xticklabels(cats, rotation=315, fontsize=FONT_SIZE_X_AXIS)

        bar_ax.set_xlim(np.min(cats_pos), np.max(cats_pos) + 1)

        bar_ax.set_ylabel("count", fontsize=8)
        plot_ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

        plot_ax.set_ylabel("event rate", fontsize=8)

    def _histogram_grouped_plot(self, bar_ax, plot_ax, data, valid=True):
        # Final groups plot
        # clear axes
        bar_ax.clear()
        plot_ax.clear()

        if not valid:
            bar_ax.text(
                0.5 * (bar_ax.get_xlim()[0] + bar_ax.get_xlim()[1]),
                0.5 * (bar_ax.get_ylim()[0] + bar_ax.get_ylim()[1]),
                "Error",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=20,
                color="red",
            )
            return

        bins = data.index.values
        counts = data["pop"]

        bin_colors = []
        c = 0
        for bin_ in bins:
            bin_colors.append(BAR_COLORS[c % len(BAR_COLORS)])
            c += 1

        height = counts
        width = 1
        bins_pos = bins - 0.5

        patches = bar_ax.bar(
            bins_pos, height, width, color=bin_colors, hatch="", edgecolor="black", linewidth=0.5, align="edge"
        )

        # plot event rates
        #         brs = chartdata['def_rate'].str.strip('%').str.strip('nan').str.strip()
        brs = data["def_rate"]

        plot_ax.plot(bins, brs, marker="o", color="orangered", linestyle="dotted", markersize=3, linewidth=1.5)

        # split_points - plot vertical lines ( | )
        for b in bins:
            bar_ax.axvline(b - 0.5, color="black", linewidth=0.5)

        # xticks по границам бинов
        bar_ax.set_xticks(bins)
        bar_ax.set_xticklabels(bins, rotation=0, fontsize=FONT_SIZE_X_AXIS)

        bar_ax.set_xlim(np.min(bins_pos), np.max(bins_pos) + 1)

        bar_ax.set_ylabel("count", fontsize=8)
        plot_ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
        plot_ax.set_ylabel("event rate", fontsize=8)
