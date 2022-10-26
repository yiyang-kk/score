"""
Home Credit Python Scoring Library and Workflow
Copyright © 2017-2019, Pavel Sůva, Marek Teller, Martin Kotek, Jan Zeller,
Marek Mukenšnabl, Kirill Odintsov, Elena Kuchina, Jan Coufalík, Jan Hynek, Nada Horka and
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
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from ipywidgets import FloatProgress
import gc
import plotly.express as px
from IPython.display import display


#
# def get_score_dist(data, score, weight, title=None, savefile=None):
#     """
#     Function for computing of score distribution
#     Args:
#         data (pd.DataFrame): dataset containing the score and the weight column
#         score (str): name of the score column in data
#         weight (str): name of the weight column in data
#         title: sets the title of the plot
#         savefile: if existent, will save the plot to the desired place
#
#     Returns:
#
#     """
#     df_predprob = pd.DataFrame(data[[score, weight]])
#
#     df_predprob['WT'] = 1
#
#     fig = plt.figure(figsize=(10, 5))
#     ax = fig.add_subplot()
#
#     ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
#     ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
#     ax.set_ylabel("f(Z)")
#     ax.set_xlabel("2*P(Z =1) -1")
#     ax.set_facecolor('xkcd:ice')
#     ax.grid(True)
#     ax.set_xlim([-1, 1])
#     ax.hist(df_predprob[score], bins=100, cumulative=0, density=False, alpha=0.2, stacked=False, color='black',
#             weights=df_predprob[weight] / df_predprob[weight].sum())
#
#     if title:
#         plt.title(title)
#     if savefile:
#         plt.savefig(savefile, )
#     plt.show()


def get_quantiles(df, perf_measure, percentile):
    """
    Computes the quantiles.
    Args:
        df (pd.DataFrame): the dataset with histories
        perf_measure (str): measure on which we compute the quantile - caasb avg or score avg
        percentile (float): desired percentile

    Returns:
        pd.DataFrame: quantiles based on histories (surprise surprise, for n_histories = 1 will be all the same)
    """
    quantiles = df.groupby('t-t0 (days)'). \
        apply(lambda row: np.percentile(row[perf_measure], q=percentile))
    quantiles = pd.DataFrame(quantiles)
    quantiles.rename(columns={0: perf_measure}, inplace=True)
    quantiles = quantiles.reset_index(drop=False)
    quantiles.rename(columns={'index': 't-t0 (days)'}, inplace=True)

    return quantiles


class CollModelImpactAnalysis:
    """

    Args:
        data (pandas.DataFrame): the data frame with following attributes
        weight (str):  name of the column with weights in df
        uplift_metrics (list): a list of column names in df containing the metrics we want to compute
            (usually score deltas or caasb differences)
        treatment (str): name of the column with the 0/1 control (low) /treatment (high) information
        outcome (str): name of the column containing the original target (paid=1, unpaid=0, no matter of treatment)
        target (str): name of the column containing the new target (dependent on payment AND treatment)
        caasb (str): column name, contains the real CAASB, the money outcome of collections
        n_bins (int, optional): number of bins to which will be q-cut the data for the cumulative computations, default 10
        n_bootstraps (int, optional): number of the desired bootstraps, default 100
        use_caasb (bool, optional): True for the resulting average real caasb, False for average response ratio, default True
        alpha (int, optional): alpha for the confidence interval, default 2
    """

    def __init__(self, data, weight, base, uplift_metrics, treatment, outcome, target, time,
                 caasb, n_bins=10, n_bootstraps=50, use_caasb=True, n_histories=100, alpha=2):
        self.data = data
        self.base = base
        self.uplift_metrics = uplift_metrics
        self.treatment = treatment
        self.outcome = outcome
        self.target = target
        self.time = time
        self.caasb = caasb
        self.n_bins = n_bins
        self.n_bootstraps = n_bootstraps
        self.use_caasb = use_caasb
        self.weight = weight
        if self.n_bootstraps <= 1:
            self.n_histories = 1  # we cannot do multiple histories with just one data for one day
        else:
            self.n_histories = n_histories
        self.alpha = alpha

    def _model_impact_curves(self, bootstrap):
        """
           Function to compute the uplift curves for a set of metrics (score, caasb) - shows the average value of real
           outcome depending on the ratio of classifiables sent to the treatment/higher treatment.

           Arguments:
               bootstrap (pandas.DataFrame): bootstrapped dataset of the self.data

           Returns:
               dict: a dictionary with the impact curves for all metrics
           """
        if self.use_caasb:
            _outcome = self.caasb
        else:
            _outcome = self.outcome

        # _________________________________________________________________

        FULL_POPULATION_SIZE = bootstrap[self.weight].sum()
        Curves = {}

        # ------------------------outer loop through metrics -------------------------------------------------

        for metric in self.uplift_metrics:
            # First compute the cumulative loss if NO ONE is set to TREATMENT(HIGH), but all go to CONTROL(LOW):
            # -------------------------------------------------------------

            Recovery_rate_LOW = bootstrap[bootstrap[self.treatment] == 0].groupby([self.base]) \
                .apply(lambda row: (1.0 * row[_outcome] * row[self.weight]).sum() / row[self.weight].sum()).values[0]

            CAASB = Recovery_rate_LOW

            curve = [CAASB]

            bootstrap['Uplift_bin'] = pd.qcut(bootstrap[metric], self.n_bins, labels=False, retbins=False,
                                              duplicates='drop')

            bootstrap['Uplift_boundaries'] = pd.qcut(bootstrap[metric], self.n_bins, duplicates='drop')

            # create descending bins top down, e.g. [9,8,7,...,0]
            Uplift_bins = sorted(list(bootstrap['Uplift_bin'].value_counts().index), reverse=True)

            # --------------------------inner loop  evaluate each  self.metric bins group---------------------------
            for i in Uplift_bins:

                # CAASB for HIGH segment (uplift self.metric at or above threshold)
                if bootstrap[(bootstrap[self.treatment] == 1) & (bootstrap['Uplift_bin'] >= i)].shape[0] > 0:
                    Recovery_rate_HIGH = \
                        bootstrap[(bootstrap[self.treatment] == 1) & (bootstrap['Uplift_bin'] >= i)].groupby(
                            [self.base]) \
                            .apply(
                            lambda row: (1.0 * row[_outcome] * row[self.weight]).sum() / row[self.weight].sum()).values[
                            0]

                    HIGH_SEGMENT_SIZE = bootstrap[(bootstrap['Uplift_bin'] >= i)][self.weight].sum()

                    # CAASB per HIGH capita (uplift)
                    CAASB = Recovery_rate_HIGH

                    # CAASB per POPULATION capita (every 'HIGH_SEGMENT_SIZE/FULL_POPULATION_SIZE' % of population gets to HIGH)
                    CAASB_HIGH_SEGMENT = (HIGH_SEGMENT_SIZE / FULL_POPULATION_SIZE) * CAASB
                else:
                    CAASB_HIGH_SEGMENT = 0
                # ---------------------------------------------------------------------------------#
                # CAASB for LOW segment (uplift self.metric below threshold)

                if i != 0:  # (lowest bin):
                    if bootstrap[(bootstrap[self.treatment] == 0) & (bootstrap['Uplift_bin'] < i)].shape[0] > 0:
                        # print('btstrp_size:', bootstrap[(bootstrap[self.treatment] == 0) & (bootstrap['Uplift_bin'] < i)].shape[0])
                        Recovery_rate_LOW = \
                            bootstrap[(bootstrap[self.treatment] == 0) & (bootstrap['Uplift_bin'] < i)].groupby(
                                [self.base]) \
                                .apply(
                                lambda row: (1.0 * row[_outcome] * row[self.weight]).sum() / row[
                                    self.weight].sum()).values[
                                0]

                        LOW_SEGMENT_SIZE = bootstrap[(bootstrap['Uplift_bin'] < i)][self.weight].sum()

                        # CAASB per LOW capita (uplift)
                        CAASB = Recovery_rate_LOW

                        # CAASB per POPULATION capita (every 'LOW_SEGMENT_SIZE/FULL_POPULATION_SIZE' % of population gets to LOW)

                        CAASB_LOW_SEGMENT = (LOW_SEGMENT_SIZE / FULL_POPULATION_SIZE) * CAASB
                    else:  # switch to higher bin, we have no low segment contracts here
                        CAASB_LOW_SEGMENT = 0

                else:  # LOW SEGMENT SIZE is zero
                    CAASB_LOW_SEGMENT = 0

                CAASB_POPULATION = CAASB_HIGH_SEGMENT + CAASB_LOW_SEGMENT

                curve.append(CAASB_POPULATION)
            #    ------------------------------------inner loop end--------------------------------------------
            Curves[metric] = curve
            # ----------------------------------------Outer loop end--------------------------------------------
        return Curves

    def bootstrap_impact_analysis(self, by_day=None, progress_bar=True):
        """

        Args:
            by_day (pd.DataFrame, optional): the subset of self.data on which we compute impact analysis, if None, we compute
                impact analysis on whole dataset self.data.
                Defaults to None.
            progress_bar (bool, optional): True if you want to display the progress bar, False otherwise.
                Defaults to True

        Returns:
            dict: bootstrap_CAASB_curves_collection_by_metric - a dictionary with resulting curves for each bootstrap
        """

        f = FloatProgress(description='Progress:', min=0, max=self.n_bootstraps)
        if progress_bar:
            # TODO: check grouping.transform for display of FloatProgress
            display(f)

        bootstrap_CAASB_curves_collection = {}

        # for use of day-by-day computations in historical daily winner distributions
        if by_day is not None:
            n = by_day.shape[0]
            df = by_day
        else:
            n = self.data.shape[0]  # desired number of rows in a bootstrap
            df = self.data

        for i in range(0, self.n_bootstraps):
            if self.n_bootstraps > 0:
                rnd_mask = np.random.randint(n, size=n)
                # display(df[[self.target] + [self.weight, self.base] + self.uplift_metrics + [self.treatment]
                #                    + [self.outcome] + [self.caasb]])
                BTSTRP = df[[self.target] + [self.weight, self.base] + self.uplift_metrics + [self.treatment]
                            + [self.outcome] + [self.caasb]].loc[df.index.intersection(rnd_mask)]
                BTSTRP = BTSTRP.reset_index(drop=True)
                # !! very important to reset index before making any sub-selections self.based on mask

            else:
                BTSTRP = df[[self.target] + [self.weight, self.base] + self.uplift_metrics + [self.treatment]
                            + [self.outcome] + [self.caasb]]
                BTSTRP = BTSTRP.reset_index(drop=True)

            bootstrap_CAASB_curves_collection[i] = self._model_impact_curves(BTSTRP)

            f.value += 1

        del BTSTRP
        gc.collect()
        return bootstrap_CAASB_curves_collection

    def plot_impact_analysis(self, to_plot, savefile=None):
        """Displays impact analysis plot.

        Args:
            to_plot (str): name of the dictionary with the data
            savefile (str, optional): name of path + file name for saving the plot.
                Defaults to None.
        """

        fig = plt.figure(figsize=(17, 10))
        ax = fig.add_subplot()
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.xaxis.set_ticks(np.arange(0, 1.05, 0.05))  # ticks and frequency
        ax.set_facecolor('xkcd:ice')
        ax.grid(True)

        if self.use_caasb:
            plt.title('MIXING : predicted delta CAASB > cutoff --> send to HIGH, else send to LOW ', fontsize=16)
            ax.set_ylabel(" actual CAASB per capita in mixed population", fontsize=16)
            ax.set_xlabel(" top % population by predicted (CAASB_HIGH -CAASB_LOW) from top to low ", fontsize=16)
        else:
            plt.title('MIXING : predicted uplift > cutoff --> send to HIGH, else send to LOW ', fontsize=16)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            ax.set_ylabel(" actual response rate per capita in mixed population", fontsize=16)
            ax.set_xlabel(" top % population by predicted uplift (P_recover@HIGH - P_recover@Low) from top to low ",
                          fontsize=16)

        to_plot_by_metric = {}

        for metric in self.uplift_metrics:
            to_plot_by_metric[metric] = {}
            for k in to_plot:
                to_plot_by_metric[metric][k] = to_plot[k][metric]

        for metric in self.uplift_metrics:
            # https://stackoverflow.com/questions/19736080/creating-dataframe-from-a-dictionary-where-entries-have-different-lengths
            _Curves = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in to_plot_by_metric[metric].items()]))

            # ger 2pct , mean and 98pct of statistic(uplift, gain, etc) for each uplift_metric bin (0,1,..n_bins-1)
            m = _Curves.mean(1)
            m = np.array(m).reshape(-1, 1)
            down = np.nanpercentile(_Curves, 2, axis=1, keepdims=True)
            up = np.nanpercentile(_Curves, 98, axis=1, keepdims=True)

            m = pd.DataFrame(m)
            m.rename(columns={0: 'mean'}, inplace=True)
            down = pd.DataFrame(down)
            down.rename(columns={0: '2pct'}, inplace=True)
            up = pd.DataFrame(up)
            up.rename(columns={0: '98pct'}, inplace=True)

            _Curves_stats = pd.DataFrame(
                [pd.DataFrame(down)['2pct'], pd.DataFrame(m)['mean'], pd.DataFrame(up)['98pct']]).T

            # ---------------------make index of _Curve_stats go from 0 to 100% ---------------------------
            _Curves_stats = _Curves_stats.reset_index(drop=False)
            multiplier = len(_Curves_stats['index'])
            _Curves_stats['index'] = _Curves_stats['index'].apply(lambda x: (1.0 / (multiplier - 1)) * (x))
            _Curves_stats.set_index('index', inplace=True)
            # --------------------------------------------------------------------------------------------

            ax.plot(_Curves_stats['mean'], linestyle='--', linewidth=3, marker='x', markersize=15,
                    label='CAASB based on: ' + metric + ' (2pct, mean ,98pct) ')
            ax.fill_between(_Curves_stats.index, _Curves_stats['2pct'], _Curves_stats['98pct'], alpha=0.1,
                            )

            ax.legend(loc="best", fontsize=10, )

        if savefile is not None:
            plt.savefig(savefile, bbox_inches='tight', dpi=72)

        plt.show()

    def _daily_winner_distribution(self,
                                   _seed,
                                   t0,
                                   challenger_models,
                                   challenger_name,
                                   by_day=None,
                                   dynamic_cutoff=True,
                                   fixed_cutoff=0.5):

        """

        Args:
            t0 (datetime): initial date of the dataframe in which we compute the daily distributions
            challenger_models (list): list of models' names
            challenger_name (str): the chosen name for challenger strategy
            by_day (pd.DataFrame, optional): the subset of self.data, for each day, usually inherited from
                get_hist_daily_winner_distribution.
                Dfeaults to None.
            dynamic_cutoff (bool, optional): True for finding the best cutoff in data, False for fixed defined cutoff.
                Defaults to True.
            fixed_cutoff (float, optional): The fixed share of contracts entering the High milestone,
                Defaults to 0.5.

        Returns:
            pd.DataFrame: maximized result for a by_day/self.data dataset
                For multiple models in one challenger the return is the winner model, either absolute
                or on the defined fixed_cutoff level.

        """
        #
        np.random.seed(_seed)

        daily_CAASB_curves_collection = \
            self.bootstrap_impact_analysis(by_day=by_day, progress_bar=False)

        winner_distribution = pd.DataFrame()

        # ------------------OUTER LOOP  : get the winner of each [bootstrap] version of the day -----------------#

        for item in daily_CAASB_curves_collection.items():
            # Daily_CAASB_curves_collection.items():
            # item[0] bootstrap index
            # item[1][metric]  metric curve (with n_bins +1 points)

            bs = item[0]

            optimal = {}
            # ----------------INNER LOOP : report optimal values of each challenger--------------------#

            for metric in challenger_models:

                optimal[metric] = {}

                if dynamic_cutoff:
                    cutoff = np.argmax(item[1][metric])  # for which cutoff is the highest value on the curve
                else:
                    cutoff = int(fixed_cutoff * self.n_bins)  # fixed cutoff re-set to bin number

                # TODO this setting and resetting cutoff values is rather messy, for sure could be done in a clearer way.

                maximum = item[1][metric][cutoff]  # the max value for the above found/defined cutoff
                cutoff = (1.0 / self.n_bins) * cutoff  # reset back to share in 0 - 1
                optimal[metric] = [maximum, cutoff]

            # -------------------INNER LOOP END--------------------------------------------------------#

            # -----get winner in in the  bootstrap----:

            # optimal.items():
            # item[0]  metric
            # item[1][0]  metric_value
            # item[1][1]  cutoff_point

            if by_day is not None:
                t = max(by_day[self.time])
            else:
                t = max(self.data[self.time])

            max_i = max(optimal.items(), key=lambda x: x[1][0])
            winner = {0: {'day': t, 't-t0 (days)': (t - t0).days, 'bootstrap': bs, 'winner': max_i[0],
                          'CAASB_per_capita at t': max_i[1][0], 'top_POP%_cutoff(HIGH)': max_i[1][1],
                          'type': challenger_name}}
            winner = pd.DataFrame(winner).T
            winner_distribution = winner_distribution.append(winner)

            del optimal, winner

            # -------------------------------------------END OUTER LOOP------------------------------#

        return winner_distribution

    def get_hist_daily_winner_distribution(self,
                                           seed,
                                           challengers,
                                           dynamic_cutoff=True,
                                           fixed_cutoff=0.5,
                                           progress_bar=False
                                           ):
        """Daily winner distribution for each day of the data mask.

        Args:
            seed (int): to set the seed for the random, to be able to reset the random
            challengers (dict): the set of challenger models or sets (or both)
            dynamic_cutoff (bool, optional): True for finding the best cutoff in data, False for fixed defined cutoff.
                Defaults to True.
            fixed_cutoff (float, optional): The fixed share of contracts entering the High milestone,
                Defaults to 0.5.
            progress_bar (bool, optional): True for showing the Progress bar in the notebook.
                Defaults to False.

        Returns:
            pd.DataFrame: df with historical daily winner distribution
        """

        days = sorted(list(self.data[self.time].unique()))
        t0 = days[0]  # day 0

        f = FloatProgress(description='Progress', min=0, max=len(days))
        if progress_bar:
            display(f)

        historical_daily_winner_distribution = pd.DataFrame()

        # -----------------------------LOOP : go through days ---------------------------------------------#
        for day in days:
            # create a dataset for each day to send it to _daily_winner_distribution
            X_day = self.data[self.data[self.time] == day]
            X_day = X_day.reset_index(drop=True)

            # get winner distribution for each challenger configuration:

            for key in challengers.keys():
                winner_distribution = self._daily_winner_distribution(seed,
                                                                      t0,
                                                                      challengers[key]['challengers'],
                                                                      challengers[key]['_type'],
                                                                      by_day=X_day,
                                                                      dynamic_cutoff=dynamic_cutoff,
                                                                      fixed_cutoff=fixed_cutoff,

                                                                      )

                historical_daily_winner_distribution = historical_daily_winner_distribution.append(winner_distribution)

            f.value += 1
            # -----------------------------LOOP : go through days ---------------------------------------------#

        # add the cumulative time average of 'CAASB_per_capita at t' from t0 --> t:

        del X_day
        gc.collect()

        # Compute and print out the checksums:
        len_days = len(days)
        len_chals = len(challengers.keys())
        if self.n_bootstraps == 0:
            n_btstrp = 1
        else:
            n_btstrp = self.n_bootstraps

        print('Rows in the historical daily distribution: ', historical_daily_winner_distribution.shape[0])
        print(f'Checksum: days*challengers_nr*bootstraps: {len_days} * {len_chals} * {n_btstrp} = ',
              len_days * len_chals * self.n_bootstraps)

        return historical_daily_winner_distribution

    def create_trajectories(self, historical_distribution, _seed, challengers):

        """Function to simulate the day-after day set of segmented contracts/classifiables.
        For each history of n_histories, result from a randomly changed bootstrap is chosen, for each day.

        Args:
            historical_distribution (pd.DataFrame): the result of get_hist_daily_winner_distribution
            _seed (int): random seed number
            challengers (dict): which challengers to take into simulation of trajectories

        Returns:
            pd.DataFrame
        """

        np.random.seed(_seed)

        days = sorted(list(self.data[self.time].unique()))
        collection_winner_histories = pd.DataFrame()

        f = FloatProgress(description='Trajectory : ', min=0, max=self.n_histories)
        display(f)

        # ----------------GENERATE multi-day trajectories (HISTORIES) FROM DAILY WINNERS DISTRIBUTIONS-----------------#

        for history in range(0, self.n_histories):

            winner_history = pd.DataFrame()

            for key in challengers.keys():

                winner_history_per_type = pd.DataFrame()

                # ---------------SAMPLE A RANDOM WINNER FROM EACH DAY's WINNERS DISTRIBUTION-------------------#

                for day in days:
                    df = historical_distribution[(historical_distribution['day'] == day) &
                                                 (historical_distribution['type'] == challengers[key]['_type'])]

                    df = df.reset_index(drop=True)
                    mask = np.random.randint(df.shape[0], size=1)
                    df = df.loc[mask]  # <--------for each day pickup a random point from its winner distribution
                    winner_history_per_type = winner_history_per_type.append(df)

                # -----------------------------------END SAMPLE-------------------------------------------------#

                winner_history_per_type.sort_values(by=['t-t0 (days)'], ascending=[True], na_position='first',
                                                    inplace=True)
                # cumulative mean for each new day
                winner_history_per_type['CAASB_per_capita averaged over t-t0'] = \
                    winner_history_per_type['CAASB_per_capita at t'].expanding(1).mean()

                winner_history = winner_history.append(winner_history_per_type)

            winner_history['history_id'] = history
            winner_history.set_index('history_id', inplace=True)
            winner_history = winner_history.reset_index(drop=False)
            winner_history.rename(columns={'index': 'history_id'}, inplace=True)
            collection_winner_histories = collection_winner_histories.append(winner_history)
            collection_winner_histories['weight'] = 1

            f.value += 1

        # Compute and print out the checksums:
        len_days = len(days)
        len_chals = len(challengers.keys())

        print('Rows in the historical daily distribution: ', collection_winner_histories.shape[0])
        print(f'Checksum: days*challengers_nr*histories_nr: {len_days} * {len_chals} * {self.n_histories} = ',
              len_days * len_chals * self.n_histories)

        return collection_winner_histories

    def _get_quantiles_cumulative(self,
                                  collection_winner_histories,
                                  challenger,
                                  champion,
                                  perf_measure):
        """

        Args:
            collection_winner_histories: (pd.DataFrame): the dataset which we cumulate, originates in create_trajectories
            challenge (str): the name of the challenger model/set of models
            champion (str): the name of the champion model/set of models
            perf_measure (str): the column name of the measure to get percentiles for

        """
        # names = ['challenger', 'champion']
        names = [challenger, champion]
        df = [collection_winner_histories[collection_winner_histories['type'] == challenger],
              collection_winner_histories[collection_winner_histories['type'] == champion]]
        quantiles_all = pd.DataFrame()
        for name, x in zip(names, df):
            for i in [self.alpha, 50, (100 - self.alpha)]:  # we want to have percentiles AND median
                quantiles = get_quantiles(x, perf_measure, percentile=i)
                quantiles['type'] = name + '_' + str(i) + '_quantile'
                quantiles_all = quantiles_all.append(quantiles)

        # for the sake of nice dynamic plots - can be turned off in case of static
        quantiles_histories = pd.DataFrame()
        for i in range(0, self.n_histories):
            temp = quantiles_all.copy()
            temp['history_id'] = i
            quantiles_histories = quantiles_histories.append(temp)
            del temp
        quantiles_histories = quantiles_histories.reset_index(drop=True)
        collection_winner_histories = collection_winner_histories.append(quantiles_histories)

        return collection_winner_histories

    def get_deltas_difference(self,
                              collection_winner_histories,
                              challenger,
                              champion,
                              perf_measure='CAASB_per_capita averaged over t-t0'):
        """Method for adding the deltas between champion and challenger model (sets of models) and computing its conf.int.

        Args:
            collection_winner_histories (pd.DataFrame): the outcome of get_quantiles_cumulative
            challenger (str): the name of the challenger model/set of models
            champion (str): the name of the champion model/set of models
            perf_measure (str, optional): the column name of the measure to get percentiles for.
                Defaults to 'CAASB_per_capita averaged over t-t0'.

        Returns:
            pd.DataFrame: dataset with appended deltas and deltas' quantiles
        """

        label = "'" + challenger + "'" + ' vs ' + "'" + champion + "'"
        coll_delta_histories = self._get_quantiles_cumulative(collection_winner_histories, challenger, champion,
                                                              perf_measure)

        df = coll_delta_histories[
            (coll_delta_histories['type'] == challenger) | (coll_delta_histories['type'] == champion)].copy()

        df.loc[df['type'] == challenger, 'weight'] = 1  # for summing by grouping it later
        df.loc[df['type'] == champion, 'weight'] = -1

        delta = pd.DataFrame(df.groupby(['history_id', 't-t0 (days)']).
                             apply(lambda row: (1.0 * row[perf_measure] * row['weight']).sum()))
        delta = delta.reset_index(drop=False)
        delta.rename(columns={0: perf_measure}, inplace=True)
        delta['type'] = label
        delta_quantiles = pd.DataFrame()
        for i in [self.alpha, 50, (100 - self.alpha)]:  # we want to have percentiles AND median
            quantiles = get_quantiles(delta, perf_measure, percentile=i)
            quantiles['type'] = 'delta' + '_' + str(i) + '_quantile'
            delta_quantiles = delta_quantiles.append(quantiles)

        # delta_histories = pd.DataFrame()
        for i in range(0, self.n_histories):
            temp = delta_quantiles.copy()
            temp['history_id'] = i
            delta = delta.append(temp)
            del temp
        delta = delta.reset_index(drop=True)
        delta['category'] = 'differences'
        coll_delta_histories['category'] = 'absolute'

        coll_delta_histories = coll_delta_histories.append(delta)

        return coll_delta_histories

    def plot_dynamic(self, df, challenger, champion, show='delta', perf_measure='CAASB_per_capita averaged over t-t0'):
        """

        Args:
            df (pd.DataFrame): dataset returned from get_deltas_difference
            challenger (str): a name of the challenger stragety
            champion (str): a name of the champion strategy
            show (str, optional): 'delta' shows a delta between champion and challenger strategies
                'challenger' shows the results for challenger
                'champion' shows the results for champion
                throws error otherwise.
                Defaults to 'delta'.
            perf_measure (str, optional): column name of the performance measure, usually caasb or score
                Defaults to 'CAASB_per_capita averaged over t-t0'.
            
        """

        # color setting
        label = "'" + challenger + "'" + ' vs ' + "'" + champion + "'"
        _color_discrete_map = {'delta_' + str(self.alpha) + '_quantile': 'orangered',
                               'delta_50_quantile': 'darkmagenta',
                               'delta_' + str(100 - self.alpha) + '_quantile': 'slateblue',

                               challenger + '_' + str(self.alpha) + '_quantile': 'blue',
                               challenger + '_50_quantile': 'blue',
                               challenger + '_' + str(100 - self.alpha) + '_quantile': 'blue',

                               champion + '_' + str(self.alpha) + '_quantile': 'red',
                               champion + '_50_quantile': 'red',
                               champion + '_' + str(100 - self.alpha) + '_quantile': 'red',

                               label: 'darkmagenta',
                               challenger: 'blue',
                               champion: 'red'}

        # ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
        _line_dash_map = {'delta_' + str(self.alpha) + '_quantile': 'dot',
                          'delta_50_quantile': 'solid',
                          'delta_' + str(100 - self.alpha) + '_quantile': 'dot',

                          challenger + '_' + str(self.alpha) + '_quantile': 'dot',
                          challenger + '_50_quantile': 'dot',
                          challenger + '_' + str(100 - self.alpha) + '_quantile': 'dot',

                          champion + '_' + str(self.alpha) + '_quantile': 'dot',
                          champion + '_50_quantile': 'dot',
                          champion + '_' + str(100 - self.alpha) + '_quantile': 'dot',

                          label: 'dashdot',
                          challenger: 'dashdot',
                          champion: 'dashdot'}

        # type of plot setting
        if show == 'delta':
            _title = 'delta CAASB :' + ' Challenger (' + challenger + ')' + ' - Champion (' + champion + ')'
            mask = (df['category'] == 'differences')
        elif show == 'challenger':
            _title = challenger + ': CAASB per capita (avg. over time)'
            mask = (df['category'] == 'absolute') & \
                   (df['type'].isin([challenger, challenger + '_' + str(self.alpha) + '_quantile',
                                     challenger + '_' + str(100 - self.alpha) + '_quantile',
                                     challenger + '_50_quantile']))
        elif show == 'champion':
            _title = champion + ': CAASB per capita (avg. over time)'
            mask = (df['category'] == 'absolute') & \
                   (df['type'].isin([champion, champion + '_' + str(self.alpha) + '_quantile',
                                     champion + '_' + str(100 - self.alpha) + '_quantile',
                                     champion + '_50_quantile']))
        else:
            raise ValueError(f'Value {show} is not in accepted values for show. The accepted values are: '
                             f'delta, {challenger} and {champion}.')

        df = df.loc[mask]
        valuemin = np.nanmin(df[perf_measure].values)
        valuemax = np.nanmax(df[perf_measure].values)

        fig = px.line(df, x='t-t0 (days)',
                      y=perf_measure,
                      animation_frame="history_id",
                      color="type",
                      line_dash="type",
                      color_discrete_map=_color_discrete_map,
                      line_dash_map=_line_dash_map,
                      template="seaborn",
                      facet_col='category',
                      title=_title,
                      range_y=[valuemin, valuemax],
                      labels={perf_measure: 'delta in CAASB (cumulative over time)',
                              't-t0 (days)': 'days since start of daily segmentations'},
                      line_shape='spline',
                      height=800)

        fig.show()

    def plot_cutoffs(self, df, _metrics=None, savefile=None):
        """A plot of cutoffs as a boxplot showing how big ratio is suitable for a High workflow.

        Args:
            df (pd.DataFrame): dataset, origin in  create_trajectories.
            _metrics (list, optional): the list of names of models, must be the same as in df['type'].
                Defaults to None.
            savefile (str, optional): the file path and filename, None means the plot is not saved
                Defaults to None.

        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot()
        minimum = -0.1
        maximum = 1.1

        ax.set_title('Segmentation cutoff: POP%  <= cutoff go to HIGH, POP% > cutoff go to LOW  | winner', fontsize=16)
        ax.set_ylim([minimum, maximum])
        # ax2.yaxis.set_major_formatter(formatter)
        ax.set_facecolor('xkcd:ice')
        ax.grid(True)

        data_to_plot = []
        ticklabels = []

        if _metrics is None:
            _metrics = self.uplift_metrics

        for metric in _metrics:
            D = df[df['winner'] == metric]
            collectn = list(D['top_POP%_cutoff(HIGH)'].values)
            data_to_plot.append(collectn)
            ticklabels.append(metric)

        ax.set_xticklabels(ticklabels, fontsize=12, rotation='vertical')
        bp = ax.boxplot(data_to_plot, patch_artist=True)

        ## change outline color, fill color and linewidth of the boxes
        for box in bp['boxes']:
            # change outline color
            box.set(color='#7570b3', linewidth=2, alpha=0.5)
            # change fill color
            box.set(facecolor='#1b9e77')

        ## change color and linewidth of the whiskers
        for whisker in bp['whiskers']:
            whisker.set(color='#7570b3', linewidth=2)

        ## change color and linewidth of the caps
        for cap in bp['caps']:
            cap.set(color='#7570b3', linewidth=2)

        ## change color and linewidth of the medians
        for median in bp['medians']:
            median.set(color='#b2df8a', linewidth=2)

        ## change the style of fliers and their fill
        for flier in bp['fliers']:
            flier.set(marker='o', color='#e7298a', alpha=0.5)

        if savefile:
            plt.savefig(savefile, )
        plt.show()

    def plot_caasb(self, df, _metrics=None, savefile=None):
        """A plot of average CAASB per capita in the currency of the dataset.

        Args:
            df (pd.DataFrame): a dataset with all the models, in single strategies ideally. Origin: create_trajectories
            _metrics (list, optional): the list of names of models, must be the same as in df['type']
                Defaults to None.
            savefile (str, optional): the file path and filename, None means the plot is not saved
                Defaults to None.
            
        """

        # http://blog.bharatbhole.com/creating-boxplots-with-matplotlib/

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot()

        minimum = df['CAASB_per_capita at t'].values.min()
        maximum = df['CAASB_per_capita at t'].values.max()

        ax.set_title('actual CAASB per capita in portfolio | winner', fontsize=16)
        ax.set_ylim([1.1 * minimum, 1.1 * maximum])
        ax.set_facecolor('xkcd:ice')
        ax.grid(True)

        data_to_plot = []
        ticklabels = []

        if _metrics is None:
            _metrics = self.uplift_metrics

        for metric in _metrics:
            D = df[df['type'] == metric]  # was 'winner'
            collectn = list(D['CAASB_per_capita at t'].values)
            data_to_plot.append(collectn)
            ticklabels.append(metric)

        ax.set_xticklabels(ticklabels, fontsize=12)
        bp = ax.boxplot(data_to_plot, patch_artist=True)

        ## change outline color, fill color and linewidth of the boxes
        for box in bp['boxes']:
            # change outline color
            box.set(color='#7570b3', linewidth=2, alpha=0.5)
            # change fill color
            box.set(facecolor='#1b9e77')

        ## change color and linewidth of the whiskers
        for whisker in bp['whiskers']:
            whisker.set(color='#7570b3', linewidth=2)

        ## change color and linewidth of the caps
        for cap in bp['caps']:
            cap.set(color='#7570b3', linewidth=2)

        ## change color and linewidth of the medians
        for median in bp['medians']:
            median.set(color='#b2df8a', linewidth=2)

        ## change the style of fliers and their fill
        for flier in bp['fliers']:
            flier.set(marker='o', color='#e7298a', alpha=0.5)

        if savefile:
            plt.savefig(savefile, )
        plt.show()

    def plot_winning(self, df, savefile=None):
        """Plots a ratio of the day-winnig strategies for the entires date span
        A plot with days on x-axis and percentage of the winners on y-axis. Sum of all ratios is one.

        Args:
            df (pd.DataFrame): a dataset with strategy which contains multiple models (otherwise the winner is the model).
            savefile (str, optional): the file path and filename, None means the plot is not saved

        """
        fig = plt.figure(figsize=(12, 8))

        ax = fig.add_subplot()

        ax.set_title('Winning approach', fontsize=16)
        ax.set_ylabel('pop%', fontsize=16)
        ax.set_ylim([0, 1])
        ax.set_facecolor('xkcd:ice')
        ax.grid(True)

        ax.hist(df['winner'], bins=10, color='#1b9e77', alpha=0.5, cumulative=0, density=False, stacked=False,
                weights=df['weight'] / df['weight'].sum(), rwidth=10, label='pop%')
        if savefile:
            plt.savefig(savefile, )
        plt.show()

    def plot_distribution_caasb(self,
                                df,
                                challenger,
                                champion,
                                show='delta',
                                perf_measure='CAASB_per_capita averaged over t-t0',
                                animation=True,
                                savefile=None):
        """

        Args:
            df (pd.DataFrame): dataset
            challenger (str): name of the challenger model/set of models
            champion (str): name of the champion model/set of models
            show (str, optional): values 'delta' for plotting the differences between champion and challenger
                'absolute' for plotting the absolute values for both
                Defaults to 'delta averaged over t-t0'.
            perf_measure (str, optional): column name for the performance measure which we display
                Defaults to 'CAASB_per_capita averaged over t-t0'.
            animation (bool, optional): True if we want to have the animated results (by days)
                Defaults to True.
            savefile (str, optional): path and name to the file
                Defaults to None.

        """
        label = "'" + challenger + "'" + ' vs ' + "'" + champion + "'"
        _color_discrete_map = {'delta_' + str(self.alpha) + '_quantile': 'orangered',
                               'delta_50_quantile': 'darkmagenta',
                               'delta_' + str(100 - self.alpha) + '_quantile': 'slateblue',
                               label: 'darkmagenta',
                               challenger: 'blue', champion: 'red'}

        if show == 'delta':
            _title = 'Distribution of delta CAASB :' + ' Challenger (' + challenger + ')' + ' - Champion (' + champion + ')'
            mask = (df['category'] == 'differences') & ~ (df['type'].isin(['delta_' + str(self.alpha) + '_quantile',
                                                                           'delta_50_quantile',
                                                                           'delta_' + str(
                                                                               100 - self.alpha) + '_quantile']))
        elif show == 'absolute':
            _title = 'Distribution of' + perf_measure
            mask = (df['category'] == 'absolute') & (df['type'].isin([challenger, champion]))

        else:
            raise ValueError(f'Value {show} is not in accepted values for show. The accepted values are: '
                             f'delta and absolute.')

        df = df.loc[mask]
        if animation:
            animation_frame = 't-t0 (days)'
        else:
            animation_frame = None

        fig1 = px.histogram(df, x=perf_measure,
                            nbins=100, opacity=0.5,
                            color="type",
                            color_discrete_map=_color_discrete_map,
                            animation_frame=animation_frame,
                            title='distribution of : ' + perf_measure,
                            marginal="box",
                            template="seaborn",
                            facet_col='category',
                            labels={perf_measure: 'CAASB per capita (CZK)'},
                            height=600)

        if savefile:
            plt.savefig(savefile, )

        fig1.show()
