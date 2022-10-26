import abc
import math as math
import re
import warnings
from os import path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import ks_2samp
from scipy.special import logit, expit

import xgboost as xgb

from .auxiliary import RussianImputer, is_number

from IPython.display import display


class ProjectParameters(object):
    """
    Storage of metadata for usage in Doctools and wrappers for most used
    methods of producing plots and outputs.
    Metadata need to be manualy set and plotting methods then use them 
    to calculate and export plots.

    """
    sample_dict = {}
    rowid_variable = "SKP_CREDIT_CASE"
    targets = []
    time_variable = "DATE_DECISION"
    segments = []
    predictors_woe = []
    predictors_grp = []
    scores = []
    masks = {}
    weight = None

    def _processSample(self, samples):
        """Returns a tuple (sample[pd.sr], name[str]) of mask and name for given sample or multiple samples.      

        Args:
            samples ([str, list(str)]): Names of samples, need to exist in ProjectParameters.masks dictionary.
        
        Returns:
            tuple (sample[pd.sr], name[str]): mask sample or multiple samples
        """
        from functools import reduce

        if isinstance(samples, str):
            if samples in self.sample_dict.keys():
                return tuple([self.sample_dict[samples], samples])
            elif samples.upper() in ['ALL']:
                # get copy of a random mask and set all to True
                all_mask = next(iter(self.sample_dict.values())).copy()
                all_mask.loc[:] = True
                return tuple([all_mask, "All"])
            else:
                raise ValueError(f"Sample `{samples}` not in documentations samples.")
        elif isinstance(samples, list):
            for sample in samples:
                if isinstance(sample, str):
                    if sample not in self.sample_dict.keys():
                        raise ValueError(f"Sample `{sample}` not in documentations samples.")
                else:
                    raise ValueError(f"Sample `{sample}` is not valid sample name.")
            masks = [self.sample_dict[name] for name in samples]
            merged_masks = reduce(lambda a, b: a | b, masks)
            merged_names = '+'.join(samples)
            return tuple([merged_masks, merged_names])
        else:
            raise ValueError(f"Sample `{samples}` not a valid sample.")

    def _processMasks(self, masks):
        """Returns a list of tuples [(sample[pd.sr], name[str]), ] of mask and name for given sample or multiple samples..
        Args:
           masks ([str, list(str)]): Names of masks, need to exist in ProjectParameters.masks dictionary.

        Returns:
            list of tuples [(sample[pd.sr], name[str]), ]: mask sample or multiple samples
        """
        # from functools import reduce

        if isinstance(masks, str):
            if masks in self.sample_dict.keys():
                return tuple([self.sample_dict[masks], masks])
            elif masks.upper() in ['ALL']:
                # get copy of a random mask and set all to True
                all_mask = next(iter(self.sample_dict.values())).copy()
                all_mask.loc[:] = True
                return tuple([all_mask, "All"])
            else:
                raise ValueError(f"Sample `{masks}` not in documentations masks.")
        elif isinstance(masks, list):
            joined_masks = []
            for mask in masks:
                if isinstance(mask, str):
                    if mask not in self.sample_dict.keys():
                        raise ValueError(f"Mask `{mask}` not in documentations masks.")
                else:
                    raise ValueError(f"Mask `{mask}` is not valid sample name.")

            joined_masks = [self.sample_dict[name] for name in masks]

            return list(zip(joined_masks, masks))
        else:
            raise ValueError(f"Mask `{masks}` not a valid sample.")

    # def _processSample(self, samples):
    #     """Returns a tuple (sample[pd.df], name) for specified sample
    #     or union of samples

    #     Args:
    #         samples ([str, list(str)]): Sample names or list of names. Must exist in documentation

    #     Returns:
    #         ([pd.df],[str]): A tuple of sample and name of sample
    #     """
    #     if isinstance(samples, str):
    #         if samples in self.sample_dict.keys():
    #             return tuple([self.sample_dict[samples], samples])
    #         elif samples.upper() in ['ALL']:
    #             merged_samples = pd.concat([sample for name, sample in self.sample_dict.items()])
    #             return tuple([merged_samples, "All"])
    #         else:
    #             raise ValueError(f"Sample `{samples}` not in documentations samples.")
    #     elif isinstance(samples, list):
    #         for sample in samples:
    #             if isinstance(sample, str):
    #                 if sample not in self.sample_dict.keys():
    #                     raise ValueError(f"Sample `{sample}` not in documentations samples.")
    #             else:
    #                 raise ValueError(f"Sample `{sample}` is not valid sample name.")
    #         merged_samples = pd.concat([self.sample_dict[sample] for sample in samples])
    #         merged_names = '+'.join(samples)
    #         return tuple([merged_samples, merged_names])
    #     else:
    #         raise ValueError(f"Sample `{samples}` not a valid sample.")

    def _processTarget(self, targets):
        """"Returns a tuple (target, base) of column names."
        Args:
            targets (list):
        Returns:
            [tuple]: list of tuples of column names (target,base)
        """
        try:
            target_tuple = [(tar, base) for tar, base in self.targets if tar == targets][0]
        except:
            raise IndexError(f"The given target {targets} is not defined in doctools.targets.")
        return target_tuple

    def ContinuousEvaluation(self, data, predictor, sample, target, output_folder=None, show_plot=False):
        """

        Args:
            data:
            predictor (str): predictor column name in the data
            sample:
            target:
            output_folder:
            show_plot:

        Returns:

        """
        cec = ContinuousEvaluationCalculator(self)
        # plt.rcParams["lines.marker"] = "D"

        sample_mask, sample_name = self._processSample(sample)
        sample_tuple = (data[sample_mask], sample_name)

        if self.rowid_variable not in sample_tuple[0].columns:
            pd.set_option('mode.chained_assignment', None)
            sample_tuple[0][self.rowid_variable] = sample_tuple[0].index
            pd.set_option('mode.chained_assignment', 'warn')

        target_tuple = self._processTarget(target)
        cec.s([sample_tuple]).p([predictor]).t([target_tuple])
        cec.calculate().get_visualization(output_folder=output_folder, show_plot=show_plot)

    def GroupedEvaluation(self, data, predictor, sample, target, grouping=None, use_weight=False, show_gini=True,
                          output_folder=None, display_table=False, show_plot=True):
        """
        Args:
            data (pd.DataFrame): the dataset
            predictor (str): predictor column name in the data
            sample (list of str): name or names in list of sample defined in self.sample_dict.keys()
            target (str): column name for target
            grouping (object, optional): object NewGrouping from scoring
            use_weight (bool, optional):
            show_gini (bool, optional):
            output_folder (str, optional):
            display_table (bool, optional):
            show_plot (bool, optional):

        """
        gec = GroupingEvaluationCalculator(self, use_weight=use_weight)
        # plt.rcParams["lines.marker"] = "D"

        sample_mask, sample_name = self._processSample(sample)
        sample_tuple = (data[sample_mask], sample_name)
        if self.rowid_variable not in sample_tuple[0].columns:
            pd.set_option('mode.chained_assignment', None)
            sample_tuple[0][self.rowid_variable] = sample_tuple[0].index
            pd.set_option('mode.chained_assignment', 'warn')

        target_tuple = self._processTarget(target)
        gec.s([sample_tuple]).p([predictor]).t([target_tuple]).g(grouping).w(self.weight)
        gec.calculate().get_visualization(show_gini=show_gini, output_folder=output_folder, show_plot=show_plot)
        if display_table:
            display(gec.get_table(beautiful=True))

    def Correlations(self, data, predictors, sample, use_weight=False, output_folder=None, filename=None,
                     show_plot=True):
        """ Creates a correlation matrix for given predictors. Has weighted and non weighted versions implemented.

        Args:
            data (pd.DataFrame): the dataset
            predictors (list): List of all predictors which are taken in account for correlation matrix.
            sample (list, str): String or list of strings with the sample mask(s) which will be used on dataset.
            use_weight (bool, optional): If True, weighted correlation is computed.
            output_folder (str, optional): Name of output folder. If not given, plot is not saved.
            filename (str, optional): if not given, default name is used (correlation.png)
            show_plot (bool, optional): If True, shows plot in Jupyter Notebook. Default True.
        """
        sample_mask, sample_name = self._processSample(sample)
        sample_tuple = (data[sample_mask], sample_name)

        # weighted correlation
        if use_weight:
            def covariance(x, y, w):
                return np.average((x - np.average(x, weights=w)) * (y - np.average(y, weights=w)), weights=w)

            def correlation(x, y, w):
                return covariance(x, y, w) / np.sqrt(covariance(x, x, w) * covariance(y, y, w))

            corr_matrix = np.empty((len(predictors), len(predictors)), dtype=np.float)
            for i, pred in enumerate(sorted(predictors)):
                for j, pred2 in enumerate(sorted(predictors)):
                    if i >= j:
                        corr = correlation(x=sample_tuple[0][pred], y=sample_tuple[0][pred2],
                                           w=sample_tuple[0][self.weight])
                        corr_matrix[i, j] = corr
                        corr_matrix[j, i] = corr
                    if i == j:
                        corr_matrix[i, j] = 1.0
                    else:
                        continue
            cormat_full = pd.DataFrame(corr_matrix, index=sorted(predictors), columns=sorted(predictors))

        else:
            cormat_full = sample_tuple[0][sorted(predictors)].fillna(0).corr()

        a4_dims = (12, 10)

        fig, ax = plt.subplots(figsize=a4_dims, dpi=50)
        fig.suptitle('Correlations of Variables', fontsize=25)
        sns.heatmap(cormat_full, ax=ax, annot=True, fmt="0.1f", linewidths=.5, annot_kws={"size": 15}, cmap="OrRd")
        plt.tick_params(labelsize=15)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        if output_folder is not None:
            if not filename:
                filename = "correlation.png"
            plt.savefig(path.join(output_folder, filename), bbox_inches='tight', dpi=200, )
        if show_plot:
            plt.show()
        plt.clf()
        plt.close()

    def ScoreCalibration(self, data, score, sample, target, bins=50, use_weight=False, output_folder=None,
                         swap_probability=False, filename="calibration.png", show_plot=True):
        """
        Args:
            data (pd.DataFrame): the dataset with score, target, (weight)
            score (str): column name for score (only one score for one function call)
            sample (list of str): name or names in list of sample defined in self.sample_dict.keys()
            target (str): column name for target
            bins (int, optional): number of bins for percentiles' creation
            use_weight (bool, optional): True for using the self.weight, False for calculating without weights
            output_folder (str, optional): name of the output folder. If None, plot is not saved anywhere
            swap_probability (bool, optional): If True, average score is changed to (1-avg(score))
            filename (str, optional): Name of the plot. If none specified, a default is used
            show_plot (bool, optional): True for displaying plot in notebook
        """
        sample_mask, sample_name = self._processSample(sample)
        sample_tuple = (data[sample_mask], sample_name)

        scores = []
        brs = []

        if use_weight:
            score_weight = sample_tuple[0][[self.weight, score]]
            score_weight = score_weight.sort_values(by=score)
            weighted_sum = sum(score_weight[self.weight])
            weight_cum = np.cumsum(score_weight[self.weight])
            weight_perc = weight_cum / weighted_sum
            weight_target = sample_tuple[0][target]
            bins = np.percentile(weight_perc, np.linspace(0, 100, bins + 1))

            for b in zip(bins[:-1], bins[1:]):
                if swap_probability:
                    scores += [1 - score_weight[(weight_perc >= b[0]) & (weight_perc < b[1])][score].mean()]
                else:
                    scores += [score_weight[(weight_perc >= b[0]) & (weight_perc < b[1])][score].mean()]
                brs += [weight_target[(weight_perc >= b[0]) & (weight_perc < b[1])].mean()]
        else:
            score = sample_tuple[0][score]
            target = sample_tuple[0][target]
            bins = np.percentile(score, np.linspace(0, 100, bins + 1))
            for b in zip(bins[:-1], bins[1:]):
                if swap_probability:
                    scores += [1 - score[(score >= b[0]) & (score < b[1])].mean()]
                else:
                    scores += [score[(score >= b[0]) & (score < b[1])].mean()]
                brs += [target[(score >= b[0]) & (score < b[1])].mean()]

        plt.scatter(scores, brs)
        upperlimit = np.nanmax(scores + brs)
        plt.xlim([0, upperlimit])
        plt.ylim([0, upperlimit])
        plt.plot(np.linspace(0, upperlimit, 1000), np.linspace(0, upperlimit, 1000), color='red', marker="None")
        plt.grid()
        plt.ylabel('default rate')
        plt.xlabel('prediction')
        if output_folder is not None:
            plt.savefig(path.join(output_folder, filename), bbox_inches='tight', dpi=200)
        if show_plot:
            plt.show()
        plt.close()

    def ScoreComparison(self, data, scores, sample, target, output_folder=None, filename=None, show_plot=False):
        scc = ScoreComparisonCalculator(self)
        # plt.rcParams["lines.marker"] = "D"

        sample_mask, sample_name = self._processSample(sample)
        sample_tuple = (data[sample_mask], sample_name)
        if self.rowid_variable not in sample_tuple[0].columns:
            pd.set_option('mode.chained_assignment', None)
            sample_tuple[0][self.rowid_variable] = sample_tuple[0].index
            pd.set_option('mode.chained_assignment', 'warn')

        target_tuple = self._processTarget(target)
        scc.s([sample_tuple]).p(scores).t([target_tuple])
        scc.calculate().get_visualization(output_folder=output_folder, filename=filename, show_plot=show_plot)

    def GroupingPlots(self, data, predictors, sample, target, grouping, use_weight=False, output_folder=None):
        sample_mask, sample_name = self._processSample(sample)
        sample_tuple = (data[sample_mask], sample_name)
        if self.rowid_variable not in sample_tuple[0].columns:
            pd.set_option('mode.chained_assignment', None)
            sample_tuple[0][self.rowid_variable] = sample_tuple[0].index
            pd.set_option('mode.chained_assignment', 'warn')

        target_tuple = self._processTarget(target)

        for col in predictors:
            if col not in grouping.bins_data_.keys():
                raise ValueError(f"{col} is missing in grouping.")

        predictors_categorical = [pred for pred in predictors if pred in grouping.cat_columns]
        predictors_numerical = [pred for pred in predictors if pred in grouping.columns]

        # create a mask that contains whole sample
        fake_mask = [True] * sample_tuple[0].shape[0]

        if use_weight:
            weight = self.weight
        else:
            weight = None

        grouping.plot_bins(data=sample_tuple[0],
                           cols_pred_num=predictors_numerical,
                           cols_pred_cat=predictors_categorical,
                           mask=fake_mask,
                           col_target=target_tuple[0],
                           output_folder=output_folder,
                           col_weight=weight
                           )

    def PlotDataset(self, data, sample, target, segment_col=None, zero_ylim=True, use_weight=False, output_folder=None,
                    filename=None, show_plot=False):
        """Plots the count of contracts and badrate in months (or other defined time intervals).
        Calls :py:meth:`~scoring.doctools.calculators.PlotDatasetCalculator`

        Args:
            data (pd.DataFrame): the dataset with target, (weight)
            sample (list of str): name or names in list of sample defined in ProjectParameters
            target (str): column name for target
            segment_col (str, optional): Segment-by column. If given, the whole plot is drawn with distinct curves for each
                of the column's values.
            zero_ylim (bool): True means, the y-axis starts with 0.
            use_weight (bool): If True, the weight defined in ProjectParameters is used for calculating weighted version.
            If False, no weight is used.
            output_folder (str, optional): if given, the plot is saved to the folder, if None, plot is not saved
            filename (str, optional): if not given, default 'data_badrate.png' is used
            show_plot (bool, optional): Default True. If True, the plot is shown in iPython.
        """
        pld = PlotDatasetCalculator(self, segment_col=segment_col, zero_ylim=zero_ylim, use_weight=use_weight)

        sample_mask, sample_name = self._processSample(sample)
        sample_tuple = (data[sample_mask], sample_name)
        if self.rowid_variable not in sample_tuple[0].columns:
            pd.set_option('mode.chained_assignment', None)
            sample_tuple[0][self.rowid_variable] = sample_tuple[0].index
            pd.set_option('mode.chained_assignment', 'warn')

        target_tuple = self._processTarget(target)
        pld.s([sample_tuple]).t([target_tuple]).w(self.weight)
        pld.calculate().get_visualization(output_folder=output_folder, filename=filename, show_plot=show_plot)

    def DataSampleSummary(self, data, sample, target, segment_col=None, use_weight=False, output_folder=None,
                          filename=None):
        """Creates summary table about the data (number of observations, bad rate, by segment, by time).
        Calls :py:meth:`~scoring.doctools.calculators.DataSampleSummaryCalculator`

        Args:
            data (pd.DataFrame): the dataset with score, target, (weight)
            sample (list of str): name or names in list of sample defined in self.sample_dict.keys()
            target (str): column name for target
            segment_col (str, optional): column name for segment. Defaults to None.
            use_weight (bool, optional): True for using the self.weight, False for calculating without weights
            output_folder (str, optional): name of the output folder. If None, plot is not saved anywhere
            filename (str, optional): Name of the plot. If none specified, a default is used

        Returns:
            pd.DataFrame: summary table
        """
        dssc = DataSampleSummaryCalculator(self, segment_col=segment_col, use_weight=use_weight)

        sample_mask, sample_name = self._processSample(sample)
        sample_tuple = (data[sample_mask], sample_name)
        if self.rowid_variable not in sample_tuple[0].columns:
            pd.set_option('mode.chained_assignment', None)
            sample_tuple[0][self.rowid_variable] = sample_tuple[0].index
            pd.set_option('mode.chained_assignment', 'warn')

        target_tuple = self._processTarget(target)
        dssc.s([sample_tuple]).t([target_tuple]).w(self.weight).tm(self.time_variable)
        dssc.calculate()
        results = dssc.get_table()
        if output_folder:
            results.to_csv(path.join(output_folder, filename))
        return results

    def PredictorPowerAnalysis(self, data, sample, predictors, target, sort_by=None, use_weight=False, masks=None,
                               output_folder=None, filename="covariates.csv"):
        """
        Gini and Information Value analysis for each predictor.
        Calls :py:meth:`~scoring.doctools.calculators.PredictorGiniIVCalculator`

        Args:
            data (pandas DataFrame): dataset
            sample (str, list): String or list of strings which defines all samples used for the computation. The samples
                are merged into one and then used for calculation. If you want to see the samples one by one, use the attribute
            masks (described lower). Sample data is used only in case that masks are None.
                Must be the same name as in self.sample_dict.keys().
            predictors (list): List of predictors for the power analysis, usually weight of evidence.
            target (str): string of the target, must be defined in doctools.ProjectParameters()
            sort_by (str, optional): Name of the mask according to which the values will be sorted.
            use_weight (bool, optional): If True, the weight from doctools.ProjectParameters() is used.
            masks (list, optional): List of strings, for showing multiple samples (test, train, out of time...)
            output_folder (str, optional): The folder name. If not given, output is not saved.
            filename (str, optional): Filename, if not given, default value is used (covariates.csv).
        """
        ppg = PredictorGiniIVCalculator(self, use_weight=use_weight)
        target_tuple = self._processTarget(target)
        if masks:
            masks_all = self._processMasks(masks)
            results = []
            for mask, mask_name in masks_all:
                if mask.sum() > 0:
                    mask_tuple = (data[mask], mask_name)
                    if self.rowid_variable not in mask_tuple[0].columns:
                        pd.set_option('mode.chained_assignment', None)
                        mask_tuple[0][self.rowid_variable] = mask_tuple[0].index
                        pd.set_option('mode.chained_assignment', 'warn')

                    ppg.s([mask_tuple]) \
                        .t([target_tuple]) \
                        .w(self.weight) \
                        .p(predictors)

                    results.append(ppg.calculate().get_table())
            results = [item for la in results for item in la]
        else:
            sample_mask, sample_name = self._processSample(sample)
            sample_tuple = (data[sample_mask], sample_name)

            if self.rowid_variable not in sample_tuple[0].columns:
                pd.set_option('mode.chained_assignment', None)
                sample_tuple[0][self.rowid_variable] = sample_tuple[0].index
                pd.set_option('mode.chained_assignment', 'warn')

            ppg.s([sample_tuple]) \
                .t([target_tuple]) \
                .w(self.weight) \
                .p(predictors)

            results = ppg.calculate().get_table()

        results = pd.DataFrame(results).pivot(index='predictor_name', columns='metrics', values=['gini', 'iv'])
        results = results.dropna(axis=1, how='all')
        if sort_by:
            results = results.sort_values(by=[('gini', "Gini " + sort_by)], ascending=False)

        if output_folder is not None:
            results.to_csv(path.join(output_folder, filename))

        return results

    def LibrariesVersion(self, psw_version, output_folder=None, filename='libraries_version.csv'):
        """ Shows versions of all required python libraries listed in __init__ requirements which were used for the workflow.

        Args:
            psw_version (str): Python Scoring Workflow version used for modelling
            output_folder (str, optional): Name of output folder. If None, the list of versions is NOT saved
            filename (str, optional): Filename. If not given, the default will be used.

        Returns:
            list_versions (pandas DataFrame): DataFrame with libraries versions and their name as index
        """
        from scoring import check_version  # to get to the REQUIREMENTS - maintained version
        list_versions = check_version(psw_version, list_versions_noprint=True)

        list_versions = pd.DataFrame(list_versions).set_index('name')
        if output_folder is not None:
            list_versions.to_csv(path.join(output_folder, filename))
        return list_versions

    def ModelPerformanceGiniLiftKS(self, data, sample, target, scores, lift_perc=10, use_weight=False, masks=None,
                                   output_folder=None, filename='performance.csv'):
        """
            Shows the gini, lift and Kolmogorov-Smirnov statistics.
            Calls :py:meth:`~scoring.doctools.calculators.ScoreModelPerformanceCalculator`

        Args:
            data (numpy DataFrame): the dataset
            sample (str, list of str): list or string of the sample name(s) defined in self.sample_dict.keys()
            target (str): string of the target, must be defined in ProjectParameters()
            scores (str, list of str): list or string of the score(s), must be defined in ProjectParameters()4
            lift_perc (float): Percentile of lift.
            masks (list of str, optional): list of the samples' masks, eg. train, valid, names must be defined in
                ProjectParameters()
            use_weight (bool, optional): If True, the weight defined in ProjectParameters() is used, default False.
            output_folder (str): output folder name, if None, the plot is not saved
            filename (str): filename, if None, default value "roc.png" is used
        """
        ppg = ScoreModelPerformanceCalculator(self, use_weight=use_weight, lift_perc=lift_perc)

        if isinstance(scores, list):
            if len(scores) == len(set(scores)):
                pass
            else:
                raise ValueError(f"Some of the scores: {scores} are duplicated.Please check and do not use same column "
                                 f"as different scores or disable the Old scores Comparison(s).")

        if masks:
            results = []
            for mask in masks:
                sample_mask, sample_name = self._processSample(mask)
                if sample_mask.sum() > 0:
                    sample_tuple = (data[sample_mask], sample_name)
                    if self.rowid_variable not in sample_tuple[0].columns:
                        pd.set_option('mode.chained_assignment', None)
                        sample_tuple[0][self.rowid_variable] = sample_tuple[0].index
                        pd.set_option('mode.chained_assignment', 'warn')

                    target_tuple = self._processTarget(target)
                    ppg.s([sample_tuple]).t([target_tuple]).w(self.weight).sc(scores)
                    results.append(ppg.calculate().get_table())
            results = [item for la in results for item in la]
        else:
            sample_mask, sample_name = self._processSample(sample)
            if sample_mask.sum() > 0:
                sample_tuple = (data[sample_mask], sample_name)
                if self.rowid_variable not in sample_tuple[0].columns:
                    pd.set_option('mode.chained_assignment', None)
                    sample_tuple[0][self.rowid_variable] = sample_tuple[0].index
                    pd.set_option('mode.chained_assignment', 'warn')

                target_tuple = self._processTarget(target)
                ppg.s([sample_tuple]).t([target_tuple]).w(self.weight).sc(scores)
                results = ppg.calculate().get_table()

        results = pd.DataFrame(results).pivot(index='sample', columns='score',
                                              values=['gini', f'lift_{lift_perc}', 'KS'])
        if isinstance(scores, str):
            results = results.droplevel("score", axis=1)

        if output_folder is not None:
            results.to_csv(path.join(output_folder, filename))
        return results

    def LiftCurve(self, data, sample, scores, target, masks=None, use_weight=False, lift_perc=10,
                  list_lift=[5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], output_folder=None,
                  filename=None, show_plot=True):
        """ Visualise the lift curve for given list of distinct percentages. Can be called for multiple masks.
        Calls :py:meth:`~scoring.doctools.calculators.ScoreLiftBySegmentsCalculator`

        Args:
            data (pd.DataFrame): the dataset with score, target, (weight)
            scores (list of str): column name for score (only one score for one function call)
            sample (list of str): name or names in list of sample defined in self.sample_dict.keys()
            target (str): column name for target
            masks (list, optional): if any, the plot will have one curve for each mask and each score
            use_weight (bool, optional): True for using the self.weight, False for calculating without weights
            lift_perc (int, optional): percentage of lift, for our usage obsolete.
            list_lift (list of int, optional): List of desired lift percentages, in range (0, 100].If not given, default
                values are used.
            output_folder (str, optional): name of the output folder. If None, plot is not saved anywhere
            filename (str, optional): Name of the plot. If none specified, a default is used
            show_plot (bool, optional): True for displaying plot in notebook
        """
        target_tuple = self._processTarget(target)
        if masks:
            sample_masks = self._processMasks(masks)
            sample_tuple = [(data[sample_mask], sample_name) for sample_mask, sample_name in sample_masks]
            for i in range(len(masks)):
                if self.rowid_variable not in sample_tuple[i][0].columns:
                    pd.set_option('mode.chained_assignment', None)
                    sample_tuple[i][0][self.rowid_variable] = sample_tuple[i][0].index
                    pd.set_option('mode.chained_assignment', 'warn')

        else:
            sample_mask, sample_name = self._processSample(sample)
            sample_tuple = (data[sample_mask], sample_name)
            if self.rowid_variable not in sample_tuple[0].columns:
                pd.set_option('mode.chained_assignment', None)
                sample_tuple[0][self.rowid_variable] = sample_tuple[0].index
                pd.set_option('mode.chained_assignment', 'warn')

        lift = ScoreLiftBySegmentsCalculator(self, list_lift=list_lift, use_weight=use_weight, lift_perc=lift_perc,
                                             masks=masks)

        lift.s([sample_tuple]).t([target_tuple]).w(self.weight).sc(scores)

        lift.calculate().get_visualization(output_folder=output_folder, filename=filename, show_plot=show_plot)

    def ROCCurve(self, data, sample, scores, target, masks=None, use_weight=False, output_folder=None, filename=None,
                 show_plot=True):
        """
        Visualise the ROC curve. Calls :py:meth:`~scoring.doctools.calculators.ScoreROCBySegmentsCalculator`
        Calls :py:meth:`~scoring.doctools.calculators.ScoreROCBySegmentsCalculator`

        Args:
            data (numpy DataFrame): the dataset
            sample (str, list of str): list or string of the sample name(s) defined in self.sample_dict.keys()
            scores (str, list of str): list or string of the score(s), must be defined in doctools.ProjectParameters()
            target (str): string of the target, must be defined in doctools.ProjectParameters()
            masks (list of str, optional): list of the samples' masks, eg. train, valid, names must be defined in
            doctools.ProjectParameters()
            use_weight (bool, optional): If True, the weight defined in doctools.ProjectParameters() is used, default False
            output_folder (str): output folder name, if None, the plot is not saved
            filename (str): filename, if None, default value "roc.png" is used
            show_plot (bool): If True, plot will be shown in Jupyter notebook. Default True.
        """
        target_tuple = self._processTarget(target)
        if masks:
            sample_masks = self._processMasks(masks)
            sample_tuple = [(data[sample_mask], sample_name) for sample_mask, sample_name in sample_masks]
            for i in range(len(masks)):
                if self.rowid_variable not in sample_tuple[i][0].columns:
                    pd.set_option('mode.chained_assignment', None)
                    sample_tuple[i][0][self.rowid_variable] = sample_tuple[i][0].index
                    pd.set_option('mode.chained_assignment', 'warn')

        else:
            sample_mask, sample_name = self._processSample(sample)
            sample_tuple = (data[sample_mask], sample_name)
            if self.rowid_variable not in sample_tuple[0].columns:
                pd.set_option('mode.chained_assignment', None)
                sample_tuple[0][self.rowid_variable] = sample_tuple[0].index
                pd.set_option('mode.chained_assignment', 'warn')

        roc = ScoreROCBySegmentsCalculator(self, use_weight=use_weight, masks=masks)

        roc.s([sample_tuple]).t([target_tuple]).w(self.weight).sc(scores)

        roc.calculate().get_visualization(output_folder=output_folder, filename=filename, show_plot=show_plot)

    def ScoreGiniBootstrap(self, data, sample, scores, target, masks=None, use_weight=False, n_iter=100, ci_range=5,
                           random_seed=42, col_score_ref=None, output_folder=None,
                           filename="bootstrap_performance.csv"):
        """ Calls :py:meth:`~scoring.doctools.calculators.BootstrapGiniCalculator`

        Args:
            data (numpy DataFrame): the dataset
            sample (str, list of str): list or string of the sample name(s) defined in self.sample_dict.keys()
            scores (str, list of str): list or string of the score(s), must be defined in doctools.ProjectParameters()
            target (str): string of the target, must be defined in doctools.ProjectParameters()
            use_weight (bool): True for using weight defined in ProjectParameters
            masks (list of strings, optional): If filled, the distinct ROC curve is drawn for each masked sample
            n_iter (int, optional): number of iterations for the bootstrap. Default 100.
            ci_range (float, optional): percent of confidence intervals, ci_range and (1 - ci_range). Default 5.
            random_seed (int, optional): random seed for the random function, can be set the same way to reconstruct the old results.
                Default... you already know it, right?
            col_score_ref (str, optional): A reference score column in the dataset.
                If given, the whole gini bootstrap is computed as a difference to this score.
            output_folder (str, optional): output folder name, if None, the plot is not saved
            filename (str, optional): filename, if None, default value "bootstrap_performance.csv" is used

        Returns: Resulting statistics as a pandas DataFrame.

        """
        target_tuple = self._processTarget(target)
        if masks:
            sample_masks = self._processMasks(masks)
            sample_tuple = [(data[sample_mask], sample_name) for sample_mask, sample_name in sample_masks]
            for i in range(len(sample_tuple)):
                if self.rowid_variable not in sample_tuple[i][0].columns:
                    pd.set_option('mode.chained_assignment', None)
                    sample_tuple[i][0][self.rowid_variable] = sample_tuple[i][0].index
                    pd.set_option('mode.chained_assignment', 'warn')

        else:
            sample_mask, sample_name = self._processSample(sample)
            sample_tuple = (data[sample_mask], sample_name)
            if self.rowid_variable not in sample_tuple[0].columns:
                pd.set_option('mode.chained_assignment', None)
                sample_tuple[0][self.rowid_variable] = sample_tuple[0].index
                pd.set_option('mode.chained_assignment', 'warn')

        bgc = BootstrapGiniCalculator(self, use_weight=use_weight, masks=masks, n_iter=n_iter, ci_range=ci_range,
                                      random_seed=random_seed, col_score_ref=col_score_ref)

        bgc.s([sample_tuple]).t([target_tuple]).w(self.weight).sc(scores)

        results = bgc.calculate().get_table()
        results = pd.DataFrame(results).sort_values(by="Score name")

        if output_folder is not None:
            results.to_csv(path.join(output_folder, filename), index=False)

        return results

    def GiniLiftInTimeScore(self, data, sample, scores, target, masks=None, use_weight=False,
                            lift_perc=15, get_gini=True, get_lift=True, show_plot=True, output_folder=None,
                            filename_gini=None, filename_lift=None):
        """Plots Gini or/and lift values in time, depending on set values of get_lift and get_gini. Lift percentage is
        settable. Calls :py:meth:`~scoring.doctools.calculators.ScoreGiniIVCalculator`

        Args:
            data (pandas DataFrame): the dataset
            sample (str, list): String or list of sample(s).
            scores (str, list): String or list of scores. Do not set duplicities in scores.
            target (str, list): String or list of targets. Do not set duplicities even in targets.
            masks (list, optional): List of masks' names, must be defined in ProjectParameters(). For each mask is drawn
                distinct line in a plot.
            use_weight (bool, optional): Default False. If True, weight defined in ProjectParameters() is used.
            lift_perc (number, optional): Default 15 %. Change to any value from (0,100) if needed.
            get_gini (bool, optional): Default True. Computes gini for given time variable (usually months).
            get_lift (bool, optional): Default True. Computes lift for given time variable (usually months).
            show_plot (bool, optional): Default True. If True, plot(s) will be drawn in Jupyter Notebook.
            output_folder (str, optional): If None, plots are not saved
            filename_gini (str, optional): If None, default value is used ()
            filename_lift (str, optional): If None, default value is used ()
        """
        if isinstance(target, list):
            if len(target) == len(set(target)):  # Validation for multiple target columns' duplicities
                pass
            else:
                raise ValueError(f"Some of the targets: {target} are duplicated."
                                 f"Please check and do not use same column as different targets or disable the Short Target Comparison(s).")

            target_tuple = []
            for tgt in target:
                target_tuple.append(self._processTarget(tgt))
        else:
            target_tuple = self._processTarget(target)

        if isinstance(scores, list):
            if len(scores) == len(set(scores)):  # Validation for multiple scores columns' duplicities
                pass
            else:
                raise ValueError(f"Some of the scores: {scores} are duplicated."
                                 f"Please check and do not use same column as different scores or disable the Old scores Comparison(s).")

        if masks:
            sample_masks = self._processMasks(masks)
            sample_tuple = [(data[sample_mask], sample_name) for sample_mask, sample_name in sample_masks]
            for i in range(len(masks)):
                if self.rowid_variable not in sample_tuple[i][0].columns:
                    pd.set_option('mode.chained_assignment', None)
                    sample_tuple[i][0][self.rowid_variable] = sample_tuple[i][0].index
                    pd.set_option('mode.chained_assignment', 'warn')

        else:
            sample_mask, sample_name = self._processSample(sample)
            sample_tuple = (data[sample_mask], sample_name)
            if self.rowid_variable not in sample_tuple[0].columns:
                pd.set_option('mode.chained_assignment', None)
                sample_tuple[0][self.rowid_variable] = sample_tuple[0].index
                pd.set_option('mode.chained_assignment', 'warn')

        sgit = ScoreGiniIVCalculator(use_weight=use_weight, lift_perc=lift_perc, masks=masks)

        if isinstance(target, list):
            sgit.s([sample_tuple]).t(target_tuple).w(self.weight).sc(scores).tm(self.time_variable)
        else:
            sgit.s([sample_tuple]).t([target_tuple]).w(self.weight).sc(scores).tm(self.time_variable)

        sgit.calculate()
        sgit.get_visualization(get_gini=get_gini, get_lift=get_lift, output_folder=output_folder,
                               filename_gini=filename_gini, filename_lift=filename_lift,
                               show_plot=show_plot)

    # def MarginalContribution(self, data_train, data_valid, sample, model, target,
    #                          predictors_to_add=[], output_folder=None, filename="marginal_cont.csv"):
    #     target_tuple = self._processTarget(target)
    #     sample_mask, sample_name = self._processSample(sample)
    #     sample_tuple = (data[sample_mask], sample_name)
    #     if self.rowid_variable not in sample_tuple[0].columns:
    #         pd.set_option('mode.chained_assignment', None)
    #         sample_tuple[0][self.rowid_variable] = sample_tuple[0].index
    #         pd.set_option('mode.chained_assignment', 'warn')
    #
    #     results = model.marginal_contribution(data_train, data_train[target_tuple[0][0]],
    #                                         data_valid, data_valid[target_tuple[0][0]],
    #                                         predictors_to_add=predictors_to_add)
    #     if output_folder:
    #         results.to_csv(path.join(output_folder, filename))

    def TransitionMatrix(self, data, score_old, score_new, sample, target, use_weight=False, quantiles_count=10,
                         observed=None, show_plot=True, draw_default_matrix=True, draw_transition_matrix=True,
                         output_folder=None, filename_default=None, filename_transition=None):
        """ Plots the transition matrix and default matrix for old and new score.
        Calls :py:meth:`~scoring.doctools.calculators.TransitionMatrixCalculator`

        Args:
            data (pandas DataFrame):
            score_old (str): Name of the old score attribute in the dataset.
            score_new (str): Name of the new score attribute in the dataset.
            sample (str, list): String or list of sample(s).
            target (str, list): String or list of targets. Do not set duplicities even in targets.
            use_weight (bool, optional): Default False. If True, weight defined in ProjectParameters() is used.
            observed (series, 1/0 mask for dataset): If given, overrides the the default base column for this calculator.
            quantiles_count (int): Number of desired quantiles for the matrix. Default is 10.
            show_plot (bool, optional): Default True. If True, plot(s) will be drawn in Jupyter Notebook.
            draw_default_matrix (bool): If True, performance (default) matrix is drawn.
            draw_transition_matrix (bool): If True, transition matrix is drawn.
            output_folder (str): if given, the plot is saved to the folder, if None, plot is not saved
            filename_default (str): if not given, default 'performance_matrix.png' is used
            filename_transition (str): if not given, default 'transition_matrix.png' is used
        """
        target_tuple = self._processTarget(target)
        sample_mask, sample_name = self._processSample(sample)
        sample_tuple = (data[sample_mask], sample_name)
        if self.rowid_variable not in sample_tuple[0].columns:
            pd.set_option('mode.chained_assignment', None)
            sample_tuple[0][self.rowid_variable] = sample_tuple[0].index
            pd.set_option('mode.chained_assignment', 'warn')
        scores = [score_old, score_new]
        tm = TransitionMatrixCalculator(quantiles_count=quantiles_count, observed=observed, use_weight=use_weight)
        tm.s([sample_tuple]).t([target_tuple]).sc(scores).w(self.weight)
        # results = tm.calculate().get_table()
        tm.calculate().get_visualization(draw_default_matrix=draw_default_matrix,
                                         draw_transition_matrix=draw_transition_matrix,
                                         show_plot=show_plot,
                                         output_folder=output_folder,
                                         filename_default=filename_default,
                                         filename_transition=filename_transition
                                         )

    def ScoreHistogram(self, data, sample, target, score, use_weight=None, observed=None, n_bins=25, use_logit=False,
                       min_score=None, max_score=None, output_folder=None, filename=None, show_plot=True):
        """Shows histogram of score variable. Calls :py:meth:`~scoring.doctools.calculators.ScorePlotDistCalculator`

        Args:
            data (pd.DataFrame): the dataset
            sample (list of str): list of or string of the sample name(s) defined in self.sample_dict.keys()
            target (str): string of column name of the target
            score (str): string of column name of score (must be exactly one)
            use_weight (bool, optional): True for using the self.weight, False for calculating without weights. Defaults to None.
            observed (str, optional): base column - the alternative 1/0 column name of contracts which should be counted in. If None,
                column of 1 is used. Defaults to None.
            n_bins (int, optional): number of bins the score should be binned to. Defaults to 25
            use_logit (bool, optional): whether score should be logistically transformed. Defaults to False.
            min_score (float, optional): minimal score value for binning. Defaults to None.
            max_score (float, optional): maximal score value for binning. Defaults to None.
            output_folder (str, optional): name of the output folder, if None, the plot is not saved to file. Defaults to None.
            filename (str, optional): name of the filename, if None (and existing output_folder), default filename is used. Defaults to None.
            show_plot (bool, optional): True for showing the plot in notebook. Defaults to True.
        """
        target_tuple = self._processTarget(target)
        sample_mask, sample_name = self._processSample(sample)
        sample_tuple = (data[sample_mask], sample_name)
        if self.rowid_variable not in sample_tuple[0].columns:
            pd.set_option('mode.chained_assignment', None)
            sample_tuple[0][self.rowid_variable] = sample_tuple[0].index
            pd.set_option('mode.chained_assignment', 'warn')

        spdc = ScorePlotDistCalculator(use_weight=use_weight, observed=observed, n_bins=n_bins, min_score=min_score,
                                       max_score=max_score, use_logit=use_logit)
        spdc.s([sample_tuple]).t([target_tuple]).sc(score).w(self.weight)
        return spdc.calculate().get_visualization(output_folder=output_folder, filename=filename, show_plot=show_plot)

    def CalibrationDistribution(self, data, sample, target, score, shift=0, scale=1, apply_logit=False,
                                empty_representations=[np.nan], use_weight=False, bins=30, vertical_lines=None,
                                output_folder=None, filename=None, show_plot=False):
        """Shows distribution chart of score variable. Compares empirical and theoretical probability of default in each
        bin of score variable. Calls :py:meth:`~scoring.doctools.calculators.CalibrationDistributionCalculator`

        Args:
            data (pd.DataFrame): the dataset
            sample (list of str): list of or string of the sample name(s) defined in self.sample_dict.keys()
            target (str): string of column name of the target
            score (str): string of column name of score (must be exactly one)
            shift (int, optional): Shift (intercept) to be added to the score before its calibration is evaluated. Defaults to 0.
            scale (int, optional): Scale (multiplier) to be applied to the score before its calibration is evaluated. Defaults to 1.
            apply_logit (bool, optional): Whether logistic transformation must be applied to the score (i.e. whether score is in expit form). Defaults to False.
            empty_representations (list, optional): List of values of the score that encode special (empty) observations. Defaults to [np.nan].
            use_weight (bool, optional): True for using the self.weight, False for calculating without weights. Defaults to False.
            bins (int, optional): Number of bins the score should be binned to. Defaults to 30.
            vertical_lines (list of float, optional): x coordinates of vertical lines to be added to the chart. Defaults to None.
            output_folder (str, optional): name of the output folder, if None, the plot is not saved to file. Defaults to None.
            filename (str, optional): name of the filename, if None (and existing output_folder), default filename is used. Defaults to None.
            show_plot (bool, optional): True for showing the plot in notebook. Defaults to False.
        """
        target_tuple = self._processTarget(target)
        sample_mask, sample_name = self._processSample(sample)
        sample_tuple = (data[sample_mask], sample_name)
        if self.rowid_variable not in sample_tuple[0].columns:
            pd.set_option('mode.chained_assignment', None)
            sample_tuple[0][self.rowid_variable] = sample_tuple[0].index
            pd.set_option('mode.chained_assignment', 'warn')

        cdc = CalibrationDistributionCalculator(self, shift=shift, scale=scale, apply_logit=apply_logit,
                                                empty_representations=empty_representations, use_weight=use_weight,
                                                bins=bins, vertical_lines=vertical_lines)
        cdc.s([sample_tuple]).t([target_tuple]).p([score]).w(self.weight)
        return cdc.calculate().get_visualization(output_folder=output_folder, filename=filename, show_plot=show_plot)

    def EmptyInTime(self, data, sample, predictors, empty_representations=[np.nan], use_weight=False,
                    output_folder=None, filename=None, show_plot=True):
        """Shows charts with share of empty/NaN values of the predictors per time unit.

        Args:
            data (pd.DataFrame): the dataset
            sample (list of str): list of or string of the sample name(s) defined in self.sample_dict.keys()
            predictors (list of str): list of predictors to analyze
            empty_representations (list, optional): List of values of the predictors that encode special (empty) observations. Defaults to [np.nan].
            use_weight (bool, optional): True for using the self.weight, False for calculating without weights. Defaults to False.
            output_folder (str, optional): name of the output folder, if None, the plot is not saved to file. Defaults to None.
            filename (str, optional): name of the filename, if None (and existing output_folder), default filename is used. Defaults to None.
            show_plot (bool, optional): True for showing the plot in notebook. Defaults to True.
        """
        sample_mask, sample_name = self._processSample(sample)
        sample_tuple = (data[sample_mask], sample_name)
        if self.rowid_variable not in sample_tuple[0].columns:
            pd.set_option('mode.chained_assignment', None)
            sample_tuple[0][self.rowid_variable] = sample_tuple[0].index
            pd.set_option('mode.chained_assignment', 'warn')

        eitc = EmptyInTimeCalculator(self, empty_representations=empty_representations, use_weight=use_weight)
        if (type(predictors) is str):
            predictors = [predictors]
            eitc.s([sample_tuple]).p(predictors).w(self.weight).tm(self.time_variable)
            eitc.calculate()
            eitc.get_visualization(output_folder=output_folder, show_plot=show_plot)
            result = eitc.get_table()
        else:
            for i, predictor in enumerate(predictors):
                eitc.s([sample_tuple]).p([predictor]).w(self.weight).tm(self.time_variable)
                eitc.calculate()
                eitc.get_visualization(output_folder=output_folder, show_plot=show_plot)
                batch_table_addition = eitc.get_table()[['empty_share']].copy()
                batch_table_addition['predictor'] = predictor
                if i == 0:
                    batch_table = batch_table_addition
                else:
                    batch_table = pd.concat([batch_table, batch_table_addition])
            result = pd.pivot_table(batch_table.reset_index(), index='predictor', columns=self.time_variable,
                                    values='empty_share')
        if output_folder:
            if filename is None:
                filename = 'EmptyInTime.csv'
            result.to_csv(output_folder + '/' + filename)
        return result

    def PartialDependencePlot(self, data, sample, target, predictor, use_weight=False,
                              output_folder=None, filename=None, show_plot=False, show_table=False):
        """PDP (Partial Dependency Plots) are showing overall trend of the model output (prediction) related to one particular predictor. We calculate PDP's for each predictor's values.

        First, we group the predictor values into several bins (corresponding to the splits inside the decision trees).
        Then for each observation (more precisely, for a reasonably sized random subsample) calculate the model output in hypothetical situation
        when the predictor would change its value to be in the particular bin and all the other varibles' values would remain the same.
        Average of these values over all observations for each particular bin is mean Partial Dependency value.
        When these values are plotted with the bins on x-axis, the PDP plot is formed.
        This plot shows how the mean of the prediction changes when the variable changes (and all other variables remain the same).

        We don't calculate just mean Partial Dependency but also its quantiles and median.

        Args:
            data (pd.DataFrame): the dataset
            sample (list of str): list of or string of the sample name(s) defined in self.sample_dict.keys()
            target (str): string of column name of the target
            predictor (str): string of column name of predictor
            use_weight (bool, optional): True for using the self.weight, False for calculating without weights. Defaults to False.
            output_folder (str, optional): name of the output folder, if None, the plot is not saved to file. Defaults to None.
            filename (str, optional): name of the filename, if None (and existing output_folder), default filename is used. Defaults to None.
            show_plot (bool, optional): Output the graphics? Defaults to False.
            show_table (bool, optional): Output the underlying table? Defaults to False.

        Returns:
            [type]: [description]
        """
        target_tuple = self._processTarget(target)
        sample_mask, sample_name = self._processSample(sample)
        sample_tuple = (data[sample_mask], sample_name)
        if self.rowid_variable not in sample_tuple[0].columns:
            pd.set_option('mode.chained_assignment', None)
            sample_tuple[0][self.rowid_variable] = sample_tuple[0].index
            pd.set_option('mode.chained_assignment', 'warn')

        pdpc = PartialDependencePlotCalculator(self, use_weight=use_weight)
        pdpc.s([sample_tuple]).p([predictor]).w(self.weight)
        pdpc.calculate().get_visualization(output_folder=output_folder, filename=filename, show_plot=show_plot)
        if show_table:
            display(pdpc.get_table())
        return pdpc.get_table()

    def IceplotRu(self, data, sample, target, predictor, use_weight=False, sample_size=1000, frac_to_plot=0.1,
                  output_folder=None, filename=None):
        """Individual Conditional Expectations.

        For each observation, we show how the prediction would change if one particular variable was chagning its values (and all the other variables remained the same)
        and we draw these lines all into one chart (in our case there are lines for 250 randomly chosen observations).
        
        There is also mean PDP showed by a thick line in the chart.

        Args:
            data (pd.DataFrame): the dataset
            sample (list of str): list of or string of the sample name(s) defined in self.sample_dict.keys()
            target (str): string of column name of the target
            predictor (str): string of column name of predictor
            use_weight (bool, optional): True for using the self.weight, False for calculating without weights. Defaults to False.
            sample_size (int, optional): Number of obserations to be used to calculate PDP. Defaults to 1000.
            frac_to_plot (float, optional): Fraction of sample size to be plotted to ICE plot. Defaults to 0.1.
            output_folder (str, optional): name of the output folder, if None, the plot is not saved to file. Defaults to None.
            filename (str, optional): name of the filename, if None (and existing output_folder), default filename is used. Defaults to None.
            
        """
        if use_weight:
            warnings.warn("use_weight=True, but weights are not implemented in this calculator")
        target_tuple = self._processTarget(target)
        sample_mask, sample_name = self._processSample(sample)
        sample_tuple = (data[sample_mask], sample_name)
        if self.rowid_variable not in sample_tuple[0].columns:
            pd.set_option('mode.chained_assignment', None)
            sample_tuple[0][self.rowid_variable] = sample_tuple[0].index
            pd.set_option('mode.chained_assignment', 'warn')

        iprc = IceplotRuCalculator(self, sample_size=sample_size, frac_to_plot=frac_to_plot)
        iprc.s([sample_tuple]).p([predictor])
        return iprc.calculate().get_visualization(output_folder=output_folder, filename=filename)

    def ExpectedApprovalRate(self, data, sample, score, query_subset, reference_ar=[0.50], use_weight=False,
                             def_by_score_ascending=False, output_folder=None, filename=None):
        """For each population of interest we calculate theoretical approval rate. The estimation goes as follows:

        We define a reference approval rate for the whole population of incoming customers
        We calculate a cutoff value which corresponds to this targetted approval rate
        We set the same cutoff for the subpopulation
        We evaluate what would the approval rate on just this subpopulation be when the cutoff is applied.
        If the subpopulation approval rate is different to the reference approval rate, it ususally means that the subpopulation is shifted, i.e. the estimated probability of default of such customers is different from probability of default of a typical customer.

        Args:
            data (pd.DataFrame): the dataset
            sample (list of str): list of or string of the sample name(s) defined in self.sample_dict.keys()
            score (str): string of column name of score (must be exactly one)
            query_subset (list of str): list of pandas queries defining subsets of interests from data
            reference_ar (list of float, optional): List of reference approval rates. Defaults to [0.50].
            use_weight (bool, optional): True for using the self.weight, False for calculating without weights. Defaults to False.
            def_by_score_ascending (bool, optional): Whether score is increasing with probabiity of default or decreasing. Defaults to False.
            output_folder (str, optional): name of the output folder, if None, the plot is not saved to file. Defaults to None.
            filename (str, optional): name of the filename, if None (and existing output_folder), default filename is used. Defaults to None.
        """
        sample_mask, sample_name = self._processSample(sample)
        sample_tuple = (data[sample_mask], sample_name)
        if self.rowid_variable not in sample_tuple[0].columns:
            pd.set_option('mode.chained_assignment', None)
            sample_tuple[0][self.rowid_variable] = sample_tuple[0].index
            pd.set_option('mode.chained_assignment', 'warn')

        earc = ExpectedApprovalRateCalculator(query_subset=query_subset,
                                              reference_ar=reference_ar,
                                              use_weight=use_weight,
                                              def_by_score_ascending=def_by_score_ascending)
        earc.sc([score]).s([sample_tuple]).w(self.weight)
        earc.calculate()
        result = earc.get_table()
        if output_folder:
            if filename is None:
                filename = 'approval_rates.csv'
            result.to_csv(output_folder + '/' + filename)
        return result


class Calculator(object, metaclass=abc.ABCMeta):
    """Base abstract class for usage in Project Parameters.

        Defines a structure for all Calculators and enforces
        methods that need to be implemented:

        calculate(self):
            performs all needed calculations    

        get_table(self):
            exports calculated data
            in tabular form

        get_visualization(self):
            exports calculated data
            in visual form

        get_description(self):
            exports description of 
            calculator in string form


    """
    samples = []
    targets = []
    predictors = []
    scores = []
    predictors_to_add = []

    description = None
    rowid_variable = None
    time_variable = None
    weight = None

    projectParameters = None

    def __init__(self, projectParameters=ProjectParameters()):
        self.projectParameters = projectParameters
        self.rowid_variable = projectParameters.rowid_variable
        self.time_variable = projectParameters.time_variable
        self.targets = projectParameters.targets
        self.masks = projectParameters.masks
        # self.scores = (
        #     projectParameters.scores
        # )
        #  # [(projectParameters.sample_dict[sample_name], sample_name) for sample_name in projectParameters.sample_ordering]

    def s(self, samples):
        """sets array of sample s"""
        ## add input checking

        self.samples = samples
        return self

    def t(self, targets):
        """sets array of targets"""
        self.targets = targets
        return self

    def p(self, predictors):
        """sets array of predictors"""
        self.predictors = predictors
        return self

    def sc(self, scores):
        """sets array of scores"""
        self.scores = scores
        return self

    def sg(self, segments):
        """sets array of segmentations"""
        self.segments = segments
        return self

    def padd(self, predictors_to_add):
        """sets array of predictors to add for marginal contribution"""
        self.predictors_to_add = predictors_to_add
        return self

    def tm(self, time_variable):
        """overrides the project-lvl time variable"""
        self.time_variable = time_variable
        return self

    def w(self, weight_col):
        """set weight"""
        self.weight = weight_col
        return self

    def g(self, grouping):
        """set grouping"""
        self.grouping = grouping
        return self

    def rwid(self, rowid_variable):
        """set rowID"""
        self.rowid_variable = rowid_variable
        return self

    def get_hash(self):
        """returns hash of the description"""
        import hashlib

        hash_object = hashlib.md5(self.get_description())
        return hash_object.hexdigest()

    def replace_legend(self, woe, predictor, grouping=None):
        """
        Tries to replace a woe value with interval or list of values if grouping was provided.
        If no grouping was provided returns original value.

        Args:
            woe (float): woe value to be replaced
            predictor ():
            grouping (, optional):
        """
        grouping_dictionary = grouping.export_dictionary()
        woe = np.float32(woe)

        if grouping:
            new_value = grouping_dictionary[predictor][round(woe, 5)]

            # if value is list of strings lets join them
            if type(new_value) is list:
                group_name = [str(i) for i in new_value]
                # split into chunks of five
                group_name = [",".join(group_name[i:i + 5]) for i in range(0, len(group_name), 5)]
                # each chuck get a new line
                group_name = "\n".join([str(i) for i in group_name])

                return group_name
            else:
                return new_value
        else:
            return woe

    # these need to be implemented by further developers
    @abc.abstractmethod
    def calculate(self):
        return

    @abc.abstractmethod
    def get_table(self):
        return

    @abc.abstractmethod
    def get_visualization(self):
        return

    @abc.abstractmethod
    def get_description(self):
        return


# expects one sample
# accepts multiple targets


class SampleDescriptionCalculator(Calculator):
    def calculate(self):
        pivots = []

        # declaration of all used external variables
        # ------------------------------------------
        df = self.samples[0][0]
        targets = self.targets
        rowid_col = self.rowid_variable
        time_variable = self.time_variable
        # ---------------------------------

        a = pd.pivot_table(df, index=time_variable, values=rowid_col, aggfunc=[len], dropna=False, fill_value=0)
        a.columns = pd.Index(["Observations"])
        pivots.append(a)

        for target in targets:
            p = pd.pivot_table(
                df[df[target[1]] == 1],
                index=time_variable,
                values=target[0],
                columns=target[1],
                aggfunc=[len, np.sum, np.mean],
                dropna=False,
                fill_value=0,
            )
            p.columns = pd.Index([target[0] + " measurable", target[0] + " bads", target[0] + " default rate"])
            pivots.append(p)

        r = pd.concat(pivots, axis=1, sort=False)
        r2 = pd.DataFrame(r.apply(np.sum)).T

        for target in targets:
            r2[target[0] + " default rate"] = r2[target[0] + " bads"] / r2[target[0] + " measurable"]

        r2 = r2.set_index(pd.Index(["All"]))
        self.table = pd.concat([r, r2], sort=False)
        self.table.columns = pd.MultiIndex.from_product([["Sample description"], self.table.columns])

        return self

    def get_table(self):
        return self.table

    def get_visualization(self):
        return self

    def get_description(self):
        return "Description of sample " + self.samples[0][1]


# Matus adjusted sample description by segments
class SampleDescriptionSegmentsCalculator(Calculator):
    def calculate(self):
        pivots = []

        # declaration of all used external variables
        # ------------------------------------------
        df = self.samples[0][0]
        targets = self.targets
        predictors = self.predictors
        rowid_col = self.rowid_variable
        segment = self.segments[0]
        # ---------------------------------

        a = pd.pivot_table(df, index=segment, values=rowid_col, aggfunc=[len], dropna=False, fill_value=0)
        a["rel"] = a.len / a.len.sum()
        a.columns = pd.Index(["Observations", "Share"])
        pivots.append(a)

        for target in targets:
            p = pd.pivot_table(
                df[df[target[1]] == 1],
                index=segment,
                values=target[0],
                columns=target[1],
                aggfunc=[len, np.sum, np.mean],
                dropna=False,
                fill_value=0,
            )
            p.columns = pd.Index([target[0] + " measurable", target[0] + " bads", target[0] + " default rate"])
            pivots.append(p)
            for predictor in predictors:
                p = pd.pivot_table(
                    df[df[target[1]] == 1],
                    index=segment,
                    values=predictor,
                    columns=target[1],
                    aggfunc=[np.mean],
                    dropna=False,
                    fill_value=0,
                )
                p.columns = pd.Index(["AVG(" + predictor + ") on " + target[0] + " meas."])
                pivots.append(p)

        r = pd.concat(pivots, axis=1, sort=False)
        r2 = pd.DataFrame(r.apply(np.sum)).T

        for target in targets:
            r2[target[0] + " default rate"] = r2[target[0] + " bads"] / r2[target[0] + " measurable"]
            for predictor in predictors:
                r2["AVG(" + predictor + ") on " + target[0] + " meas."] = (
                        r["AVG(" + predictor + ") on " + target[0] + " meas."].multiply(
                            r[target[0] + " measurable"]).sum()
                        / r2[target[0] + " measurable"]
                )

        r2 = r2.set_index(pd.Index(["All"]))
        self.table = pd.concat([r, r2], sort=False)
        self.table.columns = pd.MultiIndex.from_product([["Sample description"], self.table.columns])

        return self

    def get_table(self):
        return self.table

    def get_visualization(self, ax=None):
        return self

    def get_description(self):
        return "Description of sample " + self.samples[0][1] + " by segments of " + self.segments[0]


# import Calculator

# expect only one sample
# expects only one target variable
# expects only one predictor


class PredictorRiskInTimeCalculator(Calculator):
    def calculate(self):
        # declaration of all used external variables
        # -----------------------------------------
        df = self.samples[0][0]
        target = self.targets[0]
        predictor = self.predictors[0]
        rowid_col = self.rowid_variable
        time_variable = self.time_variable
        # ---------------------------------
        self.table = pd.pivot_table(
            df[df[target[1]] == 1],
            index=time_variable,
            values=target[0],
            columns=predictor,
            aggfunc=[len, np.sum, np.mean],
            margins=True,
        )

        pred_cat = list(set(self.table.columns.values))
        pred_cat = self.table.columns.levels[1].values
        stat = ["Meas. obs.", "Bads", "Bad rate"]
        self.table.columns = pd.MultiIndex.from_product([stat, pred_cat])
        return self

    def calculate_weighted(self):
        # declaration of all used external variables
        # -----------------------------------------
        df = self.samples[0][0].copy()
        target = self.targets[0]
        predictor = self.predictors[0]
        rowid_col = self.rowid_variable
        time_variable = self.time_variable
        weight_col = self.weight

        target_weighted = f"{target}_{weight_col}"

        df[target_weighted] = df[target[0]] * df[weight_col]
        # ---------------------------------
        self.table = pd.pivot_table(
            df[df[target[1]] == 1],
            index=time_variable,
            values=[target_weighted, weight_col],
            columns=predictor,
            aggfunc=np.sum,
            margins=True,
        )

        self.table = self.table[[weight_col, target_weighted]]

        table_weighted_mean = self.table[target_weighted] / self.table[weight_col]
        table_weighted_mean.columns = pd.MultiIndex.from_product([["w_mean"], table_weighted_mean.columns.values])
        self.table = pd.concat([self.table, table_weighted_mean], axis=1)
        pred_cat = list(set(self.table.columns.values))
        pred_cat = self.table.columns.levels[1].values
        stat = ["Meas. obs.", "Bads", "Bad rate"]
        self.table.columns = pd.MultiIndex.from_product([stat, pred_cat])
        return self

    def get_table(self):
        return self.table

    def get_visualization(self):
        import matplotlib.pyplot as plt

        br = self.table["Bad rate"]
        yaxislim = math.ceil(1.2 * br.max().max() * 40) / 40
        del br["All"]
        br.drop("All")
        ax = br.plot(ylim=(0, yaxislim), title="Risk of " + self.predictors[0] + " on " + self.targets[0][0])
        ax.set_xticks(list(np.arange(br.shape[0])))
        ax.set_xticklabels(br.index)
        plt.show()

    def get_description(self):
        return (
                "Risk of predictor "
                + self.predictors[0]
                + " on target "
                + self.targets[0][0]
                + " on sample "
                + self.samples[0][1]
        )


# import Calculator

# expect only one sample
# expects only one predictor


class PredictorShareInTimeCalculator(Calculator):
    def calculate(self):
        # declaration of all used external variables
        # -----------------------------------------
        df = self.samples[0][0]
        predictor = self.predictors[0]
        rowid_col = self.rowid_variable
        time_variable = self.time_variable
        # ---------------------------------

        cnt = pd.pivot_table(df, index=time_variable, columns=predictor, values=rowid_col, aggfunc=len, margins=True)
        share = cnt.apply(lambda x: x / x["All"], axis=1)

        self.table = pd.concat([cnt, share], axis=1, sort=False)

        pred_cat = list(share.columns.values)
        stat = ["All obs.", "Share"]
        self.table.columns = pd.MultiIndex.from_product([stat, pred_cat])
        return self

    def calculate_weighted(self):
        # declaration of all used external variables
        # -----------------------------------------
        df = self.samples[0][0]
        predictor = self.predictors[0]
        rowid_col = self.rowid_variable
        time_variable = self.time_variable
        weight_col = self.weight
        # ---------------------------------

        cnt = pd.pivot_table(df, index=time_variable, columns=predictor, values=weight_col, aggfunc=sum, margins=True)
        share = cnt.apply(lambda x: x / x["All"], axis=1)

        self.table = pd.concat([cnt, share], axis=1, sort=False)

        pred_cat = list(share.columns.values)
        stat = ["All obs.", "Share"]
        self.table.columns = pd.MultiIndex.from_product([stat, pred_cat])
        return self

    def get_table(self):
        return self.table

    def get_visualization(self):
        import matplotlib.pyplot as plt

        piv = self.table["Share"]
        del piv["All"]
        piv.drop("All")
        ax = piv.plot(ylim=(0, 1), title="Share of " + self.predictors[0])
        ax.set_xticks(list(np.arange(piv.shape[0])))
        ax.set_xticklabels(piv.index)

        plt.show()
        return

    def get_description(self):
        return "Share of predictor " + self.predictors[0] + " on sample " + self.samples[0][1]


# import Calculator

# expect only one sample
# expects only one target variable
# expects one or more predictors


class PredictorGiniInTimeCalculator(Calculator):
    def calculate(self):

        # declaration of all used external variables
        # -----------------------------------------
        df = self.samples[0][0]
        targets = self.targets
        predictors = self.predictors
        rowid_col = self.rowid_variable
        time_variable = self.time_variable

        # ---------------------------------
        def gini(x):
            from sklearn import metrics
            import pandas as pd
            import numpy as np

            def unzip_scoretarget_tuple(x):
                prep = list(zip(*x.values))
                target_ = prep[0]
                prediction_ = prep[1]
                return [target_, prediction_]

            target_, prediction_ = unzip_scoretarget_tuple(x)
            if np.min(target_) < np.max(target_):  # we need 2 classes to calculate gini
                return 100 * (metrics.roc_auc_score(target_, prediction_) * 2 - 1)
            else:
                return np.nan  # gini not defined

        df = df.copy()

        # if the predictor is categorical one, encode it with badrate first
        # for predictor in predictors:

        # prepare combinations of predictor and target

        arr_t = []
        for target in targets:
            arr_p = []
            for predictor in predictors:
                if df[predictor].dtype == "O":
                    df_piv = pd.pivot_table(
                        df, index=predictor, columns=target[0], values=rowid_col, aggfunc=len, fill_value=0
                    )
                    df_piv["br"] = df_piv.apply(lambda x: 1.00 * x[1] / (x[0] + x[1]), axis=1)
                    df_piv = pd.DataFrame(df_piv.reset_index().values, columns=[predictor, "goods", "bads", "br"])
                    df = (
                        df[[rowid_col, time_variable, target[0], target[1]] + predictors]
                            .copy()
                            .reset_index()
                            .merge(df_piv, on=predictor, how="left")
                            .set_index("index")
                    )
                    df[predictor] = df["br"]

                # ri = RussianImputer().fit(df[df[target[1]] == 1][predictor], df[df[target[1]] == 1][target[0]])
                # df[predictor] = ri.transform(df[df[target[1]] == 1][predictor])

                target_predictor_col = target[0] + predictor + "_tuple"
                df[target_predictor_col] = df[[target[0], predictor]].apply(
                    lambda x: (x[target[0]], x[predictor]), axis=1
                )
                # display(df[(df[target[1]] == 1) & (~df[predictor].isnull())])
                gini_results = pd.pivot_table(
                    df[(df[target[1]] == 1) & ~df[predictor].isnull()], index=time_variable,
                    values=target_predictor_col, aggfunc=gini, margins=True
                )

                overall_gini_sign = gini_results.loc["All"][0]
                gini_results = gini_results.apply(lambda x: x * 1 if overall_gini_sign >= 0 else -1 * x)
                arr_p.append(gini_results)

            t = pd.concat(arr_p, axis=1, sort=False)
            t.columns = pd.MultiIndex.from_product([["Gini " + target[0]], predictors])
            arr_t.append(t)

            for predictor in predictors:
                del df[target[0] + predictor + "_tuple"]

        # return final results
        self.table = pd.concat(arr_t, axis=1, sort=False)

        return self

    def calculate_weighted(self):

        # declaration of all used external variables
        # -----------------------------------------
        df = self.samples[0][0]
        targets = self.targets
        predictors = self.predictors
        rowid_col = self.rowid_variable
        time_variable = self.time_variable
        weight_col = self.weight

        # ---------------------------------

        def gini(x):
            from sklearn import metrics
            import pandas as pd
            import numpy as np

            def unzip_scoretargetweight_tuple(x):
                target, prediction, weight = list(zip(*x.values))
                return target, prediction, weight

            target_, prediction_, weight_ = unzip_scoretargetweight_tuple(x)
            if np.min(target_) < np.max(target_):  # we need 2 classes to calculate gini
                return 100 * (metrics.roc_auc_score(target_, prediction_, sample_weight=weight_) * 2 - 1)
            else:
                return np.nan  # gini not defined

        df = df.copy()

        # if the predictor is categorical one, encode it with badrate first
        # for predictor in predictors:

        # prepare combinations of predictor and target

        arr_t = []
        for target in targets:
            arr_p = []
            for predictor in predictors:

                if df[predictor].dtype == "O":
                    df_piv = pd.pivot_table(
                        df, index=predictor, columns=target[0], values=rowid_col, aggfunc=len, fill_value=0
                    )
                    df_piv["br"] = df_piv.apply(lambda x: 1.00 * x[1] / (x[0] + x[1]), axis=1)
                    df_piv = pd.DataFrame(df_piv.reset_index().values, columns=[predictor, "goods", "bads", "br"])
                    df = (
                        df[[rowid_col, time_variable, target[0], target[1]] + predictors]
                            .copy()
                            .reset_index()
                            .merge(df_piv, on=predictor, how="left")
                            .set_index("index")
                    )
                    df[predictor] = df["br"]

                # ri = RussianImputer().fit(df[df[target[1]] == 1][predictor], df[df[target[1]] == 1][target[0]])
                # df[predictor] = ri.transform(df[df[target[1]] == 1][predictor])

                target_predictor_col = target[0] + predictor + weight_col + "_tuple"
                df[target_predictor_col] = df[[target[0], predictor, weight_col]].apply(
                    lambda x: (x[target[0]], x[predictor], x[weight_col]), axis=1
                )
                # display(df[(df[target[1]] == 1) & (~df[predictor].isnull())])
                gini_results = pd.pivot_table(
                    df[(df[target[1]] == 1) & ~df[predictor].isnull()], index=time_variable,
                    values=target_predictor_col, aggfunc=gini, margins=True
                )

                overall_gini_sign = gini_results.loc["All"][0]
                gini_results = gini_results.apply(lambda x: x * 1 if overall_gini_sign >= 0 else -1 * x)
                arr_p.append(gini_results)

            t = pd.concat(arr_p, axis=1, sort=False)
            t.columns = pd.MultiIndex.from_product([["Gini " + target[0]], predictors])
            arr_t.append(t)

            for predictor in predictors:
                del df[target[0] + predictor + weight_col + "_tuple"]

        # return final reults
        self.table = pd.concat(arr_t, axis=1, sort=False)

        return self

    def get_table(self):
        return self.table

    def get_visualization(self):
        import matplotlib.pyplot as plt

        gini_results = self.table
        # print(gini_results)
        gini_results = gini_results.drop("All")
        ax = gini_results.plot(title="Gini of " + self.predictors[0] + " on " + self.targets[0][0], ylim=(0, 100))
        ax.set_xticks(list(np.arange(gini_results.shape[0])))
        ax.set_xticklabels(gini_results.index)
        plt.show()
        return self

    def get_description(self):
        if len(self.predictors) == 1 and len(self.targets) == 1:
            return (
                    "Gini of predictor "
                    + self.predictors[0]
                    + " on target "
                    + self.targets[0][0]
                    + " on sample "
                    + self.samples[0][1]
            )
        elif len(self.predictors) > 1 and len(self.targets) > 1:
            return "Gini of multiple predictors " + " on multiple targets " + "on sample " + self.samples[0][1]
        elif len(self.predictors) > 1:
            return "Gini of multiple predictors on target " + self.targets[0][0] + " on sample " + self.samples[0][1]
        elif len(self.targets) > 1:
            return (
                    "Gini of predictor " + self.predictors[0] + " on multiple targets " + " on sample " +
                    self.samples[0][1]
            )


# Matus adjusted - adding Gini by segments calculator
class PredictorGiniBySegmentsCalculator(Calculator):
    def calculate(self):

        # declaration of all used external variables
        # -----------------------------------------
        df = self.samples[0][0]
        targets = self.targets
        predictors = self.predictors
        rowid_col = self.rowid_variable
        segment = self.segments[0]

        # ---------------------------------

        def gini(x):
            from sklearn import metrics
            import pandas as pd
            import numpy as np

            def unzip_scoretarget_tuple(x):
                prep = list(zip(*x.values))
                target_ = prep[0]
                prediction_ = prep[1]
                return [target_, prediction_]

            target_, prediction_ = unzip_scoretarget_tuple(x)
            if np.min(target_) < np.max(target_):  # we need 2 classes to calculate gini
                return 100 * (metrics.roc_auc_score(target_, prediction_) * 2 - 1)
            else:
                return np.nan  # gini not defined

        df = df.copy()

        # if the predictor is categorical one, encode it with badrate first
        # for predictor in predictors:

        # prepare combinations of predictor and target

        arr_t = []
        for target in targets:
            arr_p = []
            for predictor in predictors:

                if df[predictor].dtype == "O":
                    df_piv = pd.pivot_table(
                        df, index=predictor, columns=target[0], values=rowid_col, aggfunc=len, fill_value=0
                    )
                    df_piv["br"] = df_piv.apply(lambda x: 1.00 * x[1] / (x[0] + x[1]), axis=1)
                    df_piv = pd.DataFrame(df_piv.reset_index().values, columns=[predictor, "goods", "bads", "br"])
                    df = (
                        df[[rowid_col, segment, target[0], target[1]] + predictors]
                            .copy()
                            .reset_index()
                            .merge(df_piv, on=predictor, how="left")
                            .set_index("index")
                    )
                    df[predictor] = df["br"]

                # ri = RussianImputer().fit(df[df[target[1]] == 1][predictor], df[df[target[1]] == 1][target[0]])
                # df[predictor] = ri.transform(df[df[target[1]] == 1][predictor])

                target_predictor_col = target[0] + predictor + "_tuple"
                df[target_predictor_col] = df[[target[0], predictor]].apply(
                    lambda x: (x[target[0]], x[predictor]), axis=1
                )
                gini_results = pd.pivot_table(
                    df[df[target[1]] == 1], index=segment, values=target_predictor_col, aggfunc=gini, margins=True
                )

                overall_gini_sign = gini_results.loc["All"][0]
                gini_results = gini_results.apply(lambda x: x * 1 if overall_gini_sign >= 0 else -1 * x)
                arr_p.append(gini_results)

            t = pd.concat(arr_p, axis=1, sort=False)
            t.columns = pd.MultiIndex.from_product([["Gini " + target[0]], predictors])
            arr_t.append(t)

            for predictor in predictors:
                del df[target[0] + predictor + "_tuple"]

        # return final results
        self.table = pd.concat(arr_t, axis=1, sort=False)

        return self

    def get_table(self):
        return self.table

    def get_visualization(self):
        import matplotlib.pyplot as plt

        gini_results = self.table
        # print(gini_results)
        gini_results = gini_results.drop("All")
        ax = gini_results.plot(
            title="Gini of "
                  + self.predictors[0]
                  + " on "
                  + self.targets[0][0]
                  + "\n by segments of "
                  + self.segments[0],
            ylim=(0, 100),
        )
        ax.set_xticks(list(np.arange(gini_results.shape[0])))
        ax.set_xticklabels(gini_results.index)
        plt.show()
        return self

    def get_description(self):
        if len(self.predictors) == 1 and len(self.targets) == 1:
            return (
                    "Gini of predictor "
                    + self.predictors[0]
                    + " on target "
                    + self.targets[0][0]
                    + " on sample "
                    + self.samples[0][1]
                    + " by segments of "
                    + self.segments[0]
            )
        elif len(self.predictors) > 1 and len(self.targets) > 1:
            return (
                    "Gini of multiple predictors "
                    + " on multiple targets "
                    + "on sample "
                    + self.samples[0][1]
                    + " by segments of "
                    + self.segments[0]
            )
        elif len(self.predictors) > 1:
            return (
                    "Gini of multiple predictors on target "
                    + self.targets[0][0]
                    + " on sample "
                    + self.samples[0][1]
                    + " by segments of "
                    + self.segments[0]
            )
        elif len(self.targets) > 1:
            return (
                    "Gini of predictor "
                    + self.predictors[0]
                    + " on multiple targets "
                    + " on sample "
                    + self.samples[0][1]
                    + " by segments of "
                    + self.segments[0]
            )


# import Calculator

# expect only one sample
# expects only one target variable
# expects one or more predictors (only continuous)


class PredictorLiftInTimeCalculator(Calculator):
    def liftN(self, x, q):
        def unzip_scoretarget_tuple(x):
            prep = list(zip(*x.values))
            target_ = prep[0]
            prediction_ = prep[1]
            return [target_, prediction_]

        target_, prediction_ = unzip_scoretarget_tuple(x)
        df = pd.DataFrame([pd.Series(target_, name="target"), pd.Series(prediction_, name="prediction")]).T
        # print(df)
        quant_ = df["prediction"].quantile(q)
        pop_dr = df["target"].mean()
        q_dr = df["target"][df["prediction"] >= quant_].mean()
        if pop_dr == 0:
            return np.nan
        else:
            lift = 1.0 * q_dr / pop_dr
        return lift

    def lift05(self, x):
        return self.liftN(x, 0.95)

    def lift10(self, x):
        return self.liftN(x, 0.9)

    def lift20(self, x):
        return self.liftN(x, 0.8)

    def lift30(self, x):
        return self.liftN(x, 0.7)

    def lift40(self, x):
        return self.liftN(x, 0.6)

    def lift50(self, x):
        return self.liftN(x, 0.5)

    def lift60(self, x):
        return self.liftN(x, 0.4)

    def lift70(self, x):
        return self.liftN(x, 0.3)

    def lift80(self, x):
        return self.liftN(x, 0.2)

    def lift90(self, x):
        return self.liftN(x, 0.1)

    def get_all_lifts(self):
        return [
            self.lift05,
            self.lift10,
            self.lift20,
            self.lift30,
            self.lift40,
            self.lift50,
            self.lift60,
            self.lift70,
            self.lift80,
            self.lift90,
        ]

    def calculate(self):

        # declaration of all used external variables
        # -----------------------------------------
        df = self.samples[0][0]
        target = self.targets[0]
        predictors = self.predictors
        rowid_col = self.rowid_variable
        time_variable = self.time_variable
        # ---------------------------------

        # if the predictor is categorical one, encode it with badrate first
        # for predictor in predictors:
        #    if df[predictor].dtype == 'O':
        #        df_piv = pd.pivot_table(df, index=predictor, columns=target[0], values=rowid_col, aggfunc=len, fill_value=0)
        #        df_piv['br'] = df_piv.apply(lambda x: 1.00 * x[1] / ( x[0] + x[1] ), axis=1)
        #        df_piv = pd.DataFrame(df_piv.reset_index().values, columns=[predictor,'goods','bads','br'])
        #        df = df[[rowid_col, time_variable, target[0], target[1]] + predictors].copy().reset_index().merge(df_piv, on=predictor, how='left').set_index('index')
        #        df[predictor] = df['br']

        # prepare combinations of predictor and target

        pd.options.mode.chained_assignment = None  # default='warn'

        arr = []
        for predictor in predictors:
            target_predictor_col = target[0] + predictor + "_tuple"
            df[target_predictor_col] = df[[target[0], predictor]].apply(lambda x: (x[target[0]], x[predictor]), axis=1)
            lift_results = pd.pivot_table(
                df[df[target[1]] == 1],
                index=time_variable,
                values=target_predictor_col,
                aggfunc=self.lift10,
                margins=True,
            )
            lift_results.columns = ["Lift"]
            lift_results.name = predictor
            arr.append(lift_results)

        # cleanup
        for predictor in predictors:
            del df[target[0] + predictor + "_tuple"]

        # return final reults
        self.table = pd.concat(arr, axis=1, sort=False)
        self.table = self.table
        self.table.columns = pd.MultiIndex.from_product([["Lift10"], self.predictors])

        return self

    def get_table(self):
        return self.table

    def get_visualization(self):
        import matplotlib.pyplot as plt

        gini_results = self.table
        # print(gini_results)
        gini_results = gini_results.drop("All")
        ax = gini_results.plot(title="Lift10 of " + self.predictors[0] + " on " + self.targets[0][0], ylim=(0, 7))
        ax.set_xticks(list(np.arange(gini_results.shape[0])))
        ax.set_xticklabels(gini_results.index)

        plt.show()
        return self

    def get_description(self):
        if len(self.predictors) == 1:
            return (
                    "Lift10 of predictor "
                    + self.predictors[0]
                    + " on target "
                    + self.targets[0][0]
                    + " on sample "
                    + self.samples[0][1]
            )
        else:
            return (
                    "Lift10 of multiple predictors "
                    + " on target "
                    + self.targets[0][0]
                    + " on sample "
                    + self.samples[0][1]
            )


# Matus adjusted - adding Gini by segments calculator
class PredictorLiftBySegmentsCalculator(Calculator):
    def liftN(self, x, q):
        def unzip_scoretarget_tuple(x):
            prep = list(zip(*x.values))
            target_ = prep[0]
            prediction_ = prep[1]
            return [target_, prediction_]

        target_, prediction_ = unzip_scoretarget_tuple(x)
        df = pd.DataFrame([pd.Series(target_, name="target"), pd.Series(prediction_, name="prediction")]).T
        # print(df)
        quant_ = df["prediction"].quantile(q)
        pop_dr = df["target"].mean()
        q_dr = df["target"][df["prediction"] > quant_].mean()
        if pop_dr == 0:
            return np.nan
        else:
            lift = 1.0 * q_dr / pop_dr
        return lift

    def lift05(self, x):
        return self.liftN(x, 0.95)

    def lift10(self, x):
        return self.liftN(x, 0.9)

    def lift20(self, x):
        return self.liftN(x, 0.8)

    def lift30(self, x):
        return self.liftN(x, 0.7)

    def lift40(self, x):
        return self.liftN(x, 0.6)

    def lift50(self, x):
        return self.liftN(x, 0.5)

    def lift60(self, x):
        return self.liftN(x, 0.4)

    def lift70(self, x):
        return self.liftN(x, 0.3)

    def lift80(self, x):
        return self.liftN(x, 0.2)

    def lift90(self, x):
        return self.liftN(x, 0.1)

    def get_all_lifts(self):
        return [
            self.lift05,
            self.lift10,
            self.lift20,
            self.lift30,
            self.lift40,
            self.lift50,
            self.lift60,
            self.lift70,
            self.lift80,
            self.lift90,
        ]

    def calculate(self):

        # declaration of all used external variables
        # -----------------------------------------
        df = self.samples[0][0]
        target = self.targets[0]
        predictors = self.predictors
        rowid_col = self.rowid_variable
        segment = self.segments[0]
        # ---------------------------------

        # if the predictor is categorical one, encode it with badrate first
        # for predictor in predictors:
        #    if df[predictor].dtype == 'O':
        #        df_piv = pd.pivot_table(df, index=predictor, columns=target[0], values=rowid_col, aggfunc=len, fill_value=0)
        #        df_piv['br'] = df_piv.apply(lambda x: 1.00 * x[1] / ( x[0] + x[1] ), axis=1)
        #        df_piv = pd.DataFrame(df_piv.reset_index().values, columns=[predictor,'goods','bads','br'])
        #        df = df[[rowid_col, time_variable, target[0], target[1]] + predictors].copy().reset_index().merge(df_piv, on=predictor, how='left').set_index('index')
        #        df[predictor] = df['br']

        # prepare combinations of predictor and target
        arr = []
        for predictor in predictors:
            target_predictor_col = target[0] + predictor + "_tuple"
            df[target_predictor_col] = df[[target[0], predictor]].apply(lambda x: (x[target[0]], x[predictor]), axis=1)
            lift_results = pd.pivot_table(
                df[df[target[1]] == 1], index=segment, values=target_predictor_col, aggfunc=self.lift10, margins=True
            )
            lift_results.columns = ["Lift"]
            lift_results.name = predictor
            arr.append(lift_results)

        # cleanup
        for predictor in predictors:
            del df[target[0] + predictor + "_tuple"]

        # return final reults
        self.table = pd.concat(arr, axis=1, sort=False)
        self.table = self.table
        self.table.columns = pd.MultiIndex.from_product([["Lift10"], self.predictors])

        return self

    def get_table(self):
        return self.table

    def get_visualization(self):
        import matplotlib.pyplot as plt

        gini_results = self.table
        # print(gini_results)
        gini_results = gini_results.drop("All")
        ax = gini_results.plot(
            title="Lift10 of "
                  + self.predictors[0]
                  + " on "
                  + self.targets[0][0]
                  + "\n by segments of "
                  + self.segments[0],
            ylim=(0, 7),
        )
        ax.set_xticks(list(np.arange(gini_results.shape[0])))
        ax.set_xticklabels(gini_results.index)

        plt.show()
        return self

    def get_description(self):
        if len(self.predictors) == 1:
            return (
                    "Lift10 of predictor "
                    + self.predictors[0]
                    + " on target "
                    + self.targets[0][0]
                    + " on sample "
                    + self.samples[0][1]
                    + " by segments of "
                    + self.segments[0]
            )
        else:
            return (
                    "Lift10 of multiple predictors "
                    + " on target "
                    + self.targets[0][0]
                    + " on sample "
                    + self.samples[0][1]
                    + " by segments of "
                    + self.segments[0]
            )


class PredictorLiftCalculator(PredictorLiftInTimeCalculator):
    def calculate(self):

        df = self.samples[0][0]
        target = self.targets[0]
        predictors = self.predictors
        rowid_col = self.rowid_variable
        time_variable = self.time_variable

        arr = []
        for predictor in predictors:
            target_predictor_col = target[0] + predictor + "_tuple"
            df[target_predictor_col] = df[[target[0], predictor]].apply(lambda x: (x[target[0]], x[predictor]), axis=1)
            df["one"] = predictor
            lift_results = pd.pivot_table(
                df[df[target[1]] == 1],
                index="one",
                values=target_predictor_col,
                aggfunc=self.get_all_lifts(),
                margins=False,
            )
            lift_results.name = predictor
            a = lift_results.T.reset_index()[["level_0", predictor]]
            a.columns = ["Lift", predictor]
            a = a.set_index("Lift")
            arr.append(a)

        # cleanup
        for predictor in predictors:
            del df[target[0] + predictor + "_tuple"]
            # del df['one']

        # return final reults
        self.table = pd.concat(arr, axis=1, sort=False)
        self.table = self.table
        # self.table.column

    def get_table(self):
        return self.table

    def get_visualization(self):
        import matplotlib.pyplot as plt

        ax = self.table.plot(title="Lift")
        ax.set_xticks(list(np.arange(self.table.shape[0])))
        ax.set_xticklabels(self.table.index)

        plt.show()

    def get_description(self):
        if len(self.predictors) == 1:
            return (
                    "Lift of predictor "
                    + self.predictors[0]
                    + " on target "
                    + self.targets[0][0]
                    + " on sample "
                    + self.samples[0][1]
            )
        else:
            return (
                    "Lift of multiple predictors " + " on target " + self.targets[0][0] + " on sample " +
                    self.samples[0][1]
            )


# import Calculator

# expect only one sample
# expects only one target variable
# expects one or more predictors


class PredictorROCCalculator(Calculator):
    def calculate(self):

        # declaration of all used external variables
        # -----------------------------------------
        df = self.samples[0][0]
        target = self.targets[0]
        predictors = self.predictors
        rowid_col = self.rowid_variable
        time_variable = self.time_variable
        # ---------------------------------

        # if the predictor is categorical one, encode it with badrate first
        for predictor in predictors:
            if df[predictor].dtype == "O":
                df_piv = pd.pivot_table(
                    df, index=predictor, columns=target[0], values=rowid_col, aggfunc=len, fill_value=0
                )
                df_piv["br"] = df_piv.apply(lambda x: 1.00 * x[1] / (x[0] + x[1]), axis=1)
                df_piv = pd.DataFrame(df_piv.reset_index().values, columns=[predictor, "goods", "bads", "br"])
                df = (
                    df[[rowid_col, time_variable, target[0], target[1]] + predictors]
                        .copy()
                        .reset_index()
                        .merge(df_piv, on=predictor, how="left")
                        .set_index("index")
                )
                df[predictor] = df["br"]

        # prepare combinations of predictor and target
        arr = []
        from sklearn.metrics import roc_curve

        for predictor in predictors:
            fpr, tpr, cutoff = roc_curve(df[df[target[1]] == 1][target[0]], df[df[target[1]] == 1][predictor])
            auc = pd.DataFrame(
                [
                    pd.Series(fpr, name="False positive rate"),
                    pd.Series(tpr, name="True positive rate"),
                    pd.Series(cutoff, name="Cutoff"),
                ]
            ).T
            auc.columns = pd.MultiIndex.from_product([[predictor], auc.columns])
            arr.append(auc)

        # return final reults
        # print(auc)
        self.table = pd.concat(arr, axis=1, sort=False)
        return self

    def get_table(self):
        return self.table

    def get_visualization(self):
        import matplotlib.pyplot as plt

        for predictor in self.predictors:
            plt.plot(
                self.get_table()[predictor]["False positive rate"],
                self.get_table()[predictor]["True positive rate"],
                label=predictor,
            )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.plot([0, 1], [0, 1], color="grey", lw=2, linestyle="--")
        plt.title("ROC")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")

        plt.show()
        return self

    def get_description(self):
        if len(self.predictors) == 1:
            return (
                    "Gini of predictor "
                    + self.predictors[0]
                    + " on target "
                    + self.targets[0][0]
                    + " on sample "
                    + self.samples[0][1]
            )
        else:
            return (
                    "Gini of multiple predictors " + " on target " + self.targets[0][0] + " on sample " +
                    self.samples[0][1]
            )


# import Calculator

# expect only one sample
# expects only one target variable
# expects only one predictor


class GroupingEvaluationCalculator(Calculator):
    """
    Class for visualising statistics of a binned predictor.
    Includes risk in time, share in time, gini in time.

    Extends Calculator class and uses it's init.

    Args:
        projectParameters (ProjectParameters): definitions of doctools metadata
        use_weight (bool): uses weight if True (default: False)
        
    """
    def __init__(self, *args, use_weight=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_weight = use_weight

    def calculate(self):

        # declaration of all used external variables
        # -----------------------------------------
        df = self.samples[0][0]
        target = self.targets[0]
        predictor = self.predictors[0]
        rowid_col = self.rowid_variable
        time_variable = self.time_variable
        # ---------------------------------

        self.pr = PredictorRiskInTimeCalculator(self.projectParameters)

        self.pr.samples = self.samples
        self.pr.targets = self.targets
        self.pr.predictors = self.predictors
        self.pr.rowid_variable = self.rowid_variable
        self.pr.time_variable = self.time_variable
        self.pr.weight = self.weight

        self.ps = PredictorShareInTimeCalculator(self.projectParameters)

        self.ps.samples = self.samples
        self.ps.targets = self.targets
        self.ps.predictors = self.predictors
        self.ps.rowid_variable = self.rowid_variable
        self.ps.time_variable = self.time_variable
        self.ps.weight = self.weight

        self.pg = PredictorGiniInTimeCalculator(self.projectParameters)

        self.pg.samples = self.samples
        self.pg.targets = self.targets
        self.pg.predictors = self.predictors
        self.pg.rowid_variable = self.rowid_variable
        self.pg.time_variable = self.time_variable
        self.pg.weight = self.weight

        if self.use_weight:
            self.pr.calculate_weighted()
            self.ps.calculate_weighted()
            self.pg.calculate_weighted()
        else:
            self.pr.calculate()
            self.ps.calculate()
            self.pg.calculate()

        self.table = pd.concat([self.pr.get_table(), self.ps.get_table(), self.pg.get_table()], axis=1, sort=False)
        return self

    def get_table(self, beautiful=False):
        if not beautiful:
            return self.table
        else:
            table = self.table.copy()
            table.index.rename(self.pr.predictors[0], inplace=True)
            table = table[['Share', 'Bad rate']].T
            table = table[table.index.get_level_values(1) != 'All']
            table['WOE'] = table.index.get_level_values(1)
            table = table[['WOE'] + list(table.columns[:-1])]
            group_reindex = []
            for idx in table.index.get_level_values(1):
                try:
                    idx = self.replace_legend(float(idx), self.pr.predictors[0], self.grouping)
                except:
                    pass
                if idx not in group_reindex:
                    group_reindex.append(idx)
            table.index.set_levels(group_reindex, level=1, inplace=True)
            cm1 = sns.light_palette("Grey", as_cmap=True)
            cm2 = sns.diverging_palette(150, 20, n=7, as_cmap=True)
            table = table.fillna(0).style \
                .background_gradient(cmap=cm1, subset=('Share', [col for col in table.columns if col not in ['WOE']]),
                                     axis=None) \
                .background_gradient(cmap=cm2,
                                     subset=('Bad rate', [col for col in table.columns if col not in ['WOE']]),
                                     axis=None) \
                .highlight_null(null_color='white') \
                .format("{:.2%}", subset=[col for col in table.columns if col not in ['WOE']])
            return table

    def get_visualization(self, show_gini=True, output_folder=None, show_plot=True):
        from matplotlib import pyplot as plt

        if show_gini:
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 4))
        else:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        badrate = self.pr.table["Bad rate"]
        del badrate["All"]
        badrate = badrate.drop("All")
        badrate.index = badrate.index.astype(str)
        yaxislim = math.ceil(1.2 * badrate.max().max() * 40) / 40
        badrate.plot(
            ylim=(0, yaxislim),
            title="Risk of " + self.pr.predictors[0] + " on " + self.pr.targets[0][0],
            ax=axes[0],
            use_index=True,
        )
        axes[0].set_xticks(list(np.arange(badrate.shape[0])))
        axes[0].set_xticklabels(badrate.index)
        h, l = axes[0].get_legend_handles_labels()

        if self.grouping:
            try:
                l = [self.replace_legend(float(i), self.pr.predictors[0], self.grouping) for i in l]
            except:
                pass
        else:
            try:
                l = [f"{float(i):6.3f}" for i in l]
            except:
                pass
        axes[0].legend(h, l, bbox_to_anchor=(-0.08, 0.5), loc="right")

        share = self.ps.table["Share"]
        del share["All"]
        share = share.drop("All")
        share.index = share.index.astype(str)
        share.plot(ylim=(0, 1), title="Share of " + self.pr.predictors[0], ax=axes[1])
        axes[1].set_xticks(list(np.arange(share.shape[0])))
        axes[1].set_xticklabels(share.index)
        axes[1].get_legend().remove()

        if show_gini:
            gini = self.pg.table
            gini = gini.drop("All")
            gini.index = gini.index.astype(str)
            gini.plot(title="Gini of " + self.predictors[0] + " on " + self.targets[0][0], ylim=(0, 100), ax=axes[2])
            axes[2].set_xticks(list(np.arange(gini.shape[0])))
            axes[2].set_xticklabels(gini.index)
            axes[2].get_legend().remove()

        if output_folder:
            plt.savefig(path.join(output_folder, f"{self.predictors[0]}.PNG"), bbox_inches='tight', dpi=200)

        if show_plot:
            plt.show()
        else:
            plt.close()

        return self

    def get_description(self):
        return (
                "Grouping evaluation of predictor "
                + self.predictors[0]
                + " on target "
                + self.targets[0][0]
                + " on sample "
                + self.samples[0][1]
        )


# import Calculator

# expect only one sample
# expects only one target variable
# expects only one predictor


class ContinuousEvaluationCalculator(Calculator):
    """
    Class for visualising statistics of a continuous predictor.
    Includes risk in time, stats in time, gini in time, missing in time and histogram.

    Extends Calculator class and uses it's init.

    Args:
        projectParameters (ProjectParameters): definitions of doctools metadata
        
    """
    def calculate(self):

        # declaration of all used external variables
        # -----------------------------------------
        df = self.samples[0][0]
        target = self.targets[0]
        predictor = self.predictors[0]
        rowid_col = self.rowid_variable
        time_variable = self.time_variable
        # ---------------------------------

        self.pr = PredictorLogriskInTimeCalculator(self.projectParameters)

        self.pr.samples = self.samples
        self.pr.targets = self.targets
        self.pr.predictors = self.predictors
        self.pr.rowid_variable = self.rowid_variable
        self.pr.time_variable = self.time_variable
        self.pr.calculate()

        self.ps = PredictorStatsInTimeCalculator(self.projectParameters)

        self.ps.samples = self.samples
        self.ps.targets = self.targets
        self.ps.predictors = self.predictors
        self.ps.rowid_variable = self.rowid_variable
        self.ps.time_variable = self.time_variable
        self.ps.calculate()

        self.pg = PredictorGiniInTimeCalculator(self.projectParameters)

        self.pg.samples = self.samples
        self.pg.targets = self.targets
        self.pg.predictors = self.predictors
        self.pg.rowid_variable = self.rowid_variable
        self.pg.time_variable = self.time_variable
        self.pg.calculate()

        self.ph = PredictorHistogramCalculator(self.projectParameters)

        self.ph.samples = self.samples
        self.ph.targets = self.targets
        self.ph.predictors = self.predictors
        self.ph.rowid_variable = self.rowid_variable
        self.ph.time_variable = self.time_variable
        self.ph.calculate()

        self.pm = MissingInTimeCalculator(self.projectParameters)

        self.pm.samples = self.samples
        self.pm.targets = self.targets
        self.pm.predictors = self.predictors
        self.pm.rowid_variable = self.rowid_variable
        self.pm.time_variable = self.time_variable
        self.pm.calculate()

        self.table = pd.concat(
            [self.pr.get_table(), self.ps.get_table(), self.pm.get_table()[["Share", "Bad rate"]], self.pg.get_table()],
            axis=1,
            sort=False,
        )
        return self

    def get_table(self):
        return self.table

    def get_visualization(self, output_folder=None, show_plot=False):
        from matplotlib import pyplot as plt

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 4))

        # a = self.pr.get_table()["Logrisk on bin"]
        # a = a.drop('All')
        # a.index = a.index.astype(str)
        # a.T.plot(title="Logisk of " + self.pr.predictors[0] + " on " + self.pr.targets[0][0], ax=axes[0])
        # axes[0].set_xticks(list(np.arange(a.T.shape[0])))
        # axes[0].set_xticklabels(a.T.index)

        b = self.ps.get_table()
        b = b.drop("All")
        b.index = b.index.astype(str)
        b["Statistics"][["mean", "25%", "50%", "75%", "90%"]].plot(title="Quantiles " + self.predictors[0], ax=axes[0])
        axes[0].legend(bbox_to_anchor=(-0.08, 0.5), loc="right")
        # axes[1].set_xticks(list(np.arange(b.shape[0])))
        # axes[1].set_xticklabels(b.index)

        gini = self.pg.table
        gini = gini.drop("All")
        gini.index = gini.index.astype(str)
        gini.plot(title="Gini of " + self.predictors[0] + " on " + self.targets[0][0], ylim=(0, 100), ax=axes[2])
        # axes[3].set_xticks(list(np.arange(gini.shape[0])))
        # axes[3].set_xticklabels(gini.index)

        # self.ph.get_table().plot(kind="bar", title="Distribution", color="blue", width=1, ax=axes[3])
        # plt.xticks([], [])

        ax1 = axes[1]
        ax2 = ax1.twinx()

        ax1.set_ylabel("Risk")
        ax2.set_ylabel("Share")

        try:
            risk = pd.DataFrame(self.table["Bad rate"]["Missing"])
            risk.columns = ["Risk of missings"]
            risk = risk.drop("All")
            risk.index = risk.index.astype(str)

            risk.plot(ax=ax1, ylim=(0, 1), color="orange", linestyle="dashed")

            share = pd.DataFrame(self.table["Share"]["Missing"])
            share.columns = ["Share of missings"]
            share = share.drop("All")
            share.index = share.index.astype(str)
            share.plot(ax=ax2, ylim=(0, 1), color="orange", title="Share and risk of missings")

            # ax2.set_xticks(list(np.arange(share.shape[0])))
            # ax2.set_xticklabels(share.index)
        except:
            None

        try:
            share = pd.DataFrame(self.table["Share"]["Not missing"])
            share.columns = ["Share of non missings"]
            share = share.drop("All")
            share.index = share.index.astype(str)
            share.plot(ax=ax2, ylim=(0, 1), color="grey", title="Share and risk of missings")

            risk = pd.DataFrame(self.table["Bad rate"]["Not missing"])
            risk.columns = ["Risk of non missings"]
            risk = risk.drop("All")
            risk.index = risk.index.astype(str)
            risk.plot(ax=ax1, ylim=(0, 1), color="grey", linestyle="dashed")

            # ax1.set_xticks(list(np.arange(risk.shape[0])))
            # ax1.set_xticklabels(risk.index)
        except:
            None

        ax1.legend(loc=4)
        ax2.legend(loc=0)

        if output_folder:
            plt.savefig(path.join(output_folder, f"{self.predictors[0]}.PNG"), bbox_inches='tight', dpi=200)
        if show_plot:
            plt.show()

        return self

    def get_description(self):
        return (
                "Evaluation of "
                + self.predictors[0]
                + " on target "
                + self.targets[0][0]
                + " on sample "
                + self.samples[0][1]
        )


# import Calculator

# expect only one sample
# expects only one target variable
# expects multi predictor


class ScoreComparisonCalculator(Calculator):
    def calculate(self):

        # declaration of all used external variables
        # -----------------------------------------
        df = self.samples[0][0]
        target = self.targets[0]
        predictor = self.predictors[0]
        rowid_col = self.rowid_variable
        time_variable = self.time_variable
        # ---------------------------------

        self.pg = PredictorGiniInTimeCalculator(self.projectParameters)

        self.pg.samples = self.samples
        self.pg.targets = self.targets
        self.pg.predictors = self.predictors
        self.pg.rowid_variable = self.rowid_variable
        self.pg.time_variable = self.time_variable
        self.pg.calculate()

        self.ph = PredictorLiftInTimeCalculator(self.projectParameters)
        self.ph.samples = self.samples
        self.ph.targets = self.targets
        self.ph.predictors = self.predictors
        self.ph.rowid_variable = self.rowid_variable
        self.ph.time_variable = self.time_variable
        self.ph.calculate()

        self.sdc = SampleDescriptionCalculator(self.projectParameters)

        self.sdc.samples = self.samples
        self.sdc.targets = self.targets
        self.sdc.predictors = self.predictors
        self.sdc.rowid_variable = self.rowid_variable
        self.sdc.time_variable = self.time_variable
        self.sdc.calculate()

        self.plc = PredictorLiftCalculator(self.projectParameters)

        self.plc.samples = self.samples
        self.plc.targets = self.targets
        self.plc.predictors = self.predictors
        self.plc.rowid_variable = self.rowid_variable
        self.plc.time_variable = self.time_variable
        self.plc.calculate()

        self.procc = PredictorROCCalculator(self.projectParameters)

        self.procc.samples = self.samples
        self.procc.targets = self.targets
        self.procc.predictors = self.predictors
        self.procc.rowid_variable = self.rowid_variable
        self.procc.time_variable = self.time_variable
        self.procc.calculate()

        self.table = pd.concat([self.sdc.get_table(), self.pg.get_table(), self.ph.get_table()], axis=1, sort=False)
        return self

    def get_table(self):
        return self.table

    def get_visualization(self, output_folder=None, filename=None, show_plot=False):
        from matplotlib import pyplot as plt

        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(25, 5))

        gini = self.pg.table
        gini = gini.drop("All")
        gini.index = gini.index.astype(str)
        gini.plot(title="Gini of " + self.predictors[0] + " on " + self.targets[0][0], ylim=(0, 100), ax=axes[0])
        # axes[0].set_xticks(list(np.arange(gini.shape[0])))
        # axes[0].set_xticklabels(gini.index)

        lift = self.ph.get_table()
        lift = lift.drop("All")
        lift.index = lift.index.astype(str)
        lift.plot(title="Lift10 of " + self.predictors[0] + " on " + self.targets[0][0], ylim=(0, 7), ax=axes[1])
        # axes[1].set_xticks(list(np.arange(lift.shape[0])))
        # axes[1].set_xticklabels(lift.index)

        lift_static = self.plc.get_table()
        lift_static.plot(title="Lift of " + self.predictors[0] + " on " + self.targets[0][0], ax=axes[2])
        axes[2].set_xticks(list(np.arange(lift_static.shape[0])))
        axes[2].set_xticklabels(lift_static.index)

        roc = self.procc.get_table()
        for predictor in self.predictors:
            plt.plot(
                roc[predictor]["False positive rate"],
                roc[predictor]["True positive rate"],
                label=predictor,
                axes=axes[3],
                marker="None",
            )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.plot([0, 1], [0, 1], color="grey", lw=2, linestyle="--")
        plt.title("ROC")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")

        if output_folder:
            if not filename:
                filename = "score_comparison.PNG"

            plt.savefig(path.join(output_folder, filename), bbox_inches='tight', dpi=200)
        if show_plot:
            plt.show()
        return self

    def get_description(self):
        return "Comparison of scores" + " on target " + self.targets[0][0] + " on sample " + self.samples[0][1]


# Matus adjusted - Score by segments calculator
class ScoreSegmentsCalculator(Calculator):
    def calculate(self):

        # declaration of all used external variables
        # -----------------------------------------
        df = self.samples[0][0]
        target = self.targets[0]
        predictor = self.predictors[0]
        segment = self.segments[0]
        rowid_col = self.rowid_variable
        segment = self.segments[0]
        # ---------------------------------

        self.pg = PredictorGiniBySegmentsCalculator(self.projectParameters)  # done

        self.pg.samples = self.samples
        self.pg.targets = self.targets
        self.pg.predictors = self.predictors
        self.pg.rowid_variable = self.rowid_variable
        self.pg.segments = self.segments
        self.pg.calculate()

        self.ph = PredictorLiftBySegmentsCalculator(self.projectParameters)  # done
        self.ph.samples = self.samples
        self.ph.targets = self.targets
        self.ph.predictors = self.predictors
        self.ph.rowid_variable = self.rowid_variable
        self.ph.segments = self.segments
        self.ph.calculate()

        self.sdc = SampleDescriptionSegmentsCalculator(self.projectParameters)  # done

        self.sdc.samples = self.samples
        self.sdc.targets = self.targets
        self.sdc.predictors = self.predictors
        self.sdc.rowid_variable = self.rowid_variable
        self.sdc.segments = self.segments
        self.sdc.calculate()

        self.table = pd.concat([self.sdc.get_table(), self.pg.get_table(), self.ph.get_table()], axis=1, sort=False)
        return self

    def get_table(self):
        return self.table

    def get_visualization(self):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))

        targets = self.targets
        predictors = self.predictors
        segments = self.segments

        desc = self.sdc.get_table()["Sample description"][[targets[0][0] + " measurable", targets[0][0] + " bads"]]
        desc[targets[0][0] + " goods"] = desc[targets[0][0] + " measurable"] - desc[targets[0][0] + " bads"]
        desc = desc.drop("All")
        N = desc.shape[0]

        avg_scores = self.sdc.get_table()["Sample description"][
            [targets[0][0] + " default rate"]
            + ["AVG(" + predictor + ") on " + targets[0][0] + " meas." for predictor in predictors]
            ]
        avg_scores = avg_scores.drop("All")

        ind = np.arange(N)  # the x locations for the groups
        width = 0.25  # the width of the bars: can also be len(x) sequence

        ax1 = fig.axes[0]
        ax1.bar(ind, desc[targets[0][0] + " bads"], width, color="tomato", label="Bads")
        ax1.bar(
            ind,
            desc[targets[0][0] + " goods"],
            width,
            bottom=desc[targets[0][0] + " bads"],
            color="skyblue",
            label="Goods",
        )
        ax1.set_ylabel("Count of measurable cases")
        ax1.set_xlabel(segments[0])
        ax1.set_xticks(ind)
        ax1.set_xticklabels(desc.index)

        ax2 = ax1.twinx()
        yaxislim = math.ceil(1.4 * avg_scores.max().max() * 40) / 40
        ax2.set_ylim([0, yaxislim])
        ax2.plot(ind, avg_scores[targets[0][0] + " default rate"], label="Default rate")
        for predictor in predictors:
            ax2.plot(
                ind, avg_scores["AVG(" + predictor + ") on " + targets[0][0] + " meas."], label="AVG(" + predictor + ")"
            )

        ax1.set_title("Calibration on " + targets[0][0])
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")

        gini = self.pg.get_table()
        gini = gini.drop("All")
        ax3 = fig.axes[1]

        width = 0.15

        for i in np.arange(len(self.predictors)):
            ax3.bar(
                ind + width * (2 * i - len(self.predictors) + 1) / 2,
                gini["Gini " + self.targets[0][0]][predictors[i]],
                width=width,
                label=predictors[i],
            )

        ax3.set_title("Gini of predictors on " + self.targets[0][0])
        ax3.set_ylim(0, 100)
        ax3.set_xlabel(segments[0])
        ax3.set_xticks(ind)
        ax3.set_xticklabels(desc.index)
        ax3.legend(loc="upper right")

        lift = self.ph.get_table()
        lift = lift.drop("All")

        ax4 = fig.axes[2]

        width = 0.15

        for i in np.arange(len(self.predictors)):
            ax4.bar(
                ind + width * (2 * i - len(self.predictors) + 1) / 2,
                lift["Lift10"][predictors[i]],
                width=width,
                label=predictors[i],
            )

        ax4.set_title("Lift10 of predictors on " + self.targets[0][0])
        ax4.set_ylim(0, 6)
        ax4.set_xlabel(segments[0])
        ax4.set_xticks(ind)
        ax4.set_xticklabels(desc.index)
        ax4.legend(loc="upper right")

        for ax in fig.axes:
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)

        fig.subplots_adjust(wspace=0.3)

        for ax in axes.flatten():
            for label in ax.get_xticklabels():
                label.set_rotation(45)

        plt.show()

        return self

    def get_description(self):
        return (
                "Comparison of scores"
                + " on target "
                + self.targets[0][0]
                + " on sample "
                + self.samples[0][1]
                + " by segments of "
                + self.segments[0]
        )


# import Calculator

# expect only one sample
# expects only one predictor


class PredictorStatsInTimeCalculator(Calculator):
    def calculate(self):
        # declaration of all used external variables
        # -----------------------------------------
        df = self.samples[0][0]
        predictor = self.predictors[0]
        rowid_col = self.rowid_variable
        time_variable = self.time_variable
        # ---------------------------------

        b = df[[predictor]].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).T
        b.set_index(pd.MultiIndex.from_product([["All"]]), inplace=True)

        a = (
            df[[time_variable, predictor]]
                .groupby(time_variable)[predictor]
                .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        )
        # print(a)
        # a = a.reset_index()
        # a = a.rename(columns={'level_1':'statistics'})
        # print(a)
        # res = pd.pivot_table(a, index=time_variable, columns='statistics',values=predictor, aggfunc=np.max)
        res = a[b.columns]

        self.table = pd.concat([res, b], axis=0, sort=False)
        self.table.index.name = time_variable
        self.table.columns = pd.MultiIndex.from_product([["Statistics"], res.columns])

        return self

    def get_table(self):
        return self.table

    def get_visualization(self):
        import matplotlib.pyplot as plt

        a = self.table
        a.drop("All")
        ax = a["Statistics"][["mean", "std"]].plot(title="Mean and standard deviation of " + self.predictors[0])
        ax.set_xticks(list(np.arange(a["Statistics"][["mean", "std"]].shape[0])))
        ax.set_xticklabels(a["Statistics"][["mean", "std"]].index)
        plt.show()
        return self

    def get_description(self):
        return "Statistics of predictor " + self.predictors[0] + " on sample " + self.samples[0][1]


# import Calculator

# expect only one sample
# expects only one predictor


class PredictorHistogramCalculator(Calculator):
    def calculate(self):
        # declaration of all used external variables
        # -----------------------------------------
        df = self.samples[0][0]
        predictor = self.predictors[0]
        rowid_col = self.rowid_variable
        time_variable = self.time_variable
        # ---------------------------------

        self.table = pd.cut(df[predictor], bins=100).value_counts().sort_index(ascending=True)
        return self

    def get_table(self):
        return self.table

    def get_visualization(self):
        import matplotlib.pyplot as plt

        ax = self.samples[0][0][self.predictors[0]].plot(
            kind="hist", bins=100, title="Distribution of " + self.predictors[0]
        )
        ax.set_xticks(list(np.arange(self.samples[0][0][self.predictors[0]].shape[0])))
        ax.set_xticklabels(self.samples[0][0][self.predictors[0]].index)

        plt.show()
        return self

    def get_description(self):
        return "Histogram of predictor " + self.predictors[0] + " on sample " + self.samples[0][1]


# import Calculator

# expect only one sample
# expects only one target
# expects one predictor


class PredictorLogriskInTimeCalculator(Calculator):
    def calculate(self):

        # declaration of all used external variables
        # -----------------------------------------
        df = self.samples[0][0]
        predictor = self.predictors[0]
        target = self.targets[0]
        rowid_col = self.rowid_variable
        time_variable = self.time_variable
        # ---------------------------------

        df = df.copy()

        # ri = RussianImputer().fit(df[df[target[1]] == 1][predictor], df[df[target[1]] == 1][target[0]])
        # df[predictor] = ri.transform(df[predictor])

        # if df[df[target[1]]==1][predictor].value_counts().shape[0]<=10:
        #    a = df[df[target[1]]==1][predictor]
        # else:
        try:
            a = pd.cut(df[df[target[1]] == 1][predictor], bins=5, labels=False)
        except:
            try:
                a = pd.cut(df[df[target[1]] == 1][predictor], bins=5, labels=False)
            except:
                a = pd.cut(df[df[target[1]] == 1][predictor], bins=2, labels=False)

        gb = df[df[target[1]] == 1][[target[0], time_variable]].groupby([time_variable, a])

        self.table = pd.DataFrame(gb.apply(lambda x: (len(x))))
        self.table = pd.concat([self.table, gb.apply(lambda x: (np.sum(x[target[0]])))], axis=1, sort=False)
        self.table = pd.concat([self.table, gb.apply(lambda x: (np.mean(x[target[0]])))], axis=1, sort=False)
        self.table.columns = ["Meas. obs. on bin", "Bads on bin", "Risk on bin"]
        self.table["Logrisk on bin"] = self.table["Risk on bin"].apply(
            lambda p: -np.inf if p == 1 else np.log(p / (1 - p))
        )

        gb_nt = df[df[target[1]] == 1][[target[0], time_variable]].groupby([a])

        gb_nt_a = pd.DataFrame(gb_nt.apply(lambda x: (len(x))))
        gb_nt_a = pd.concat([gb_nt_a, gb_nt.apply(lambda x: (np.sum(x[target[0]])))], axis=1, sort=False)
        gb_nt_a = pd.concat([gb_nt_a, gb_nt.apply(lambda x: (np.mean(x[target[0]])))], axis=1, sort=False)
        gb_nt_a.columns = ["Meas. obs. on bin", "Bads on bin", "Risk on bin"]
        gb_nt_a["Logrisk on bin"] = gb_nt_a["Risk on bin"].apply(lambda p: -np.inf if p == 1 else np.log(p / (1 - p)))

        gb_nt_a.set_index(pd.MultiIndex.from_product([["All"], gb_nt_a.index.values]), inplace=True)

        self.table = pd.concat([self.table, gb_nt_a], axis=0, sort=False)

        return self

    def get_table(self):
        return self.table.unstack()

    def get_visualization(self):
        import matplotlib.pyplot as plt

        # self.get_table()['logrisk'].plot( title='Association of '+ self.predictors[0] + ' with logrisk')
        # self.get_table()['risk'].plot( title='Association of '+ self.predictors[0] + ' with risk')
        ax = self.table["Risk on bin"].unstack().T.plot()
        ax.set_xticks(list(np.arange(self.table["Risk on bin"].unstack().T.shape[0])))
        ax.set_xticklabels(self.table["Risk on bin"].unstack().T.index)
        plt.show()
        return self

    # def get_calculator_parameters(self):
    #     return self.calculatorParameters

    def get_description(self):
        return (
                "Logrisk in time of predictor "
                + self.predictors[0]
                + " on target "
                + self.targets[0][0]
                + " on sample "
                + self.samples[0][1]
        )


# import Calculator

# expect only one sample
# expects only one target
# expects one predictor


class PredictorLogriskCalculator(Calculator):
    def calculate(self):

        # declaration of all used external variables
        # -----------------------------------------
        df = self.samples[0][0]
        predictor = self.predictors[0]
        target = self.targets[0]
        rowid_col = self.rowid_variable
        time_variable = self.time_variable
        # ---------------------------------

        if df[df[target[1]] == 1][predictor].value_counts().shape[0] <= 10:
            a = df[df[target[1]] == 1][predictor]
        else:
            try:
                a = pd.qcut(df[df[target[1]] == 1][predictor], q=10)
            except:
                a = pd.qcut(df[df[target[1]] == 1][predictor], q=5)

        gb = df[df[target[1]] == 1][[target[0], time_variable]].groupby([a])
        self.table = pd.DataFrame(gb.apply(lambda x: (len(x))))
        self.table = pd.concat([self.table, gb.apply(lambda x: (np.sum(x[target[0]])))], axis=1, sort=False)
        self.table = pd.concat([self.table, gb.apply(lambda x: (np.mean(x[target[0]])))], axis=1, sort=False)
        self.table.columns = ["cnt", "bads", "risk"]
        self.table["logrisk"] = self.table["risk"].apply(lambda p: np.log(p / (1 - p)))
        return self

    def get_table(self):
        return self.table

    def get_visualization(self):
        import matplotlib.pyplot as plt

        # self.get_table()['logrisk'].plot( title='Association of '+ self.predictors[0] + ' with logrisk')
        # self.get_table()['risk'].plot( title='Association of '+ self.predictors[0] + ' with risk')
        ax = self.table["logrisk"].plot()
        ax.set_xticks(list(np.arange(self.table["logrisk"].shape[0])))
        ax.set_xticklabels(self.table["logrisk"].index)
        plt.show()
        return self

    def get_description(self):
        return (
                "Logrisk of predictor "
                + self.predictors[0]
                + " on target "
                + self.targets[0][0]
                + " on sample "
                + self.samples[0][1]
        )


# expect list of at least 2 samples (i.e. a sample is a tuple of the form (dataset,dataset's_name)) with the convention that first sample in the list is the train sample and second sample is the test sample
# expects only one target variable (i.e. a tuple of the form (target_column, target's_name))
# expects all predictors of the proposed model
# expects one score, the proposed model's score, with the expectation that it is the first one in list of scores


class MarginalContributionsCalculator(Calculator):
    def calculate(self):

        # declaration of all used external variables
        #
        # -----------------------------------------
        samples = self.samples
        target = self.targets
        predictors = [pred for pred in self.predictors if "WOE" in pred]
        proposed_score = self.scores[0]
        # ---------------------------------

        # calculating the tuple of Ginis of the whole proposed model
        from sklearn import metrics
        from sklearn import linear_model

        lg = linear_model.LogisticRegression(solver="lbfgs")

        t = ("Nothing is removed",)

        for sample, name in samples:
            for tgt_col, tgt_obs in target:
                df = sample.loc[sample[tgt_obs] == 1, :]

                if len(df[tgt_col].unique()) > 1:
                    gini_sample = round(100 * (metrics.roc_auc_score(df[tgt_col], df[proposed_score]) * 2 - 1), 2)
                else:
                    gini_sample = float("NaN")

                t += (gini_sample,)

        self.table = [t]

        # calculating the tuples of Ginis of the submodels
        for pred in predictors:
            if pred != "Intercept":

                # defining training dataset
                df = samples[0][0]
                df = df.loc[df[target[0][1]] == 1, :]
                df["Intercept"] = 1

                # fitting model without one predictor on the training dataset
                res = lg.fit(df[list(set(predictors + ["Intercept"]) - set([pred]))], df[target[0][0]])

                # creating and filling a tuple
                t = (pred,)

                for sample, name in samples:
                    for tgt_col, tgt_obs in target:

                        df = sample.loc[sample[tgt_obs] == 1, :]
                        df["Intercept"] = 1
                        if len(df[tgt_col].unique()) > 1:
                            gini_sample = round(
                                100
                                * (
                                        metrics.roc_auc_score(
                                            df[tgt_col],
                                            res.predict_proba(df[list(set(predictors + ["Intercept"]) - set([pred]))])[
                                            :, 1
                                            ],
                                        )
                                        * 2
                                        - 1
                                ),
                                2,
                            )
                        else:
                            gini_sample = float("NaN")

                        t += (gini_sample,)

            # appending the tuple to the table
            self.table.append(t)

        # polishing the table
        self.table = pd.DataFrame(self.table)
        self.table = self.table.rename(columns={0: "PREDICTOR REMOVED"})
        self.table = self.table.set_index("PREDICTOR REMOVED")
        self.table.columns = pd.MultiIndex.from_product(
            [[name for sample, name in samples], [tgt_name for tgt_name, meas in self.targets]]
        )

        # self.table.columns = ['Predictor removed'] + [name+':'+tgt_col for sample,name in samples for tgt_col,tgt_name in target]
        self.table = self.table.sort_values(by=self.table.columns[1], ascending=False)

        return self

    def get_table(self):
        return self.table

    def get_visualization(self):
        return self

    def get_description(self):
        return "Marginal contributions of predictors."


# Adjusted by Matus
class MarginalContributionsAddCalculator(Calculator):
    def calculate(self):

        # declaration of all used external variables
        #
        # -----------------------------------------
        samples = self.samples
        target = self.targets
        predictors = self.predictors
        predictors_to_add = self.predictors_to_add
        proposed_score = self.scores[0]
        # ---------------------------------

        # calculating the tuple of Ginis of the whole proposed model
        from sklearn import metrics
        from sklearn import linear_model

        lg = linear_model.LogisticRegression()

        t = ("Nothing is added",)

        for sample, name in samples:
            for tgt_col, tgt_obs in target:
                df = sample.loc[sample[tgt_obs] == 1, :]

                if len(df[tgt_col].unique()) > 1:
                    gini_sample = round(100 * (metrics.roc_auc_score(df[tgt_col], df[proposed_score]) * 2 - 1), 2)
                else:
                    gini_sample = float("NaN")

                t += (gini_sample,)

        base_model = [t]
        self.table = []

        # calculating the tuples of Ginis of the submodels
        for pred in predictors_to_add:
            # defining training dataset
            df = samples[0][0]
            df = df.loc[df[target[0][1]] == 1, :]
            df["Intercept"] = 1

            # fitting model without one predictor on the training dataset
            res = lg.fit(df[predictors + ["Intercept"] + [pred]], df[target[0][0]])

            # creating and filling a tuple
            t = (pred,)
            for sample, name in samples:
                for tgt_col, tgt_obs in target:

                    df = sample.loc[sample[tgt_obs] == 1, :]
                    df["Intercept"] = 1

                    if len(df[tgt_col].unique()) > 1:
                        gini_sample = round(
                            100
                            * (
                                    metrics.roc_auc_score(
                                        df[tgt_col], res.predict_proba(df[predictors + ["Intercept"] + [pred]])[:, 1]
                                    )
                                    * 2
                                    - 1
                            ),
                            2,
                        )
                    else:
                        gini_sample = float("NaN")

                    t += (gini_sample,)

            # appending the tuple to the table
            self.table.append(t)

        # polishing the table
        base_model = pd.DataFrame(base_model)
        base_model = base_model.rename(columns={0: "PREDICTOR ADDED"})
        base_model = base_model.set_index("PREDICTOR ADDED")
        base_model.columns = pd.MultiIndex.from_product(
            [[name for sample, name in samples], [tgt_name for tgt_name, meas in self.targets]]
        )

        self.table = pd.DataFrame(self.table)
        self.table = self.table.rename(columns={0: "PREDICTOR ADDED"})
        self.table = self.table.set_index("PREDICTOR ADDED")
        self.table.columns = pd.MultiIndex.from_product(
            [[name for sample, name in samples], [tgt_name for tgt_name, meas in self.targets]]
        )

        # self.table.columns = ['Predictor removed'] + [name+':'+tgt_col for sample,name in samples for tgt_col,tgt_name in target]
        self.table.sort_values(by=self.table.columns[0], inplace=True, ascending=False)

        frames = [base_model, self.table]
        self.table = pd.concat(frames, sort=False)

        return self

    def get_table(self):
        return self.table

    def get_visualization(self):
        return self

    def get_description(self):
        return "Marginal contributions of predictors."


class MarginalLiftsCalculator(Calculator):
    def calculate(self):

        # declaration of all used external variables
        #
        # -----------------------------------------
        samples = self.samples
        target = self.targets
        predictors = self.predictors
        proposed_score = self.scores[0]

        # ---------------------------------

        def liftN_loc(target_, prediction_, q):

            dframe = pd.DataFrame([pd.Series(target_, name="target"), pd.Series(prediction_, name="prediction")]).T
            # print(dframe)
            quant_ = dframe["prediction"].quantile(q)
            pop_dr = dframe["target"].mean()
            q_dr = dframe["target"][dframe["prediction"] > quant_].mean()
            if pop_dr == 0:
                return np.nan
            else:
                lift = 1.0 * q_dr / pop_dr
                return lift

        # calculating the tuple of Ginis of the whole proposed model
        from sklearn import metrics
        from sklearn import linear_model

        lg = linear_model.LogisticRegression()

        t = ("Nothing is removed",)

        for sample, name in samples:
            for tgt_col, tgt_obs in target:
                df = sample.loc[sample[tgt_obs] == 1, :]

                if len(df[tgt_col].unique()) > 1:
                    lift_sample = round(liftN_loc(df[tgt_col], df[proposed_score], 0.9), 5)
                else:
                    lift_sample = float("NaN")

                t += (lift_sample,)

        self.table = [t]

        # calculating the tuples of Ginis of the submodels
        for pred in predictors:
            if pred != "Intercept":

                # defining training dataset
                df = samples[0][0]
                df = df.loc[df[target[0][1]] == 1, :]
                df["Intercept"] = 1

                # fitting model without one predictor on the training dataset
                res = lg.fit(df[list(set(predictors + ["Intercept"]) - set([pred]))], df[target[0][0]])

                # creating and filling a tuple
                t = (pred,)

                for sample, name in samples:
                    for tgt_col, tgt_obs in target:

                        dframe = sample.loc[sample[tgt_obs] == 1, :].reset_index().drop(["index"], axis=1)

                        if len(dframe[tgt_col].unique()) > 1:
                            lift_sample = round(
                                liftN_loc(
                                    dframe[tgt_col],
                                    res.predict_proba(dframe[list(set(predictors + ["Intercept"]) - set([pred]))])[
                                    :, 1
                                    ],
                                    0.9,
                                ),
                                5,
                            )
                        else:
                            lift_sample = float("NaN")

                        t += (lift_sample,)

            # appending the tuple to the table
            self.table.append(t)

        # polishing the table
        self.table = pd.DataFrame(self.table)
        self.table = self.table.rename(columns={0: "PREDICTOR REMOVED"})
        self.table = self.table.set_index("PREDICTOR REMOVED")
        self.table.columns = pd.MultiIndex.from_product(
            [[name for sample, name in samples], [tgt_name for tgt_name, meas in self.targets]]
        )

        # self.table.columns = ['Predictor removed'] + [name+':'+tgt_col for sample,name in samples for tgt_col,tgt_name in target]
        self.table = self.table.sort_values(by=self.table.columns[1], ascending=False)

        return self

    def get_table(self):
        return self.table

    def get_visualization(self):
        return self

    def get_description(self):
        return "Marginal contributions of predictors."


# expects one sample
# expects all predictors of the proposed model


class CorrelationCalculator(Calculator):
    """
    Class for visualising correlation of predictors.

    Extends Calculator class and uses it's init.

    Args:
        projectParameters (ProjectParameters): definitions of doctools metadata
        
    """
    def calculate(self):
        # declaration of all used external variables
        #
        # -----------------------------------------
        samples = self.samples
        target = self.targets
        predictors = self.predictors
        df = samples[0][0]
        # ---------------------------------

        self.table = df[predictors].corr(method="spearman")

        return self

    def get_table(self):
        return self.table

    def get_visualization(self):
        ## missing implementation?
        return self

    def get_description(self):
        return "Correlation matrix on" + self.samples[0][0]


# expects one sample
# expects two scores


class TransitionCalculator(Calculator):
    def calculate(self):
        # declaration of all used external variables
        #
        # -----------------------------------------
        samples = self.samples
        target = self.targets
        predictors = self.predictors
        rowid_variable = self.rowid_variable
        df = samples[0][0]

        # ---------------------------------

        scoreA = self.scores[0]
        scoreB = self.scores[1]
        dfx = df.copy()
        dfx[scoreA + "_bin"] = pd.qcut(dfx[scoreA], q=5)
        dfx[scoreB + "_bin"] = pd.qcut(dfx[scoreB], q=5)
        self.table = pd.pivot_table(
            dfx,
            index=scoreA + "_bin",
            columns=dfx[scoreB + "_bin"],
            aggfunc=len,
            values=rowid_variable,
            dropna=False,
            fill_value=0,
        )

        return self

    def get_table(self):
        return self.table

    def get_visualization(self):
        return self

    def get_description(self):
        return "Transition matrix from " + self.scores[1] + " to " + self.scores[0] + " on " + self.samples[0][1]


class XgbPredictorEvaluationCalculator(Calculator):
    def calculate(self):

        # declaration of all used external variables
        # -----------------------------------------
        df = self.samples[0][0]
        target = self.targets[0]
        predictor = self.predictors[0]
        rowid_col = self.rowid_variable
        time_variable = self.time_variable
        # ---------------------------------

        self.pr = PredictorLogriskInTimeCalculator(self.projectParameters)

        self.pr.samples = self.samples
        self.pr.targets = self.targets
        self.pr.predictors = self.predictors
        self.pr.rowid_variable = self.rowid_variable
        self.pr.time_variable = self.time_variable
        self.pr.calculate()

        self.ps = PredictorStatsInTimeCalculator(self.projectParameters)

        self.ps.samples = self.samples
        self.ps.targets = self.targets
        self.ps.predictors = self.predictors
        self.ps.rowid_variable = self.rowid_variable
        self.ps.time_variable = self.time_variable
        self.ps.calculate()

        self.pg = PredictorGiniInTimeCalculator(self.projectParameters)

        self.pg.samples = self.samples
        self.pg.targets = self.targets
        self.pg.predictors = self.predictors
        self.pg.rowid_variable = self.rowid_variable
        self.pg.time_variable = self.time_variable
        self.pg.calculate()

        self.ph = PredictorHistogramCalculator(self.projectParameters)

        self.ph.samples = self.samples
        self.ph.targets = self.targets
        self.ph.predictors = self.predictors
        self.ph.rowid_variable = self.rowid_variable
        self.ph.time_variable = self.time_variable
        self.ph.calculate()

        self.ipc = IceplotCalculator(self.projectParameters)
        self.ipc.samples = self.samples
        self.ipc.predictors = self.predictors
        self.ipc.calculate()

        self.pm = MissingInTimeCalculator(self.projectParameters)
        self.pm.samples = self.samples
        self.pm.targets = self.targets
        self.pm.predictors = self.predictors
        self.pm.rowid_variable = self.rowid_variable
        self.pm.time_variable = self.time_variable
        self.pm.calculate()

        self.table = pd.concat(
            [self.pr.get_table(), self.ps.get_table(), self.pm.get_table()[["Share", "Bad rate"]], self.pg.get_table()],
            axis=1,
            sort=False,
        )
        return self

    def get_table(self):
        return self.table

    def get_visualization(self):
        from matplotlib import pyplot as plt

        fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(30, 5))

        a = self.pr.get_table()["Logrisk on bin"]
        # a = a.drop('All')
        a.T.plot(title="Logisk of " + self.pr.predictors[0] + " on " + self.pr.targets[0][0], ax=axes[0])
        axes[0].set_xticks(list(np.arange(a.T.shape[0])))
        axes[0].set_xticklabels(a.T.index)

        b = self.ps.get_table()
        b = b.drop("All")
        b["Statistics"][["mean", "25%", "50%", "75%", "90%"]].plot(title="Quantiles " + self.predictors[0], ax=axes[1])
        axes[1].set_xticks(list(np.arange(b["Statistics"][["mean", "25%", "50%", "75%", "90%"]].shape[0])))
        axes[1].set_xticklabels(b["Statistics"][["mean", "25%", "50%", "75%", "90%"]].index)

        gini = self.pg.table
        gini = gini.drop("All")
        gini.plot(title="Gini of " + self.predictors[0] + " on " + self.targets[0][0], ylim=(0, 100), ax=axes[3])
        axes[3].set_xticks(list(np.arange(gini.shape[0])))
        axes[3].set_xticklabels(gini.index)

        self.ph.get_table().plot(kind="bar", title="Distribution", color="blue", width=1, ax=axes[5])
        plt.xticks([], [])

        ax1 = axes[2]
        ax2 = ax1.twinx()

        ax1.set_ylabel("Risk")
        ax2.set_ylabel("Share")

        try:
            risk = pd.DataFrame(self.table["Bad rate"]["Missing"])
            risk.columns = ["Risk of missings"]
            risk = risk.drop("All")
            risk.plot(ax=ax1, ylim=(0, 1), color="orange", linestyle="dashed")

            share = pd.DataFrame(self.table["Share"]["Missing"])
            share.columns = ["Share of missings"]
            share = share.drop("All")
            share.plot(ax=ax2, ylim=(0, 1), color="orange", title="Share and risk of missings")

            ax2.set_xticks(list(np.arange(share.shape[0])))
            ax2.set_xticklabels(share.index)
        except:
            None

        try:
            share = pd.DataFrame(self.table["Share"]["Not missing"])
            share.columns = ["Share of non missings"]
            share = share.drop("All")
            share.plot(ax=ax2, ylim=(0, 1), color="grey", title="Share and risk of missings")

            risk = pd.DataFrame(self.table["Bad rate"]["Not missing"])
            risk.columns = ["Risk of non missings"]
            risk = risk.drop("All")
            risk.plot(ax=ax1, ylim=(0, 1), color="grey", linestyle="dashed")

            ax1.set_xticks(list(np.arange(risk.shape[0])))
            ax1.set_xticklabels(risk.index)
        except:
            None

        ax1.legend(loc=4)
        ax2.legend(loc=0)

        self.ipc.get_table().plot(ax=axes[4], title="Ice plot of " + self.predictors[0])

        plt.show()

        return self

    def get_description(self):
        return (
                "Evaluation of "
                + self.predictors[0]
                + " on target "
                + self.targets[0][0]
                + " on sample "
                + self.samples[0][1]
        )


class MissingInTimeCalculator(Calculator):
    def calculate(self):

        # declaration of all used external variables
        # -----------------------------------------
        df = self.samples[0][0]
        predictor = self.predictors[0]
        target = self.targets[0]
        rowid_col = self.rowid_variable
        time_variable = self.time_variable
        # ---------------------------------

        df = df.copy()
        df[predictor] = df[predictor].isnull().apply(lambda x: "Missing" if x == True else "Not missing")
        self.samples[0] = (df, self.samples[0][1])

        self.pr = PredictorRiskInTimeCalculator(self.projectParameters)

        self.pr.samples = self.samples
        self.pr.targets = self.targets
        self.pr.predictors = self.predictors
        self.pr.rowid_variable = self.rowid_variable
        self.pr.time_variable = self.time_variable
        self.pr.calculate()

        self.ps = PredictorShareInTimeCalculator(self.projectParameters)

        self.ps.samples = self.samples
        self.ps.targets = self.targets
        self.ps.predictors = self.predictors
        self.ps.rowid_variable = self.rowid_variable
        self.ps.time_variable = self.time_variable
        self.ps.calculate()

        self.table = pd.concat([self.pr.get_table(), self.ps.get_table()], axis=1, sort=False)

        return self

    def get_table(self):
        return self.table

    def get_visualization(self):
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()

        ax1.set_ylabel("Risk")
        ax2.set_ylabel("Share")

        try:
            risk = pd.DataFrame(self.table["Bad rate"]["Missing"])
            risk.columns = ["Risk of missings"]
            risk = risk.drop("All")
            risk.plot(ax=ax1, ylim=(0, 1), color="orange", linestyle="dashed")

            share = pd.DataFrame(self.table["Share"]["Missing"])
            share.columns = ["Share of missings"]
            share = share.drop("All")
            share.plot(ax=ax2, ylim=(0, 1), color="orange", title="Share and risk of missings")

            ax2.set_xticks(list(np.arange(share.shape[0])))
            ax2.set_xticklabels(share.index)
        except:
            None

        try:
            share = pd.DataFrame(self.table["Share"]["Not missing"])
            share.columns = ["Share of non missings"]
            share = share.drop("All")
            share.plot(ax=ax2, ylim=(0, 1), color="grey", title="Share and risk of missings")

            risk = pd.DataFrame(self.table["Bad rate"]["Not missing"])
            risk.columns = ["Risk of non missings"]
            risk = risk.drop("All")
            risk.plot(ax=ax1, ylim=(0, 1), color="grey", linestyle="dashed")

            ax1.set_xticks(list(np.arange(risk.shape[0])))
            ax1.set_xticklabels(risk.index)
        except:
            None

        ax1.legend(loc=4)
        ax2.legend(loc=0)

        plt.show()

    def get_description(self):
        return 1


# expect one sample, one predictor, no target
# expect a scoring object with function predict_proba


class IceplotCalculator(Calculator):
    def calculate(self):
        # declaration of all used external variables
        # -----------------------------------------
        df = self.samples[0][0]
        predictor = self.predictors[0]
        xgb_model = self.projectParameters.model
        rowid_col = self.rowid_variable
        time_variable = self.time_variable
        # ---------------------------------

        predictor_stats = df[predictor].describe().T
        # print(predictor_stats)
        predictor_min = predictor_stats["min"]
        predictor_max = predictor_stats["max"]

        res = []
        to_test = list(np.linspace(predictor_min, predictor_max, num=20))
        # to_test.append(np.nan)
        for i in to_test:
            df = df.copy().sample(2000, random_state=42)
            df[predictor] = i

            if hasattr(xgb_model[0], 'predict_proba'):
                m = pd.Series(xgb_model[0].predict_proba(df[self.projectParameters.predictors_continuous])[:, 1]).mean()
            else:
                m = pd.Series(xgb_model[0].predict(df[self.projectParameters.predictors_continuous])).mean()
            if i != i:
                res.append((predictor_min - 1, m))
            else:
                res.append((i, m))

        t = pd.DataFrame(res)
        t.columns = [predictor, xgb_model[1]]
        t = t.set_index(predictor)
        self.table = t
        return self

    def get_table(self):
        return self.table

    def get_visualization(self):
        import matplotlib.pyplot as plt

        ax = self.table.plot(
            title="Association of " + self.predictors[0] + " with score " + self.projectParameters.model[1]
        )
        ax.set_xticks(list(np.arange(self.table.shape[0])))
        ax.set_xticklabels(self.table.index)
        plt.show()

    def get_description(self):
        return (
                "Association of predictor "
                + self.predictors[0]
                + " with score "
                + self.projectParameters.model[1]
                + " on sample "
                + self.samples[0][1]
        )


class IceplotMatrixCalculator(Calculator):
    def calculate(self):

        # declaration of all used external variables
        # -----------------------------------------
        df = self.samples[0][0]
        predictors = self.predictors
        xgb_model = self.projectParameters.model
        rowid_col = self.rowid_variable
        time_variable = self.time_variable
        # ---------------------------------

        # def  f(predictor):
        #    ipc = IceplotCalculator(self.projectParameters)
        #    ipc.s([self.samples[0]]).p([predictor]).calculate()

        import multiprocessing as mp

        arr = []
        for predictor in predictors:
            ipc = IceplotCalculator(self.projectParameters)
            ipc.s([self.samples[0]]).p([predictor])
            arr.append(ipc)

        pool = mp.Pool(processes=2)
        arr = pool.map(self.f, arr)
        pool.close()
        # print(result_list)

        self.table = arr
        return self

    def get_table(self):
        return None

    def get_visualization(self):
        from matplotlib import pyplot as plt
        import math

        arr = self.table
        nrows_ = math.ceil(len(arr) / 5)
        fig, axes = plt.subplots(nrows=nrows_, ncols=5, figsize=(nrows_ * 2, 60))

        for i in range(0, len(arr)):
            arr[i].get_table().plot(
                ax=axes[int(i / 5)][i % 5], use_index=True, title=arr[i].get_table().index.name
            ).set_xlabel("")
        plt.show()

    def get_description(self):
        return (
                "Association of multiple predictors "
                + "with score "
                + self.projectParameters.model[1]
                + " on sample "
                + self.samples[0][1]
        )

    def f(self, x):
        return x.calculate()


class IceplotRuCalculator(Calculator):
    """Doctools calculator. Individual Conditional Expectations.

    For each observation, we show how the prediction would change if one particular variable was chagning its values (and all the other variables remained the same)
    and we draw these lines all into one chart (in our case there are lines for 250 randomly chosen observations).
    
    There is also mean PDP showed by a thick line in the chart.

    Args:
        projectParameters (ProjectParameters): definitions of doctools metadata
        sample_size (int, optional): Number of obserations to be used to calculate PDP. Defaults to 1000.
        frac_to_plot (float, optional): Fraction of sample size to be plotted to ICE plot. Defaults to 0.1.

	Methods:
        calculate(): calculates table for the visualisation
        get_table(): returns the calculated table as pd.DataFrame
        get_visualisation(showPlot=False, outputFolder=None, fileName=None): plots the visualisation either to ipython or to file or both
    """

    def __init__(self, *args, sample_size=1000, frac_to_plot=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_size = sample_size
        self.frac_to_plot = frac_to_plot

    def _get_grid_points(self, x, num_grid_points, add_nan):
        if num_grid_points is None:
            return x.unique()
        elif add_nan:
            return np.append(x.quantile(np.linspace(0, 1, num_grid_points)).unique(), [np.nan])
        else:
            return x.quantile(np.linspace(0, 1, num_grid_points)).unique()

    def _to_ice_data(self, df, column, x_s):
        ice_data = pd.DataFrame(np.repeat(df.values, x_s.size, axis=0), columns=df.columns)
        data_column = ice_data[column].copy()
        ice_data[column] = np.tile(x_s, df.shape[0])

        for col in df.columns:
            ice_data[col] = ice_data[col].astype(df[col].dtype)

        return ice_data, data_column

    def _get_quantiles(self, x):
        return np.greater.outer(x, x).sum(axis=1) / x.size

    def _pdp(self):
        return self.ice_data.mean(axis=1)

    def calculate(self):
        # declaration of all used external variables
        # -----------------------------------------
        df = self.samples[0][0]
        predictor = self.predictors[0]
        model = self.projectParameters.model
        rowid_col = self.rowid_variable
        time_variable = self.time_variable
        # ---------------------------------

        from pandas.api.types import is_numeric_dtype
        self.is_cat = not is_numeric_dtype(df[predictor])
        self.has_nan = df[predictor].isnull().sum()

        if len(df[predictor]) > self.sample_size:
            df = df.sample(self.sample_size, random_state=241)

        if self.is_cat:
            x_s = self._get_grid_points(df[predictor], None, False)
        else:
            x_s = self._get_grid_points(df[predictor], 20, self.has_nan)

        self.ice_data = self._to_ice_data(df[model[2]], predictor, x_s)[0]

        if hasattr(model[0], 'predict_proba'):
            self.ice_data['ice_y'] = model[0].predict_proba(self.ice_data[model[2]])[:, 1]
        else:
            self.ice_data['ice_y'] = model[0].predict(self.ice_data[model[2]])

        other_columns = [column for column in model[2] if column != predictor]

        for col in self.ice_data.columns:
            self.ice_data[col] = self.ice_data[col].astype('object')
        if self.is_cat:
            self.ice_data.fillna('NaN', inplace=True)
        else:
            self.fill_value = self.ice_data[predictor].max() + 1
            self.ice_data.fillna(self.fill_value, inplace=True)
        self.ice_data = self.ice_data.pivot_table(values='ice_y', index=other_columns, columns=predictor).T

        if not self.is_cat:
            if not self.ice_data.index.is_monotonic_increasing:
                self.ice_data = self.ice_data.sort_index()

        return self

    def get_table(self):
        pass

    def get_visualization(self, output_folder=None, filename=None, **kwargs):
        import matplotlib.pyplot as plt
        from matplotlib import rc

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)

        if self.frac_to_plot < 1.:
            n_cols = self.ice_data.shape[1]
            icols = np.random.choice(
                n_cols,
                size=math.floor(self.frac_to_plot * n_cols),
                replace=False
            )
            plot_ice_data = self.ice_data.iloc[:, icols]
        else:
            plot_ice_data = self.ice_data

        if self.is_cat:
            x = range(0, len(self.ice_data.index))
            x_ticks = x
            x_labels = self.ice_data.index
        else:
            x = self._get_quantiles(self.ice_data.index[self.ice_data.index < self.fill_value])
            x_ticks = np.linspace(x.min(), x.max(), 10)
            x_labels = ['{:0.3f}'.format(num) for num in self.ice_data.index[self.ice_data.index < self.fill_value][
                [len(x) - len(x[x >= q]) for q in x_ticks]]]
            if self.has_nan:
                x = np.append(x, [1.1])
                x_ticks = np.append(x_ticks, [1.1])
                x_labels = np.append(x_labels, ['NaN'])

        ax.plot(x, plot_ice_data, **kwargs)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=45)

        pdp_data = self._pdp()
        ax.plot(x, pdp_data, lw=5, ls='--', c='k')
        plt.title(self.predictors[0])

        if output_folder is not None:
            if filename is None:
                filename = "ice_" + self.predictors[0] + ".png"
            plt.savefig(output_folder + "/" + filename, format="png", dpi=200,
                        bbox_inches="tight")

        plt.show()

        return self

    def get_description(self):
        return (
                "Association of predictor "
                + self.predictors[0]
                + " with score "
                + self.projectParameters.model[1]
                + " on sample "
                + self.samples[0][1]
        )


# expect one sample, one predictor, no target
# computes and plots partial dependence plots (including missing values) and number of observations in each interval
# expect a booster (scikit-learn), model name and a list of model features
# - XGB - pp.model = (booster_xgb, 'XGB', booster_xgb.get_booster().feature_names)
# - LGBM - pp.model = (booster_lgbm, 'LGBM', booster_lgbm.booster_.feature_name())
# optionally you can set output folder (pp.output_folder) where to save plots


class PartialDependencePlotCalculator(Calculator):
    """Doctools calculator.
    PDP (Partial Dependency Plots) are showing overall trend of the model output (prediction) related to one particular predictor. We calculate PDP's for each predictor's values.

    First, we group the predictor values into several bins (corresponding to the splits inside the decision trees).
    Then for each observation (more precisely, for a reasonably sized random subsample) calculate the model output in hypothetical situation
    when the predictor would change its value to be in the particular bin and all the other varibles' values would remain the same.
    Average of these values over all observations for each particular bin is mean Partial Dependency value.
    When these values are plotted with the bins on x-axis, the PDP plot is formed.
    This plot shows how the mean of the prediction changes when the variable changes (and all other variables remain the same).

    We don't calculate just mean Partial Dependency but also its quantiles and median.

    Args:
        projectParameters (ProjectParameters): definitions of doctools metadata
        use_weight (bool, optional): True for using the self.weight, False for calculating without weights. Defaults to False.

    Methods:
        calculate(): calculates table for the visualisation
        get_table(): returns the calculated table as pd.DataFrame
        get_visualisation(showPlot=False, outputFolder=None, fileName=None): plots the visualisation either to ipython or to file or both
    """

    def __init__(self, *args, use_weight=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_weight = use_weight

    def calculate(self):
        warnings.filterwarnings("ignore")

        # declaration of all used external variables
        # -----------------------------------------
        df = self.samples[0][0]
        predictor = self.predictors[0]
        model = self.projectParameters.model
        rowid_col = self.rowid_variable
        time_variable = self.time_variable
        if self.use_weight:
            weight = self.weight
        else:
            weight = None
        # ---------------------------------

        from pandas.api.types import is_numeric_dtype
        self.is_cat = not is_numeric_dtype(df[predictor])

        booster = model[0]
        feature_names = model[2]
        ale = []
        inf_finite = (-999999, 999999)  # values to be used as a proxy for -inf/+inf
        precision = 4  # number of decimals

        if not self.is_cat:
            # find splits in xgb model, get unique values rounded to 4 decimals
            if str(type(booster)).endswith("'xgboost.sklearn.XGBClassifier'>"):
                xgdump = booster.get_booster().get_dump()
                values = []
                regexp = re.compile("\[{0}<([\d.Ee+-]+)\]".format(predictor))
                for i in range(len(xgdump)):
                    m = re.findall(regexp, xgdump[i])
                    values.extend(map(float, m))
                values = np.round(values, precision)
                values = np.unique(values)

                grid = [inf_finite[0]] + list(values) + [inf_finite[1]] + ["missing"]

            # use quantile grid for other models (LightGBM, etc.)
            else:
                predictor_min = df[predictor].quantile(q=0.01)  # min
                predictor_max = df[predictor].quantile(q=0.99)  # max
                grid = (
                        [inf_finite[0]]
                        + list(np.round(np.linspace(predictor_min, predictor_max, num=20), precision))
                        + [inf_finite[1]]
                        + ["missing"]
                )

        else:
            grid = sorted(list(df[pd.notnull(df[predictor])][predictor].unique()))
            if df[predictor].isnull().sum() > 0:
                grid = grid + ["missing"]

        for i in range(len(grid)):

            # Category bin
            if grid[i] == "missing":
                category = "missing"
            elif self.is_cat:
                category = grid[i]
            else:
                left = grid[i]
                if i >= len(grid) - 2:
                    right = np.inf
                else:
                    right = grid[i + 1]
                category = pd.Interval(left, right, closed="left")

            # SAMPLE TO CALCULATE PDP ON
            if df.shape[0] > 2000:
                df2 = df.sample(2000, random_state=42)
            else:
                df2 = df.copy()
            if i == 0:
                ice_df = pd.DataFrame()
                if weight is not None:
                    ice_df_w = df2[weight].copy()

            # PDP, ICE
            if (self.is_cat) and (grid[i] == "missing"):
                df2.loc[::-1, predictor] = np.nan
            elif grid[i] == "missing":
                df2.loc[:, predictor] = np.nan
            elif self.is_cat:
                df2.loc[::-1, predictor] = grid[i]
            else:
                df2.loc[:, predictor] = grid[i]

            if hasattr(booster, 'predict_proba'):
                ice_s = pd.DataFrame(booster.predict_proba(df2[feature_names])[:, 1], index=df2.index,
                                     columns=[category]).transpose()
            else:
                ice_s = pd.DataFrame(booster.predict(df2[feature_names]), index=df2.index,
                                     columns=[category]).transpose()
            ice_df = ice_df.append(ice_s)

            # ALE
            if grid[i] == "missing":
                ale.append((category, np.nan))

            elif self.is_cat:
                ale.append((category, np.nan))

            elif i == 0 and i < len(grid) - 1:
                ale.append((category, 0))

            elif i > 0 and i < len(grid) - 1:
                df2 = df.loc[(df[predictor] >= grid[i - 1]) & (df[predictor] < grid[i]), :]

                if weight is not None:
                    df2 = df2.merge(df[weight], how='inner')

                if df2.shape[0] > 0:
                    df2.loc[:, predictor] = grid[i - 1]
                    if hasattr(booster, 'predict_proba'):
                        pred_lower = booster.predict_proba(df2[feature_names])[:, 1]
                    else:
                        pred_lower = booster.predict(df2[feature_names])
                    df2.loc[:, predictor] = grid[i]
                    if hasattr(booster, 'predict_proba'):
                        pred_upper = booster.predict_proba(df2[feature_names])[:, 1]
                    else:
                        pred_upper = booster.predict(df2[feature_names])

                    if weight is not None:
                        ale.append((category, ((pred_upper - pred_lower) * df2[weight]).sum() / df2[weight].sum()))
                    else:
                        ale.append((category, (pred_upper - pred_lower).mean()))

                else:
                    ale.append((category, np.nan))

            del df2

        ale = pd.DataFrame(ale, columns=["category", "LE"]).set_index("category")
        ale.loc[:, "ALE"] = ale["LE"].cumsum()
        if hasattr(booster, 'predict_proba'):
            ale.loc[:, "ALE"] = ale["ALE"] - ale["ALE"].mean() + booster.predict_proba(df[feature_names])[:, 1].mean()
        else:
            ale.loc[:, "ALE"] = ale["ALE"] - ale["ALE"].mean() + booster.predict(df[feature_names]).mean()

        if not self.is_cat:
            grid = sorted(list(set(grid[:-1])))
            cut = (
                pd.cut(df[predictor], grid + [np.inf], precision=precision, right=False)
                    .values.add_categories("missing")
                    .fillna(value="missing")
            )
            if weight is not None:
                obs = df.groupby(cut)[[weight]].sum()
                obs = obs.rename(columns={weight: "Observations"})
            else:
                obs = df.groupby(cut)[[rowid_col]].count()
                obs = obs.rename(columns={rowid_col: "Observations"})
        else:
            if weight is not None:
                obs = df[[predictor, weight]].astype('object').fillna("missing").groupby(predictor)[[weight]].sum()
                obs = obs.rename(columns={weight: "Observations"})
            else:
                obs = df[[predictor, rowid_col]].astype('object').fillna("missing").groupby(predictor)[
                    [rowid_col]].count()
                obs = obs.rename(columns={rowid_col: "Observations"})

        if weight is not None:
            ice_df = ice_df.transpose()
            table = pd.DataFrame()
            for category in ice_df.columns:
                mean_in_category = (ice_df[category] * ice_df_w).sum() / ice_df_w.sum()
                ice_df = ice_df.sort_values(category)
                ice_df_cum_w = ice_df_w[ice_df.index].cumsum() / ice_df_w.sum()
                median_in_category = ice_df.loc[ice_df_cum_w[ice_df_cum_w >= 0.5].index[0], category]
                q25_in_category = ice_df.loc[ice_df_cum_w[ice_df_cum_w >= 0.25].index[0], category]
                q75_in_category = ice_df.loc[ice_df_cum_w[ice_df_cum_w >= 0.75].index[0], category]
                new_row = pd.DataFrame({"PDP": [mean_in_category],
                                        "PDP - median": [median_in_category],
                                        "PDP - 0.25 quantile": [q25_in_category],
                                        "PDP - 0.75 quantile": [q75_in_category], },
                                       index=[category])
                table = table.append(new_row)
        else:
            table = pd.DataFrame(ice_df.mean(axis=1), columns=["PDP"])
            table.loc[:, "PDP - median"] = ice_df.median(axis=1)
            table.loc[:, "PDP - 0.25 quantile"] = ice_df.quantile(q=0.25, axis=1)
            table.loc[:, "PDP - 0.75 quantile"] = ice_df.quantile(q=0.75, axis=1)
        table = table.join(ale["ALE"], how="left")
        table = table.join(obs, how="left")
        self.table_orig = table

        # Group by over PDP to remove duplicates, category intervals must be changed accordingly
        agg_dict = {
            "PDP": np.mean,
            "PDP - median": np.mean,
            "PDP - 0.25 quantile": np.mean,
            "PDP - 0.75 quantile": np.mean,
            "ALE": np.mean,
            "Observations": np.sum,
            "left": np.min,
            "right": np.max,
        }

        if not self.is_cat:
            table_agg = table[:-1].copy()
            table_agg["left"] = [float(col.left) for col in table_agg.index]
            table_agg["right"] = [float(col.right) for col in table_agg.index]
            table_agg = table_agg.groupby("PDP").agg(agg_dict)
            table_agg["category"] = table_agg.apply(lambda x: pd.Interval(x["left"], x["right"], closed="left"), axis=1)
            table_agg = table_agg.set_index("category").sort_index()
            table_agg = table_agg.append(table.loc["missing", :])
            table = table_agg.drop(["left", "right"], axis=1)
        else:
            table_agg = table.copy()

        self.table = table
        self.ice_df = ice_df
        self.ale = ale
        self.obs = obs
        warnings.resetwarnings()
        return self

    def get_table(self):
        return self.table

    def get_visualization(self, output_folder=None, filename=None, show_plot=False):
        model = self.projectParameters.model

        fig, ax1 = plt.subplots(figsize=(10, 5))
        plt.xticks(range(len(self.table)), self.table.index, rotation=90)
        ax1.plot(range(len(self.table)), self.table["PDP"], color="k", label="PDP - mean")  # mean - PDP
        ax1.plot(range(len(self.table)), self.table["PDP - median"], linestyle="--", label="PDP - median")  # median
        ax1.plot(
            range(len(self.table)), self.table["PDP - 0.25 quantile"], linestyle="--", label="PDP - 0.25 quantile"
        )  # quantile 0.25
        ax1.plot(
            range(len(self.table)), self.table["PDP - 0.75 quantile"], linestyle="--", label="PDP - 0.75 quantile"
        )  # quantile 0.75
        # ax1.plot(range(len(self.table)), self.ice_df.iloc[:,0:20], alpha=0.5, linewidth=1) # ICE
        ax1.plot(range(len(self.table)), self.table["ALE"], color="y", marker="o", label="ALE")  # ALE
        ax1.set_ylabel("Prediction")
        ax1.legend(loc="center left", bbox_to_anchor=(1.15, 0.5))

        ax2 = ax1.twinx()
        ax2.grid(False)
        ax2.bar(range(len(self.table)), self.table["Observations"], width=0.5, alpha=0.5)
        ax2.set_ylabel("Observations")

        plt.title(self.get_description())

        if output_folder is not None:
            if filename is None:
                filename = "pdp_" + self.predictors[0] + ".png"
            plt.savefig(output_folder + "/" + filename, format="png", dpi=200,
                        bbox_inches="tight")

        if show_plot:
            plt.show()
        else:
            plt.close()
        return self

    def get_description(self):
        return (
                "Association of predictor "
                + self.predictors[0]
                + " with score "
                + self.projectParameters.model[1]
                + " on sample "
                + self.samples[0][1]
        )


class ScoreGiniInTimeCalculator(Calculator):
    def __init__(self, *args, dropna=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropna = dropna

    def calculate(self):

        # declaration of all used external variables
        # -----------------------------------------
        df = self.samples[0][0]
        targets = self.targets
        scores = self.scores
        rowid_col = self.rowid_variable
        time_variable = self.time_variable
        masks = self.masks
        dropna = self.dropna

        # ---------------------------------

        def gini(y_true, y_pred, dropna=False):
            if dropna:
                df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
                y_true = df["y_true"]
                y_pred = df["y_pred"]

            try:
                return 2 * roc_auc_score(y_true, y_pred) - 1
            except ValueError:
                return np.nan

        results = pd.DataFrame()
        if not masks:
            masks = {"All": df.index.notnull()}

        for mask in masks:
            data_masked = df.loc[masks[mask], [t[0] for t in targets] + scores + [time_variable]]

            for target in targets:
                for score in scores:
                    grouped = data_masked.groupby(time_variable, axis=0).apply(
                        lambda x: gini(x[target[0]], x[score], dropna=dropna)
                    )
                    grouped = grouped.append(
                        pd.Series(
                            gini(data_masked[target[0]], data_masked[score], dropna=dropna),
                            index=[pd.Timestamp("2099-01-01")],
                        )
                    )

                    results = results.join(pd.DataFrame(grouped, columns=[[mask], [target[0]], [score]]), how="outer")

        # return final results
        results.columns = pd.MultiIndex.from_tuples(results.columns)
        results = results.sort_index()
        self.table = results

        return self

    def get_table(self):
        return self.table

    def get_visualization(self):
        plt.figure(figsize=(8.0, 5.5))
        for i in self.table.columns:
            plt.plot(self.table[:-1][i], label=i, marker="o")

        plt.xticks(rotation=90)
        plt.ylim([0, 1])
        plt.title("Gini in Time")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.ylabel("Gini")
        plt.show()
        return self

    def get_description(self):
        if len(self.scores) == 1 and len(self.targets) == 1:
            return (
                    "Gini of score "
                    + self.scores[0]
                    + " on target "
                    + self.targets[0][0]
                    + " on sample "
                    + self.samples[0][1]
            )
        elif len(self.scores) > 1 and len(self.targets) > 1:
            return "Gini of multiple scores " + " on multiple targets " + "on sample " + self.samples[0][1]
        elif len(self.scores) > 1:
            return "Gini of multiple scores on target " + self.targets[0][0] + " on sample " + self.samples[0][1]
        elif len(self.targets) > 1:
            return "Gini of score " + self.scores[0] + " on multiple targets " + " on sample " + self.samples[0][1]


class ScoreLiftInTimeCalculator(Calculator):
    def __init__(self, *args, dropna=True, lift_perc=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropna = dropna
        self.lift_perc = lift_perc

    def calculate(self):

        # declaration of all used external variables
        # -----------------------------------------
        df = self.samples[0][0]
        targets = self.targets
        scores = self.scores
        rowid_col = self.rowid_variable
        time_variable = self.time_variable
        masks = self.masks
        dropna = self.dropna
        lift_perc = self.lift_perc

        # ---------------------------------

        def lift(y_true, y_pred, lift_perc, dropna=False):
            if dropna:
                df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
                y_true = df["y_true"]
                y_pred = df["y_pred"]

            try:
                cutoff = np.percentile(y_pred, lift_perc)
                return y_true[y_pred <= cutoff].mean() / y_true.mean()
            except:
                return np.nan

        results = pd.DataFrame()
        if not masks:
            masks = {"All": df.index.notnull()}

        for mask in masks:
            data_masked = df.loc[masks[mask], [t[0] for t in targets] + scores + [time_variable]]

            for target in targets:
                for score in scores:
                    grouped = data_masked.groupby(time_variable, axis=0).apply(
                        lambda x: lift(x[target[0]], -x[score], lift_perc, dropna=dropna)
                    )
                    grouped = grouped.append(
                        pd.Series(
                            lift(data_masked[target[0]], -data_masked[score], lift_perc, dropna=dropna),
                            index=[pd.Timestamp("2099-01-01")],
                        )
                    )

                    results = results.join(pd.DataFrame(grouped, columns=[[mask], [target[0]], [score]]), how="outer")

        # return final reults
        results.columns = pd.MultiIndex.from_tuples(results.columns)
        results = results.sort_index()
        self.table = results

        return self

    def get_table(self):
        return self.table

    def get_visualization(self):
        plt.figure(figsize=(8.0, 5.5))
        for i in self.table.columns:
            plt.plot(self.table[:-1][i], label=i, marker="o")

        plt.ylim([1, 5])
        plt.xticks(rotation=90)
        plt.title(str(self.lift_perc) + "% Cumulative Lift in Time")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.ylabel(str(self.lift_perc) + "% cumulative lift")
        plt.show()
        return self

    def get_description(self):
        if len(self.scores) == 1 and len(self.targets) == 1:
            return (
                    "Lift of score "
                    + self.scores[0]
                    + " on target "
                    + self.targets[0][0]
                    + " on sample "
                    + self.samples[0][1]
            )
        elif len(self.scores) > 1 and len(self.targets) > 1:
            return "Lift of multiple scores " + " on multiple targets " + "on sample " + self.samples[0][1]
        elif len(self.scores) > 1:
            return "Lift of multiple scores on target " + self.targets[0][0] + " on sample " + self.samples[0][1]
        elif len(self.targets) > 1:
            return "Lift of score " + self.scores[0] + " on multiple targets " + " on sample " + self.samples[0][1]


class PlotDatasetCalculator(Calculator):
    """Shows statistics of the dataset - counts of contracts by time interval and badrate. Can be shown for different
        segments (e.g. men/women, education, etc.)

    Args:
        projectParameters (ProjectParameters): definitions of doctools metadata
        segment_col (str, optional): Segment-by column. If given, the whole plot is drawn with distinct curves for each
            of the column's values.
        zero_ylim (bool): True means, the y-axis starts with 0.
        use_weight (bool): If True, the weight defined in ProjectParameters is used for calculating weighted version.
            If False, no weight is used.
    """
    def __init__(self, *args, segment_col=None, zero_ylim=True, use_weight=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.segment_col = segment_col
        self.zero_ylim = zero_ylim
        self.use_weight = use_weight

    def calculate(self):
        """ Calculates the counts and badrates by time variable. """
        # declaration of all used external variables
        # -----------------------------------------
        df = self.samples[0][0]
        target = self.targets[0][0]
        # rowid_col = self.rowid_variable
        time_variable = self.time_variable
        if self.use_weight:
            weight = self.weight
        else:
            weight = None
        # ---------------------------------

        if self.segment_col:
            gr = df.groupby([time_variable, self.segment_col], axis=0)
        else:
            gr = df.groupby(time_variable, axis=0)
        res = gr.apply(
            lambda x: pd.Series(data=(len(x), 1. * len(x[x[target] == 1]) / len(x)), index=['count', 'bad rate']))

        if self.segment_col:
            res.reset_index(level=self.segment_col, inplace=True)
        res.index = res.index.astype(str)
        self.table = res
        self.table['type'] = 'basic'

        if weight:
            res_w = gr.apply(lambda x: pd.Series(data=(x[weight].sum(), x[x[target] == 1][weight].sum()),
                                                 index=['count', 'bad rate']))

            res_w['bad rate'] = res_w['bad rate'] / res_w['count']
            res_w.loc[res_w['count'] == 0, 'bad rate'] = 0

            if self.segment_col:
                res_w.reset_index(level=self.segment_col, inplace=True)
            res_w.index = res_w.index.astype(str)
            res_w['type'] = 'weighted'
            self.table = self.table.append(res_w)
        return self

    def get_table(self):
        """Returns the output table as a pandas DataFrame."""
        return self.table

    def get_visualization(self, output_folder=None, filename=None, show_plot=False):
        """ Plots the counts and badrates by months (or another defined time interval).

        Args:
            output_folder (str, optional): if given, the plot is saved to the folder, if None, plot is not saved
            filename (str, optional): if not given, default 'data_badrate.png' is used
            show_plot (bool, optional): Default True. If True, the plot is shown in iPython.
        """

        def _plot_badrates(df, plot_ax, time_variable, segment_col=None, _segment_names=None, zero_ylim=True,
                           weighted=False,
                           title=''):
            """
            Function to set the plot_dataset() plots - as the code is repeating
            Args:
                df (pd.DataFrame): dataset input from plot_dataset
                plot_ax (matplotlib.axes._subplots.AxesSubplot): subplot object
                time_variable (str): name of month/other time interval column
                segment_col (str, optional): name of the segment by which we want to group the plot
                _segment_names (list, optional): a list of the unique segment names from segment_col
                zero_ylim (bool, optional): the Y axis will or will not contain zero for bad rate
                weighted (bool, optional): True if the plot is with weight_col, False if not
                title (str, optional): the title - different for weighted and not weighted

            Returns: plot

            """
            x_axis = df.index.get_level_values(time_variable).unique()

            ax2 = plot_ax.twinx()
            if segment_col:
                length = x_axis.shape[0]  # for the 'bottom' values for stacked bar chart
                the_bottom = np.zeros(length)
                for name in _segment_names:
                    df_1 = df[df[segment_col] == name]['count']
                    df_1 = df_1.reindex(x_axis).fillna(0)  # getting 0 for the missing values
                    df_2 = df[df[segment_col] == name]['bad rate']
                    df_2 = df_2.reindex(x_axis)
                    plot_ax.bar(x_axis, df_1, label='count ' + str(name), bottom=the_bottom, alpha=0.8)
                    df_2.plot(ax=ax2, marker='o', label='bad rate ' + str(name))
                    the_bottom += df_1
                plt.legend(loc='best')
            else:
                df_1 = df['count']
                df_1 = df_1.reindex(x_axis).fillna(0)
                df_2 = df['bad rate']
                df_2 = df_2.reindex(x_axis)
                plot_ax.bar(x_axis, df_1)
                df_2.plot(ax=ax2, marker='o', label='count', color='r')

            if zero_ylim:
                max_badrate = max(df['bad rate'])
                ax2.set_ylim(0, 1.05 * max_badrate)

            for tl in ax2.get_yticklabels():
                tl.set_color('r')

            plot_ax.set_xlabel('time')
            plot_ax.set_xticks(range(len(x_axis)))
            plot_ax.set_xticklabels(x_axis, rotation=90)
            plt.xlim(-0.5, len(x_axis) - 0.5)

            if weighted:
                plot_ax.set_ylabel('weighted bad rate', color='b')
                ax2.set_ylabel('weighted count', color='r')
            else:
                plot_ax.set_ylabel('bad rate', color='b')
                ax2.set_ylabel('count', color='r')

            for tl in plot_ax.get_yticklabels():
                tl.set_color('b')
            plot_ax.set_title(title)
            return plot_ax

        if self.segment_col:
            segment_names = self.samples[0][0][self.segment_col].unique()
        else:
            segment_names = []
        if self.use_weight:
            fig, (ax, bx) = plt.subplots(nrows=2, ncols=1)  # two plots in case of weighted
            fig.set_figheight(9)
            _plot_badrates(self.table[self.table['type'] == 'weighted'], bx, self.time_variable, self.segment_col,
                           segment_names, self.zero_ylim, weighted=True, title='Weighted')

        else:
            fig, ax = plt.subplots()  # one plot in case of not weighted

        for tl in ax.get_yticklabels():
            tl.set_color('b')
        _plot_badrates(self.table[self.table['type'] == 'basic'], ax, self.time_variable, self.segment_col,
                       segment_names, self.zero_ylim, weighted=False, title='Count & Bad Rate')

        if output_folder is not None:
            if not filename:
                filename = "data_badrate.PNG"
            plt.savefig(path.join(output_folder, filename), bbox_inches='tight', dpi=200)
        if show_plot:
            plt.show()
        plt.close()
        return self

    def get_description(self):
        """Returns a simple class description."""
        return ('Plot of Badrates on sample ' + self.samples[0][1])


class PredictorGiniIVCalculator(Calculator):
    """
    Creates the Predictor Power Analysis (predictor gini and information value) table.

    Args:
        projectParameters (ProjectParameters): definitions of doctools metadata
        use_weight (bool): If True, the weight defined in ProjectParameters is used for calculating weighted version.
            If False, no weight is used.
    """

    def __init__(self, *args, use_weight, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_weight = use_weight

    def calculate(self):
        """
        Calculates Gini and IV.

        Methods:
            iv (y_true, y_pred):  Returns Information Value of a binned predictor
            gini (y_true, y_pred, sample_weight): Returns Gini coefficient (linear transformation of Area Under Curve)
        """
        # declaration of all used external variables
        # -----------------------------------------
        df = self.samples[0][0]
        targets = self.targets
        mask = self.samples[0][1]
        predictors = self.predictors
        rowid_col = self.rowid_variable
        if self.use_weight:
            weight = df[self.weight]
        else:
            weight = None

        # ---------------------------------

        def iv(y_true, y_pred):
            """
            Returns Information Value of a binned predictor

            Args:
                y_true: true values of target
                x_pred: binned predictor

            Returns:
                float: Information Value

            """
            woe = {}
            lin = {}
            iv = 0
            for v in np.unique(y_pred):
                woe[v] = (1. * (len(y_pred[(y_pred == v) & (y_true == 0)]) + 1) / (len(y_pred[y_true == 0]) + 1)) / (
                        1. * (len(y_pred[(y_pred == v) & (y_true == 1)]) + 1) / (len(y_pred[y_true == 1]) + 1))
                woe[v] = math.log(woe[v])
                lin[v] = (1. * (len(y_pred[(y_pred == v) & (y_true == 0)]) + 1) / (len(y_pred[y_true == 0]) + 1)) - (
                        1. * (len(y_pred[(y_pred == v) & (y_true == 1)]) + 1) / (len(y_pred[y_true == 1]) + 1))
                iv = iv + woe[v] * lin[v]
            return iv

        def gini(y_true, y_pred, sample_weight=None):
            """
            Returns Gini coefficient (linear transformation of Area Under Curve)

            Args:
                y_true: true values of target
                y_pred: predicted score
                sample_weight: weight column

            Returns:
                float: value of gini
            """

            return 2 * roc_auc_score(y_true, y_pred, sample_weight=sample_weight) - 1

        ginis = []
        for predictor in predictors:
            gini_predictor = gini(df[targets[0][0]], -1.0 * df[predictor], sample_weight=weight)
            iv_predictor = iv(df[targets[0][0]], df[predictor])
            ginis.append({'predictor_name': predictor, 'gini': gini_predictor, 'metrics': 'Gini ' + mask})
            ginis.append({'predictor_name': predictor, 'iv': iv_predictor, 'metrics': 'IV ' + mask})

        self.table = ginis
        return self

    def get_table(self):
        """Returns the output table as a list of dictionaries."""
        return self.table

    def get_visualization(self, output_folder=None, filename=None):
        """ not implemented for this class """
        warnings.warn("Visualisation not implemented.")

    def get_description(self):
        """Returns the class description."""
        return f"Predictor Power Analysis: Gini and Information Value for each predictor in {self.predictors} " \
               f"on sample {self.samples[0][1]}."


class ScoreModelPerformanceCalculator(Calculator):
    """
    Calculates Gini, lift and Kolmogorov-Smirnov test.

    Args:
        projectParameters (ProjectParameters): definitions of doctools metadata
        use_weight (bool): If True, the weight defined in ProjectParameters is used for calculating weighted version.
            If False, no weight is used.
        lift_perc (float): Percentile of lift.
        data (pandas DataFrame): By adding own dataset, you can replace the original one. Used internally for e.g.
            :py:meth:`~scoring.doctools.calculators.ScoreGiniIVCalculator`. Default None.
        mask_name (str): If stated, the mask_name acts like sample in other classes. Used internally for e.g.
            :py:meth:`~scoring.doctools.calculators.ScoreGiniIVCalculator`. Default None.
    """
    def __init__(self, *args, use_weight, lift_perc, data=None, mask_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_weight = use_weight
        self.lift_perc = lift_perc
        self.mask_name = mask_name
        if data is None:
            self.data = None
        else:
            self.data = data

    def calculate(self):
        """
        Calculates gini, lift and KS.

        Methods:
            gini (y_true, y_pred, sample_weight): computes gini out of true and predicted values.
            kolmogorov_smirnov (y_true, y_pred): computes kolmogorov-smirnov out of true and predicted values.
            lift (y_true, y_pred, lift_perc): computes lift out of true and predicted values, for given percentile.
        """
        # declaration of all used external variables
        # -----------------------------------------
        if self.data is None:
            df = self.samples[0][0]
            mask = self.samples[0][1]
        else:
            df = self.data
            mask = self.mask_name
        targets = self.targets
        predictors = self.predictors
        rowid_col = self.rowid_variable
        scores = self.scores
        if self.use_weight:
            weight = df[self.weight]
        else:
            weight = None

        # ---------------------------------

        def gini(y_true, y_pred, sample_weight=None):
            """
            Returns Gini coefficient (linear transformation of Area Under Curve)
            Args:
                y_true: true values of target
                y_pred: predicted score
                sample_weight: weight column

            Returns:
                float: value of gini
            """

            return 2 * roc_auc_score(y_true, y_pred, sample_weight=sample_weight) - 1

        def kolmogorov_smirnov(y_true, y_pred):
            """
            Returns a results of Kolmogorov-smirnov test on goodness of fit using scipy.ks_2sample test.

            Args:
                y_true: true values of target
                y_pred: predicted score

            Returns:
                float: value of test

            """
            return ks_2samp(y_pred[y_true == 1], y_pred[y_true == 0]).statistic

        def lift(y_true, y_pred, lift_perc=10):
            """
            Returns Lift of prediction

            Args:
                y_pred: predicted score
                y_true: true values of target
                lift_perc: lift level (i.e. for 10%-Lift, set lift_perc = 10)

            Returns:
                float: value of lift

            """
            cutoff = np.percentile(y_pred, lift_perc)
            return y_true[y_pred <= cutoff].mean() / y_true.mean()

        results = []
        if isinstance(scores, str):
            scores = [scores]
        for score in scores:
            for tgt in targets:
                if df[tgt[0]].nunique() > 1:
                    results.append({
                        'gini': gini(df[tgt[0]], df[score], sample_weight=weight),
                        f'lift_{self.lift_perc}': lift(df[tgt[0]], -df[score], self.lift_perc),
                        'KS': kolmogorov_smirnov(df[tgt[0]], df[score]),
                        'score': score,
                        'sample': mask,
                        'target': tgt[0]
                    })
                else:
                    results.append(
                        {'gini': np.nan, f'lift_{self.lift_perc}': np.nan, 'KS': np.nan, 'score': score, 'sample': mask,
                         'target': tgt[0]})
        self.table = results
        return self

    def get_table(self):
        """Returns a result as list of dictionaries."""
        return self.table

    def get_visualization(self, output_folder=None, filename=None):
        """Visualisation is not defined here, only the table output."""
        return self

    def get_description(self):
        """Returns a basic class description."""
        return f"Performance: gini, KS, lift_{self.lift_perc} of scores {self.scores} on target {self.targets[0][0]}, on sample {self.samples[0][1]}."


class ScoreROCCalculator(Calculator):
    """
        Doctools calculator for ROC curve for given dataset.

        Args:
            projectParameters (ProjectParameters): definitions of doctools metadata
            use_weight (bool): if True, calculator will use the weight defined in ProjectParameters
            df (pandas DataFrame): default None, if defined, rewrites the general dataset (used for multiple sample
                masks ROC curve)
    """

    def __init__(self, *args, use_weight, df=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_weight = use_weight
        self.df = df

    def calculate(self):
        """ Calculates gini and goods/bads cumulatives """
        # declaration of all used external variables
        # -----------------------------------------
        df = self.df.copy()
        targets = self.targets
        scores = self.scores
        if self.use_weight:
            weight = self.weight
        else:
            weight = 'wt'
            df[weight] = 1

        # ---------------------------------
        output = {}
        if isinstance(scores, str):
            scores = [scores]
        for score in scores:
            output[score] = {}
            data = df[[targets[0][0], targets[0][1], score, weight]].copy()
            data['wbt'] = data[weight] * data[targets[0][0]] * data[targets[0][1]]
            data['wb'] = data[weight] * data[targets[0][1]]

            d1 = data[data[targets[0][1]] > 0].groupby(score)['wb'].sum()
            d2 = data[data[targets[0][1]] > 0].groupby(score)['wbt'].sum()
            data = pd.concat([d1, d2], axis=1)
            data.sort_index(inplace=True, ascending=False)

            # weight + base + (1 - target)
            data["wbtn"] = data["wb"] - data["wbt"]

            # cumulative characteristics
            data["cum_cnt"] = data["wb"].cumsum()
            # data["cum_perc"] = 100 * data["cum_cnt"] / data["wb"].sum()
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

            outdata = data[["cum_good", "cum_bad"]]
            output[score]["gini"] = gini
            output[score]["points"] = outdata
        self.table = output

        return self

    def get_table(self):
        """
        Returns:
            dictionary for each score with gini and cumulative goods and bads
        """
        return self.table

    def get_visualization(self, output_folder=None, filename=None):
        """ plots the visualisation either to iPython or to file or both
        Args:
            output_folder (str, optional): if given, the plot is saved to the folder, if None, plot is not saved
            filename (str, optional): if not given, default 'roc.png' is used
        """
        plt.figure(figsize=(7, 7))
        plt.axis([0, 1, 0, 1])
        for curve_name, curve in self.table.items():
            plt.plot(
                [0] + list(curve["points"]["cum_good"]),
                [0] + list(curve["points"]["cum_bad"]),
                label=curve_name,
            )
        plt.plot(list(range(0, 101)), list(range(0, 101)), color="k")
        plt.xlabel("Cumulative good count")
        plt.ylabel("Cumulative bad count")
        plt.legend(loc="lower right")

        if output_folder is not None:
            if not filename:
                filename = "roc.png"
            plt.savefig(path.join(output_folder, filename), bbox_inches='tight', dpi=200)
        plt.show()
        plt.close()

    def get_description(self):
        """ writes the short description of the class """
        if len(self.scores) == 1 and len(self.targets) == 1:
            return (
                    "ROC of score "
                    + self.scores[0]
                    + " on target "
                    + self.targets[0][0]
                    + " on sample "
                    + self.samples[0][1]
            )
        elif len(self.scores) > 1 and len(self.targets) > 1:
            return "ROC of multiple scores " + " on multiple targets " + "on sample " + self.samples[0][1]
        elif len(self.scores) > 1:
            return "ROC of multiple scores on target " + self.targets[0][0] + " on sample " + self.samples[0][1]
        elif len(self.targets) > 1:
            return "ROC of score " + self.scores[0] + " on multiple targets " + " on sample " + self.samples[0][1]


class ScoreROCBySegmentsCalculator(Calculator):
    """ Doctool calculator for ROC for given dataset. Plots ROC curve by segments defined in masks.

    Args:
        projectParameters (ProjectParameters): definitions of doctools metadata
        use_weight (bool): if True, the doctools.ProjectParameters() weight will be used.
        masks (list of strings, optional): If filled, the distinct ROC curve is drawn for each masked sample
    """

    def __init__(self, *args, use_weight, masks=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_weight = use_weight
        self.masks = masks

    def calculate(self):
        """Calculates table for the visualisation.
        Can be used with or without sample masks.
        """
        # declaration of all used external variables
        # -----------------------------------------
        masks = self.masks
        results = []
        if isinstance(self.scores, str):
            self.scores = [self.scores]

        if not masks:
            df = self.samples[0][0]

            self.ROC = ScoreROCCalculator(self.projectParameters, use_weight=self.use_weight, df=df)

            self.ROC.samples = self.samples
            self.ROC.targets = self.targets
            self.ROC.weight = self.weight
            self.ROC.scores = self.scores

            results.append({"name": self.samples[0][1], "values": self.ROC.calculate().get_table()})
        if masks:
            for i, name in enumerate(masks):
                df = (self.samples[0][i][0])
                if len(df) > 0:
                    self.ROC = ScoreROCCalculator(self.projectParameters, use_weight=self.use_weight, df=df)

                    self.ROC.samples = self.samples
                    self.ROC.targets = self.targets
                    self.ROC.weight = self.weight
                    self.ROC.scores = self.scores

                    results.append({"name": name, "values": self.ROC.calculate().get_table()})
                else:
                    pass
        self.table = results
        return self

    def get_table(self):
        """returns list of dictionaries with name of sample/mask and ROC values."""
        return self.table

    def get_visualization(self, output_folder=None, filename=None, show_plot=True):
        """
        Plots the visualisation either to iPython or to file or both

        Args:
            output_folder (str, optional): if given, the plot is saved to the folder, if None, plot is not saved
            filename (str, optional): if not given, default 'roc.png' is used
            show_plot (bool, optional): If True, plots to iPython.
        """
        fig = plt.figure(figsize=(7, 7))
        plt.axis([0, 1, 0, 1])
        for curve in self.table:
            for score in self.scores:
                plt.plot(
                    [0] + list(curve["values"][score]["points"]["cum_good"]),
                    [0] + list(curve["values"][score]["points"]["cum_bad"]),
                    label=f'{score} {curve["name"]}',
                )
        plt.plot(list(range(0, 101)), list(range(0, 101)), color="k")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Cumulative good count")
        plt.ylabel("Cumulative bad count")
        plt.legend(loc="lower right")

        if output_folder is not None:
            if not filename:
                filename = "roc.PNG"
            plt.savefig(path.join(output_folder, filename), bbox_inches='tight', dpi=200)
        if show_plot:
            plt.show()
        plt.close(fig)

    def get_description(self):
        """ writes the short description of the class """
        return


class ScoreLiftCalculator(Calculator):
    """ Doctools Calculator for Lift curve for given dataset and segment.

        Args:
            projectParameters (ProjectParameters): definitions of doctools metadata
             use_weight (bool): if True, calculator will use the weight defined in ProjectParameters
            df (pandas DataFrame): default None, if defined, rewrites the general dataset (used for multiple sample
                masks ROC curve)
            lift_perc (float): Lift percentile. For the curve visualisation, this is useless but we left it here from
                historical reasons. Default 10.
            list_lift (list of floats): List of percentages for which we want to compute and show the lift. List is here
                for 'smoothening' the plot, especially the initial first few percentiles.
    """
    def __init__(self, *args, use_weight, lift_perc=10, list_lift=[5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], df=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.lift_perc = lift_perc
        self.df = df
        self.use_weight = use_weight
        self.list_lift = list_lift

    def calculate(self):
        """Calculates lift for each percentile given in list_lift."""
        # declaration of all used external variables
        # -----------------------------------------
        df = self.df.copy()
        targets = self.targets
        scores = self.scores
        if isinstance(scores, str):
            scores = [scores]
        if self.use_weight:
            weight = self.weight
        else:
            weight = 'wt'
            df[weight] = 1

        # ---------------------------------
        output = {}
        if max(self.list_lift) < 100:
            self.list_lift += [100]
        list_percentiles = [100 - self.list_lift[i] for i in
                            range(len(self.list_lift))]  # as the score is the other way round

        for score in scores:

            output[score] = {}
            data = df[[targets[0][0], targets[0][1], score, weight]].copy()
            data['wbt'] = data[weight] * data[targets[0][0]] * data[targets[0][1]]
            data['wb'] = data[weight] * data[targets[0][1]]

            perc_table = list(zip(self.list_lift, np.percentile(data[score], list_percentiles)))

            _data = pd.DataFrame()

            _tgr = data[(data[score] >= perc_table[0][1])][[score, 'wbt', 'wb']]
            _tgr_mean = {"score": _tgr[score].mean(),
                         "wbt": _tgr["wbt"].sum(),
                         "wb": _tgr["wb"].sum(),
                         "LIFT_PERC": perc_table[0][0]}
            _data = _data.append(_tgr_mean, ignore_index=True)

            for high, low in zip(perc_table[:-1], perc_table[1:]):
                _tgr = data[(data[score] >= low[1]) & (data[score] < high[1])][[score, 'wbt', 'wb']]
                _tgr_mean = {"score": _tgr[score].mean(),
                             "wbt": _tgr["wbt"].sum(),
                             "wb": _tgr["wb"].sum(),
                             "LIFT_PERC": low[0]}
                _data = _data.append(_tgr_mean, ignore_index=True)

            # original definition, left for reference
            # d1 = data[data[targets[0][1]] > 0].groupby(score)['wb'].sum()
            # d2 = data[data[targets[0][1]] > 0].groupby(score)['wbt'].sum()
            # data = pd.concat([d1, d2], axis=1)
            # data.sort_index(inplace=True, ascending=False)

            data = _data
            # cumulative characteristics
            data["cum_cnt"] = data["wb"].cumsum()
            data["cum_perc"] = 100 * data["cum_cnt"] / data["wb"].sum()

            # calculate lift
            data["cum_bad_cnt"] = data["wbt"].cumsum()
            data["cum_lift"] = (data["cum_bad_cnt"] / data["cum_cnt"]) / (
                    data["wbt"].sum() / data["wb"].sum()
            )

            if isinstance(self.lift_perc, list):
                lift = []
                for p in self.lift_perc:
                    lift_tmp = data[data["cum_perc"] >= p]
                    lift_tmp = lift_tmp[lift_tmp["cum_perc"] == lift_tmp["cum_perc"].min()]
                    lift.append(lift_tmp["cum_lift"].min())
            else:
                lift_tmp = data[data["cum_perc"] >= self.lift_perc]
                lift_tmp = lift_tmp[lift_tmp["cum_perc"] == lift_tmp["cum_perc"].min()]
                lift = lift_tmp["cum_lift"].min()

            outdata = data[["cum_perc", "cum_lift"]]
            output[score]["lift"] = lift
            output[score]["points"] = outdata
        self.table = output

        return self

    def get_table(self):
        """ Returns the output as a dictionary. """
        return self.table

    def get_visualization(self, output_folder=None, filename=None, show_plot=False):
        """ Creates lift plots.

        Args:
            output_folder (str, optional): if given, the plot is saved to the folder, if None, plot is not saved
            filename (str, optional): if not given, default 'lift.png' is used
            show_plot (bool, optional): if True, plot is shown in iPython.
        """
        fig = plt.figure(figsize=(10, 5))
        max_lift = []
        for curve in self.table:
            for score in self.scores:
                max_lift += [max(curve["values"][score]["points"]["cum_lift"])]
                plt.plot(
                    [0] + list(curve["values"][score]["points"]["cum_perc"]),
                    [0] + list(curve["values"][score]["points"]["cum_lift"]),
                    label=f'{score} {curve["name"]}',
                )
        plt.xlabel("Cumulative count [%]")
        plt.axis([min(curve["values"][score]["points"]["cum_perc"]), 100, 0, max(max_lift) + 0.5])
        plt.xticks(ticks=self.list_lift, labels=self.list_lift)
        plt.ylabel("Lift")
        plt.legend(loc="upper right")

        if output_folder is not None:
            if not filename:
                filename = "lift.PNG"
            plt.savefig(path.join(output_folder, filename), bbox_inches='tight', dpi=200)
        if show_plot:
            plt.show()
        plt.close(fig)

    def get_description(self):
        """Returns a simple class description. """
        if len(self.scores) == 1 and len(self.targets) == 1:
            return (
                    "ROC of score "
                    + self.scores[0]
                    + " on target "
                    + self.targets[0][0]
                    + " on sample "
                    + self.samples[0][1]
            )
        elif len(self.scores) > 1 and len(self.targets) > 1:
            return "ROC of multiple scores " + " on multiple targets " + "on sample " + self.samples[0][1]
        elif len(self.scores) > 1:
            return "ROC of multiple scores on target " + self.targets[0][0] + " on sample " + self.samples[0][1]
        elif len(self.targets) > 1:
            return "ROC of score " + self.scores[0] + " on multiple targets " + " on sample " + self.samples[0][1]


class ScoreLiftBySegmentsCalculator(Calculator):
    """ Calculates lift curve by segments. Calls :py:meth:`~scoring.doctools.calculators.ScoreLiftCalculator`

        Args:
            projectParameters (ProjectParameters): definitions of doctools metadata
            use_weight (bool): if True, calculator will use the weight defined in ProjectParameters
            lift_perc (float): Lift percentile. For the curve visualisation, this is useless but we left it here from
                historical reasons. Default 10.
            list_lift (list of floats): List of percentages for which we want to compute and show the lift. List is here
                for 'smoothening' the plot, especially the initial first few percentiles. Default value
                [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    """
    def __init__(self, *args, use_weight, lift_perc=10, list_lift=[5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                 masks=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lift_perc = lift_perc
        self.use_weight = use_weight
        self.masks = masks
        self.list_lift = list_lift

    def calculate(self):
        """Defines df - a dataset for each mask given, then calls :py:meth:`~scoring.doctools.calculators.ScoreLiftCalculator`
            to calculate lift curve for each dataset. Finally appends all results in one output for visualisation. """
        # declaration of all used external variables
        # -----------------------------------------
        masks = self.masks
        if isinstance(self.scores, str):
            self.scores = [self.scores]
        results = []

        if not masks:
            df = self.samples[0][0]

            self.lift = ScoreLiftCalculator(self.projectParameters, use_weight=self.use_weight,
                                            lift_perc=self.lift_perc, df=df, list_lift=self.list_lift)

            self.lift.samples = self.samples
            self.lift.targets = self.targets
            self.lift.weight = self.weight
            self.lift.scores = self.scores
            self.lift.lift_perc = self.lift_perc
            self.lift.list_lift = self.list_lift

            results.append({"name": self.samples[0][1], "values": self.lift.calculate().get_table()})
        if masks:
            for i, name in enumerate(masks):
                df = (self.samples[0][i][0])
                if len(df) > 0:
                    self.lift = ScoreLiftCalculator(self.projectParameters, use_weight=self.use_weight,
                                                    lift_perc=self.lift_perc, df=df)

                    self.lift.samples = self.samples
                    self.lift.targets = self.targets
                    self.lift.weight = self.weight
                    self.lift.scores = self.scores

                    results.append({"name": name, "values": self.lift.calculate().get_table()})
                else:
                    pass
        self.table = results
        return self

    def get_table(self):
        """Returns the output as a list of dictionaries. """
        return self.table

    def get_visualization(self, output_folder=None, filename=None, show_plot=False):
        """Creates the lift plot.

        Args:
            output_folder (str, optional): if given, the plot is saved to the folder, if None, plot is not saved
            filename (str, optional): if not given, default 'lift.png' is used
            show_plot (bool, optional): if True, plot is shown in iPython.
        """

        fig = plt.figure(figsize=(10, 5))
        max_lift = []
        for curve in self.table:
            for score in self.scores:
                max_lift += [max(curve["values"][score]["points"]["cum_lift"])]
                plt.plot(
                    [0] + list(curve["values"][score]["points"]["cum_perc"]),
                    [0] + list(curve["values"][score]["points"]["cum_lift"]),
                    label=f'{score} {curve["name"]}',
                )
        plt.xlabel("Cumulative count [%]")
        plt.xticks(ticks=self.list_lift, labels=self.list_lift)
        plt.axis([min(curve["values"][score]["points"]["cum_perc"]), 100, 0, max(max_lift) + 0.5])
        plt.ylabel("Lift")
        plt.legend(loc="upper right")

        if output_folder is not None:
            if not filename:
                filename = "lift.PNG"
            plt.savefig(path.join(output_folder, filename), bbox_inches='tight', dpi=200)
        if show_plot:
            plt.show()
        plt.close(fig)

    def get_description(self):
        """returns simple class description."""
        return


class BootstrapGiniCalculator(Calculator):
    """
        Creates a bootstrapped gini table, with mean, standard deviation and confidence intervals.
        The dataset is randomly sampled n_iter-times with replacements, by a random_seed and its increments.
        For all samples are computed and stored gini values, then the basic statistics is computed.

        Args:
            projectParameters (ProjectParameters): definitions of doctools metadata
            use_weight (bool): True for using weight defined in ProjectParameters
            masks (list of strings, optional): If filled, the distinct ROC curve is drawn for each masked sample
            n_iter (int, optional): number of iterations for the bootstrap. Default 100.
            ci_range (float, optional): percent of confidence intervals, ci_range and (1 - ci_range). Default 5.
            random_seed (int, optional): random seed for the random function, can be set the same way to reconstruct the old results.
                Default... you already know it, right?
            col_score_ref (str, optional): A reference score column in the dataset.
                If given, the whole gini bootstrap is computed as a difference to this score.
    """
    def __init__(self, *args, use_weight, masks=None, n_iter=100, ci_range=5,
                 random_seed=42, col_score_ref=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_iter = n_iter
        self.masks = masks
        self.ci_range = ci_range
        self.random_seed = random_seed
        self.use_weight = use_weight
        self.col_score_ref = col_score_ref

    def calculate(self):
        """ Calculates the bootstrap. """
        # declaration of all used external variables
        # -----------------------------------------
        targets = self.targets
        scores = self.scores
        masks = self.masks
        if isinstance(scores, str):
            scores = [scores]
        # ----------------------------------------
        # bootstrapping iterations
        loop_iterator = range(self.n_iter)
        output = []
        rand_backup = self.random_seed
        #  the backup is for restarting the random_seed back to its original value for each mask.
        #  Not necessary, I guess, useful just if you repeat the bootstrapping with
        #  a different set of masks but want results consistent with the previous...
        if masks:
            for i, name in enumerate(masks):
                df = (self.samples[0][i][0])
                if len(df) > 0:
                    for score in scores:
                        results = {}
                        gini_bootstrap = []
                        for _ in loop_iterator:
                            sampled_data = df.sample(frac=1, replace=True,
                                                     random_state=self.random_seed)  # sampling with replacement, i.e. bootstrapping
                            if self.use_weight:
                                sampled_weight = sampled_data[self.weight]
                            else:
                                sampled_weight = None
                            gini_value = 2 * roc_auc_score(sampled_data[targets[0][0]], 1.0 * sampled_data[score],
                                                           sample_weight=sampled_weight) - 1

                            if self.col_score_ref:
                                gini_ref = 2 * roc_auc_score(sampled_data[targets[0][0]],
                                                             1.0 * sampled_data[self.col_score_ref],
                                                             sample_weight=sampled_weight) - 1
                                gini_value = gini_value - gini_ref
                            gini_bootstrap += [gini_value]
                            if self.random_seed is not None:
                                self.random_seed += 1

                        results["mask"] = name
                        results["Score name"] = score
                        results["Gini mean"] = np.mean(gini_bootstrap)
                        results["Gini std"] = np.std(gini_bootstrap)
                        results["Confidence Interval " + str(self.ci_range) + ' %'] = np.percentile(gini_bootstrap,
                                                                                                    self.ci_range)
                        results["Confidence Interval " + str(100 - self.ci_range) + ' %'] = np.percentile(
                            gini_bootstrap,
                            100 - self.ci_range)
                        output.append(results)
                        self.random_seed = rand_backup
                else:
                    for score in scores:
                        output.append({'mask': name, 'Score name': score})
        if not masks:
            df = self.samples[0][0]
            name = self.samples[0][1]
            for score in scores:
                results = {}
                gini_bootstrap = []
                for _ in loop_iterator:
                    sampled_data = df.sample(frac=1, replace=True,
                                             random_state=self.random_seed)  # sampling with replacement, i.e. bootstrapping
                    if self.use_weight:
                        sampled_weight = sampled_data[self.weight]
                    else:
                        sampled_weight = None
                    gini_value = 2 * roc_auc_score(sampled_data[targets[0][0]], 1.0 * sampled_data[score],
                                                   sample_weight=sampled_weight) - 1

                    if self.col_score_ref:
                        gini_ref = 2 * roc_auc_score(sampled_data[targets[0][0]],
                                                     1.0 * sampled_data[self.col_score_ref],
                                                     sample_weight=sampled_weight) - 1
                        gini_value = gini_value - gini_ref
                    gini_bootstrap += [gini_value]
                    if self.random_seed is not None:
                        self.random_seed += 1

                results["mask"] = name
                results["Score name"] = score
                results["Gini mean"] = np.mean(gini_bootstrap)
                results["Gini std"] = np.std(gini_bootstrap)
                results["Confidence Interval " + str(self.ci_range) + ' %'] = np.percentile(gini_bootstrap,
                                                                                            self.ci_range)
                results["Confidence Interval " + str(100 - self.ci_range) + ' %'] = np.percentile(gini_bootstrap,
                                                                                                  100 - self.ci_range)
                output.append(results)

        self.table = output
        return self

    def get_table(self):
        """ Returns the table as list of dictionaries."""
        return self.table

    def get_visualization(self):
        """Not implemented here, the output is a table."""
        return self

    def get_description(self):
        """ A Simple class description."""
        return f"Bootstrapping of Gini for scores {self.scores} on masks {self.masks}, number of bootstraps = {self.n_iter}."


class ScoreGiniIVCalculator(Calculator):
    """
    Class for visualising gini and lift in time.
    Can be extended to visualise the KS test easily, too.

    Args:
        projectParameters (ProjectParameters): definitions of doctools metadata
        use_weight (bool): True for using weight defined in ProjectParameters
        masks (list, optional): If given, the measures are calculated for each sample distinctly. Masks defined in ProjectParameters
        lift_perc (float): Lift percentile for the lift in time plot. Default 15.
    """
    def __init__(self, *args, use_weight, masks=None, lift_perc=15, **kwargs):
        super().__init__(*args, **kwargs)
        self.masks = masks
        self.use_weight = use_weight
        self.lift_perc = lift_perc

    def calculate(self):
        """Calls :py:meth:`~scoring.doctools.calculators.ScoreModelPerformanceCalculator` for calculating Gini and lift.
        """
        # declaration of all used external variables
        # -----------------------------------------
        # target = self.targets[0]
        scores = self.scores
        # rowid_col = self.rowid_variable
        time_variable = self.time_variable
        masks = self.masks
        # -----------------------------------------
        results = []
        if masks:
            for i, name in enumerate(masks):
                df = (self.samples[0][i][0])
                if len(df) > 0:
                    time_range = df[time_variable].unique()
                    for time in time_range:
                        data = df[df[time_variable] == time]
                        self.SMPC = ScoreModelPerformanceCalculator(self, data=data, use_weight=self.use_weight,
                                                                    lift_perc=self.lift_perc, mask_name=name)

                        self.SMPC.samples = self.samples
                        self.SMPC.targets = self.targets
                        self.SMPC.weight = self.weight
                        self.SMPC.scores = self.scores

                        row_result = self.SMPC.calculate().get_table()
                        for j in range(len(row_result)):
                            row_result[j][time_variable] = time
                        results.append(row_result)
        if not masks:
            df = self.samples[0][0]
            time_range = df[self.time_variable].unique()
            name = self.samples[0][1]
            for time in time_range:
                data = df[df[time_variable] == time]
                self.SMPC = ScoreModelPerformanceCalculator(self, data=data, use_weight=self.use_weight,
                                                            lift_perc=self.lift_perc, mask_name=name)

                self.SMPC.samples = self.samples
                self.SMPC.targets = self.targets
                self.SMPC.weight = self.weight
                self.SMPC.scores = self.scores

                row_result = self.SMPC.calculate().get_table()
                for j in range(len(row_result)):
                    row_result[j][time_variable] = time
                results.append(row_result)

        results = [item for la in results for item in la]  # to get rid of the extra list structure inside
        self.table = results
        return self

    def get_table(self):
        """ Returns the output as a list of dictionaries. """
        # to_display = pd.DataFrame(self.table).set_index([self.time_variable])
        # display(to_display)
        return self.table

    def get_visualization(self, get_gini=True, get_lift=True, show_plot=False, output_folder=None, filename_gini=None,
                          filename_lift=None):
        """Plots of gini, lift in time.
        The KS in time can be added easily as it is a part of the ScoreModelPerformanceCalculator table.

        Args:
            filename_lift (str, optional): file name for lift plot
            filename_gini (str, optional): filename for gini plot
            get_gini (bool, optional): create plot for gini in time
            get_lift (bool, optional): create plot for lift in time
            show_plot (bool, optional): show plot(s) in a notebook (True) or save a file only (False)
            output_folder (str, optional): the string for the output folder
        """
        scores = self.scores
        if isinstance(scores, str):
            scores = [scores]
        plot_data = pd.DataFrame(self.table).set_index([self.time_variable])
        x_axis = np.sort(plot_data.index.get_level_values(self.time_variable).unique()).tolist()
        # creates the whole interval in months
        x_axis_period = pd.period_range(start=min(x_axis), end=max(x_axis), freq='M').strftime('%Y%m')
        x_axis_period_num = pd.to_numeric(x_axis_period)
        x_format = ticker.FixedFormatter(x_axis_period)
        x_locator = ticker.FixedLocator(range(len(x_axis_period)))
        if get_gini:
            fig, ax = plt.subplots(figsize=(10, 7))
            if self.masks:
                for name in self.masks:
                    for score in scores:
                        for tgt in self.targets:
                            df = plot_data[(plot_data["sample"] == name) & (plot_data["score"] == score) &
                                           (plot_data["target"] == tgt[0])]["gini"]
                            df = df.reindex(x_axis_period_num)  # getting 0 for the missing values
                            ax.plot(x_axis_period, df, label=f"{score} {name} {tgt[0]}", linewidth=2.0, marker='o')
            if not self.masks:
                name = self.samples[0][1]
                for score in scores:
                    for tgt in self.targets:
                        df = plot_data[(plot_data["sample"] == name) & (plot_data["score"] == score) &
                                       (plot_data["target"] == tgt[0])]["gini"]
                        df = df.reindex(x_axis_period_num)  # getting rows for the missing months
                        ax.plot(x_axis_period, df, label=f"{score} {name} {tgt[0]}", linewidth=2.0, marker='o')
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            fig.suptitle(f"Gini in time for target {self.targets[0][0]}", fontsize=20)
            ax.xaxis.set_major_locator(x_locator)
            ax.xaxis.set_major_formatter(x_format)
            ax.set_xlabel(self.time_variable)
            ax.tick_params(labelrotation=45)
            ax.set_ylim([0, 1])
            ax.set_ylabel("gini")

            if output_folder is not None:
                if not filename_gini:
                    filename_gini = "ginistability.PNG"
                plt.savefig(path.join(output_folder, filename_gini), bbox_inches='tight', dpi=200)

            if show_plot:
                plt.show()
            plt.close(fig)

        if get_lift:
            fig, ax = plt.subplots(figsize=(10, 7))
            fig.suptitle(f"Lift in time for target {self.targets[0][0]}", fontsize=20)
            if self.masks:
                for name in self.masks:
                    for score in scores:
                        for tgt in self.targets:
                            df = plot_data[(plot_data["sample"] == name) & (plot_data["score"] == score) &
                                           (plot_data["target"] == tgt[0])][f"lift_{self.lift_perc}"]
                            df = df.reindex(x_axis_period_num)  # getting rows for the missing months
                            ax.plot(x_axis_period, df, label=f"{score} {name} {tgt[0]}", linewidth=2.0, marker='o')
            if not self.masks:
                name = self.samples[0][1]
                for score in scores:
                    for tgt in self.targets:
                        df = plot_data[(plot_data["sample"] == name) & (plot_data["score"] == score) &
                                       (plot_data["target"] == tgt[0])][f"lift_{self.lift_perc}"]
                        df = df.reindex(x_axis_period_num)  # getting 0 for the missing values
                        ax.plot(x_axis_period, df, label=f"{score} {name} {tgt[0]}", linewidth=2.0, marker='o')
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax.xaxis.set_major_locator(x_locator)
            ax.xaxis.set_major_formatter(x_format)
            ax.tick_params(labelrotation=45)
            ax.set_xlabel(self.time_variable)
            y_max = plot_data[f"lift_{self.lift_perc}"].max(skipna=True)
            ax.set_ylim([0, y_max + 0.5])
            ax.set_ylabel(f"lift_{self.lift_perc}")

            if output_folder is not None:
                if not filename_lift:
                    filename_lift = "liftstability.PNG"
                plt.savefig(path.join(output_folder, filename_lift), bbox_inches='tight', dpi=200)

            if show_plot:
                plt.show()
            plt.close(fig)

    def get_description(self):
        """Returns a basic class description."""
        return f"Gini and Lift in time for score(s) {self.scores} on sample(s) {self.masks}."


class TransitionMatrixCalculator(Calculator):
    """ Shows the transition and default matrices for new and old score. All rows are sorted into n bins by each score
    value, where n is defined by quantiles_count. Then the n*n matrix is created, showing either the ratio of default for
    each old*new score combination (default matrix) or ratio of the contracts' count, showing how many contracts from
    old score's n-th bin is in new score's m'th bin (transition matrix).

    Args:
        projectParameters (ProjectParameters): definitions of doctools metadata
        use_weight (bool): True for using weight defined in ProjectParameters
        observed (series, 1/0 mask for dataset): If given, overrides the the default base column for this calculator.
        quantiles_count (int): Number of desired quantiles for the matrix. Default is 10.
    """
    def __init__(self, *args, use_weight=False, observed=None, quantiles_count=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantiles_count = quantiles_count
        self.use_weight = use_weight
        self.observed = observed

    def calculate(self):
        """
        Calculates the transition matrix between old and new score.
        """
        # declaration of all used external variables
        # -----------------------------------------
        target = self.targets
        scores = self.scores
        df = self.samples[0][0]
        observed = self.observed
        if observed is None:
            observed = target[0][1]
        if self.use_weight:
            weight = self.weight
        # -----------------------------------------

        if not self.use_weight:
            newscore_dec = pd.DataFrame(pd.qcut(df[scores[1]], self.quantiles_count, labels=False, duplicates='drop'))
            oldscore_dec = pd.DataFrame(pd.qcut(df[scores[0]], self.quantiles_count, labels=False, duplicates='drop'))
            dec_data = pd.concat([oldscore_dec, newscore_dec, df[[target[0][0], target[0][1], observed]]], axis=1)
            dec_data.columns = ['oldscore', 'newscore', 'target', 'base', 'obs']
        else:
            weighted_target = df[target[0][0]] * df[weight]
            weighted_base = df[target[0][1]] * df[weight]
            weighted_observations = df[observed] * df[weight]
            weighted_df = pd.concat(
                [df[scores[1]], df[scores[0]], df[weight], weighted_target, weighted_base, weighted_observations],
                axis=1)
            weighted_df.columns = ['newscore_raw', 'oldscore_raw', 'weight', 'target', 'base', 'obs']
            weighted_df.sort_values(['newscore_raw'], inplace=True)
            weighted_df['cum_weight_by_newscore'] = weighted_df['weight'].cumsum() / weighted_df[
                'weight'].sum() * self.quantiles_count
            weighted_df['newscore'] = np.ceil(weighted_df['cum_weight_by_newscore']) - 1
            weighted_df.loc[weighted_df['newscore'] < 0, 'newscore'] = 0
            weighted_df.sort_values(['oldscore_raw'], inplace=True)
            weighted_df['cum_weight_by_oldscore'] = weighted_df['weight'].cumsum() / weighted_df[
                'weight'].sum() * self.quantiles_count
            weighted_df['oldscore'] = np.ceil(weighted_df['cum_weight_by_oldscore']) - 1
            weighted_df.loc[weighted_df['oldscore'] < 0, 'oldscore'] = 0
            dec_data = weighted_df[['oldscore', 'newscore', 'target', 'base', 'obs']]
        dec_data_agg = dec_data.groupby(['oldscore', 'newscore'])['target', 'base', 'obs'].sum()
        dec_data_agg2 = dec_data.groupby(['oldscore'])['obs'].sum()
        dec_data_all = dec_data_agg.reset_index().join(dec_data_agg2, on=['oldscore'], rsuffix='_all').set_index(
            ['oldscore', 'newscore'])
        dec_data_all['default rate'] = dec_data_all['target'] / dec_data_all['base']
        dec_data_all['share'] = dec_data_all['obs'] / dec_data_all['obs_all']

        self.table = dec_data_all
        return self

    def get_table(self):
        """Returns the output as pandas DataFrame."""
        return self.table

    def get_visualization(self, draw_default_matrix=True, draw_transition_matrix=True, show_plot=True,
                          output_folder=None, filename_default=None, filename_transition=None):
        """ Creates the visualisation.

        Args:
            draw_default_matrix (bool): If True, performance (default) matrix is drawn.
            draw_transition_matrix (bool): If True, transition matrix is drawn.
            show_plot (bool): If True, plots to iPython. Default True.
            output_folder (str): if given, the plot is saved to the folder, if None, plot is not saved
            filename_default (str): if not given, default 'performance_matrix.png' is used
            filename_transition (str): if not given, default 'transition_matrix.png' is used
        """
        if draw_default_matrix:
            matrix_DR = np.array(self.table.unstack()[['default rate']])
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.heatmap(matrix_DR, annot=True, cmap=sns.cubehelix_palette(light=1, as_cmap=True))
            ax.set_ylabel('old score decile')
            ax.set_xlabel('new score decile')
            plt.title('Default rate by deciles')
            if output_folder is not None:
                if filename_default is None:
                    filename_default = "performance_matrix.png"
                plt.savefig(path.join(output_folder, filename_default), bbox_inches='tight', dpi=200)
            if show_plot:
                plt.show()
            plt.close(fig)

        if draw_transition_matrix:
            matrix_OS = np.array(self.table.unstack()[['share']])
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.heatmap(matrix_OS, annot=True, cmap=sns.cubehelix_palette(light=1, as_cmap=True))
            ax.set_ylabel('old score decile')
            ax.set_xlabel('new score decile')
            plt.title('Transition matrix by deciles')
            if output_folder is not None:
                if filename_transition is None:
                    filename_transition = "transition_matrix.png"
                plt.savefig(path.join(output_folder, filename_transition), bbox_inches='tight', dpi=200)
            if show_plot:
                plt.show()
            plt.close(fig)

    def get_description(self):
        """Returns a short class description. """
        return f"Transition and Default Matrix for old score: {self.scores[0]} and new score: {self.scores[1]}."


class ScorePlotDistCalculator(Calculator):
    """
    Calculator for Score distribution. The interval between minimal and maximal score is divided into n_bins equal bins.
    Then the good and bads are computed for each bin.

    Args:
        projectParameters (ProjectParameters): definitions of doctools metadata
        use_weight (bool): True for using weight defined in ProjectParameters
        observed (str, optional): base column - the alternative 1/0 column of contracts which should be counted in.
            If None, column of 1's is used.
        n_bins (int, optional): number of bins the score should be binned to (default: 25)
        min_score (float, optional): minimal score value for binning (default: None)
        max_score (float, optional): maximal score value for binning (default: None)
        use_logit (bool, optional) : If True, the log(score/(1-score)) is used instead of score. Default False.
        labels (list of str, optional): list of two strings - labels for levels [0, 1] of col_target (default:
            ['good','bad'])
    """
    def __init__(self, *args, use_weight, observed=None, n_bins=25, min_score=None, max_score=None, use_logit=False,
                 labels=["good", "bad"], **kwargs):
        super().__init__(*args, **kwargs)
        self.use_weight = use_weight
        self.observed = observed
        self.min_score = min_score
        self.max_score = max_score
        self.n_bins = n_bins
        self.labels = labels
        self.use_logit = use_logit

    def calculate(self):
        """
        Plots charts with distribution/density of score values.
        """

        # declaration of all used external variables
        # -----------------------------------------
        targets = self.targets
        scores = self.scores
        df = self.samples[0][0].copy()
        observed = self.observed
        if observed is None:
            observed = "observed"
            df[observed] = 1
        if isinstance(scores, str):
            scores = [scores]
        # -----------------------------------------
        if self.use_logit:
            df["score_logit"] = np.log(df[scores[0]] / (1 - df[scores[0]]))
            scores[0] = "score_logit"

        if self.use_weight:
            data = df[df[observed] == 1][[scores[0], targets[0][0], self.weight]].copy()
        else:
            data = df[df[observed] == 1][[scores[0], targets[0][0]]].copy()

        if self.min_score is None:
            self.min_score = min(data[scores[0]])
        if self.max_score is None:
            self.max_score = max(data[scores[0]])

        bin_border = []
        for i in range(0, self.n_bins):
            bin_border += [self.min_score + i * (self.max_score - self.min_score) / self.n_bins]
        bin_border += [self.max_score + 0.00001]

        data['bin'] = np.zeros(len(data)).astype(int)
        bin_str = []
        for i in range(0, self.n_bins):
            data['bin'] = np.where((data[scores[0]] >= bin_border[i]) & (data[scores[0]] < bin_border[i + 1]),
                                   i + 1, data['bin'])
            bin_str += ['[' + str(round(bin_border[i], 2)) + ';' + str(round(bin_border[i + 1], 2)) + ')']

        if self.use_weight:
            data_grp = data[['bin', targets[0][0], self.weight]].groupby([targets[0][0], 'bin'])[[self.weight]].sum()
        else:
            data_grp = data[['bin', targets[0][0], scores[0]]].groupby([targets[0][0], 'bin'])[[scores[0]]].count()

        bins_base = pd.DataFrame(np.arange(1, self.n_bins + 1), columns=['bin'])
        bins_base.set_index('bin', inplace=True)
        good_hist = data_grp.loc[0]
        good_hist.columns = ['good_cnt']
        bad_hist = data_grp.loc[1]
        bad_hist.columns = ['bad_cnt']
        dt_hist = bins_base.join(good_hist, how='left').join(bad_hist, how='left').fillna(0)
        dt_hist['good_cnt_norm'] = dt_hist['good_cnt'] / (dt_hist['good_cnt'] + dt_hist['bad_cnt'])
        dt_hist['bad_cnt_norm'] = dt_hist['bad_cnt'] / (dt_hist['good_cnt'] + dt_hist['bad_cnt'])

        self.table = dt_hist
        self.bin_str = bin_str
        return self

    def get_table(self):
        """Returns:
            pandas.DataFrame: dataframe with the underlying data for the plot"""
        return self.table

    def get_visualization(self, output_folder=None, filename=None, show_plot=True):
        """ This is the plotting function of this class.

        Args:
            show_plot (bool): If True, plots to iPython. Default True.
            output_folder (str): if given, the plot is saved to the folder, if None, plot is not saved
            filename (str): if not given, default 'distribution_chart.png' is used
        """
        fig = plt.subplots(figsize=(15, 6))
        plt.subplot(121)
        plt.bar(range(1, self.n_bins + 1), self.table['bad_cnt'], label=self.labels[1], color='r')
        plt.bar(range(1, self.n_bins + 1), self.table['good_cnt'], bottom=self.table['bad_cnt'].values,
                label=self.labels[0], color='b')
        plt.xticks(range(1, self.n_bins + 1), self.bin_str, rotation=90)
        plt.xlabel('Score')
        plt.ylabel('Frequency')

        plt.legend(loc="best")

        plt.subplot(122)
        plt.bar(range(1, self.n_bins + 1), self.table['bad_cnt_norm'], label=self.labels[1], color='r')
        plt.bar(range(1, self.n_bins + 1), self.table['good_cnt_norm'], bottom=self.table['bad_cnt_norm'].values,
                label=self.labels[0], color='b')
        plt.xticks(range(1, self.n_bins + 1), self.bin_str, rotation=90)
        plt.xlabel('Score')
        plt.ylabel('Normalized frequency')

        plt.legend(loc="best")

        if output_folder is not None:
            if not filename:
                filename = "distribution_chart.png"
            plt.savefig(path.join(output_folder, filename), bbox_inches='tight', dpi=200)
        if show_plot:
            plt.show()
        plt.close()

    def get_description(self):
        """Returns short class description. """
        return f"Score Distribution Plot."


class EmptyInTimeCalculator(Calculator):
    """Doctools calculator. Shows charts with share of empty/NaN values of the predictors per time unit.

    Args:
        projectParameters (ProjectParameters): definitions of doctools metadata
        empty_representations (list, optional): List of values of the predictors that encode special (empty) observations. Defaults to [np.nan].
        use_weight (bool, optional): True for using the self.weight, False for calculating without weights. Defaults to False.

	Methods:
        calculate(): calculates table for the visualisation
        get_table(): returns the calculated table as pd.DataFrame
        get_visualisation(showPlot=False, outputFolder=None, fileName=None): plots the visualisation either to ipython or to file or both
    """

    def __init__(self, *args, empty_representations=[np.nan], use_weight=False, **kwargs):
        super().__init__(*args, **kwargs)
        if type(empty_representations) is not list:
            empty_representations = [empty_representations]
        self.empty_representations = empty_representations
        self.use_weight = use_weight

    def calculate(self):
        # declaration of all used external variables
        # -----------------------------------------
        df = self.samples[0][0]
        predictor = self.predictors[0]
        rowid_col = self.rowid_variable
        time_variable = self.time_variable
        if self.use_weight:
            weight = self.weight
        else:
            weight = None
        # ---------------------------------

        if weight is not None:
            w = df[weight]
        else:
            w = pd.Series(1, index=df.index)

        X = pd.DataFrame({'all': w}, index=df.index)
        X['empty'] = 0
        for empty_value in self.empty_representations:
            if pd.isnull(empty_value):
                X.loc[pd.isnull(df[predictor]), 'empty'] = X.loc[pd.isnull(df[predictor]), 'all']
            else:
                X.loc[df[predictor] == empty_value, 'empty'] = X.loc[df[predictor] == empty_value, 'all']
        self.table = X.groupby(df[time_variable])[['empty', 'all']].sum()
        self.table['empty_share'] = self.table['empty'] / self.table['all']
        self.table.loc[self.table['all'] == 0, 'empty_share'] = 0.0
        return self

    def get_table(self):
        return self.table

    def get_visualization(self, show_plot=False, output_folder=None, filename=None):
        plt.plot(list(self.table['empty_share']), linewidth=3)
        plt.xticks(range(len(self.table.index)), self.table.index, rotation=45, fontsize=12)
        plt.ylim(-0.05, 1.05)
        plt.title(f'{self.predictors[0]} - empty share')
        if output_folder is not None:
            if filename is None:
                filename = self.predictors[0]
            plt.savefig(path.join(output_folder, filename), bbox_inches='tight', dpi=200)
        if show_plot:
            plt.show()
        plt.close()
        return

    def get_description(self):
        return


class CalibrationDistributionCalculator(Calculator):
    """Doctools calculator. Shows distribution chart of score variable. Compares empirical and theoretical probability of default in each bin of score variable.

    Args:
        projectParameters (ProjectParameters): definitions of doctools metadata
        shift (int, optional): Shift (intercept) to be added to the score before its calibration is evaluated. Defaults to 0.
        scale (int, optional): Scale (multiplier) to be applied to the score before its calibration is evaluated. Defaults to 1.
        apply_logit (bool, optional): Whether logistic transformation must be applied to the score (i.e. whether score is in expit form). Defaults to False.
        empty_representations (list, optional): List of values of the score that encode special (empty) observations. Defaults to [np.nan].
        use_weight (bool, optional): True for using the self.weight, False for calculating without weights. Defaults to False.
        bins (int, optional): Number of bins the score should be binned to. Defaults to 30.
        vertical_lines (list of float, optional): x coordinates of vertical lines to be added to the chart. Defaults to None.

    Methods:
        calculate(): calculates table for the visualisation
        get_table(): returns the calculated table as pd.DataFrame
        get_visualisation(showPlot=False, outputFolder=None, fileName=None): plots the visualisation either to ipython or to file or both
    """

    def __init__(self, *args, shift=0, scale=1, apply_logit=False, empty_representations=[np.nan], use_weight=False,
                 bins=30, vertical_lines=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.shift = shift
        self.scale = scale
        self.apply_logit = apply_logit
        if type(empty_representations) is not list:
            empty_representations = [empty_representations]
        self.empty_representations = empty_representations
        self.use_weight = use_weight
        self.bins = bins
        if vertical_lines is not None and (type(vertical_lines) is not list):
            vertical_lines = [vertical_lines]
        self.vertical_lines = vertical_lines

    def calculate(self):
        # declaration of all used external variables
        # -----------------------------------------
        df = self.samples[0][0].copy()
        target = self.targets[0][0]
        predictor = self.predictors[0]
        rowid_col = self.rowid_variable
        if self.use_weight:
            weight = self.weight
        else:
            weight = None
        # ---------------------------------

        if weight is None:
            weight = '_weight'
            df[weight] = 1
        df['_wt'] = df[weight] * df[target]

        for empty_value in self.empty_representations:
            df.loc[df[predictor] == empty_value, predictor] = np.nan

        if len(df[(df[target].notnull()) & (df[predictor].isnull())]) > 0:
            self.def_rx_nohit = df[(df[target].notnull()) & (df[predictor].isnull())]['_wt'].sum() / \
                                df[(df[target].notnull()) & (df[predictor].isnull())][weight].sum()
        else:
            self.def_rx_nohit = np.nan

        mask = df[target].notnull() & df[predictor].notnull()

        bin_ranges = [
            df[mask][predictor].min() + k * (df[mask][predictor].max() - df[mask][predictor].min()) / self.bins for k in
            range(self.bins + 1)]
        bin_means = [(bin_ranges[k] + bin_ranges[k + 1]) / 2 for k in range(self.bins)]
        bin_ranges[-1] = np.inf
        def_rx = []
        cnts = []
        for k in range(self.bins):
            if df[mask & (df[predictor] >= bin_ranges[k]) & (df[predictor] < bin_ranges[k + 1])][weight].sum() > 0:
                def_rx.append(
                    df[mask & (df[predictor] >= bin_ranges[k]) & (df[predictor] < bin_ranges[k + 1])]['_wt'].sum() /
                    df[mask & (df[predictor] >= bin_ranges[k]) & (df[predictor] < bin_ranges[k + 1])][weight].sum())
                cnts.append(
                    df[mask & (df[predictor] >= bin_ranges[k]) & (df[predictor] < bin_ranges[k + 1])][weight].sum())
            else:
                def_rx.append(np.nan)
                cnts.append(0)

        if not self.apply_logit:
            df['_pd'] = expit(self.scale * df[predictor] + self.shift)
            # df[predictor] = self.scale * df[predictor] + self.shift
        else:
            df['_pd'] = expit(self.scale * logit(df[predictor]) + self.shift)
            # df[predictor] = self.scale * logit(df[predictor]) + self.shift

        df['_wpd'] = df[weight] * df['_pd']
        g = 200 * roc_auc_score(df[mask][target], df[mask][predictor], sample_weight=df[mask][weight]) - 100
        if self.scale < 0:
            g = -g

        self.additional = {
            'Avg predicted default rate': 100 * df[mask]["_wpd"].sum() / df[mask][weight].sum(),
            'Avg hit default rate': 100 * df[mask]["_wt"].sum() / df[mask][weight].sum(),
            'Avg non-hit default rate': 100 * self.def_rx_nohit,
            'Hit Gini': g
        }

        self.table = pd.DataFrame({
            'bin_ranges': bin_ranges[:-1],
            'bin_means': bin_means,
            'def_rx': def_rx,
            'cnts': cnts,
        })
        return self

    def get_table(self):
        return self.table

    def get_visualization(self, show_plot=False, output_folder=None, filename=None):

        _, ax1 = plt.subplots()
        plt.bar(self.table['bin_means'], self.table['cnts'], color='r',
                width=(self.table['bin_ranges'][1] - self.table['bin_ranges'][0]) * 0.9)
        plt.xlabel('score')
        plt.ylabel('frequency')

        ax2 = ax1.twinx()
        ax2.plot(self.table['bin_means'], self.table['def_rx'], 'o-', color='b')
        if self.def_rx_nohit:
            ax2.plot(self.table['bin_means'], np.ones(len(self.table['bin_means'])) * self.def_rx_nohit, '--',
                     color='g')

        if not self.apply_logit:
            sc_to_prob = 1 / (1 + np.exp(-np.array(self.shift + self.scale * self.table['bin_means'])))
        else:
            sc_to_prob = 1 / (1 + np.exp(-np.array(self.shift + self.scale * self.table['bin_means'].apply(logit))))
        ax2.plot(self.table['bin_means'], sc_to_prob, '--', color='black')

        if self.vertical_lines is not None:
            for line_y in self.vertical_lines:
                max_vline = self.table['def_rx'].max()
                if max(sc_to_prob) > max_vline:
                    max_vline = max(sc_to_prob)
                ax2.plot([line_y, line_y], [0, max_vline], '--', color='c')

        plt.xlabel('score')
        plt.ylabel('default rate')
        plt.title(self.predictors[0])

        additional_text = ''
        for k, v in self.additional.items():
            additional_text += f'{k} : {v:.2f}%' + '\n'
        plt.figtext(0.5, -0.2, additional_text, ha="center", fontsize=10)

        if output_folder is not None:
            if filename is None:
                filename = self.predictors[0] + '.png'
            plt.savefig(output_folder + '/' + filename, bbox_inches='tight', dpi=200)
        if show_plot:
            plt.show()
        plt.close()
        return

    def get_description(self):
        return


class ExpectedApprovalRateCalculator(Calculator):
    """For each population of interest we calculate theoretical approval rate. The estimation goes as follows:

    We define a reference approval rate for the whole population of incoming customers
    We calculate a cutoff value which corresponds to this targetted approval rate
    We set the same cutoff for the subpopulation
    We evaluate what would the approval rate on just this subpopulation be when the cutoff is applied.
    If the subpopulation approval rate is different to the reference approval rate, it ususally means that the subpopulation is shifted, i.e. the estimated probability of default of such customers is different from probability of default of a typical customer.

    Args:
        projectParameters (ProjectParameters): definitions of doctools metadata
        query_subset (list of str): list of pandas queries defining subsets of interests from data
        reference_ar (list of float, optional): List of reference approval rates. Defaults to [0.50].
        use_weight (bool, optional): True for using the self.weight, False for calculating without weights. Defaults to False.
        def_by_score_ascending (bool, optional): Whether score is increasing with probabiity of default or decreasing. Defaults to False.

	Methods:
        calculate(): calculates the results
        get_table(): returns the calculated table as pd.DataFrame
    """

    def __init__(self, *args, query_subset, reference_ar=[0.50], use_weight=False, def_by_score_ascending=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(query_subset, list):
            self.query_subset = query_subset
        else:
            self.query_subset = [query_subset]
        self.use_weight = use_weight
        if isinstance(reference_ar, list):
            self.reference_ar = reference_ar
        else:
            self.reference_ar = [reference_ar]
        self.def_by_score_ascending = def_by_score_ascending

    def calculate(self):
        # declaration of all used external variables
        # -----------------------------------------
        df = self.samples[0][0].copy()
        score = self.scores[0]
        if self.use_weight:
            weight = self.weight
        else:
            weight = None
        # ---------------------------------

        if weight is None:
            weight = '__weight'
            df[weight] = 1
        data_ref = df[[score, weight]].sort_values(score).copy()
        data_ref['__cum_weight'] = data_ref[weight].cumsum() / data_ref[weight].sum()
        query_subset, reference_ar, cutoff, subset_ar = [], [], [], []

        for sub in self.query_subset:
            data_subset = df.query(sub, engine='python')[[score, weight]].copy()
            for ref in self.reference_ar:
                query_subset.append(sub)
                reference_ar.append(ref)
                if self.def_by_score_ascending:
                    cutoff.append(data_ref[data_ref['__cum_weight'] > 1 - ref][score].min())
                    subset_ar.append(
                        data_subset[data_subset[score] > cutoff[-1]][weight].sum() / data_subset[weight].sum())
                else:
                    cutoff.append(data_ref[data_ref['__cum_weight'] <= ref][score].max())
                    subset_ar.append(
                        data_subset[data_subset[score] <= cutoff[-1]][weight].sum() / data_subset[weight].sum())

        self.table = pd.DataFrame({
            'query_subset': query_subset,
            'reference_ar': reference_ar,
            'cutoff': cutoff,
            'subset_ar': subset_ar,
        })

    def get_table(self):
        return self.table

    def get_visualization(self):
        warnings.warn("Visualisation not implemented.")

    def get_description(self):
        return


class DataSampleSummaryCalculator(Calculator):
    """Doctools calculator. Creates summary table about the data (number of observations, bad rate, by segment, by time).

    Args:
        projectParameters (ProjectParameters): definitions of doctools metadata
        use_weight (bool, optional): True for using the self.weight, False for calculating without weights. Defaults to False.
        segment_col (str, optional): Column to segment the data by. Defaults to None.

    Methods:
        calculate(): calculates table with results
        get_table(): returns the calculated table as pd.DataFrame
    """

    def __init__(self, *args, use_weight=False, segment_col=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_weight = use_weight
        self.segment_col = segment_col

    def calculate(self):
        df = self.samples[0][0].copy()
        target = self.targets[0][0]
        base = self.targets[0][1]
        obs = 'Observations'
        if self.segment_col:
            segment_col = self.segment_col
        else:
            segment_col = '__data_sample'
        if self.use_weight:
            weight = self.weight
        time_variable = self.time_variable

        if segment_col not in df.columns:
            df[segment_col] = 'all'
        df[obs] = 1
        if self.use_weight:
            df[target] = df[target] * df[weight]
            df[base] = df[base] * df[weight]
            df[obs] = df[obs] * df[weight]
        df_grouped = df.groupby([time_variable, segment_col])[[target, base, obs]].sum()
        df_grouped[target + ' rate'] = df_grouped[target] / df_grouped[base]

        self.table = df_grouped.reset_index(level=segment_col).pivot(columns=segment_col).round(6).fillna("")

    def get_table(self):
        return self.table

    def get_visualization(self):
        warnings.warn("Visualisation not implemented.")

    def get_description(self):
        return
