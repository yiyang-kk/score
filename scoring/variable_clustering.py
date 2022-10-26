
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

import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from IPython.display import display
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import FeatureAgglomeration, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.utils import as_float_array, assert_all_finite, check_array
from sklearn.utils.validation import (check_consistent_length, check_is_fitted,
                                      column_or_1d)


class CorrVarClus():
    """
    Class for clustering variables using their correlations. It uses hierarchical clustering, i.e. it starts with each
    variable being its own cluster and in each iteration, it joins two closest (most correlated) clusters together.

    Args:
        max_correlation (float, optional): stopping criterion. If correlation between two clusters is lower than this,
        they will not be joint together. (default 0.5)
        max_clusters (int, optional): (maximal) number of clusters to be created. (default None)
        standardize (bool, optional): Should the input data be standardized? (default True)
        sample_size (int, optional): with large data frames, for correlation matrix calculation, it is usually necessary
        to sample the data (the computation is demanding). Here the size of such sample (number of observations in the sample)
        can be set. Default value of 0 means that no sampling is performed and the data frame is used in its original size.

    Properties:
        * variables\_ (list): Created by fit() method. List of names of variables that entered the clustering
        * labels\_ (list): Created by fit() method. List of cluster numbers corresponding to variables from variables\_ list
        * cluster_table\_ (pandas data frame): Created by fit() method. One table with all variables, their cluster numbers and gini

    Example:
        >>> cc = CorrVarClus(max_correlation = 0.5)
        >>> cc.fit(data[train_mask][cols_woe], data[train_mask][col_target])
        >>> cc.draw(output_file = output_folder + '/analysis/clustering_dendrogram.png')
        >>> cc.display(output_file = output_folder + '/predictors/predictor_clusters_correlation.csv')
        >>> cc.bestVariables()
    """

    def __init__(self, max_correlation=0.5, max_clusters=None, standardize=True, sample_size=0):
        """
        Initialization method. See args definitions of the class.
        """
        self.max_correlation = max_correlation
        self.max_clusters = max_clusters
        self.standardize = standardize
        self.sample_size = sample_size

        return

    def fit(self, X, y):
        '''
        This method is used to fit the clusters using a training data set and training target array. Must be run before any other method.

        Args:
            X (pd.DataFrame): data frame with perdictors only (no target, no time or other variables which don't have the role of predictors)
            y (pd.Series): array of target variable which is coded as 0 and 1
        '''

        self.variables_ = list(X.columns)
        # sample the data set (for lower size in memory) and transpose it (observations -> variables, variables -> observations ... this is needed as we want to cluster variables and not observations)
        if (self.sample_size > 0) and (len(X.index) > self.sample_size):
            print('Sampling input dataset...')
            X = X.sample(self.sample_size)
            y = y[X.index]

        if self.standardize:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                print('Standardazing data...')
                standardizer = StandardScaler()
                X = pd.DataFrame(standardizer.fit_transform(X), columns=self.variables_)

        # if a variable has variance = 0, it would cause error, so we need to remove such items
        print('Checking data for columns with zero variance...')
        for name, col in X.iteritems():
            if col.nunique() < 2:
                print('{0} column has only one value in the sample. It won\'t be used for correlation dendrogram.'.format(name))
                X = X.drop(columns=name)

        self.X_t = X.fillna(0).transpose()
        self.variables_ = list(self.X_t.index)

        # fitting hierarchical clustering
        self.Z = linkage(self.X_t, method='average', metric='correlation')
        if self.max_correlation is not None:
            clusters = fcluster(self.Z, 1-self.max_correlation, criterion='distance')
            self.correlation_line = 1-self.max_correlation
        if self.max_clusters is not None:
            clusters_max = fcluster(self.Z, self.max_clusters, criterion='maxclust')
            if max(clusters_max) < max(clusters):
                clusters = clusters_max
                if self.Z.shape[0] < self.max_clusters:
                    index_of_line_threshold = -self.Z.shape[0]
                else:
                    index_of_line_threshold = -self.max_clusters
                self.correlation_line = self.Z[index_of_line_threshold,2]

        # save the results of clustering to internal variables
        self.labels_ = list(clusters)

        # calculate gini because we need to find out which variables are strong in each cluster
        print('Calculating Gini of variables...')
        self.ginis_ = []
        for v in self.variables_:
            self.ginis_.append(-(2*roc_auc_score(y, X[v])-1))

        print('Calculating order of variables in clusters...')
        self.cluster_table_ = pd.DataFrame(
            {'Variable': self.variables_, 'Cluster': self.labels_, 'Gini': self.ginis_}, index=self.variables_)
        self.cluster_table_ = self.cluster_table_.sort_values(['Cluster', 'Gini'], ascending=[True, False])
        self.cluster_table_['Order'] = self.cluster_table_.sort_values(
            'Gini', ascending=False).groupby(['Cluster']).cumcount() + 1
        self.cluster_table_.drop(['Variable'], inplace=True, axis=1)

        print('Done! Use method draw() to visualise clusters, display() to display results in a table, bestVariables() to get list of the best variables from each cluster')
        return

    def draw(self, output_file=None):
        '''
        Draws dendrogram (tree scheme of the hierarchical clustering).

        Args:
            output_file (str, optional): folder where the chart is save to. Defaults to None.
        '''

        check_is_fitted(self, ['variables_', 'labels_', 'cluster_table_'])
        a4_dims = (10, int(len(self.labels_)/4))
        fig, ax = plt.subplots(figsize=a4_dims, dpi=70)
        dendrogram(self.Z, labels=[a+': '+str(b) for a, b in zip(self.variables_, self.labels_)], orientation='right')
        plt.axvline(x=self.correlation_line, c='k')
        plt.xlabel('correlation')
        plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], [1, 0.8, 0.6, 0.4, 0.2, 0])
        if output_file is not None:
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.show()
        plt.clf()
        plt.close()

    def display(self, output_file=None):
        '''
        Displays dataframe showing cluster number and gini of each variable. Within each cluster, the variables are sorted descending by gini.

        Args:
            output_file (str, optional): filename where dataframe is saved to as csv file. Defaults to None.
        '''

        check_is_fitted(self, ['variables_', 'labels_', 'cluster_table_'])
        display(self.cluster_table_)
        if output_file is not None:
            self.cluster_table_.to_csv(output_file)
        return

    def bestVariables(self):
        '''
        Returns list of best variables (one per each clusters, with highest gini in the clusters).

        Returns:
            list of str: best variables
        '''

        check_is_fitted(self, ['variables_', 'labels_', 'cluster_table_'])
        return list(self.cluster_table_[self.cluster_table_['Order'] == 1].index)


class FeatureAggVarClus:
    """
    Class for clustering variables using k-means algorithm.

    Args:
        n_cluster (float): how many clusters should be created
        standardize (bool, optional): Should the input data be standardized? (default True)
        sample_size (int, optional): with large data frames to reduce memory usage and computation time, 
        user can specify size of sample the algorithm is performed on.
        Default setting is 0, meaning that the data are not sampled and whole data set is used.
        connectivity (bool, optional): should there be imposed connectivity graph to capture local structure?
        for more info, see:
        https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_clustering.html
        (default True)

    Properties:
        * variables\_ (list): Created by fit() method. List of names of variables that entered the clustering
        * labels\_ (list): Created by fit() method. List of cluster numbers corresponding to variables from variables\_ list
        * cluster_table\_ (pandas data frame): Created by fit() method. One table with all variables, their cluster numbers and gini

    Example:
        >>> agglomeration = variable_clustering.FeatureAggVarClus(n_cluster=4,
        >>>                                              standardize = False,
        >>>                                               sample_size=50000)
        >>>
        >>> agglomeration.fit(data[train_mask][cols_woe], data[train_mask][col_target])
        >>> agglomeration.display(output_file=os.path.join(output_folder,
        >>>                                                'predictors/predictor_clusters_kmeans.csv')
        >>> agglomeration.bestVariables()
        >>> agglomeration.draw(output_file=os.path.join(output_folder,'/analysis/clustering_kmeans.png')

    """

    def __init__(self, n_cluster, standardize=True, sample_size=0, connectivity=True):
        """
        Initialization method. See args and kwargs definitions of the class.
        """
        self.n_cluster = n_cluster
        self.standardize = standardize
        self.sample_size = sample_size
        self.connectivity = connectivity

    def fit(self, X, y, nearest_neighbours=None):
        '''
        This method is used to fit the clusters using a training data set and training target array. Must be run before any other method.

        Args:
            X (pandas data frame): data frame with perdictors only (no target, no time or other variables which don't have the role of predictors)
            y (array): array of target variable which is coded as 0 and 1
            nearest_neigbours (, optional):
        '''

        self.variables_ = list(X.columns)
        # sample the data set (for lower size in memory) and transpose it (observations -> variables, variables -> observations ... this is needed as we want to cluster variables and not observations)
        if (self.sample_size > 0) and (len(X.index) > self.sample_size):
            print('Sampling input dataset...')
            X = X.sample(self.sample_size)
            y = y[X.index]

        if self.standardize:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                print('Standardazing data...')
                standardizer = StandardScaler()
                X = pd.DataFrame(standardizer.fit_transform(X), columns=self.variables_)

        if self.connectivity:
            assert nearest_neighbours is not None, \
                ('When connectivity=True, we must specify # of nearest neighbour features\n'
                 '(where nearest_neighbour < # of columns)')
            print('Using knn connectivity...')
            connectivity = kneighbors_graph(X.T, 3, include_self=False)
        else:
            connectivity = None

        # calculate correlation matrix
        self.correlations_ = X.astype(float).corr()

        # fit agglomerative clustering from scikit learn
        print('Fitting clusters...')
        self.agglomeration = FeatureAgglomeration(n_clusters=self.n_cluster,
                                                  connectivity=connectivity,
                                                  affinity="cosine",
                                                  linkage="average"
                                                  )
        self.agglomeration.fit(X)

        # calculate linkage - needed for vizualization
        self.X_t = X.fillna(0).transpose()
        self.Z = linkage(self.X_t, method='average', metric='cosine')

        # save the results of clustering to internal variables
        self.labels_ = list(self.agglomeration.labels_)

        # calculate gini because we need to find out which variables are strong in each cluster
        print('Calculating Gini of variables...')
        self.ginis_ = []
        for v in self.variables_:
            self.ginis_.append(-(2*roc_auc_score(y, X[v])-1))

        print('Calculating order of variables in clusters...')
        self.cluster_table_ = pd.DataFrame(
            {'Variable': self.variables_, 'Cluster': self.labels_, 'Gini': self.ginis_}, index=self.variables_)
        self.cluster_table_ = self.cluster_table_.sort_values(['Cluster', 'Gini'], ascending=[True, False])
        self.cluster_table_['Order'] = self.cluster_table_.sort_values(
            'Gini', ascending=False).groupby(['Cluster']).cumcount() + 1
        self.cluster_table_.drop(['Variable'], inplace=True, axis=1)

        print('Done! Use method draw() to visualise clusters, display() to display results in a table, bestVariables() to get list of the best variables from each cluster')

    def draw(self, output_file=None):
        '''
        Draws dendrogram (tree scheme of the hierarchical clustering).

        Args:
            output_file (str, optional): folder where the chart is save to. Defaults to None.
        '''

        check_is_fitted(self, ['variables_', 'labels_', 'correlations_','Z'])
        used_networks = [str(l) for l in np.unique(self.labels_)]

        # Create a custom palette to identify the networks
        network_pal = sns.color_palette('hls', len(used_networks))
        network_lut = dict(zip(used_networks, network_pal))

        # Convert the palette to vectors that will be drawn on the side of the matrix
        networks = [str(l) for l in self.labels_]
        network_colors = pd.Series(networks, index=self.variables_).map(network_lut)
        # Create custom colormap
        cmap = sns.cubehelix_palette(light=1, dark=0, hue=0, as_cmap=True)
        cg = sns.clustermap(self.correlations_, cmap=cmap, linewidths=.5, row_colors=network_colors,
                            col_colors=network_colors, col_linkage=self.Z, row_linkage=self.Z)
        plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
        plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
        plt.show()

    def display(self, output_file=None):
        '''
        Displays dataframe showing cluster number and gini of each variable. 
        Within each cluster, the variables are sorted descending by gini.

        Args:
            output_file (str, optional): filename where dataframe is saved to as csv file. Defaults to None.
        '''

        check_is_fitted(self, ['variables_', 'labels_', 'cluster_table_'])
        display(self.cluster_table_)
        if output_file:
            self.cluster_table_.to_csv(output_file)

    def bestVariables(self):
        '''
        Returns list of best variables (one per each clusters, with highest gini in the clusters).

        Returns:
            list of str.
        '''

        check_is_fitted(self, ['variables_', 'labels_', 'cluster_table_'])
        return list(self.cluster_table_[self.cluster_table_['Order'] == 1].index)

    def transform(self, X):
        '''
        Transform the data - aggregate the cluster variables using np.mean
        This is just calling to the .transform method of the trained agglomeration model

        Args:
            X (pandas.DataFrame): data to be transformed

        Returns:
            pandas.DataFrame: transformed data
        '''

        return self.agglomeration.transform(X)


class KMeansVarClus():
    """
    .. warning::
        DON'T USE! CURSE OF DIMENSIONALITY.

    Class for clustering variables using k-means algorithm.

    Args:
        k_param (float): k parameter of k-means algorithm, i.e. how many clusters should be created
        standardize (bool, optional): whether the data should be standardized first
            Defaults to True.
        pca (bool, optional): whether PCA should be performed to reduce dimensionality before clustering
            Defaults to True.
        n_components (int, optional): number of components for PCA
            Defaults to 10.
        sample_size (int, optional): with large data frames to reduce memory usage and computation time, user can specify size of sample the algorithm is performed on. Default setting is 0, meaning that the data are not sampled and whole data set is used.
            Defaults to 0.
        random_state (int, optional): random seed for the PCA solver. Defaults to 123.

    Example:
        >>> km = KMeansVarClus(k_param = 4)
        >>> km.fit(data[train_mask][cols_woe], data[train_mask][col_target])
        >>> km.draw(output_file = output_folder + '/analysis/clustering_kmeans.png')
        >>> km.display(output_file = output_folder + '/predictors/predictor_clusters_kmeans.csv')
        >>> km.bestVariables()
    """

    def __init__(self, k_param, standardize=True, pca=True, n_components=10, sample_size=0, random_state=123):
        """
        Initialization method. See args and kwargs definitions of the class.
        """
        self.k_param = k_param
        self.standardize = standardize
        self.sample_size = sample_size
        self.pca = pca
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y, **kwargs):
        """
        This method is used to fit the clusters using a training data set and training target array. Must be run before any other method.

        Args:
            X (pandas data frame): data frame with perdictors only (no target, no time or other variables which don't have the role of predictors)
            y (array): array of target variable which is coded as 0 and 1
            kwargs: arguments to be passed to PCA.

        Properties:
            * variables\_ (list): list of names of variables that entered the clustering
            * labels\_ (list): List of cluster numbers corresponding to variables from variables\_ list
            * cluster_table\_ (pandas DataFrame): One table with all variables, their cluster numbers a,nd gini
        """

        # sample the data set (for lower size in memory) and transpose it (observations -> variables, variables -> observations ... this is needed as we want to cluster variables and not observations)
        if (self.sample_size > 0) and (len(X.index) > self.sample_size):
            print('Sampling input dataset...')
            X = X.sample(self.sample_size)
            y = y[X.index]

        self.X_t = X.fillna(0).transpose()
        self.variables_ = list(self.X_t.index)

        if self.standardize:
            print('Standardazing data...')
            standardizer = StandardScaler()
            self.X_t = standardizer.fit_transform(self.X_t)

        if self.pca:
            print("Applying pca to reduce dimensions...")
            pca_reduction = PCA(n_components=self.n_components, random_state=self.random_state)
            pca_reduction.fit(self.X_t)
            self.X_t = pca_reduction.transform(self.X_t)
            self.pca_stats = dict(
                explained_variance_ratio=pca_reduction.explained_variance_ratio_,
                variance_explained=list(pca_reduction.explained_variance_),
                n_components=pca_reduction.n_components_,
                n_features=pca_reduction.n_features_
            )
        
        # fit k-means clustering from scikit learn
        print('Fitting clusters...')
        km = KMeans(n_clusters=self.k_param)
        km.fit(self.X_t)

        # save the results of clustering to internal variables
        self.labels_ = list(km.labels_)

        # calculate projection to 2 dimensions
        print('Caluclating projection to 2 dimensions...')
        pca = PCA(n_components=2).fit(self.X_t)
        self.pca_2d = pca.transform(self.X_t)

        # calculate gini because we need to find out which variables are strong in each cluster
        print('Calculating Gini of variables...')
        self.ginis_ = []
        for v in self.variables_:
            self.ginis_.append(-(2*roc_auc_score(y, X[v])-1))

        print('Calculating order of variables in clusters...')
        self.cluster_table_ = pd.DataFrame(
            {'Variable': self.variables_, 'Cluster': self.labels_, 'Gini': self.ginis_}, index=self.variables_)
        self.cluster_table_ = self.cluster_table_.sort_values(['Cluster', 'Gini'], ascending=[True, False])
        self.cluster_table_['Order'] = self.cluster_table_.sort_values(
            'Gini', ascending=False).groupby(['Cluster']).cumcount() + 1
        self.cluster_table_.drop(['Variable'], inplace=True, axis=1)

        print('Done! Use method draw() to visualise clusters, display() to display results in a table, bestVariables() to get list of the best variables from each cluster')

    def draw(self, output_file=None):
        '''
        Draws 2-dimensional projection of the variables space with color-coded clusters.

        Args:
            output_file (str, optional): folder where the chart is save to. Defaults to None.
        '''

        check_is_fitted(self, ['variables_', 'labels_', 'cluster_table_'])

        # show the projection with color encoding of clusters
        plt.scatter(self.pca_2d[:, 0], self.pca_2d[:, 1], c=self.labels_, cmap='gist_rainbow')
        height_diff = (max(self.pca_2d[:, 1])-min(self.pca_2d[:, 1]))/20
        for i, var in enumerate(self.variables_):
            plt.text(self.pca_2d[i, 0], self.pca_2d[i, 1]+height_diff, var)
        if output_file is not None:
            plt.savefig(output_file, bbox_inches='tight', dpi=72)
        plt.show()
        plt.clf()
        plt.close()

    def display(self, output_file=None):
        '''
        Displays dataframe showing cluster number and gini of each variable. Within each cluster, the variables are sorted descending by gini.

        Args:
            output_file (str, optional): filename where dataframe is saved to as csv file. Defaults to None.
        '''

        check_is_fitted(self, ['variables_', 'labels_', 'cluster_table_'])
        display(self.cluster_table_)
        if output_file is not None:
            self.cluster_table_.to_csv(output_file)

    def bestVariables(self):
        '''
        Returns list of best variables (one per each clusters, with highest gini in the clusters).

        Returns:
            list of str: best variables
        '''

        check_is_fitted(self, ['variables_', 'labels_', 'cluster_table_'])
        return list(self.cluster_table_[self.cluster_table_['Order'] == 1].index)
