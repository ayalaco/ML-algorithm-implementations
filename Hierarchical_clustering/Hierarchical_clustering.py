import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.spatial.distance import cdist
import matplotlib.colors as mcolors


class HierarchicalClustering:

    def __init__(self, df, min_clust=1, max_iter=None, max_dist=None):
        self.df = df
        self.min_clust = min_clust
        self.max_iter = max_iter
        self.max_dist = max_dist

        self.distance_matrix = cdist(df, df)
        self.initial_clusters = [[ind] for ind in df.index]
        self.cluster_levels = [self.initial_clusters]

        self.found_clusters = False

    def get_closest_clusters(self):
        # Create a copy with np.inf instead of 0 on the diagonal
        dist_inf = self.distance_matrix + np.diag([np.inf] * len(self.distance_matrix))
        # find and return the two clusters with minimum distance
        return np.unravel_index(np.argmin(dist_inf, axis=None), dist_inf.shape)

    def update_dist_mat(self):
        # get clusters with minimal distance
        cluster1, cluster2 = self.get_closest_clusters()

        # get minimal distance
        min_dist = self.distance_matrix[cluster1, cluster2]

        # calculate distances from new joint cluster
        ind_list = [x for x in range(len(self.distance_matrix)) if (x != cluster1 and x != cluster2)]
        joint_dist = []
        for ind in ind_list:
            joint_dist.append(np.max([self.distance_matrix[cluster1, ind], self.distance_matrix[cluster2, ind]]))
        else:
            joint_dist.append(0)
            joint_dist = np.array(joint_dist)

        # remove joined clusters from matrix (both rows and columns)
        self.distance_matrix = np.delete(self.distance_matrix, np.max([cluster1, cluster2]), 0)
        self.distance_matrix = np.delete(self.distance_matrix, np.min([cluster1, cluster2]), 0)
        self.distance_matrix = np.delete(self.distance_matrix, np.max([cluster1, cluster2]), 1)
        self.distance_matrix = np.delete(self.distance_matrix, np.min([cluster1, cluster2]), 1)

        # add new cluster to the distance matrix
        self.distance_matrix = np.concatenate((self.distance_matrix, joint_dist[:-1].reshape(1, -1)), axis=0)
        self.distance_matrix = np.concatenate((self.distance_matrix, joint_dist.reshape(-1, 1)), axis=1)

        return cluster1, cluster2, min_dist

    def update_cluster_list(self, cluster1, cluster2):

        previouse_clusters = self.cluster_levels[-1]

        new_clusters = previouse_clusters.copy()

        # append joint cluster at the end
        new_clusters.append(previouse_clusters[cluster1] + previouse_clusters[cluster2])

        # remove the previously separate clusters
        new_clusters.pop(np.max([cluster1, cluster2]))
        new_clusters.pop(np.min([cluster1, cluster2]))

        # append new cluster list to the cluster levels list
        self.cluster_levels.append(new_clusters)

    def stop_crit(self, n_clusters, n_iter, distance):
        if self.max_dist and distance >= self.max_dist:
            return True
        elif self.max_iter and n_iter >= self.max_iter:
            return True
        elif n_clusters <= self.min_clust + 1:
            return True
        else:
            return False

    def find_clusters(self):

        if not self.found_clusters:

            stop = False
            n_iter = 0

            while not stop:
                n_iter += 1
                n_clusters = len(self.cluster_levels[-1])
                cluster1, cluster2, distance = self.update_dist_mat()
                self.update_cluster_list(cluster1, cluster2)

                stop = self.stop_crit(n_clusters, n_iter, distance)

            self.found_clusters = True

        return self.cluster_levels

    def plot_clusters(self, col1_ind, col2_ind):

        final_clusters = self.cluster_levels[-1]

        colors = list(mcolors.TABLEAU_COLORS)

        plt.figure()

        for clust, color in zip(final_clusters, colors[:len(final_clusters)]):
            plt.scatter(self.df.iloc[clust, col1_ind], self.df.iloc[clust, col2_ind], c=color)

        plt.xlabel(f'{self.df.columns[col1_ind]}')
        plt.ylabel(f'{self.df.columns[col2_ind]}')
        plt.xlim(np.min(self.df.iloc[:, col1_ind]), np.max(self.df.iloc[:, col1_ind]))
        plt.ylim(np.min(self.df.iloc[:, col2_ind]), np.max(self.df.iloc[:, col2_ind]))
        plt.show()


if __name__ == '__main__':

    data = load_iris(as_frame=True)
    iris = data.data

    hier = HierarchicalClustering(iris, min_clust=3, max_iter=None, max_dist=None)
    hier.find_clusters()
    hier.plot_clusters(0, 1)
