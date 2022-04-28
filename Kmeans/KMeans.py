import numpy as np
from scipy.spatial.distance import cdist
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class KMeans:

    def __init__(self, data, k, threshold):
        self.data = data
        self.k = k
        self.threshold = threshold

        self.final_clusters = None
        self.final_centers = None

    def assign_cluster(self, centers):
        # calculate the distance of each point from each center
        dist_mat = cdist(self.data, centers)
        # return the index of the nearest center to each point
        return dist_mat.argmin(axis=1)

    def calc_mean(self, clusters):
        # calculate the mean of each cluster
        centers = []
        for i in range(self.k):
            centers.append(self.data[clusters == i].mean(axis=0))
        return centers

    def convergence_check(self, curr_centers, prev_centers):
        # claculate the change in each of the centers' position
        dist = np.diag(cdist(curr_centers, prev_centers))
        # return false as long as one of the centers moves more than the threshold.
        return (dist > self.threshold).sum() == 0

    def cluster_all(self):

        converged = False

        # randomly initialize the first centers
        curr_centers = np.random.rand(self.k * self.data.shape[1]).reshape((self.k, self.data.shape[1]))
        # assign each point to the its nearest center
        clust_list = self.assign_cluster(curr_centers)

        # repeat until convergence
        while not converged:
            prev_centers = curr_centers

            # calculate the means of the clusters
            curr_centers = self.calc_mean(clust_list)

            clust_list = self.assign_cluster(curr_centers)

            converged = self.convergence_check(curr_centers, prev_centers)

        self.final_clusters = clust_list
        self.final_centers = curr_centers

        return self.final_clusters, self.final_centers

    def plot_clusters(self, clusters, feature_1, feature_2):
        plt.figure()
        plt.scatter(self.data[:, feature_1], self.data[:, feature_2], c=clusters)
        plt.xlabel(f'Feature {feature_1}')
        plt.ylabel(f'Feature {feature_2}')
        plt.show()


if __name__ == '__main__':
    # get data
    iris = load_iris().data

    # scale data
    scaler = StandardScaler()
    scaler.fit(iris)
    scaled_iris = scaler.transform(iris)

    # set threshold and number of clusters
    num_k = 3
    threshold = 0.1

    # assign clusters
    kmn = KMeans(scaled_iris, num_k, threshold)
    clusters, centers = kmn.cluster_all()

    # plot
    kmn.plot_clusters(clusters, 0, 1)
