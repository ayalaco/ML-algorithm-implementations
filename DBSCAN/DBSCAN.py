import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from itertools import product


class DBSCANAlgo:

    def __init__(self, data, epsilon, min_pts):

        self.data = data
        self.epsilon = epsilon
        self.minPts = min_pts
        self.visited = np.zeros(self.data.shape[0])
        self.distance = cdist(self.data, self.data)
        self.clusters = []
        self.predictions = np.zeros(self.data.shape[0]) - 1

    def region_query(self, p_idx):
        # return indices of relevant points
        relevant_dist = self.distance[p_idx]
        return np.argwhere(relevant_dist <= self.epsilon).reshape((-1)).tolist()

    def expand_cluster(self, sphere_points):

        self.clusters[-1].extend(sphere_points)

        for i in sphere_points:

            if self.visited[i] == 0:
                self.visited[i] = 1

                new_sphere = self.region_query(i)
                if len(new_sphere) >= self.minPts:
                    self.expand_cluster(new_sphere)

    def fit(self):

        for ind in range(self.data.shape[0]):
            if self.visited[ind] == 0:
                self.visited[ind] = 1

                sphere = self.region_query(ind)
                if len(sphere) >= self.minPts:
                    self.clusters.append([])
                    self.expand_cluster(sphere)

    def predict(self):
        # return as 1D-array instead of list of lists. outliers remain labeled as -1.
        for ind, clust in enumerate(self.clusters):
            self.predictions[clust] = ind

        return self.predictions


def plot_clusters(data, predictions, feature1_ind, feature2_ind, feature1_title=None, feature2_title=None, title=None):
    plt.figure()
    # plot all clustered points
    plt.scatter(data[predictions != -1, feature1_ind],
                data[predictions != -1, feature2_ind], c=predictions[predictions != -1],
                label='clusters')
    # plot noise points
    plt.scatter(data[predictions == -1, feature1_ind],
                data[predictions == -1, feature2_ind], c='black', marker='x',
                label='noise')
    plt.legend()
    plt.xlabel(f'{feature1_title}')
    plt.ylabel(f'{feature2_title}')
    plt.xlim(np.min(data[:, feature1_ind]) - 0.2, np.max(data[:, feature1_ind]) + 0.2)
    plt.ylim(np.min(data[:, feature2_ind]) - 0.2, np.max(data[:, feature2_ind]) + 0.2)
    plt.title(title)
    plt.show()


def tune_hyperparameters(data, eps, minPts, skl=False):
    best_score = -10
    best_eps, best_minPt, best_clusters = None, None, None

    for params in product(eps, minPts):

        if skl:    # use sklearn algorithm
            dbscan = DBSCAN(eps=params[0], min_samples=params[1]).fit(data)
            labels = dbscan.labels_
        else:      # use our algorithm
            dbscan = DBSCANAlgo(data, *params)
            dbscan.fit()
            labels = dbscan.predict()

        # determine best parameters based on silhouette score
        score = silhouette_score(data, labels, metric='euclidean')
        if score > best_score:
            best_score = score
            best_eps, best_minPt = params
            best_clusters = labels

    return best_clusters, best_eps, best_minPt, best_score


if __name__ == '__main__':
    # get and scale data
    data = load_iris(as_frame=True)
    iris = data.data
    scaler = StandardScaler()
    scaled_iris = scaler.fit_transform(iris)

    # hyperparameters
    epsilons = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
    min_points = [2, 3, 4, 5, 6]

    # Use our DBSCAN algorithm
    labels, best_epsilon, best_minPts, score = tune_hyperparameters(
        scaled_iris, epsilons, min_points)

    # plot results
    title = 'DBSCAN from scratch'
    plot_clusters(scaled_iris, labels, 0, 2, feature1_title=iris.columns[0],
                  feature2_title=iris.columns[2], title=title)

    # Use sklearn
    labels_skl, best_epsilon_skl, best_minPts_skl, score_skl = tune_hyperparameters(
        scaled_iris, epsilons, min_points, skl=True)

    # plot results
    title = 'DBSCAN from sklearn'
    plot_clusters(scaled_iris, labels_skl, 0, 2, feature1_title=iris.columns[0],
                  feature2_title=iris.columns[2], title=title)
