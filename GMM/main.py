import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


class GMM:

    def __init__(self, df, k, iterations=1000):

        # data and hyperparameters
        self.df = df
        self.k = k
        self.iterations = iterations

        # initialize starting conditions (mean, variance and weights)
        self.initial_sigma = [df.cov().values] * k
        self.initial_w = [1 / k] * k
        # choose random initial centers
        center_indices = np.random.choice(df.index, 3, replace=False)
        self.initial_mu = [df.loc[ind, :].values for ind in center_indices]

        # results
        self.log_likelihoods, self.predictions = None, None
        self.found_clusters = False

    @staticmethod
    def calc_P(point, mus, sigmas):
        d = len(point)
        P = []
        for mu, sigma in zip(mus, sigmas):
            x_mu = (point.values - mu).reshape((1, -1))

            p = float(np.exp(-0.5 * x_mu @ np.linalg.inv(sigma) @ x_mu.T) /
                      (np.sqrt(np.linalg.det(sigma)) * ((2 * np.pi) ** (d / 2))))

            P.append(p)

        return P

    def expectation(self, mus, sigmas, ws):

        P = self.df.apply(self.calc_P, axis=1, result_type='expand', args=(mus, sigmas)).values

        q = P * ws

        log_likelihood = np.log(np.sum(q, axis=1).tolist()).sum()

        # normalize such that the sum of each row = 1
        q_norm = (q.T / np.sum(q, axis=1)).T

        n = np.sum(q_norm, axis=0)

        return q_norm, n, log_likelihood

    def maximization(self, q, n):
        w = []
        sigma = []
        mu = []

        for i in range(len(n)):
            w.append(n[i] / np.sum(n))

            mu.append(np.sum(self.df.values * q[:, i].reshape((-1, 1)), axis=0) / n[i])

            x_mu = self.df.values - mu[i]

            sigma.append(((x_mu * q[:, i].reshape((-1, 1))).T @ x_mu) / n[i])

        return w, mu, sigma

    def EM_iteration(self):

        if not self.found_clusters:

            mu = self.initial_mu
            sigma = self.initial_sigma
            w = self.initial_w
            self.log_likelihoods = []

            for i in range(self.iterations):

                # Expectation
                q, n, log_likelihood = self.expectation(mu, sigma, w)

                self.log_likelihoods.append(log_likelihood)

                # stopping criteria
                if (len(self.log_likelihoods) >= 2) and (
                        self.log_likelihoods[i] - self.log_likelihoods[i - 1] < 0.0001):
                    break

                # Maximization
                w, mu, sigma = self.maximization(q, n)

            self.predictions = np.argmax(q, axis=1)
            self.found_clusters = True

        return self.log_likelihoods, self.predictions

    def plot_clusters(self, feature_1, feature_2):
        plt.figure()
        plt.scatter(self.df.iloc[:, feature_1], self.df.iloc[:, feature_2], c=self.predictions)
        plt.xlabel(f'{self.df.columns[feature_1]}')
        plt.ylabel(f'{self.df.columns[feature_2]}')
        plt.show()

    def plot_3d_clusters(self, feature_1, feature_2, feature_3):
        ax = plt.axes(projection='3d')
        ax.scatter(self.df.iloc[:, feature_1], self.df.iloc[:, feature_2], self.df.iloc[:, feature_3],
                   c=self.predictions)
        ax.set_xlabel(f'{self.df.columns[feature_1]}')
        ax.set_ylabel(f'{self.df.columns[feature_2]}')
        ax.set_zlabel(f'{self.df.columns[feature_3]}')
        plt.show()

    def plot_likelihood(self):
        plt.figure()
        plt.plot(self.log_likelihoods)
        plt.xlabel('Iterations')
        plt.ylabel('Log Likelihood')
        plt.show()


if __name__ == '__main__':
    # get data
    data = load_iris()
    iris = pd.DataFrame(data.data, columns=data.feature_names)

    # hyperparameters
    n_clusters = 3
    n_iterations = 1000

    # initialize
    cls = GMM(iris, n_clusters, n_iterations)
    # cluster
    log_liklihoods, predictions = cls.EM_iteration()
    # plot
    cls.plot_clusters(0, 2)
    cls.plot_likelihood()
    cls.plot_3d_clusters(0, 1, 2)
