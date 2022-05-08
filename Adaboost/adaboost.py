import numpy as np
import pandas as pd
from tree import Node


class AdaBoost:

    def __init__(self, X, y, n_stumps):
        self.X = X
        self.y = y
        self.n_stumps = n_stumps
        self.alpha_t = None
        self.stumps = None

    @staticmethod
    def get_weighted_ind(weights):
        """
        Chooses a random index. an index that corresponds to a large weight is more likely to be chosen.
        """
        sum_weight = 0

        # generate random number between 0-1:
        num = np.random.rand()

        # return the first index in which the sum of weights reaches the random number:
        for ind, weight in enumerate(weights):
            sum_weight += weight
            if sum_weight >= num:
                return ind

    def build_weighted_df(self, df, labels, weights):
        """
        samples the inputed df according to the given weights.
        returns a new df (and labels) of the same size, but with more misclassified points.
        """
        new_df = pd.DataFrame(columns=df.columns)
        new_labels = []

        for i in range(df.shape[0]):
            ind = self.get_weighted_ind(weights)
            new_df = new_df.append(df.iloc[ind], ignore_index=True)
            new_labels.append(labels.iloc[ind])

        return new_df, pd.Series(new_labels)

    def build_stumps(self):
        max_depth = 1

        # initialize uniform weights:
        weights = pd.Series(1 / len(self.y), index=self.y.index, dtype='float64')
        weighted_x = self.X
        weighted_y = self.y

        self.alpha_t = []
        self.stumps = []

        for stump in range(self.n_stumps):

            # build stump
            root = Node(weighted_x, weighted_y, 0)
            root.build_tree(max_depth)
            self.stumps.append(root)

            # asses stump's miscalssification error
            predict = root.predict(weighted_x)
            t_error = predict != weighted_y
            error = np.sum(weights * t_error)

            # prevent division by zero error
            if error == 0:
                error = 0.00001

            # calculate the stump's alpha parameter
            self.alpha_t.append(np.log((1 - error) / error))

            # update weights to put more emphasis on misclassified points:
            weights *= np.exp(self.alpha_t[-1] * t_error.map({False: -1, True: 1}))
            # normalize weights:
            weights = weights / weights.sum()

            # Build a new dataframe by sampling the previouse one, but with a larger emphasis on the misclassified
            # points. This is an alternative to using weights in a gini calculation:
            weighted_x, weighted_y = self.build_weighted_df(weighted_x, weighted_y, weights)

            # reset to uniform weights for the new weighted dataframe:
            weights = pd.Series(1 / len(weighted_y), index=weighted_y.index, dtype='float64')

    def predict(self, X_test):
        stump_predictions = pd.DataFrame(index=X_test.index,
                                         columns=[f'Stump {i}' for i in range(1, len(self.stumps) + 1)])

        # get the prediction for each stump:
        for col, stump in enumerate(self.stumps):
            stump_predictions.iloc[:, col] = stump.predict(X_test)

        # return the sign of the sum of weighted predictions
        return stump_predictions.apply(lambda row: np.sign(np.sum(row * self.alpha_t)), axis=1)
