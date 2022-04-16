import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


class Node:

    def __init__(self, df, labels, depth=2, num_rand_features=None, forest=False):

        self.df = df
        self.labels = labels
        self.depth = depth
        self.leaf = False
        self.feature, self.value = None, None
        self.left_node, self.right_node = None, None
        self.prediction = None
        self.forest = forest
        self.num_rand_features = num_rand_features
        if self.forest and (self.num_rand_features is None):
            self.num_rand_features = np.floor(np.sqrt(df.shape[1]))

    def split_mask(self, feature_ind, value):

        return self.df[feature_ind] < value

    def find_split_points(self, values):
        """
        find all possible split points to divide a single feature's values
        """

        sorted_col = np.sort(np.unique(values))
        # split points are the average values between the sorted points of the column:
        split_points = [np.mean(sorted_col[i:i + 2]) for i in range(len(sorted_col) - 1)]
        # add edge points:
        split_points.insert(0, sorted_col[0] - 1)
        split_points.append(sorted_col[-1] + 1)

        return split_points

    def get_split(self):
        """
        finds the ideal feature and split point to divide the data based on gini score
        """

        # if we are building a forest, use a random subset of features for spliting:
        if self.forest:
            features = np.random.choice(self.df.columns, self.num_rand_features, replace=False)
        else:
            features = self.df.columns

        # initialize lowest_gini with an unrealisticly high value:
        lowest_gini = 1.0

        # iterate over available features
        for col in features:

            split_points = self.find_split_points(self.df[col].values)

            # iterate over all possible split points
            for point in split_points:

                mask = self.split_mask(col, point)

                left_labels = self.labels[mask]
                _, freq = np.unique(left_labels, return_counts=True)
                gini_left = 1 - ((freq / len(left_labels)) ** 2).sum()

                right_labels = self.labels[~mask]
                _, freq = np.unique(right_labels, return_counts=True)
                gini_right = 1 - ((freq / len(right_labels)) ** 2).sum()

                average_gini = (len(left_labels) * gini_left + len(right_labels) * gini_right) / len(self.df[col])

                if average_gini < lowest_gini:
                    lowest_gini = average_gini
                    best_feature = col
                    best_value = point

        return best_feature, best_value

    def split_one_node(self, feature, value):
        """
        divide the dataset according to the specified feature and value
        """
        mask = self.split_mask(feature, value)

        left = self.df[mask]
        right = self.df[~mask]
        left_labels = self.labels[mask]
        right_labels = self.labels[~mask]

        return left, right, left_labels, right_labels

    def is_leaf(self, max_depth, left, right):
        """
        check if we have reached maximum depth or the dataset can no longer be divided
        """
        if (self.depth >= max_depth) | (left.shape[0] == 0) | (right.shape[0] == 0):
            return True

        return False

    def build_tree(self, max_depth):
        """
        recursively construct the tree
        """
        self.feature, self.value = self.get_split()
        left, right, left_labels, right_labels = self.split_one_node(self.feature, self.value)
        del self.df

        # if a leaf is reached, set its value to the most common label
        if self.is_leaf(max_depth, left, right):
            self.leaf = True
            self.prediction = self.labels.value_counts().idxmax()
            del self.labels

        # if a leaf is not reached, define a left and right node and continue building the tree
        else:

            del self.labels

            self.left_node = Node(left, left_labels, self.depth + 1, self.num_rand_features, self.forest)
            self.left_node.build_tree(max_depth)

            self.right_node = Node(right, right_labels, self.depth + 1, self.num_rand_features, self.forest)
            self.right_node.build_tree(max_depth)

    def predict_one(self, sample):
        """
        predict the outcome of one input sample by following the tree until a leaf is reached
        """
        while True:

            if self.leaf:
                return self.prediction

            elif sample[self.feature] < self.value:
                return self.left_node.predict_one(sample)
            else:
                return self.right_node.predict_one(sample)

    def predict(self, test):
        """
        predict the outcome for several test samples
        """
        return test.apply(self.predict_one, axis=1)


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
        weights = pd.Series(1 / len(y_train), index=y_train.index, dtype='float64')
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
        stump_predictions = pd.DataFrame(index=X_test.index, columns=[f'Stump {i}' for i in range(1, len(self.stumps) + 1)])

        # get the prediction for each stump:
        for col, stump in enumerate(self.stumps):
            stump_predictions.iloc[:, col] = stump.predict(X_test)

        # return the sign of the sum of weighted predictions
        return stump_predictions.apply(lambda row: np.sign(np.sum(row * self.alpha_t)), axis=1)


if __name__ == "__main__":

    # get the data:
    data = pd.read_csv('heart.csv')

    data['target'] = data['target'].map({0: -1, 1: 1})
    data = pd.get_dummies(data, drop_first=True, columns=['cp', 'restecg', 'slope', 'ca', 'thal'])

    X = data.drop(['target'], axis=1)
    y = data['target']

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # build adaboost stumps:
    num_stumps = 3
    clf = AdaBoost(x_train, y_train, num_stumps)
    clf.build_stumps()

    # predict:
    outcome = clf.predict(x_test)

    accuracy = np.sum(outcome == y_test) / outcome.shape[0]
    print(f"adaboost model accuracy is {accuracy * 100:2.4}%")
