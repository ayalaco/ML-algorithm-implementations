import numpy as np
import pandas as pd

class Node:

    def __init__(self, df: pd.DataFrame, labels: pd.Series, depth=2, num_rand_features=None, forest=False):

        self.df = df
        self.labels = labels
        self.depth = depth
        
        # to be filled during training
        self.leaf = False
        self.feature, self.value = None, None
        self.left_node, self.right_node = None, None
        self.prediction = None
        
        # For random forest calculations
        self.forest = forest
        self.num_rand_features = num_rand_features
        if self.forest and (self.num_rand_features is None):
            self.num_rand_features = np.floor(np.sqrt(df.shape[1]))

    @staticmethod
    def find_split_points(values: np.ndarray) -> list:
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

    def split_mask(self, feature_ind: str, value: np.float) -> pd.Series:

        return self.df[feature_ind] < value

    def get_split(self) -> (str, np.float):
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

                # calculate gini for the left node
                left_labels = self.labels[mask]
                _, freq = np.unique(left_labels, return_counts=True)
                gini_left = 1 - ((freq / len(left_labels)) ** 2).sum()

                # calculate gini for the right node
                right_labels = self.labels[~mask]
                _, freq = np.unique(right_labels, return_counts=True)
                gini_right = 1 - ((freq / len(right_labels)) ** 2).sum()

                # calculate average gini
                average_gini = (len(left_labels) * gini_left + len(right_labels) * gini_right) / len(self.df[col])

                if average_gini < lowest_gini:
                    lowest_gini = average_gini
                    best_feature = col
                    best_value = point

        return best_feature, best_value

    def split_one_node(self, feature: str, value: np.float):
        """
        divide the dataset according to the specified feature and value
        """
        mask = self.split_mask(feature, value)

        left = self.df[mask]
        right = self.df[~mask]
        left_labels = self.labels[mask]
        right_labels = self.labels[~mask]

        return left, right, left_labels, right_labels

    def is_leaf(self, max_depth, left: pd.DataFrame, right: pd.DataFrame):
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
        if self.leaf:
            return self.prediction

        elif sample[self.feature] < self.value:
            return self.left_node.predict_one(sample)
        else:
            return self.right_node.predict_one(sample)

    def predict(self, test: pd.DataFrame):
        """
        predict the outcome for several test samples
        """
        return test.apply(self.predict_one, axis=1)
