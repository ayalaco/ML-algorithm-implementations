import pandas as pd
from Tree import Node


class RandomForest:

    def __init__(self, dataset, labels, size_subsample, num_rand_features, num_trees, max_depth):

        self.dataset = dataset
        self.labels = labels
        self.size_subsample = size_subsample
        self.num_rand_features = num_rand_features
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.trees = None

    def create_subsample(self):
        """
        generate random subsample
        """
        sub_data = self.dataset.sample(n=self.size_subsample, replace=True)
        sub_labels = self.labels[sub_data.index]

        return sub_data, sub_labels

    def fit(self):
        """
        populate the forest with trees built based on different random sub-samples
        """
        self.trees = []

        for tree in range(self.num_trees):
            # generate random subsample from the dataset
            subsample_x, subsample_y = self.create_subsample()

            # build a tree for the subset:
            root = Node(subsample_x, subsample_y, depth=0, num_rand_features=self.num_rand_features, forest=True)
            root.build_tree(self.max_depth)

            self.trees.append(root)

    def predict(self, X_test):
        """
        predict an outcome for all trees, then choose the most common prediction.
        """
        predictions = pd.DataFrame(index=X_test.index, columns=[f'Tree {i}' for i in range(1, len(self.trees) + 1)])

        # Get a prediction from each tree in the forest
        for col, root in enumerate(self.trees):
            predictions.iloc[:, col] = root.predict(X_test)

        # choose the most common prediction
        prediction_mode = predictions.mode(axis=1)

        # deal with 50-50 probabilities (won't happen for on odd number of trees):
        if prediction_mode.shape[1] == 2:
            ind_ambig = prediction_mode[1].notnull()
            prediction_mode.iloc[ind_ambig, 0] = 1  # better to have a false positive in this case
            prediction_mode.drop(columns=1, inplace=True)

        return prediction_mode.squeeze()
