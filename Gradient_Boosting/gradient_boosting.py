import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def train_trees(X, y, n_trees, n_leaves, lr):

    # the first tree is just a leaf with the mean value
    first_leaf = np.mean(y)

    y_predict = pd.Series(first_leaf, index=y.index)
    trees = []

    # build subsequent trees. The labels are constantly replaced by the updated residuals.
    for i in range(n_trees):

        # update residuals
        residuals = y - y_predict

        # train
        tree = DecisionTreeRegressor(max_leaf_nodes=n_leaves)
        tree.fit(X, residuals)
        trees.append(tree)

        # predict
        r_predict = tree.predict(X)

        # add the predicted residuals to the overall prediction:
        y_predict += lr * r_predict

    return first_leaf, trees


def predict(x_test, first_leaf, trees, lr):
    tree_predictions = pd.DataFrame(index=x_test.index, columns=[f'Tree {i}' for i in range(1, len(trees) + 1)])

    # calculate the prediction from each tree
    for ind, tree in enumerate(trees):
        tree_predictions.iloc[:, ind] = tree.predict(x_test)

    # sum the predictions for the residuals
    sum_r = tree_predictions.sum(axis=1)

    # return final prediction
    return first_leaf + lr * sum_r


if __name__ == '__main__':
    # get the data
    data = pd.read_csv('Fish.csv')

    data = pd.get_dummies(data, drop_first=True, columns=['Species'])

    X = data.drop(['Weight'], axis=1)
    y = data['Weight']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # set parameters
    num_trees = 100
    num_leaves = 8
    lr = 0.1

    # train
    initial_leaf, trees = train_trees(X_train, y_train, num_trees, num_leaves, lr)

    # predict
    outcome = predict(X_test, initial_leaf, trees, lr)

    # get metrics
    rmse = np.sqrt(mean_squared_error(y_test, outcome))
    r_sqrd = 1 - (((y_test - outcome) ** 2).sum()) / (((y_test - y_test.mean()) ** 2).sum())

    # plot
    plt.figure()
    plt.scatter(y_test, outcome)
    plt.xlabel('Observed Weight')
    plt.ylabel('Predicted Weight')
    plt.title(f'RMSE score: {rmse:2.4}   ,    R^2 score: {r_sqrd:2.4}')
    plt.show()
