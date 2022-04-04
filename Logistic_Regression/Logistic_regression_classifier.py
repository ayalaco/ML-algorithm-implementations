import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:

    def __init__(self, X, y, alpha=0.1, iterations=10):
        self.X = X
        self.y = y
        self.alpha = alpha
        self.iterations = iterations
        self.final_theta = None
        self.loss_log = None

    def initialize_theta(self):
        return np.random.rand(self.X.shape[1])

    def calculate_logit(self, z):
        return 1 / (1 + np.exp(-z))

    def negative_log_likelihood(self, h):
        return -np.sum(self.y * np.log(h) + (1 - self.y) * np.log(1 - h)) / len(self.y)

    def calculate_gradient(self, h):
        return self.X.T @ (h - self.y) / len(self.y)

    def fit(self):
        theta = self.initialize_theta()

        L_log = []
        for i in range(self.iterations):
            h = self.calculate_logit(self.X @ theta)

            L = self.negative_log_likelihood(h)
            L_log.append(L)

            grad_L = self.calculate_gradient(h)

            theta -= self.alpha * grad_L

        self.final_theta = theta
        self.loss_log = L_log

        return self.loss_log

    def plot_loss(self):
        plt.figure()
        plt.plot(self.loss_log)
        plt.xlabel('iterations')
        plt.ylabel('Loss')
        plt.title(f"alpha = {self.alpha}")
        plt.xlim(0, len(self.loss_log))
        plt.ylim(0, max(self.loss_log))
        plt.show()

    def predict(self, X_test):
        prob = self.calculate_logit(X_test @ self.final_theta)
        return (prob > 0.5).astype(int)


# Homebrew train-test split
def train_test_split(data, train_percent):
    data = data.sample(frac=1).reset_index(drop=True)
    split_ind = int(np.ceil(train_percent * data.shape[0]))
    trainset = data.iloc[:split_ind, :]
    testset = data.iloc[split_ind:, :]
    return trainset, testset


if __name__ == "__main__":
    iris = pd.read_csv(r"https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
    iris.columns = ['sepal_length (cm)', 'sepal_width (cm)', 'petal_length (cm)', 'petal_width (cm)', 'class']

    # We'll make this a binary problem: "Iris-setosa” - 1, or not "Iris-setosa” - 0
    iris['label'] = 0
    iris.loc[iris['class'] == 'Iris-setosa', 'label'] = 1
    iris = iris.drop(columns='class')

    # train-test split:
    train, test = train_test_split(iris, 0.7)

    x_train = train.iloc[:, :-1].values
    y_train = train.iloc[:, -1].values
    x_test = test.iloc[:, :-1].values
    y_test = test.iloc[:, -1].values

    # set parameters:
    alpha = 0.1
    iterations = 30

    # fit
    clf = LogisticRegression(x_train, y_train, alpha, iterations)
    L_log1 = clf.fit()
    clf.plot_loss()

    # predict
    prediction = clf.predict(x_test)
    test_accuracy = np.sum(prediction == y_test) / y_test.shape[0]
    print(f"The test accuracy is {test_accuracy * 100}%")
