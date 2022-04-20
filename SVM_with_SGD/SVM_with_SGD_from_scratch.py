import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class SVM:

    def __init__(self, n_iter=100, C=0.001):

        self.n_iter = n_iter
        self.C = C
        self.w = None
        self.L_log = None

    @staticmethod
    def add_intercept(x):
        return x.assign(intercept=1)

    def calc_loss(self, x, y):
        loss_assist = np.column_stack((1 - y * (x @ self.w), np.zeros(len(y))))
        loss = np.sum(np.max(loss_assist, axis=1)) / len(y)
        regularization = self.C * np.linalg.norm(self.w) ** 2
        return loss + regularization

    def fit(self, x, y):

        # add intercept column
        x = self.add_intercept(x)

        # generate initial weights equal to 0
        self.w = np.zeros(x.shape[1])

        self.L_log = []

        for i in range(self.n_iter):

            eta = 1 / (self.C * (i + 1))
            
            # the gradient for the regularization term:
            grad_reg = 2 * self.C * self.w

            # choose one random sample for stochastic gradient descent
            ind = np.random.randint(x.shape[0])

            # get prediction
            h = x.iloc[ind] @ self.w

            # update weights (depends on classification)
            if y.iloc[ind] * h >= 1:
                # if correctly classified
                self.w -= eta * grad_reg
            else:
                # if missclassified
                self.w -= eta * (grad_reg - y.iloc[ind] * x.iloc[ind])

            L = self.calc_loss(x, y)
            self.L_log.append(L)

    def predict(self, x):
        # add intercept column
        x = self.add_intercept(x)
        return np.sign(x @ self.w)

    def score(self, x, y):
        return np.sum(self.predict(x) == y) / y.shape[0]

    def plot_loss(self):
        plt.figure()
        plt.plot(self.L_log)
        plt.xlabel('iterations')
        plt.ylabel('Loss')
        plt.title(f"Iterations = {self.n_iter}, Regularization parameter = {self.C}")
        plt.xlim(0, len(self.L_log))
        plt.ylim(0, max(self.L_log))
        plt.show()


if __name__ == '__main__':
    # get data
    data = pd.read_csv('diabetes.csv')
    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1].map({1: 1, 0: -1})

    # scale data
    scaler = StandardScaler()
    scaler.fit(X)
    X_norm = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(X_norm, Y, test_size=0.2)

    # hyper parameters:
    C_param = 0.0001
    iterations = 500

    # train the data:
    svm = SVM(n_iter=iterations, C=C_param)
    svm.fit(x_train, y_train)

    # predict for train set
    train_prediction = svm.predict(x_train)
    train_accuracy = svm.score(x_train, y_train)
    print(f"Train accuracy is: {train_accuracy * 100:2.4}%")

    # predict for test set
    test_prediction = svm.predict(x_test)
    test_accuracy = svm.score(x_test, y_test)
    print(f"Test accuracy is: {test_accuracy * 100:2.4}%")

    # plot loss
    svm.plot_loss()
