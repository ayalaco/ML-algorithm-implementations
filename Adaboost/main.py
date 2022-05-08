import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from adaboost import AdaBoost

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
