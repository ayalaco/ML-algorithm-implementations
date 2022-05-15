import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Random_Forest import RandomForest

if __name__ == '__main__':
    # get data
    data = pd.read_csv('wdbc.data')
    data.columns = ['ID number', 'Diagnosis'] + [f'feature{i}' for i in range(1, data.shape[1] - 1)]

    X = data.iloc[:, 2:]
    y = data.iloc[:, 1].map({'B': 0, 'M': 1})

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # build a forest and predict
    size_subsample = int(len(y_train) / 3)
    num_rand_features = int(np.sqrt(x_train.shape[1]))
    num_trees = 50
    max_depth = 2

    forest = RandomForest(x_train, y_train, size_subsample, num_rand_features, num_trees, max_depth)
    forest.fit()

    predictions = forest.predict(x_test)

    accuracy = np.sum(predictions == y_test) / predictions.shape[0]
    print(f"Decision tree accuracy is {accuracy * 100:2.4}%")
