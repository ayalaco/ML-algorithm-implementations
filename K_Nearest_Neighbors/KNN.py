import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from scipy.spatial.distance import cdist

class KNN:
    
    def __init__(self, X, y, k):
        self.X = X
        self.y = y
        self.k = k

    def get_k_nearest(self, X_test):
        '''
        returns k nearest neighbor's indices for each point in the test set.
        '''
        dist = cdist(X_test, self.X)
            
        sort_index = np.argsort(dist, axis=1)
        k_nearest_ind = sort_index[:, :self.k]

        return k_nearest_ind
        
    def predict(self, X_test):

        k_nearest = self.get_k_nearest(X_test)

        predictions = []
        for i in range(k_nearest.shape[0]):
            
            nearest_labels = self.y[k_nearest[i]]
            unique_labels, counts = np.unique(nearest_labels, return_counts=True)
            # set the prediction to the most common label
            predictions.append(unique_labels[np.argmax(counts)])
            

        return predictions


if __name__ == '__main__':

    # load data
    data, target = load_iris(as_frame=True, return_X_y=True)

    # split to train and test sets
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)

    # reset index for convenience
    for xy_set in [x_train, x_test, y_train, y_test]:
        xy_set.reset_index(drop=True, inplace=True)

    # train and predict
    k = 9
    iris = KNN(x_train, y_train, k)
    predictions = iris.predict(x_test)
    accuracy = np.sum(predictions == y_test.values)/len(predictions)

    print(f"The accuracy of the model with {k} nearest Neighbors is {accuracy*100:2.4}%")
 
 