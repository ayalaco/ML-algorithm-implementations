import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

    
class DiabetesPredictor:
    
    def __init__(self, X, y):
        self.X_with_diab, self.y_with_diab = X[y == 1], y[y == 1]  
        self.X_no_diab, self.y_no_diab = X[y == 0], y[y == 0]
        
        self.total_len = len(y)
        
        self.prior_with_diab, self.prior_no_diab = None, None
        self.stats_with_diab, self.stats_no_diab = None, None
              
    def fit(self):
        # priors:
        self.prior_with_diab = len(self.y_with_diab)/self.total_len
        self.prior_no_diab = len(self.y_no_diab)/self.total_len
        
        # mean and std of each feature (for likelihood calculation):   
        self.stats_with_diab = self.X_with_diab.agg(['mean', 'std'])
        self.stats_no_diab = self.X_no_diab.agg(['mean', 'std'])

    def calc_gauss_prob(self, mu, sig, v):

        return np.exp(-((v-mu)**2)/(2*sig**2))/np.sqrt(2*np.pi*sig**2)

    def estimate_probabilty(self, stat_vec, prior, samples):
   
        mu = stat_vec.loc['mean'].values
        sig = stat_vec.loc['std'].values
        v = samples.values
        
        likelihoods = self.calc_gauss_prob(mu, sig, v)
        return np.prod(likelihoods, axis=1) * prior    
    
    def predict(self, v):
        
        prob_with_diab = self.estimate_probabilty(self.stats_with_diab, self.prior_with_diab, v)
        prob_no_diab = self.estimate_probabilty(self.stats_no_diab, self.prior_no_diab, v)
              
        return (prob_with_diab > prob_no_diab).astype(int)
        
        
if __name__ == "__main__":

    data = pd.read_csv(r'diabetes.csv')
    y = data['Outcome']
    X = data.drop(columns='Outcome')
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    diab = DiabetesPredictor(x_train, y_train)
    diab.fit()
    preds = diab.predict(x_test)
    accuracy = np.sum(preds == y_test.values)/len(preds)
    
    print(f"The test accuracy is {accuracy*100:2.4}%")
