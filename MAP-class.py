from sklearn import datasets
import scipy.stats as stats
import scipy.optimize as opt

import numpy as np
import matplotlib.pyplot as plt


iris = datasets.load_iris()
Xapp = iris.data
yapp = iris.target


class MAP :
    
    def fit(self, X, y):
        X = Xapp
        self.dim = np.shape(X)
        n = self.dim[0]
        m = self.dim[1]
        self.classes = np.unique(y)
        res = type('res', (), {})()
        res.pi = []
        res.mu = []
        res.var = []
        
        
        for k in self.classes:
            # working on data from class k
            posk = [i for i in range(len(y)) if target[i] == k]
            Xk = X[posk,:]
            pik = len(posk)/n
            muk = Xk.mean(axis = 0)
            vark = Xk.var(axis = 0)

            res.pi.append(pik)
            res.mu.append(muk)
            res.var.append(vark)
            print("-------------")
            print("Proba apriori for class k = " +
                  str(k) + " is " + str(pik))
            print("Means of each columns for class k = " + 
                  str(k) + " is " + str(muk))
            print("Var of each columns for class k = " + 
                  str(k) + " is " + str(vark))
        
        self.model = res
    
    def maxloglikelihood(self, X):
        posterior = []
        
        for k in self.classes:
            # get model's parameters for class k
            pik = self.model.pi[k]
            muk = self.model.mu[k]
            vark = self.model.var[k]
            pdfk = 0
            
            # estimate f(x|y)
            for i in range(self.dim[1]):
                pdfk = pdfk + stats.norm.logpdf(X[i], muk[i], vark[i])
            
            posterior.append(pdfk)
        
        return posterior.index(np.max(posterior))
            
    def predict(self, X):
        # should return the class of X, based on MLE
        # have to sum the log of normal, 
        # using mean and variance calculated in fit function
        mpa = self.maxloglikelihood(X)
        print("\n ------- \n")
        print("The preidction for " + str(X) + " is y = " + str(mpa) ) 
        

map = MAP()
map.fit(Xapp, yapp)

#run prediction for an example of iris data
map.predict([ 5.1,  3.5,  1.4,  0.2])


