import numpy as np
import itertools

from sklearn.base import BaseEstimator

class ExactKMeans(BaseEstimator):
    """Compute the exact k-means solution by brute-force search.  Suitable for small problems."""
    
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        pass
    
    def subsets(self, arr):
        """ Note this only returns non empty subsets of arr"""
        return itertools.chain(*[itertools.combinations(arr,i + 1) for i,a in enumerate(arr)])

    def k_subset(self, arr, k):
        s_arr = sorted(arr)
        return set([frozenset(i) for i in itertools.combinations(self.subsets(arr),k) 
                    if sorted(itertools.chain(*i)) == s_arr])

    def cost(self, X, assignment):
        """Compute the cost of a cluster assignment"""
        
        n, d = X.shape
        means = np.zeros( (len(assignment), d) )
        
        my_cost = 0
        for i, idx in enumerate(map(list, assignment)):
            means[i] = np.mean(X[idx, :], axis=0)
            my_cost += np.sum( (X[idx, :] - means[i])**2 )
            
        return my_cost, means
    
    def fit(self, X):
        self.fit_predict(X)
        pass
    
    def fit_predict(self, X):
        """
        input:
          - X, ndarray, shape=(n, d)
          
        output:
          - Y, ndarray, shape=(n,)
            Best cluster assignment
            
        """
        best_cost = np.inf
        best_assignment = None
        
        n = X.shape[0]
        
        for assignment in self.k_subset(range(n), self.n_clusters):
                
            # Compute the cost
            cost, means = self.cost(X, assignment)
            
            if cost < best_cost:
                best_cost = cost
                best_assignment = assignment
                self.means_ = means
        
        Y = np.zeros(n, dtype=int)
        for i, s in enumerate(best_assignment):
            Y[list(s)] = i
            
        return Y
        
