# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 15:56:31 2022

@author: Yeeun Kim

"""

import numpy as np
import pandas as pd
from metric import harmonic_number
from IsolationTree import IsolationTree, KmeansIsolationTree

class IsolationForest:
    
    def __init__(self, ntree = 100, subsample = 128, extend = False):
        self.ntree = ntree
        self.subsample = subsample
        self.extend = extend
        
    def fit(self, X):
        
        self.X = X
        self.n = X.shape[0]
                    
        self.height_limit = np.ceil(np.log2(self.subsample))
        
        path_length = pd.DataFrame(index = X.index)
        
        for b in range(self.ntree):
            # draw subsample
            X_b = X.sample(n = self.subsample)
            
            # construct tree with X_b
            itree_b = IsolationTree(self.height_limit, extend = self.extend)
            itree_b.split(X_b)
            path_length_b = itree_b.predicted
            path_length_b.columns = [b]
            
            path_length = pd.merge(left = path_length, left_index = True,
                                     right = path_length_b, right_index = True,
                                     how = 'left')
        
        self.path_length = path_length
            
    def anomaly_score(self):
        
        n = self.n
        c = 2*harmonic_number(n-1) - (2 * (n-1) / n)
        
        anomaly_score_list = []
        for xi in self.X.index:            
            h_avg = np.mean(self.path_length.loc[xi])
            s = 2**(-h_avg/c)
            anomaly_score_list.append(s)
        
        return pd.DataFrame({'anomaly_score':anomaly_score_list}, index = self.X.index)
    
class KmeansIsolationForest:
    def __init__(self, ntree = 100, subsample = 128, max_cluster = 5):
        self.ntree = ntree
        self.subsample = subsample
        self.max_cluster = max_cluster
        
    def fit(self, X):
        
        self.X = X
        self.n, self.p = X.shape
        
        self.height_limit = np.ceil(np.log2(self.subsample))
        
        score = pd.DataFrame(index = X.index)
        
        for b in range(self.ntree):
            
            # draw subsample
            X_b = X.sample(n = self.subsample)
            
            # construct tree with X_b
            itree_b = KmeansIsolationTree(X = X_b, 
                                          max_depth = self.height_limit, 
                                          max_cluster = self.max_cluster)
            itree_b.split()
            score_b = itree_b.score_value()
            score_b.columns = [b]
            
            score = pd.merge(left = score, left_index = True,
                             right = score_b, right_index = True,
                             how = 'left')
        
        self.score = score
            
    def anomaly_score(self):
        # final score
        anomaly_score_list = []
        
        for xi in self.X.index:            
            score = np.mean(self.score.loc[xi])
            anomaly_score_list.append(score)
        
        anomaly_score_list = np.array(anomaly_score_list)
        anomaly_score_list = 1 - anomaly_score_list
        
        return pd.DataFrame({'anomaly_score':anomaly_score_list}, index = self.X.index)
        