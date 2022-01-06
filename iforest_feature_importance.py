# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 01:11:02 2022

@author: Yeeun Kim
"""

import numpy as np
import pandas as pd
from math import e

def harmonic_number(n):
    return np.log(n) + e 

class IsolationTree():
    # implementaion with array
    
    def __init__(self, max_depth = None):
        self.max_depth = max_depth
        
    # for convinience
    def __isLeft(self, index):
        return index % 2 == 1
    
    def __isRight(self, index):
        return index % 2 == 0
    
    def __left_index(self, index):
        return int(index*2 + 1)
    
    def __right_index(self, index):
        return int(index*2 + 2)
    
    def __parent_index(self, index):
        if self.__isLeft(index):
            return int((index - 1)/2)
        else:
            return int((index - 2)/2)
        
    def __left(self, index):
        return self.itree[self.__left_index(index)]
    
    def __right(self, index):
        return self.itree[self.__right_index(index)]
    
    def __parent(self, index):
        return self.itree[self.__parent_index(index)]
    
    def __is_leaf(self,index):
        is_max_depth = index >= 2**self.max_depth - 1
        is_size1 = self.itree[index].shape[0] == 1
        
        return is_max_depth or is_size1
    
    def __depth(self, index):
        return int(np.log2(index + 1))
    
    # implementation
    def split(self, X):
        self.X = X
        self.n, self.p = X.shape
        
        if self.max_depth is None:
            self.max_depth = min(int(self.n - 1), 10)
            
        # initialize Tree : empty Tree
        tree_length = int(2**(self.max_depth + 1) - 1)
        self.itree = [None] * tree_length
        self.itree[0] = X
        
        # initialize path length by node
        self.path_length = [None] * tree_length
        
        # initialize path length per observation
        self.predicted = pd.DataFrame({"path_length":None}, index = X.index)
        
        # initialize history
        self.history = [None] * tree_length
        
        # initialize feature importance
        self.n_isolate = pd.DataFrame({'n_isolate':0}, index = X.columns)
        
        # split all node
        for node_idx, x in enumerate(self.itree):
            self.__split(node_idx, x)
            
        # calculate path length
        for node_idx, x in enumerate(self.itree):
            self.__path_length(node_idx)
        
    def __split(self, node_idx, x):
        if x is None or x.shape[0] == 0:
            # do nothing
            return
        
        elif self.__is_leaf(node_idx):
            # isolated node
            if self.itree[node_idx].shape[0] == 1:
                par_idx = self.__parent_index(node_idx)
                par_feature = self.history[par_idx][0]
                self.n_isolate.loc[par_feature] += 1
            
            # update path length
            self.predicted.loc[x.index] = self.__depth(node_idx) + harmonic_number(self.itree[node_idx].shape[0])
                    
        else:
            # split
            random_feature = np.random.choice(x.columns)
            x_r = x[random_feature]
            random_split = np.random.uniform(x_r.min(), x_r.max())
            
            left_x = x.loc[x_r < random_split]
            right_x = x.loc[x_r >= random_split]
            if left_x.shape[0] > 0:
                self.itree[self.__left_index(node_idx)] = left_x
            if right_x.shape[0] > 0 :
                self.itree[self.__right_index(node_idx)] = right_x
                
            self.history[node_idx] = (random_feature, random_split)
        
    def __path_length(self, index):
        
        if self.itree[index] is not None and self.__is_leaf(index):
            self.path_length[index] = self.__depth(index) + harmonic_number(self.itree[index].shape[0])
          
    def predict(self, X):
        predicted = pd.DataFrame({'path_length':None}, index = X.index)
        
        for x_idx in X.index:
            predicted.loc[x_idx] = self.__predict(X.loc[x_idx:x_idx])
            
        return predicted            
    
    def __predict(self, x):
        
        idx = 0
        while self.history[idx] is not None:
            x_r, s = self.history[idx]
        
            if (x[x_r] < s).iloc[0]:
                idx = self.__left_index(idx)
            else:
                idx = self.__right_index(idx)
        
        return self.path_length[idx]

class IsolationForest:
    def __init__(self, ntree = 100, subsample = None):
        
        self.ntree = ntree
        self.subsample = subsample
        
    def fit(self, X):
        
        self.X = X
        self.n = X.shape[0]
        
        # check subsample
        if self.subsample is None:
            self.subsample = int(self.n//2)
            
        self.height_limit = np.ceil(np.log2(self.subsample))
        
        path_length = pd.DataFrame(index = X.index)
        feature_importance = pd.DataFrame(index = X.columns)
        
        for b in range(self.ntree):
            # draw subsample
            X_b = X.sample(n = self.subsample)
            
            # construct tree with X_b
            itree_b = IsolationTree(self.height_limit)
            itree_b.split(X_b)
            path_length_b = itree_b.predicted
            feature_importance_b = itree_b.n_isolate
            path_length_b.columns = [b]
            feature_importance_b.columns = [b]
            
            path_length = pd.merge(left = path_length, left_index = True,
                                     right = path_length_b, right_index = True,
                                     how = 'left')
            feature_importance = pd.merge(left = feature_importance, left_index = True,
                                          right = feature_importance_b, right_index = True,
                                          how = 'left')
            
        
        self.path_length = path_length
        self._feature_importance = feature_importance
            
    def anomaly_score(self):
        
        c = 2*harmonic_number(self.n)
        
        anomaly_score_list = []
        for xi in self.X.index:            
            h_avg = np.mean(self.path_length.loc[xi])
            s = 2**(-h_avg/c)
            anomaly_score_list.append(s)
        
        return pd.DataFrame({'anomaly_score':anomaly_score_list}, index = self.X.index)
    
    def feature_importance(self):
        fi = self._feature_importance
        fi /= fi.sum()
        
        return fi.mean(axis = 1)
