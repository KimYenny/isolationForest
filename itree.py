# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 15:55:18 2022

@author: Yeeun Kim
"""

import pandas as pd
import numpy as np
from metric import harmonic_number

class BinaryTree_array():
    
    def __init__(self, max_depth = None):
        self.max_depth = max_depth
    
    def is_left(self, index):
        return index % 2 == 1
    
    def is_right(self, index):
        return index % 2 == 0
    
    def get_left(self, index):
        return int(index*2 + 1)
    
    def get_right(self, index):
        return int(index*2 + 2)
    
    def get_parent(self, index):
        if self.is_left(index):
            return int((index - 1)/2)
        else:
            return int((index - 2)/2)
    
    def get_depth(self, index):
        return int(np.log2(index + 1))
    
    
class IsolationTree(BinaryTree_array):
    """
    split(X) : Split the isolation tree with input data 
    
    predict(X) : 
    """
    def __init__(self, max_depth = None, extend = False):
        super().__init__(max_depth = max_depth)
        self.extend = extend
        
    def is_leaf(self, index):
        is_max_depth = index >= 2**self.max_depth - 1
        is_size1 = self.itree[index].shape[0] == 1
        
        return is_max_depth or is_size1
        
    def split(self, X):
        """ Split the isolation tree with input data 
        @ input : pandas DataFrame
        """

        self.X = X
        self.n, self.p = X.shape
        
        if self.max_depth is None:
            self.max_depth == self.n-1
            
        # initialize Tree : empty Tree
        tree_length = int(2**(self.max_depth+1) - 1)
        self.itree = [None] * tree_length
        self.itree[0] = X
        
        # initialize path length by node
        self.path_length = [None] * tree_length
        
        # initialize path length per observation
        self.predicted = pd.DataFrame({"path_length":None}, index = X.index)
        
        # initialize history
        self.history = [None] * tree_length
        
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
        
        elif self.is_leaf(node_idx):
            # update path length
            self.predicted.loc[x.index] = self.get_depth(node_idx) + harmonic_number(self.itree[node_idx].shape[0])
            return
        
        else: # split
            if not self.extend: # iforest
                random_feature = np.random.choice(x.columns)
                x_r = x[random_feature]
                random_split = np.random.uniform(x_r.min(), x_r.max())
                
                left_x = x.loc[x_r < random_split]
                right_x = x.loc[x_r >= random_split]
                if left_x.shape[0] > 0:
                    self.itree[self.get_left(node_idx)] = left_x
                if right_x.shape[0] > 0 :
                    self.itree[self.get_right(node_idx)] = right_x
                    
                self.history[node_idx] = (random_feature, random_split)
                
            else: # extended iforest
                n, p = x.shape
                
                m = x.min().min()
                M = x.max().max()
                
                # random intercept and slope
                intercept = np.random.uniform(size = (n, 1) ,low = m, high = M)
                slope = np.random.normal(size = (p, 1))
                
                # split rule
                split_index = ((x - intercept) @ slope <= 0 )[0]
                
                #split
                left_x = x.loc[split_index]
                right_x = x.loc[np.logical_not(split_index)]
                if left_x.shape[0] > 0:
                    self.itree[self.get_left(node_idx)] = left_x
                if right_x.shape[0] > 0 :
                    self.itree[self.get_right(node_idx)] = right_x
                    
                self.history[node_idx] = (intercept, slope)
            
        
    def __path_length(self, index):
        
        if self.itree[index] is not None and self.is_leaf(index):
            self.path_length[index] = self.get_depth(index) + harmonic_number(self.itree[index].shape[0])
          
    def predict(self, X):
        """ Split the isolation tree with input data 
        @ input : pandas DataFrame
        """
        predicted = pd.DataFrame({'path_length':None}, index = X.index)
        
        for x_idx in X.index:
            predicted.loc[x_idx] = self.__predict(X.loc[x_idx:x_idx])
            
        return predicted            
    
    def __predict(self, x):
        
        idx = 0
        while self.history[idx] is not None:
            x_r, s = self.history[idx]
        
            if (x[x_r] < s).iloc[0]:
                idx = self.left_index(idx)
            else:
                idx = self.right_index(idx)
        
        return self.path_length[idx]
    
