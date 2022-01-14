# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 15:55:18 2022

@author: Yeeun Kim
"""

import pandas as pd
import numpy as np

# for Isolation Tree
from metric import harmonic_number

# for Kmeans based Isolation Tree
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
        

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
    
    
class KmeansIsolationTree():
    def __init__(self, X,
                 depth = 0, max_depth = None, max_cluster = 5):
        
        self.X = X
        self.n, self.p = X.shape
        
        self.depth = depth
        self.child = list()
        
        self.max_depth = max_depth
        self.max_cluster = max_cluster
        
        if depth == 0:
            self.score_value_ = pd.DataFrame(index = X.index, data = {'score':0})
        
    def split(self):
        
        if self.max_depth is None:
            self.max_depth = self.n - 1
        
        # split
        if self.depth < self.max_depth and self.n > 1:
            self.__split(self.X)
            
            for _child in self.child:
                _child.split()
            
    def __split(self, x):
        # condition
        #   self.depth < self.max_depth
        #   self.n > 1
        
        c, labels = self.__elbow(x)
        
        # update score value
        self.score_value_ += self._score_values(x, labels)
                
        # split
        for idx, c_idx in enumerate(range(c)):
            
            _x = x.loc[labels == idx]
            _child = KmeansIsolationTree(X = _x, 
                                         depth = self.depth + 1,
                                         max_depth = self.max_depth)
            _child.score_value_ = self.score_value_.loc[labels == idx]
            
            # append children
            self.child.append(_child)
            
    def __elbow(self, x):
        
        if x.shape[0] == 2: # n = 2
            return 2, np.array([0,1])
        
        elif x.shape[0] == 3: # n = 3
            km = KMeans(n_clusters = 2)
            km.fit(x)
            
            return 2, km.labels_
        
        else: # n > 3
            max_cluster = min(x.shape[0]-1, self.max_cluster)
            Ks = range(2, max_cluster + 1)
            
            # Kmeans Methods with differenc k
            Km = [KMeans(n_clusters = k).fit(x) for k in Ks]
            
            # Silhouette score
            score = [silhouette_score(x, Km[ki].labels_) for ki in range(len(Ks))]
            
            # the optimal k
            k_idx = np.argmin(score)
            
            return Ks[k_idx], Km[k_idx].labels_
    
    def _score_values(self, x, labels):
        """score value for only ONE split point"""
        
        score = pd.DataFrame(index = x.index, data = {'score':None})
        
        clusters = np.unique(labels)
        for cluster in clusters:
            x_cluster = x.loc[labels == cluster]
            center_cluster = x_cluster.mean()
            max_cluster = x_cluster.max()
            min_cluster = x_cluster.min()
            width_cluster = max(self.__distance(max_cluster, center_cluster),
                                self.__distance(min_cluster, center_cluster),)
            
            for x_index in x_cluster.index:
                xi = x.loc[x_index]
                score.loc[x_index] = self.__score_values(xi,
                                                         center_cluster,
                                                         width_cluster)
        return score
                
    def __score_values(self, xi, 
                       center_cluster, width_cluster):
        """score value for only ONE data point at one split point"""
        
        dist_center = self.__distance(xi, center_cluster)
        
        if width_cluster == 0:
            return 1
        else:
            return 1 - dist_center/width_cluster
    
    def __distance(self, a, b):
        return np.sqrt(sum((a - b)**2))
    
    def score_value(self):
        """ average score values of all splits"""
        
        # find leavse
        leaves = self.__get_leaves()
                
        # average them
        score = leaves[0].score_value_ / leaves[0].depth
        
        for leaf in leaves[1:]:
            
            score = pd.concat([score, 
                               leaf.score_value_ / leaf.depth])
        
        score = score.sort_index()
        self.final_score = score
        
        return score
        
    def __get_leaves(self):
        
        descendant = self.child
        leaves = list()
        
        while len(descendant) > 0:
            c = descendant.pop()
            if c.isLeaf():
                leaves.append(c)
            else:
                for cc in c.child:
                    descendant.append(cc)
        return leaves
                    
        if self.isLeaf():
            return self.score_value_
        else:
            for c in self.child:
                c.__traverse()
        
    def isLeaf(self):
        return len(self.child) == 0
        
    def _search(self):
        for c in self.child:
            c._search()
        else:
            return 