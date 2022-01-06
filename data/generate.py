# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 20:59:32 2022

@author: user
"""

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
#%% Two MVN

X_n = np.random.multivariate_normal(size = 950, mean = np.array([1,4,2,3,6,8,4]), cov = np.eye(7))
X_an = np.random.multivariate_normal(size = 50, mean = np.array([10,4,5,3,0,8,9]), cov = np.eye(7))
X = pd.concat([pd.DataFrame(X_n), pd.DataFrame(X_an)])
X = X.sample(frac=1).reset_index(drop = True)

X.to_csv("C:/Users/user/Desktop/iforest/data/two_mvn.csv", index = False)

#%%
n = 980
m = 20
X1 = np.random.uniform(-10,10,size = n + m)
error = np.append(np.random.normal(size = n, scale = 0.1), np.random.normal(size = m, scale = 0.5))
X2 = np.sin(X1) + error
plt.scatter(X1,X2)

X = pd.DataFrame({'x1':X1, 'x2':X2})
X.to_csv("C:/Users/user/Desktop/iforest/data/sin.csv", index = False)

#%%
n = 950
m = 50

r = 4
X1 = np.random.uniform(-4,4, size = n)
sign = np.random.binomial(p = 0.5, n = 1, size = n)*2 - 1
error = np.random.normal(size = n , scale = 0.7)
X2 = sign*np.sqrt(r**2 - X1**2) + error

X_normal = pd.DataFrame({'0':X1,'1':X2})
X_anomaly = pd.DataFrame(np.random.multivariate_normal(size = m, mean = [0,0], cov = np.eye(2)*0.3))
X_normal.columns = X_anomaly.columns
X = pd.concat([X_normal, X_anomaly], ignore_index = True)

plt.scatter(X[0], X[1])

X.to_csv("C:/Users/user/Desktop/iforest/data/ring.csv", index = False)
#%%
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

pca = PCA(n_components = 2)
X_pca = pca.fit_transform(X)
plt.scatter(X_pca[:,0], X_pca[:,1], s=50, alpha = 1)