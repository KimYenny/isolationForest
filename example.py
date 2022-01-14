# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 22:10:54 2022
@author: Yeeun Kim
"""
import pandas as pd

X = pd.read_csv("/data/two_mvn.csv")

#%% Isolation Forest (2008)
from IsolationForest import IsolationForest

iforest = IsolationForest()
iforest.fit(X)
anomaly_score = iforest.anomaly_score()['anomaly_score']

#%% Extended Isolation Forest (2018)
eiforest = IsolationForest(extend = True)
eiforest.fit(X)
anomaly_score_eiforest = eiforest.anomaly_score()['anomaly_score']

#%% Kmeans based Isolation Forest (2020)
from IsolationForest import KmeansIsolationForest
kmeansIFroest = KmeansIsolationForest(ntree = 30)
kmeansIFroest.fit(X)
kmeansIFroest.anomaly_score()

#%%
from matplotlib import pyplot as plt
import seaborn as sns
cmap = sns.cubehelix_palette(as_cmap=True)

f, ax = plt.subplots()
points = plt.scatter(X['0'], X['1'], s=50, c = anomaly_score ,alpha = 0.5)
f.colorbar(points)


