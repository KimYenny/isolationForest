# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 22:10:54 2022
@author: Yeeun Kim
"""

import os
os.chdir("C:/Users/user/Desktop/iforest")

import pandas as pd
from iforest import IsolationForest

#%%
X = pd.read_csv(os.getcwd() + "/data/ring.csv")
iforest = IsolationForest(subsample = 100)
iforest.fit(X)
anomaly_score = iforest.anomaly_score()['anomaly_score']

from matplotlib import pyplot as plt
import seaborn as sns
cmap = sns.cubehelix_palette(as_cmap=True)

f, ax = plt.subplots()
points = plt.scatter(X['0'], X['1'], s=50, c = anomaly_score ,alpha = 0.5)
f.colorbar(points)


