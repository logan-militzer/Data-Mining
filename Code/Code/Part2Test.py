# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:01:44 2020

@author: Logan
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#from mpl_toolkits.basemap import Basemap

path = os.getcwd()
os.chdir(path)
print(path)

df = pd.read_json('train.json')

plt.figure(figsize=(8,8))
sns.jointplot(x="latitude",y="longitude", data= df,size=7)
plt.ylabel('longitude')
plt.xlabel('latitude')
plt.show();