import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

path = os.getcwd()
os.chdir(path)
print(path)

df = pd.read_json('train.json')

df.info()

##Bathrooms
'''
BathHist = df.hist(column = 'bathrooms')
BathHist

BathBox = df.boxplot(column = ['bathrooms'])
BathBox

BathBoxWOOL = df.boxplot(column = ['bathrooms'], showfliers = False)
BathBoxWOOL

BathCount = df[df['bathrooms'] > 7.0].count()

##Bedrooms

BedHist = df.hist(column = 'bedrooms')
BedHist

#df.sort_values(by=['price'], ascending = False)


BedBox = df.boxplot(column = ['bedrooms'])
BedBox

BedBoxWOOL = df.boxplot(column = ['bedrooms'], showfliers = False)
BedBoxWOOL

BedCount = df[df['bedrooms'] > 6.0].count()
BedCount



##Price

PriceHist = df.hist(column = 'price')
PriceHist

#df.sort_values(by=['price'], ascending = False)


PriceBox = df.boxplot(column = ['price'])
PriceBox

PriceBoxWOOL = df.boxplot(column = ['price'], showfliers = False)
PriceBoxWOOL

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()

PriceCount = df[df['price'] > .0].count()
PriceCount



#df.hist(column = 'bathrooms')
#df.boxplot(column = ['bathrooms'])
#df.boxplot(column = ['bedrooms'])
#df.boxplot(column = ['price'])

##Outliers on price > 20000
    #Price < 200

#df.sort_values(by=['bathrooms'],ascending=False)
#sns.heatmap(df.isnull(), cbar = False)
#sns.heatmap(df., cbar = False)
  
    
plt.figure(figsize = (10, 10))
sns.jointplot(x="latitude",y="longitude", data = df, size = 7)
plt.ylabel('longitude')
plt.xlabel('latitude')
plt.show();


x = df["latitude"]
y = df["longitude"]

heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.clf()
plt.imshow(heatmap.T, extent=extent, origin='lower')
plt.show()

from scipy.stats import gaussian_kde
x = df["latitude"]
y = df["longitude"]


xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

fig, ax = plt.subplots()
ax.scatter(x, y, c=z, s=100, edgecolor='')
plt.show()



x = df["latitude"]
y = df["longitude"]

heatmap, xedge, yedge = np.histogram2d(x, y, bins=20)
extent = [xedge[0], xedge[-1], yedge[0], yedge[-1]]

plt.clf()
plt.imshow(heatmap.T, xyedge = xyedge, origin='lower')
plt.show()


x = df["latitude"]
y = df["longitude"]


xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

fig, ax = plt.subplots()
ax.scatter(x, y, c=z, s=100, edgecolor='')
plt.show()

'''