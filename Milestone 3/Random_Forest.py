import os
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import sklearn
from sklearn.externals.six import StringIO

features = [
    'bathrooms', 
    'bedrooms', 
    'latitude', 
    'longitude', 
    'price',
    'Special Characters',
    'Numbers', 
    'StopWord',
    'Uppercase',
]


df = pd.read_json('cleaned_train.json')



df_X = df.drop('interest_level', axis=1)
df_y = df['interest_level']

df_X = df.drop('interest_level', axis=1)
df_X = df_X.drop('description', axis=1)
df_X = df_X.drop('display_address', axis=1)
df_X = df_X.drop('street_address', axis=1)
df_X = df_X.drop('features', axis=1)
df_X = df_X.drop('created', axis=1)
df_X = df_X.drop('listing_id', axis=1)



from sklearn.model_selection import KFold # import KFold

kf = KFold(n_splits=5) # Define the split - into 5 folds 
kf.get_n_splits(df_X) # returns the number of splitting iterations in the cross-validator

KFold(n_splits=5, random_state=None, shuffle=False)


##################

import pandas as pd
import numpy as np

df = pd.read_json('cleaned_train.json')
df.head()

#df_X = dataset.iloc[:, 0:4].values
#df_y = dataset.iloc[:, 4].values
df_y2 = pd.get_dummies(df_y)





from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y2, test_size=0.2, random_state=0)
#Maybe remove test_size and set random states to 1


###Classifier

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



from sklearn.ensemble import RandomForestRegressor

RFM = RandomForestRegressor(n_estimators=20, random_state=0)
RFM.fit(X_train, y_train)
y_pred = RFM.predict(X_test)




#Evaluation

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

####more scores
scores = cross_val_score(estimator=sc, X=df_X, y=df_y, cv=10, n_jobs=4)
scores

from sklearn.metrics import f1_score
f1_score(y_test, y_pred, average='weighted')


######### normalizing

from sklearn import preprocessing

x = df_X.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_X_normalized = pd.DataFrame(x_scaled)





#df_X_normalized

X_train2, X_test2, y_train2, y_test2 = train_test_split(df_X_normalized, df_y2, random_state=1)


sc_normal = StandardScaler()
X_train2 = sc_normal.fit_transform(X_train2)
X_test2 = sc_normal.transform(X_test2)


RFM_normal = RandomForestRegressor(n_estimators=20, random_state=0)
RFM_normal.fit(X_train2, y_train2)
y_pred2 = RFM_normal.predict(X_test2)

#Normalized score
print(confusion_matrix(y_test2, y_pred2))
print(classification_report(y_test2, y_pred2))
print(accuracy_score(y_test2, y_pred2))
