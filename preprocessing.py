# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 14:36:32 2021

@author: Aya ELgebaly
"""


#PREPROCESSING FOE DATA....................

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


veriler=pd.read_csv(r"./data.csv")

#MEAN::

kilo=veriler.iloc[:,2:3]
yas=veriler.iloc[:,3:4]

#ımputer
imputer1=SimpleImputer(missing_values=np.nan,  strategy="mean")
imputer2=SimpleImputer(missing_values=np.nan,  strategy="mean")

kilo=imputer1.fit_transform(kilo)
yas=imputer2.fit_transform(yas)

ulke=veriler.iloc[:,0:1].values
lb=LabelEncoder()
ulke[:,0]=lb.fit_transform(ulke[:,0])

ohe=OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()

ulke=pd.DataFrame(data=ulke, index=range(22), columns=["fr", "tr", "us"])
kilo=pd.DataFrame(data=kilo, index=range(22), columns=["kilo"])
yas=pd.DataFrame(data=yas, index=range(22), columns=["yas"])
cinsiyet=veriler.iloc[:,4:5]
cat_variables = veriler[["cinsiyet"]]
Y = pd.get_dummies(cat_variables,drop_first=True)


df1=pd.concat([ulke, veriler.iloc[:,1:2]], axis=1)
df2=pd.concat([df1,kilo], axis=1)
X=pd.concat([df2, yas], axis=1)

#Precossing,
'''
encoding
imputer-nan
standardizaiton, normalization.--sayısal
'''
sc=StandardScaler()
X_1=sc.fit_transform(X)

x_train, x_test, y_train, y_test= train_test_split(X, Y, test_size=0.33, random_state=0)
rfc=RandomForestRegressor(n_estimators=10, random_state=0)
rfc.fit(x_train,y_train)
tahmin=rfc.predict(x_test)
print(tahmin)

