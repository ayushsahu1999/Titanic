# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 21:12:30 2019

@author: Dell
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
ds = pd.read_csv('new.csv')
test = pd.read_csv('new_test.csv')
ds.drop('Unnamed: 0', axis=1, inplace=True)
test.drop('Unnamed: 0', axis=1, inplace=True)
pred = pd.read_csv('gender_submission.csv')

# Treating missing values
ds['Age'] = ds['Age'].fillna(ds['Age'].mean())
test['Age'] = test['Age'].fillna(test['Age'].mean())

# As mode is 'S'
ds['Embarked'] = ds['Embarked'].fillna('S')

y_train = ds.iloc[:, 0].values
X_train = ds.iloc[:, 1:8].values
X_test = test.iloc[:, 0:].values
y_test = pred.iloc[:, 1].values

# dummy variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_train[:, 1] = labelencoder_X.fit_transform(X_train[:, 1])
X_train[:, 3] = labelencoder_X.fit_transform(X_train[:, 3])
X_train[:, 4] = labelencoder_X.fit_transform(X_train[:, 4])
X_train[:, 5] = labelencoder_X.fit_transform(X_train[:, 5])
onehotencoder = OneHotEncoder(categorical_features = [1])
X_train = onehotencoder.fit_transform(X_train).toarray()
X_train = X_train[:, 1:]
onehotencoder = OneHotEncoder(categorical_features = [3])
X_train = onehotencoder.fit_transform(X_train).toarray()
X_train = X_train[:, 1:]
onehotencoder = OneHotEncoder(categorical_features = [5])
X_train = onehotencoder.fit_transform(X_train).toarray()
X_train = X_train[:, 1:]
onehotencoder = OneHotEncoder(categorical_features = [7])
X_train = onehotencoder.fit_transform(X_train).toarray()
X_train = X_train[:, 1:]

X_test[:, 1] = labelencoder_X.fit_transform(X_test[:, 1])
X_test[:, 3] = labelencoder_X.fit_transform(X_test[:, 3])
X_test[:, 4] = labelencoder_X.fit_transform(X_test[:, 4])
X_test[:, 5] = labelencoder_X.fit_transform(X_test[:, 5])
onehotencoder_X = OneHotEncoder(categorical_features = [1])
X_test = onehotencoder_X.fit_transform(X_test).toarray()
X_test = X_test[:, 1:]
onehotencoder_X = OneHotEncoder(categorical_features = [3])
X_test = onehotencoder_X.fit_transform(X_test).toarray()
X_test = X_test[:, 1:]
onehotencoder_X = OneHotEncoder(categorical_features = [5])
X_test = onehotencoder_X.fit_transform(X_test).toarray()
X_test = X_test[:, 1:]
onehotencoder_X = OneHotEncoder(categorical_features = [7])
X_test = onehotencoder_X.fit_transform(X_test).toarray()
X_test = X_test[:, 1:]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# Fitting classifier to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf')
classifier.fit(X_train, y_train)

# Predicting the Result
y_pred = classifier.predict(X_test)

# Making Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Checking the Score
classifier.score(X_test, y_test)
classifier.score(X_train, y_train)

df = pd.DataFrame(y_pred)
df['PassengerId'] = pred['PassengerId']
df['Survived'] = df[0]
df.drop(0, axis=1, inplace=True)
df.to_csv('submission.csv')

