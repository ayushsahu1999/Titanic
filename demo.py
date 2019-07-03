# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 19:56:48 2019

@author: Dell
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import string

# Importing the dataset
dataset = pd.read_csv('train.csv')

# Fixing the Name columns
def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if ((substring in big_string) == True):
            return substring
    
    return np.nan

titles = ['Mr', 'Mrs', 'Miss.', 'Master', 'Dr.', 'Ms.', 'Major', 'Rev', 'Mlle', 'Col', 
          'Capt.', 'Mme', 'Countess', 'Don', 'Jonkheer']

dataset['Title'] = dataset['Name'].map(lambda x: substrings_in_string(x, titles))

# Generalizing in only Mr and Mrs
def generalize(x):
    title = x['Title']
    if (title in ['Mr', 'Master', 'Major', 'Rev', 'Col', 'Capt.', 'Don', 'Jonkheer']):
        return 'Mr'
    elif (title in ['Mrs', 'Mme', 'Countess']):
        return 'Mrs'
    elif (title in ['Miss.', 'Ms.', 'Mlle']):
        return 'Miss'
    elif (title == 'Dr.'):
        if (x['Sex'] == 'male'):
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title
    
dataset['Title'] = dataset.apply(generalize, axis=1)

# Getting the decks
decks = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'Unknown']

dataset['Decks'] = dataset['Cabin'].map(lambda x: substrings_in_string(str(x), decks))

# Fare/Person
dataset['FperP'] = dataset['Fare']/(dataset['SibSp']+dataset['Parch']+1)

'''
dataset.dtypes

dataset.describe()

dataset.isnull().sum()

# Analysis of decks
dataset['Decks'].value_counts()
dataset['Decks'].value_counts().plot.bar()

# Analysis of fare per person
dataset['FperP'].describe()
dataset['FperP'].plot.hist()
dataset['FperP'].plot.box()
'''

# Getting optimal values of all decks
# First A then B,C,and so on till line 91
def opt(x):
    deck = x['Decks']
    if (deck == 'G'):
        return x['FperP']
    
    return np.nan

dataset['A'] = dataset.apply(opt, axis=1)

Q1 = dataset['A'].quantile(0.25)
Q3 = dataset['A'].quantile(0.75)
IQR = Q3-Q1
lr = Q1-(1.5*IQR)
ur = Q3 + (1.5*IQR)
dataset['A'].plot.hist()
dataset['A'].plot.box()
# A
dataset.loc[dataset['A']<lr, 'FperP'] = np.mean(dataset['A'])
# B
dataset.loc[dataset['A']>ur, 'FperP'] = np.mean(dataset['A'])
dataset.loc[dataset['A']<0, 'FperP'] = np.mean(dataset['A'])
# C
dataset.loc[dataset['A']>ur, 'FperP'] = np.mean(dataset['A'])
dataset.loc[dataset['A']<0, 'FperP'] = np.mean(dataset['A'])
# D
dataset.loc[dataset['A']>ur, 'FperP'] = np.mean(dataset['A'])
# E
dataset.loc[dataset['A']<lr, 'FperP'] = np.mean(dataset['A'])
dataset.loc[dataset['A']>ur, 'FperP'] = np.mean(dataset['A'])
# G
dataset.loc[dataset['A']<lr, 'FperP'] = np.mean(dataset['A'])


# function for alloting missing decks
dataset['A']=np.where(dataset.Decks.isnull(), 
                      0, dataset.FperP)

def deck_allotment(x):
    if (x['A']==0):
        fare = x['FperP']
        if (fare>30.92 and fare<36):
            return 'A'
        elif (fare>81.88 and fare<=165.0078):
            return 'B'
        elif (fare>58.55 and fare<=81.88):
            return 'C'
        elif (fare>=36 and fare<=58.55):
            return 'D'
        elif (fare>13.25 and fare<=30.92):
            return 'E'
        elif (fare>6.72 and fare<=13.25):
            return 'F'
        elif (fare>=3.5 and fare<=6.72):
            return 'G'
        else:
            return 'T'
    else:
        return x['Decks']
     
    
dataset['Decks']=dataset.apply(deck_allotment, axis=1)

# Deleting Useless Columns
dataset.drop('A', axis=1, inplace=True)
dataset.drop('B', axis=1, inplace=True)
dataset.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin', 'FperP'], axis=1, inplace=True)

dataset['Family'] = dataset['SibSp']+dataset['Parch']
dataset.drop(['SibSp', 'Parch'], axis=1, inplace=True)

# Exporting new dataset
dataset.to_csv('new.csv')

# Test Set
test = pd.read_csv('test.csv')
test.isnull().sum()
test['Decks'].value_counts()

test['Title'] = test['Name'].map(lambda x: substrings_in_string(x, titles))
test['Title'] = test.apply(generalize, axis=1)
test['Decks'] = test['Cabin'].map(lambda x: substrings_in_string(str(x), decks))
test['FperP'] = test['Fare']/(test['SibSp']+test['Parch']+1)

# function for alloting missing decks
test['A']=np.where(test.Decks.isnull(), 
                      0, test.FperP)

test['Decks']=test.apply(deck_allotment, axis=1)

# Deleting Useless Columns
test.drop('A', axis=1, inplace=True)
test.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin', 'FperP'], axis=1, inplace=True)

test['Family'] = test['SibSp']+test['Parch']
test.drop(['SibSp', 'Parch'], axis=1, inplace=True)

# Exporting new dataset
test.to_csv('new_test.csv')
