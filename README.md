# Titanic
A machine learning model to predict if the passenger would have survived the Titanic disaster or not.

# Description
This dataset is taken from the Titanic Competition from Kaggle. I have made this model so that I can hone my skills.

# Contents
### train.csv
dataset in which model is trained.
### test.csv
dataset in which model is tested.
### gender_submission
dataset containing true predictions used to measure accuracy of predictions.

## Since the dataset is not organized and needs some pre-processing, so some pre-processing is applied in demo.py to make it organized and treat missing values as well as outliers.
### demo.py
Python file in which pre-processing is done on train and test datasets to make it organized and treat missing values.
### new.csv
Dataset containing pre-processed training dataset or treated training dataset
### new_test.csv
Dataset containing pre-processed test dataset or treated test dataset
### submission.csv
Dataset containing the predicted values by our model.

## Challenges faced
The main challenge was to treat the missing values in cabins. As out of 892 entries in training set, around 500 entries have missing cabins and according to my study of dataset, cabins **will** play a huge part in predicting accurate predictions whether the passenger will survive or not. So, what I have done is found the optimum values of all the cabins and remove outliers for all cabins seperately and then use those values to find the missing values of cabins.

## Legend
### PassengerId
Id of onboard passengers
### Survived
Dependent variable or the column to predict
0 = No
1 = Yes
### Pclass
1 = 1st class ticket
2 = 2nd class ticket
3 = 3rd class ticket
### Name
Name of passenger
### Sex
Sex of passenger
### Age
Age of passenger
### SibSp
Number of Siblings/Spouse
### Parch
Number of Parent/Children
### Ticket
Ticket number
### Fare
Total Fare
### Cabin
Cabin of the passenger
### Embarked
Port of Embarkation
C = Cherbourg
Q = Queenstown
S = Southampton
