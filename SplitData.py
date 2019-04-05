# Splits data from given HousingData.csv
# 80% goes to data_train.csv
# 20% goes to data_test.csv
import pandas as pd
from sklearn.model_selection import train_test_split
import csv

# Import data set and print to show functionality
data = pd.read_csv("HousingData.csv")
# print data.head()

# Split data into labels and features
# label = data which we want to predict
# feature = data used to predict labels
# y = target variable (dependent variable)
y = data.SalePrice
x = data.drop('SalePrice', axis = 1)

# Split data into train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# Display splits
print("\nx_train:\n")
# print(x_train.head())
# print(x_train.shape)

# print x_train

# print("\nx_test:\n")
# print(x_test.head())
# print(x_test.shape)
