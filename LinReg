import pandas as pd
import numpy as np
from sklearn import linear_model
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



data = pd.read_csv("data.csv")

# convert categorical values into integer values
data['Street'] = data.Street.map({'Grvl': 0, 'Pave': 1})

data['MSZoning'] = data.MSZoning.map({'C (all)': 0, 'FV': 1, 'RH': 2, 'RL': 3, 'RM': 4})

# doesnt work
data['Alley'] = data.Alley.map({'Grvl': 0, 'NA': 1, 'Pave': 2})

data['LotShape'] = data.LotShape.map({'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3} )

data['LandContour'] = data.LandContour.map({'Lvl': 0, 'Bnk': 1, 'HLS': 2, 'Low': 3} )

data['Utilities'] = data.Utilities.map({'AllPub': 0, 'NoSeWa': 1})

data['LotConfig'] = data.LotConfig.map({'Inside': 0, 'Corner': 1, 'CulDSac': 2, 'FR2': 3, 'FR3': 4} )

data['LandSlope'] = data.LandSlope.map({'Gtl': 0, 'Mod': 1, 'Sev': 2} )

data['Neighborhood'] = data.Neighborhood.map({'Blmngtn': 0, 'Blueste': 1, 'BrDale': 2, 'BrkSide': 3, 'ClearCr': 4, 'CollgCr': 5, 'Crawfor': 6, 'Edwards': 7, 'Gilbert': 8, 'IDOTRR': 9, 'MeadowV': 10, 'Mitchel': 11, 'NAmes': 12, 'NoRidge': 13, 'NPkVill': 14, 'NridgHt': 15, 'NWAmes': 16, 'OldTown':17, 'SWISU':18, 'Sawyer':19, 'SawyerW':20, 'Somerst': 21, 'StoneBr':22, 'Timber':23, 'Veenker':24})

data['Condition1'] = data.Condition1.map({'Artery': 0, 'Feedr': 1, 'Norm': 2, 'RRNn': 3, 'RRAn': 4, 'PosN': 5, 'PosA': 6, 'RRNe': 7, 'RRAe': 8})

data['Condition2'] = data.Condition2.map({'Artery': 0, 'Feedr': 1, 'Norm': 2, 'RRNn': 3, 'RRAn': 4, 'PosN': 5, 'PosA': 6, 'RRNe': 7, 'RRAe': 8})

data['BldgType'] = data.BldgType.map({'1Fam': 0, '2fmCon': 1, 'Duplex': 2, 'Twnhs': 3, 'TwnhsE': 4})

data['HouseStyle'] = data.HouseStyle.map({'1Story': 0, '1.5Fin': 1, '1.5Unf': 2, '2Story': 3, '2.5Fin': 4, '2.5Unf': 5, 'SFoyer': 6, 'SLvl': 7})

data['RoofStyle'] = data.RoofStyle.map({'Flat': 0, 'Gable': 1, 'Gambrel': 2, 'Hip': 3, 'Mansard': 4, 'Shed': 5})

data['RoofMatl'] = data.RoofMatl.map({'ClyTile': 0, 'CompShg': 1, 'Membran': 2, 'Metal': 3, 'Roll': 4, 'Tar&Grv': 5, 'WdShake': 6, 'WdShngl': 7})

data['Exterior1st'] = data.Exterior1st.map({'AsbShng': 0, 'AsphShn': 1, 'BrkComm': 2, 'BrkFace': 3, 'CBlock': 4, 'CemntBd': 5, 'HdBoard': 6, 'ImStucc': 7, 'MetalSd': 8, 'Other': 9, 'Plywood': 10, 'PreCast': 11, 'Stone': 12, 'Stucco': 13, 'VinylSd': 14, 'Wd Sdng': 15, 'WdShing': 16})

data['Exterior2nd'] = data.Exterior2nd.map({'AsbShng': 0, 'AsphShn': 1, 'Brk Cmn': 2, 'BrkFace': 3, 'CBlock': 4, 'CmentBd': 5, 'HdBoard': 6, 'ImStucc': 7, 'MetalSd': 8, 'Other': 9, 'Plywood': 10, 'Stone': 11, 'Stucco': 12, 'VinylSd': 13, 'Wd Sdng': 14, 'Wd Shng': 15})

# gives floats
data['MasVnrType'] = data.MasVnrType.map({'BrkCmn': 0, 'BrkFace': 1, 'NA': 2, 'None': 3, 'Stone': 4})

data['ExterQual'] = data.ExterQual.map({'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Po': 4})

data['ExterCond'] = data.ExterCond.map({'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Po': 4})

data['Foundation'] = data.Foundation.map({'BrkTil': 0, 'CBlock': 1, 'PConc': 2, 'Slab': 3, 'Stone': 4, 'Wood': 5})

# gives floats and Nan
data['BsmtQual'] = data.BsmtQual.map({'Ex': 0, 'Fa': 1, 'Gd': 2, 'NA': 3, 'TA': 4})

# gives floats and NaN
data['BsmtCond'] = data.BsmtCond.map({'Fa': 0, 'Gd': 1, 'NA': 2, 'Fa': 3, 'Po': 4, 'TA': 5})

# gives floats and NaN
data['BsmtExposure'] = data.BsmtExposure.map({'Gd': 0, 'Av': 1, 'Mn': 2, 'No': 3, 'NA': 4,})

# gives floats and Nan
data['BsmtFinType1'] = data.BsmtFinType1.map({'ALQ': 0, 'BLQ': 1, 'GLQ': 2, 'LwQ': 3, 'NA': 4, 'Rec': 5, 'Unf': 6})

# gives floats and Nan
data['BsmtFinType2'] = data.BsmtFinType2.map({'ALQ': 0, 'BLQ': 1, 'GLQ': 2, 'LwQ': 3, 'NA': 4, 'Rec': 5, 'Unf': 6})

data['Heating'] = data.Heating.map({'Floor': 0, 'GasA': 1, 'GasW': 2, 'Grav': 3, 'OthW': 4, 'Wall': 5})

data['HeatingQC'] = data.HeatingQC.map({'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Po': 4})

data['CentralAir'] = data.CentralAir.map({'N': 0, 'Y': 1})

# gives floats
data['Electrical'] = data.Electrical.map({'FuseA': 0, 'FuseF': 1, 'FuseP': 2, 'Mix': 3, 'NA': 4, 'SBrkr': 5})

data['KitchenQual'] = data.KitchenQual.map({'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Po': 4})

data['Functional'] = data.Functional.map({'Typ': 0, 'Min1': 1, 'Min2': 2, 'Mod': 3, 'Maj1': 4, 'Maj2': 5, 'Sev': 6, 'Sal': 7})

# gives floats and Nan
data['FireplaceQu'] = data.FireplaceQu.map({'Ex': 0, 'Fa': 1, 'Gd': 2, 'NA': 3, 'Po': 4, 'TA': 5})

# gives floats and Nan
data['GarageType'] = data.GarageType.map({'2Types': 0, 'Attchd': 1, 'Basment': 2, 'BuiltIn': 3, 'CarPort': 4, 'Detchd': 5, 'NA': 6})

# gives floats and Nan
data['GarageFinish'] = data.GarageFinish.map({'Fin': 0, 'NA': 1, 'RFn': 2, 'Unf': 3})

# gives floats and Nan
data['GarageQual'] = data.GarageQual.map({'Ex': 0, 'Fa': 1, 'Gd': 2, 'NA': 3, 'Po': 4, 'TA': 5})

# gives floats and Nan
data['GarageCond'] = data.GarageCond.map({'Ex': 0, 'Fa': 1, 'Gd': 2, 'NA': 3, 'Po': 4, 'TA': 5})

data['PavedDrive'] = data.PavedDrive.map({'Y': 0, 'P': 1, 'N': 2})

# gives all Nan (all NA)
data['PoolQC'] = data.PoolQC.map({'Ex': 0, 'Fa': 1, 'Gd': 2, 'NA': 3})

# lotta Nan
data['Fence'] = data.Fence.map({'GdPrv': 0, 'GdWo': 1, 'MnPrv': 2, 'MnWw': 3, 'NA': 4})

# lotta Nan
data['MiscFeature'] = data.MiscFeature.map({'Gar2': 0, 'NA': 1, 'Othr': 2, 'Shed': 3, 'TenC': 4})

data['SaleType'] = data.SaleType.map({'WD': 0, 'CWD': 1, 'VWD': 2, 'New': 3, 'COD': 4, 'Con': 5, 'ConLw': 6, 'ConLI': 7, 'ConLD': 8, 'Oth': 9})

data['SaleCondition'] = data.SaleCondition.map({'Normal': 0, 'Abnorml': 1, 'AdjLand': 2, 'Alloca': 3, 'Family': 4, 'Partial': 5})


# make all Nan values = 0
#data.fillna(0, inplace = True)
data.fillna(data.mean(), inplace=True)
# turn float values to integers
c = data.columns[data.dtypes == float]
data[c] = data[c].astype(int)

print(data.head())
#print(df.info())
#print(df.describe())


x = data[['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition']]


y = data['SalePrice']

reg = LinearRegression()

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.20, random_state = 101)

#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

reg.fit(x_train,y_train)

print(reg.coef_)
print("----------------------------------------------")

cdf = pd.DataFrame(reg.coef_, x.columns, columns = ['Coeff'])
print(cdf)

print("-----------------------------------------------")

y_predict = reg.predict(x_test)

print("\nPredicted SalePrices:\n")
print(y_predict)
print(len(y_predict))
print(y_predict.shape)

# actual SalePrice of test group
print("\nTrue SalePrices:\n")
print(y_test)
print(len(y_test))
print(y_test.shape)

print("-----------------------------------------------")

real_price_list = list(y_test)
predicted_price_list = list(y_predict)

accuracy_counter = 0
tolerance = .20
for i in range(1, len(real_price_list)):
    if (1 - tolerance) * real_price_list[i] <= predicted_price_list[i] and (1 + tolerance) * real_price_list[i] >= predicted_price_list[i]:
        accuracy_counter += 1

print('Number of prices in accuracy interval: ', accuracy_counter)

error = y_test - y_predict
print('Error per House: ')
print(abs(error))


plt.scatter(range(len(y_test)), y_test, label = 'True SalePrice')
plt.scatter(range(len(y_predict)), y_predict, label = 'Predicted SalePrice')
plt.xlabel('House Number')
plt.ylabel('Sale Price of House (in $)')
plt.title('Performance of Linear Regression Classifier')
plt.legend()
plt.show()
