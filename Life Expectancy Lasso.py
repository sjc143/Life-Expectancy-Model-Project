import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

#Loading in kaggle dataset
data = pd.read_csv("Life_Expectancy_Data.csv", low_memory = False)
data.replace('', np.nan, inplace=True)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
data.dropna(inplace=True)  #removing incomplete data entries
y = np.array(data['Life expectancy '])
x = np.array(data.loc[:, 'Adult Mortality':])

#Standardize data
x_mean = np.mean(x, axis=0)
x_std_dev = np.std(x, axis=0)
x_standardized = (x - x_mean) / x_std_dev

#Splitting the data into training and testing
x_train, x_test, y_train, y_test = train_test_split(x_standardized, y, random_state = 0)

#Training lasso model
lasso = linear_model.Lasso(alpha = 0.1)
lasso.fit(x_train, y_train)
lasso_r2 = lasso.score(x_test, y_test)
print(f"Lasso R Score : {lasso_r2:.3f}")


alphas = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
for i in alphas:
    lasso = linear_model.Lasso(alpha=i)
    lasso.fit(x_train, y_train)
    lasso_r2 = lasso.score(x_test, y_test)
    print(f"Lasso (alpha = {i}) R Score: {lasso_r2:.3f}")


#change of tolerance, plot loss function with iteration count
