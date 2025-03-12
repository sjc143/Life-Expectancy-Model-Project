import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


class ModelComparison:

    def __init__(self, test_size=0.20):
        self.test_size = test_size

    # Standardize data by removing mean and dividing by standard deviation
    def standardize(self, x):
        x_mean = np.mean(x, axis=0)
        x_std_dev = np.std(x, axis=0)
        x_standardized = (x - x_mean) / x_std_dev
        return x_standardized

    # Trains Linear, Lasso, and Ridge models from sklearn and reports R scores for default alpha = 1.0
    def train_compare(self, x, y):

        #Standardizes and splits data
        x_standardized = self.standardize(x)
        x_train, x_test, y_train, y_test = train_test_split(x_standardized, y, test_size=self.test_size, random_state=0)

        #Training linear, lasso, and ridge model
        linear =linear_model.LinearRegression()
        linear.fit(x_train, y_train)
        print(f"Linear R Score: {linear.score(x_test, y_test):.3f}")

        lasso = linear_model.Lasso(alpha = 1.0)
        lasso.fit(x_train, y_train)
        print(f"Lasso R Score : {lasso.score(x_test, y_test):.3f}")

        ridge = linear_model.Ridge(alpha = 1.0)
        ridge.fit(x_train, y_train)
        print(f"Ridge R Score : {ridge.score(x_test, y_test):.3f}")

    # Comparison of Linear, Lasso, and Ridge at varied alphas (regularization parameters)
    def alpha_range_compare(self, x, y, alphas):

        #Standardizes and splits data
        x_standardized = self.standardize(x)
        x_train, x_test, y_train, y_test = train_test_split(x_standardized, y, test_size=self.test_size, random_state=0)

        #stores r scores for each model; linear model does not vary with alpha
        testing_linear_rscore = []
        training_lasso_rscore = []
        testing_lasso_rscore = []
        training_ridge_rscore = []
        testing_ridge_rscore = []

        for i in alphas:
            lasso = linear_model.Lasso(alpha=i)
            lasso.fit(x_train, y_train)
            ridge = linear_model.Ridge(alpha=i)
            ridge.fit(x_train, y_train)

            # storing R scores for Lasso and Ridge
            training_lasso_rscore.append(lasso.score(x_train, y_train))
            testing_lasso_rscore.append(lasso.score(x_test, y_test))
            training_ridge_rscore.append(ridge.score(x_train, y_train))
            testing_ridge_rscore.append(ridge.score(x_test, y_test))

        linear = linear_model.LinearRegression()
        linear.fit(x_train, y_train)
        testing_linear_rscore = [linear.score(x_test, y_test)]*len(training_lasso_rscore)


        return testing_linear_rscore, training_lasso_rscore, testing_lasso_rscore, training_ridge_rscore, testing_ridge_rscore



#Application of Lasso_vs_Ridge for Life Expectancy
#Loading in kaggle dataset
data = pd.read_csv("Life_Expectancy_Data.csv", low_memory = False)
data.replace('', np.nan, inplace=True)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
data.dropna(inplace=True)  #removing incomplete data entries
y = np.array(data['Life expectancy '])
x = np.array(data.loc[:, 'Adult Mortality':])

#Comparison of Lasso vs Ridge models for alpha = 1.0
life_exp = ModelComparison()
life_exp.train_compare(x, y)

#Comparison of Lasso vs Ridge models for alpha range
alphas = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
(testing_linear_rscore, training_lasso_rscore, testing_lasso_rscore,
 training_ridge_rscore, testing_ridge_rscore) = life_exp.alpha_range_compare(x, y, alphas)

#Comparison plots ax1 (training vs testing lasso), ax2 (training vs testing ridge),
# ax3 (testing linear vs lasso vs ridge)
#Note: linear model does not vary with alpha and is used as a benchmark comparison in ax3 plot
fig1, ax1 = plt.subplots()
ax1.plot(alphas, training_lasso_rscore)
ax1.plot(alphas, testing_lasso_rscore)
ax1.set_xlabel("Alpha")
ax1.set_ylabel("R Score")
ax1.legend(["Training", "Testing"])
ax1.set_title("Lasso")

fig2, ax2 = plt.subplots()
ax2.plot(alphas, training_ridge_rscore)
ax2.plot(alphas, testing_ridge_rscore)
ax2.set_xlabel("Alpha")
ax2.set_ylabel("R Score")
ax2.legend(["Training", "Testing"])
ax2.set_title("Ridge")

fig3, ax3 = plt.subplots()
ax3.plot(alphas, testing_linear_rscore)
ax3.plot(alphas, testing_lasso_rscore)
ax3.plot(alphas, testing_ridge_rscore)
ax3.set_xlabel("Alpha")
ax3.set_ylabel("R Score")
ax3.legend(["Linear", "Lasso", "Ridge"])
ax3.set_title("Linear vs Lasso vs Ridge")

plt.show()
