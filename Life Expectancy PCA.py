import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt, ticker
from sklearn.metrics import explained_variance_score
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

#Finding covariance matrix and the eigendecomposition
covariance_matrix = np.cov(x_standardized.T)
eigenvals, eigenvects = np.linalg.eig(covariance_matrix)

#Sort eigenvalues & vectors from greatest to least
idxs = np.argsort(eigenvals)[::-1]
eigenvals_sorted = eigenvals[idxs]
eigenvects_sorted = eigenvects[idxs]



#Explained variance ratio test to visualize information kept by components
eiganvals_sum = sum(eigenvals_sorted)
explained_variance = [(i / eiganvals_sum) for i in eigenvals_sorted]
cumulative_explained_variance = np.cumsum(explained_variance)

print("cumulative explained varience ratio: \n", cumulative_explained_variance)
plt.plot(cumulative_explained_variance, '-o')
plt.xticks(np.arange(0, len(cumulative_explained_variance), 1))
plt.xlabel('Component Number')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.show()


#>95% of variance is captured by first 13 components --> will use k=13 for transformation
#Transform old standardized dataset via selected components (k=13)
k=13
W = eigenvects_sorted[:k]
x_projected = x_standardized.dot(W.T)


#Feeding new transformed dataset into Lasso regression
#Splitting the data into training and testing
x_train, x_test, y_train, y_test = train_test_split(x_projected, y, random_state = 0)

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


