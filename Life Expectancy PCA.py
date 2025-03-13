import textwrap

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt, ticker
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import train_test_split


class PCA:

    def __init__(self, n_components=2):
        self.n_components = n_components

    # Standardize data by removing mean and dividing by standard deviation
    def standardize(self, x):
        x_mean = np.mean(x, axis=0)
        x_std_dev = np.std(x, axis=0)
        x_standardized = (x - x_mean) / x_std_dev
        return x_standardized

    # Finds eigenvectors of system and sorts then according to greatest to least with respect to eigenvalues
    # show_ratio = True will result in the creation of a cumulative explained varience ratio graph to aid in the selection of how many components to use 
    def fit(self, x, show_ratio=True):
        x_standardized = self.standardize(x)

        #Finds covarience matrix to perform eigendecomposition
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

        #Prints out the cumulative explained varience ratio and creates a plot to visualize
        if show_ratio:
            print("cumulative explained varience ratio: \n", cumulative_explained_variance)
            print()
            plt.plot(np.arange(1, len(cumulative_explained_variance)+1, 1), cumulative_explained_variance, '-o')
            plt.xticks(np.arange(1, len(cumulative_explained_variance)+1, 1))
            plt.xlabel('Component Number')
            plt.ylabel('Cumulative Explained Variance Ratio')
            plt.show()

        return eigenvects_sorted

    # Transforms the given system x depending on k components used from eigenvects_sorted
    def transform(self, x, k, eigenvects_sorted):
        x_standardized = self.standardize(x)
        W = eigenvects_sorted[:k]
        x_projected = x_standardized.dot(W.T)

        return x_projected



#Loading in kaggle dataset
data = pd.read_csv("Life_Expectancy_Data.csv", low_memory = False)
data.columns = data.columns.str.strip()
data.replace('', np.nan, inplace=True) #removing rows with empty data entries
data.dropna(inplace=True)
numeric_columns = data.select_dtypes(include=[np.number]).columns #removing rows with filler zeros (ex. 0 population)
data = data[~(data[numeric_columns] == 0).any(axis=1)]
y = np.array(data['Life expectancy '])
x = np.array(data.loc[:, 'Adult Mortality':])



#Application of PCA onto Life Expectancy to gauge varience captured by components
Testing_PCA = PCA()
Testing_PCA.fit(x)


#>95% of variance is captured by first 12 components --> will use k=12 for transformation
#Transform old standardized dataset via selected components (k=12)
PCA = PCA(n_components=12)
eigenvects = PCA.fit(x, show_ratio=False)
x_transformed = PCA.transform(x, k=12, eigenvects_sorted=eigenvects)

#Comparing linear, lasso, and ridge performance with transformed dataset
x_train, x_test, y_train, y_test = train_test_split(x_transformed, y, random_state = 0)

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
print(f"R Scores for PCA Transformed Data:")
print(f"Linear - {linear.score(x_test, y_test):.3f}")

lasso = linear_model.Lasso(alpha = 1.0)
lasso.fit(x_train, y_train)
print(f"Lasso - {lasso.score(x_test, y_test):.3f}")

ridge = linear_model.Ridge(alpha = 1.0)
ridge.fit(x_train, y_train)
print(f"Ridge R - {ridge.score(x_test, y_test):.3f}")
print()




#Bar graph to visualize contribution of each component from PCA
#bar graph of first entry in eiganvects
labels = ['Adult Mortality', 'infant deaths',	'Alcohol', 'percentage expenditure', 'Hepatitis B',
         'Measles', 'BMI', 'under-five deaths', 'Polio', 'Total expenditure', 'Diphtheria',
         'HIV/AIDS', 'GDP',	'Population', 'thinness  1-19 years', 'thinness 5-9 years',
         'Income composition', 'Schooling']
wrapped_labels = [textwrap.fill(label, 10) for label in labels]
plt.figure(figsize=(8, 12))
plt.bar(wrapped_labels, eigenvects[0])
plt.xticks(rotation = 90, fontsize = 8)
plt.show()

#Finding the top contributing variables in first three principle components
for i in range(3):
    top_indices = np.argsort(np.abs(eigenvects[i]))[-3:][::-1]  # Sort in descending order
    print(f"Top 3 components for eigenvector {i}:")

    for j in range(3):
        print(labels[top_indices[j]])

    print()

#Training models on the 9 contributing variables found above
x_reduced = np.array(data.loc[:,[' thinness  1-19 years', 'percentage expenditure', 'Hepatitis B',
                                 'under-five deaths ', 'infant deaths', 'Population',
                                 'Income composition of resources', 'GDP', ' thinness 5-9 years']])
x_train_red, x_test_red, y_train_red, y_test_red = train_test_split(x_reduced, y, random_state = 0)

linear = linear_model.LinearRegression()
linear.fit(x_train_red, y_train_red)
print(f"R Scores for Reduced Data:")
print(f"Linear - {linear.score(x_test_red, y_test_red):.3f}")

lasso = linear_model.Lasso(alpha = 1.0)
lasso.fit(x_train_red, y_train_red)
print(f"Lasso - {lasso.score(x_test_red, y_test_red):.3f}")

ridge = linear_model.Ridge(alpha = 1.0)
ridge.fit(x_train_red, y_train_red)
print(f"Ridge - {ridge.score(x_test_red, y_test_red):.3f}")
print()
