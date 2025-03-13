# Life Expectancy Model Project

## Overview
With the anticipated population plateau in developed countries, life expectancy as a societal metric has become increasingly important due to its usage as both a metric for general quality of life in a country and a metric to direct social policies from healthcare to retirement benefits. The ability to predict life expectancy based on current socio-economic and health factors would enable researchers and policymakers to make more informed and proactive decisions. This project aims to compare linear, lasso, and ridge regression models for the prediction of life expectancy, determining the best performing regression through comparison of R scores. Additionally, as there are no strictly established indicators for life expectancy, PCA will be applied to try and determine whether the number of life expectancy indicators can be reduced.

## Data Preparation
Data for this project was used from a [Kaggle dataset](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who/data) of combined life expectancy indicators provided by the World Health Organization (WHO). This dataset is linked above and also included as the Life_Expectancy_Data.csv in the project.

Although no preprocessing is required for the Kaggle dataset to be run with the code provided, usage of the code for any other datasets would require the removal of any incomplete data entries that include empty strings or filler zeros (ex. zero population would result in removal of an entire entry row). 

## Required Packages 
The required to run the code are: NumPy, Pandas, Scikit-learn, and Matplotlib. 

## Running Instructions
1. Download [Life_Expectancy_Data.csv](Life_Expectancy_Data.csv), Life Expectancy Model Comparisons.py, and Life Expectancy PCA.py
2. Group them in a project folder together
3. Run Life Expectancy Model Comparisons.py and Life Expectancy PCA.py and view the results
