import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# === PREPROCESSING ===


# Load dataset
data = pd.read_csv("Life_Expectancy_Data.csv", low_memory=False)


# Strip whitespace from column names
data.columns = data.columns.str.strip()


# Replace empty strings with NaN
data.replace('', np.nan, inplace=True)


# Drop rows with any NaN values
data.dropna(inplace=True)


# Remove rows where ANY numeric column contains a zero (excluding categorical columns)
numeric_columns = data.select_dtypes(include=[np.number]).columns
data = data[~(data[numeric_columns] == 0).any(axis=1)]


# Save preprocessed data
data.to_csv("Preprocessed_Life_Expectancy_Data.csv", index=False)


# === MACHINE LEARNING MODEL ===


# Load preprocessed data
data = pd.read_csv("Preprocessed_Life_Expectancy_Data.csv")


# List of columns we want to use as features
columns_of_interest = [
    'Adult Mortality', 'infant deaths', 'Alcohol', 'percentage expenditure', 'Hepatitis B',
    'Measles', 'BMI', 'under-five deaths', 'Polio',
    'Total expenditure', 'Diphtheria', 'HIV/AIDS', 'GDP', 'Population',
    'thinness  1-19 years', 'thinness 5-9 years', 'Income composition of resources', 'Schooling']


# Features (X) and target (y)
if 'thinness  1-19 years' in data.columns:
    x = np.array(data[columns_of_interest])
    y = np.array(data['Life expectancy'])
else:
    print("Error: 'thinness 1-19 years' column not found in the dataset")
    exit()


# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)


# Standardize the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# Linear regression model
linear_model = LinearRegression()
linear_model.fit(x_train_scaled, y_train)


# Predictions
y_pred = linear_model.predict(x_test_scaled)


# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# === STREAMLIT INTERFACE ===


st.title("Life Expectancy Prediction (Linear Regression)")


# Reordered user input section
adult_mortality = st.number_input("Adult Mortality", min_value=0.0, step=0.1)
infant_deaths = st.number_input("Infant Deaths", min_value=0, step=1)
alcohol = st.number_input("Alcohol Consumption", min_value=0.0, step=0.1)
percentage_expenditure = st.number_input("Percentage Expenditure", min_value=0.0, step=0.1)
hepatitis_b = st.number_input("Hepatitis B Vaccination", min_value=0.0, step=0.1)
measles = st.number_input("Measles Cases", min_value=0, step=1)
bmi = st.number_input("BMI", min_value=10.0, step=0.1)
under_five_deaths = st.number_input("Under-Five Deaths", min_value=0, step=1)
polio = st.number_input("Polio Vaccination", min_value=0.0, step=0.1)
total_expenditure = st.number_input("Total Expenditure", min_value=0.0, step=0.1)
diphtheria = st.number_input("Diphtheria Vaccination", min_value=0.0, step=0.1)
hiv_aids = st.number_input("HIV/AIDS Rate", min_value=0.0, step=0.1)
gdp = st.number_input("GDP", min_value=0.0, step=0.1)
population = st.number_input("Population", min_value=0, step=1)
thinness_1_19 = st.number_input("Thinness 1-19 Years", min_value=0.0, step=0.1)
thinness_5_9 = st.number_input("Thinness 5-9 Years", min_value=0.0, step=0.1)
income_composition = st.number_input("Income Composition of Resources", min_value=0.0, step=0.1)
schooling = st.number_input("Schooling (Years)", min_value=0.0, step=0.1)


# Prepare user input data for prediction
user_input = np.array([[adult_mortality, infant_deaths, alcohol, percentage_expenditure, hepatitis_b, measles, bmi,
                        under_five_deaths, polio, total_expenditure, diphtheria, hiv_aids, gdp, population,
                        thinness_1_19, thinness_5_9, income_composition, schooling]])


# Scale user input using the same scaler
user_input_scaled = scaler.transform(user_input)


# Make prediction
predicted_life_expectancy = linear_model.predict(user_input_scaled)


# Display prediction
st.subheader(f"Predicted Life Expectancy: {predicted_life_expectancy[0]:.2f} years")
st.subheader(f"Linear Model R² Score: {r2:.3f}")


# === CONSOLE OUTPUT ===
print(f"Predicted Life Expectancy: {predicted_life_expectancy[0]:.2f} years")
print(f"Linear Model R² Score: {r2:.3f}")