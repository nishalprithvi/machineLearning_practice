# Exploring first ML model 

# Linear Regression for predicting the price of houses using scikit learn 

# pattern followed for scikit-learn or machine learning :
# *-> Import -> Instantiate -> Fit -> Predict

###############################################################################

# Importing the necessary libraries 

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Loading the dataset
# Scikit-Learn has some built-in datasets, which are great for learning.

housing = fetch_california_housing()
# print(housing)

# Let's use Pandas to make it easier to see our data 
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target # Add the target column to the dataframe

# print(df)
# print(df.info())

# Step 2: Define Features (X) and Target (y)
# We want to predict 'MedHouseVal' using all other columns.

features = housing.feature_names
X = df[features]
y = df['MedHouseVal']

print("--- Data Loaded and Defined ---")
print("Features (X) shape:", X.shape)
print("Target (y) shape:", y.shape)
print("\n")

# Step 3: Train-Test Split
# We'll use 80% of the data for training and 20% for testing.
# `random_state` ensures we get the same split every time we run the code, for reproducibility.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- Data Split ---")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("\n")


# Step 4: Initialize and Train the Model
# First, we create an instance of the Linear Regression model.

model = LinearRegression()

# Next, we train it on our training data using the .fit() method.
# The model is learning the relationship between the features in X_train and the prices in y_train.
model.fit(X_train, y_train)

print("--- Model Trained ---")
print("Model Coefficients : ", model.coef_) # I guess it means weights 
print("Model Intercept: ", model.intercept_) # this is clear "b"
print("\n")

# Step 5: Make Predictions
# Now we use the trained model to predict prices for the test set, which it has never seen.
y_pred = model.predict(X_test)

# Step 6: Evaluate the Model
# We compare our model's predictions (y_pred) with the actual answers (y_test).
mse = mean_squared_error(y_test, y_pred)

print("--- Model Evaluation ---")
print(f"The Mean Squared Error (MSE) on the test set is : {mse:.2f}")