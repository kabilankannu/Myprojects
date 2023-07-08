import pandas as pd
from pandas import read_csv
import numpy as np
filename = 'omega 0.0.csv'
data = read_csv(filename)
data.head()
X = data.iloc[:, 0:2]
y = data.iloc[:, -1]
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Regression algorithm
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

# Evaluate the performance on the testing set
y_pred = regressor.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
prd=regressor.predict([[0.08,-0.4]])

# print("R-squared: {:.2f}".format(r2))
# print("Mean Squared Error: {:.2f}".format(mse))
# print("Root Mean Squared Error: {:.2f}".format(rmse))
print(prd)