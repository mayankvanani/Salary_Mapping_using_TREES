##### 						RANDOM FOREST (ensemble learning)

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

## importing the dataset
df = pd.read_csv('Position_Salaries.csv')
print(df)

## classifying features and labels
X = np.array(df['Level']).reshape(-1,1)
y = np.array(df['Salary'])
print(y)

## label - encoding 
## train and test split
## feature scaling

## Fitting Random forest to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300, criterion='mse', random_state=0) # n_estimator is the number of trees 
																   		# criterion = 'mse' is mean squared error which is default. So no need to write
regressor.fit(X, y)
## ADDING MORE N_ESTIMATOR (TREES) MEANS GREATER THE THE TREES AVAILABLE WHOSE AVERAGE ARE TAKEN FOR A PARTICULAR PREDICTION.
## HAVING LARGE TREES IS NOT ALWAYS GOOD. N_ESTIMATOR CAN HAVE GOOD SCORE WITH 300 TREES RATHER THAN 1000 TREES.

## predicting the new result
y_pred = regressor.predict(6.5)
print(y_pred)

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Random-Forest Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show() 

