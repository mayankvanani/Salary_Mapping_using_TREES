#####						DECISION TREE REGRESSION

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

## importing dataset
df = pd.read_csv('Position_Salaries.csv')
print(df)

## classifying features and labels
X = np.array(df['Level']).reshape(-1,1)
y = np.array(df['Salary'])
# print(y)

## Fitting decision tree regerssion to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(criterion='mse', random_state=0)	
## mse is mean squared error which is by default. 
regressor.fit(X, y)

## Predicting new result
y_pred = regressor.predict(6.5)
print(y_pred)

# Visualising the Decision Tree Regression results (for higher resolution)
# we see that decision tree divides the x-axis into regural intervals and we see steps as becoz for that interval the average value computed is constant.
X_grid = np.arange(min(X), max(X), 0.01) 
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'blue')
plt.plot(X_grid, regressor.predict(X_grid), color = 'red')
plt.xlabel('Position') 
plt.ylabel('Salary')
plt.title('Decision Tree Regressor')
plt.show()
