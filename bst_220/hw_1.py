##importing packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# hard coding values

y = [1,2,3]
x = [1,3,2]


#plotting the values

plt.scatter(x,y,color = 'blue')
plt.xlabel('x')
plt.ylabel('y')

# Note: It needs our x values to be a 2D array for the x-values, otherwise it will flag an error

x_shaped = np.array(x).reshape(-1, 1)
# Fitting the simple linnear model
model = LinearRegression()
model.fit(x_shaped, y)

# Getting the coefficients
intercept = model.intercept_
slope = model.coef_[0]


# Plot data and fitted line

plt.scatter(x, y, color='blue', label='Data')
x_line = np.linspace(min(x), max(x), 100).reshape(-1, 1)
y_line = model.predict(x_line)
plt.plot(x_line, y_line, color='red', label=f'Fit: Y = {intercept:.2f} + {slope:.2f}X')

# we need the output table, and scikit learn won't produce the output table so we need to use statsmdoels
# first we need to add the contant values to the x values
X_sm = sm.add_constant(x_shaped)  # adds the intercept column w/the 1 values which is needed to calculate the y-intercpet correctly which impacts the slope
model_sm = sm.OLS(y, X_sm).fit()

print (intercept)
print(slope)
model_sm.summary().tables[1]
