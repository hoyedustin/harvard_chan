## importing packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


## loading and previewing the data

data = pd.read_csv("infants.csv")
print(data.head())


## now we need to find and define our X and y values for single linnear regression
## instructions says mother's age for x and babys weight as Y
## note: x values need to be in a 2D structure per scikit learn. I will print shapes to pracice

X = data[["mage"]]

y = data["weight"]

print(type(X))
print(X.shape)

print(type(y))
print(y.shape)


## now we fit our model

model = LinearRegression()
model.fit(X, y)

y_intercept = model.intercept_

print(model.coef_[0])
print(y_intercept)


## predicting

y_pred = model.predict(X)

## finding and printing our coefficients
beta0 = model.intercept_
beta1 = model.coef_[0]

print(f"Intercept (β0): {beta0:.4f}")
print(f"Slope (β1): {beta1:.4f}")


## plotting

plt.figure(figsize=(8, 6))

# create scatter plot of actual data
plt.scatter(data["mage"], data["weight"], color="steelblue", alpha=0.6, label="Observed data")

# draw regression line on data set
plt.plot(data["mage"], y_pred, color="red", linewidth=2)

plt.xlabel("Mother's age")
plt.ylabel("Baby's weight")
plt.legend()
plt.show()
