##importing packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import ttest_ind
import statsmodels.formula.api as smf

#importing our dataset

hw_2_data = pd.read_csv("SCCS2_v12.csv")

print(hw_2_data.head())
print(hw_2_data.columns.tolist())
print(hw_2_data["gender"].value_counts())
print(hw_2_data.isna().sum())

#problem 1 wants us to adjust for linnear and quadratic age, so lets create that quadratic age column first
#remember we also have to create our bmi colummn like last time

hw_2_data["age_squared"] = hw_2_data["age"] ** 2


height_meters = hw_2_data["height"] / 100
hw_2_data["bmi"] = hw_2_data["weight"] / (height_meters ** 2)

analytic = hw_2_data.dropna(subset=["tc", "gender", "bmi", "age"])

#ok now that we got our colmns made and our nice suset created, lets look at the models

# Model 1
y_total_cholesteral = analytic["tc"]

X1 = sm.add_constant(analytic[["gender", "age", "age_squared"]])
m1 = sm.OLS(y_total_cholesteral, X1).fit()

#now for our second model, lets add in BMI to see if it is a confounder

# Model 2
X2 = sm.add_constant(analytic[["gender", "age", "age_squared", "bmi"]])
m2 = sm.OLS(y_total_cholesteral, X2).fit()

print(m2.summary())

#Ok now we are making our third model. This time comparing bmi coefficnet before vs after gender is added
X3 = sm.add_constant(analytic[["bmi", "age", "age_squared"]])
m3 = sm.OLS(y_total_cholesteral, X3).fit()


# adding model 4 even though it should have the exact same variables as model 2
X4 = sm.add_constant(analytic[["bmi", "age", "age_squared", "gender"]])
m4 = sm.OLS(y_total_cholesteral, X4).fit()
print(m4.summary())

#ok now we are moving onto effect modification
#first we have to multiply gender and BMI into our gender_bmi column to actullly produce effect modification
analytic["gender_bmi"] = analytic["gender"] * analytic["bmi"]

#now we can add it to our model
X5 = sm.add_constant(analytic[["gender", "bmi", "gender_bmi", "age", "age_squared"]])
m5 = sm.OLS(y_total_cholesteral, X5).fit()
print(m5.summary())
