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

# Now lets work on HW 1 Continued
# First lets import the data

hw_1_continued_data = pd.read_csv("/Users/dustinh/Downloads/SCCS2_v12.csv")
hw_1_continued_data

#now lets assign x and y variables


X = hw_1_continued_data[["age"]]

y = hw_1_continued_data["tc"]

print(type(X))
print(X.shape)
print(type(y))
print(y.shape)

#As a reminder now that we have our variables defined, we need to flatten our x values to use stats models
#The reason why we need to flatten our x, is because statsmodels requires this to produce a parameter table
x_shaped = np.array(X/10).reshape(-1, 1)
#NOTE: I added the (X/10) part above because we need to fit the model on a decade change per the instructions
X_sm = sm.add_constant(x_shaped)  # adds the intercept column w/the 1 values which is needed to calculate the y-intercpet correctly which impacts the slope
model_sm_pt_2 = sm.OLS(y, X_sm).fit()

# Getting the Table
model_sm_pt_2.summary().tables[1]

# ok now we are going to calculate the pearson correlation
# note, this is very annnoying, but statsmodels required x to be a 2D but flattened array when fitting a linnear model, but to find the pearson correlation, both x and y need to be 1D
age_1d = hw_1_continued_data["age"]
chol_1d = hw_1_continued_data["tc"]

r, p_value = pearsonr(age_1d, chol_1d)
print(r)

# notw lets graph the residuals
fitted_vals = model_sm_pt_2.fittedvalues
residuals = model_sm_pt_2.resid

plt.scatter(fitted_vals, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')  # reference line at 0
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Fitted Values")
plt.show()

# now lets plot the histogram

plt.hist(residuals, bins=30, edgecolor='black')
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals")
plt.savefig("histogram_residuals.png", dpi=300, bbox_inches='tight')
plt.show()

#ok now we are moving on to a lowess smoothing curve

# age_1d and chol_1d are the 1D variables
smoothed = lowess(chol_1d, age_1d)

plt.scatter(age_1d, chol_1d, alpha=0.4, label="Data")
plt.plot(smoothed[:, 0], smoothed[:, 1], color='red', linewidth=2, label="Lowess curve")
plt.xlabel("Age")
plt.ylabel("Total Cholesterol")
plt.title("Lowess Curve: Age vs. Total Cholesterol")
plt.legend()
plt.savefig("lowess_curve.png", dpi=300, bbox_inches='tight')
plt.show()

#now the assignment is having us add a new 2 degree polynomial to the fuction
#The concept should be that if it is just a normal regression, the x squared coefficient should come out close to zero

# Create the quadratic term
age_sq = age_1d ** 2

# Stack age and age^2 as two columns for statsmodels
X_quad = np.column_stack((age_1d, age_sq))
X_quad_sm = sm.add_constant(X_quad)

model_quad = sm.OLS(chol_1d, X_quad_sm).fit()
print(model_quad.summary())

#moving on to questoin two. We have to run a varience t-test

# Split cholesterol by sex group
male_chol = hw_1_continued_data[hw_1_continued_data["gender"] == 0]["tc"]
female_chol = hw_1_continued_data[hw_1_continued_data["gender"] == 1]["tc"]

t_stat, p_value = ttest_ind(male_chol, female_chol, equal_var=True)
print(t_stat, p_value)


#retroactivly adding this for
n1 = len(male_chol)
n2 = len(female_chol)
df_ttest = n1 + n2 - 2
print(n1, n2, df_ttest)

#now we have to fit a simple linnear regression on those values
gender_1d = hw_1_continued_data["gender"]

#remember to shape our value
gender_shaped = np.array(gender_1d).reshape(-1, 1)
#we have to add the constant too
X_gender_sm = sm.add_constant(gender_shaped)

model_gender = sm.OLS(chol_1d, X_gender_sm).fit()
print(model_gender.summary())

#now we are adding another predictor - gender - to our multiple linnear regression model
#note, I didn't mention this with the 2 degree polynomial regression, but we need to use column stack command now. It takes multipe 1D arrays and turns them into a 2D array


# Build the design matrix: age, age^2, and gender together
X_full = np.column_stack((age_1d, age_1d**2, gender_1d))
X_full_sm = sm.add_constant(X_full)

model_full = sm.OLS(chol_1d, X_full_sm).fit()
print(model_full.summary())


#the next question is asking for crude vs. adjusted rates so we need less ronding
print(model_sm.params)
print(model_full.params)
print(model_quad.params)

# Create a smooth range of ages spanning your data
age_range = np.linspace(age_1d.min(), age_1d.max(), 100)

# Fitted equations for each sex
male_fitted = 3.0390 + 0.0909*age_range - 0.0007*age_range**2
female_fitted = 3.1335 + 0.0909*age_range - 0.0007*age_range**2

# Scatter plot of raw data, colored by sex
plt.figure(figsize=(8, 6))
plt.scatter(hw_1_continued_data[hw_1_continued_data["gender"] == 0]["age"],
            hw_1_continued_data[hw_1_continued_data["gender"] == 0]["tc"],
            alpha=0.4, label="Male (raw data)", color="steelblue")
plt.scatter(hw_1_continued_data[hw_1_continued_data["gender"] == 1]["age"],
            hw_1_continued_data[hw_1_continued_data["gender"] == 1]["tc"],
            alpha=0.4, label="Female (raw data)", color="lightcoral")

# Overlay fitted curves
plt.plot(age_range, male_fitted, color="darkblue", linewidth=2, label="Male (fitted)")
plt.plot(age_range, female_fitted, color="darkred", linewidth=2, label="Female (fitted)")

plt.xlabel("Age")
plt.ylabel("Total Cholesterol")
plt.title("Fitted Regression Curves by Sex, with Raw Data")
plt.legend()
plt.savefig("fitted_by_sex.png", dpi=300, bbox_inches='tight')
plt.show()

#now for D3, we need to build out an interaction model

# Create interaction terms
age_sex_interaction = age_1d * gender_1d
age_sq_sex_interaction = (age_1d**2) * gender_1d

# Build the design matrix
X_interaction = np.column_stack((age_1d, age_1d**2, gender_1d, age_sex_interaction, age_sq_sex_interaction))
X_interaction_sm = sm.add_constant(X_interaction)

model_interaction = sm.OLS(chol_1d, X_interaction_sm).fit()
print(model_interaction.summary())

#now lets plot the values

# Smooth age range (reuse the same range as before)
age_range = np.linspace(age_1d.min(), age_1d.max(), 100)

# Fitted equations from the interaction model
male_fitted_interaction = 2.7876 + 0.1138*age_range - 0.0011*age_range**2
female_fitted_interaction = 3.8630 + 0.0390*age_range + 0.0000*age_range**2

# Scatter plot of raw data, colored by sex
plt.figure(figsize=(8, 6))
plt.scatter(hw_1_continued_data[hw_1_continued_data["gender"] == 0]["age"],
            hw_1_continued_data[hw_1_continued_data["gender"] == 0]["tc"],
            alpha=0.4, label="Male (raw data)", color="steelblue")
plt.scatter(hw_1_continued_data[hw_1_continued_data["gender"] == 1]["age"],
            hw_1_continued_data[hw_1_continued_data["gender"] == 1]["tc"],
            alpha=0.4, label="Female (raw data)", color="lightcoral")

# Overlay fitted curves
plt.plot(age_range, male_fitted_interaction, color="darkblue", linewidth=2, label="Male (fitted)")
plt.plot(age_range, female_fitted_interaction, color="darkred", linewidth=2, label="Female (fitted)")

plt.xlabel("Age")
plt.ylabel("Total Cholesterol")
plt.title("Fitted Regression Curves by Sex (Interaction Model), with Raw Data")
plt.legend()
plt.savefig("fitted_interaction_by_sex.png", dpi=300, bbox_inches='tight')
plt.show()

#for question E, we need to know MSE values. Good news is that is is actually pretty easy since we already built the models. We just need to run one command to get the values


print(np.sqrt(model_sm.mse_resid))           # Model 1: age only
print(np.sqrt(model_quad.mse_resid))         # Model 2: age + age^2
print(np.sqrt(model_gender.mse_resid))       # Model 3: sex only
print(np.sqrt(model_full.mse_resid))         # Model 4: age + age^2 + sex
print(np.sqrt(model_interaction.mse_resid))  # Model 5: full interaction model

#thne do the same for the r squared values

print(model_sm.rsquared, model_sm.rsquared_adj)
print(model_quad.rsquared, model_quad.rsquared_adj)
print(model_gender.rsquared, model_gender.rsquared_adj)
print(model_full.rsquared, model_full.rsquared_adj)
print(model_interaction.rsquared, model_interaction.rsquared_adj)

#I overwrote my model 1 so just adding back for the problem
x_shaped = np.array(X).reshape(-1, 1)
X_sm = sm.add_constant(x_shaped)
model_sm = sm.OLS(y, X_sm).fit()
print(model_sm.nobs)
print(model_sm.rsquared, model_sm.rsquared_adj)

# now for question F, we are splitting the two graphs apart


# Split the dataset by sex
male_data = hw_1_continued_data[hw_1_continued_data["gender"] == 0]
female_data = hw_1_continued_data[hw_1_continued_data["gender"] == 1]

# Male model: age + age^2 only, fit on male data only
male_age = male_data["age"]
male_chol_full = male_data["tc"]
X_male = np.column_stack((male_age, male_age**2))
X_male_sm = sm.add_constant(X_male)
model_male_only = sm.OLS(male_chol_full, X_male_sm).fit()

# Female model: age + age^2 only, fit on female data only
female_age = female_data["age"]
female_chol_full = female_data["tc"]
X_female = np.column_stack((female_age, female_age**2))
X_female_sm = sm.add_constant(X_female)
model_female_only = sm.OLS(female_chol_full, X_female_sm).fit()
print(model_female_only.summary())

print(model_male_only.summary())

# now lets plot our separate equations

# Pull coefficients directly from the fitted models
male_b0, male_b1, male_b2 = model_male_only.params
female_b0, female_b1, female_b2 = model_female_only.params

# Smooth age range
age_range = np.linspace(age_1d.min(), age_1d.max(), 100)

# Fitted equations from the SEPARATE models (Part F)
male_fitted_separate = male_b0 + male_b1*age_range + male_b2*age_range**2
female_fitted_separate = female_b0 + female_b1*age_range + female_b2*age_range**2

# Scatter plot of raw data, colored by sex
plt.figure(figsize=(8, 6))
plt.scatter(male_data["age"], male_data["tc"],
alpha=0.4, label="Male (raw data)", color="steelblue")
plt.scatter(female_data["age"], female_data["tc"],
alpha=0.4, label="Female (raw data)", color="lightcoral")

# Overlay fitted curves
plt.plot(age_range, male_fitted_separate, color="darkblue", linewidth=2, label="Male (fitted)")
plt.plot(age_range, female_fitted_separate, color="darkred", linewidth=2, label="Female (fitted)")

plt.xlabel("Age")
plt.ylabel("Total Cholesterol")
plt.title("Fitted Regression Curves by Sex (Separate Models), with Raw Data")
plt.legend()
plt.savefig("fitted_separate_by_sex.png", dpi=300, bbox_inches='tight')
plt.show()

#now we move onto our final question which invovles calculating BMI

# first we need meters, then BMI
height_meters = hw_1_continued_data["height"] / 100
hw_1_continued_data["bmi"] = hw_1_continued_data["weight"] / (height_meters ** 2)

print(hw_1_continued_data["bmi"].describe())


#now that we have our new bmi column, we can create the bins

bins = [-float('inf'), 18.5, 25.0, 30.0, float('inf')]
labels = ['Underweight', 'Normal', 'Overweight', 'Obese']

hw_1_continued_data['bmi_category'] = pd.cut(hw_1_continued_data['bmi'], bins=bins, labels=labels, right=False)

print(hw_1_continued_data['bmi_category'].value_counts())

#now its time to make our first model. We are going to use continous i.e no bins

bmi_1d = hw_1_continued_data["bmi"]

bmi_shaped = np.array(bmi_1d).reshape(-1, 1)
X_bmi_sm = sm.add_constant(bmi_shaped)

model_bmi_continuous = sm.OLS(chol_1d, X_bmi_sm).fit()
print(model_bmi_continuous.summary())

#now we are going to do continous i.e bins

# need Create dummy variables. This is becasue you cant do math on words
bmi_dummies = pd.get_dummies(hw_1_continued_data["bmi_category"], drop_first=False)
bmi_dummies = bmi_dummies.drop(columns=["Normal"])


X_bmi_cat = sm.add_constant(bmi_dummies.astype(float))
model_bmi_categorical = sm.OLS(chol_1d, X_bmi_cat).fit()
print(model_bmi_categorical.summary())

#now we have to make scatter plots/regressions for the continues BMI and the categorical BMI
#first we will start with the continues regression. Standard regression here.

b0, b1 = model_bmi_continuous.params

bmi_range = np.linspace(bmi_1d.min(), bmi_1d.max(), 100)
continuous_fitted = b0 + b1*bmi_range

plt.figure(figsize=(8, 6))
plt.scatter(bmi_1d, chol_1d, alpha=0.4, color="steelblue", label="Raw data")
plt.plot(bmi_range, continuous_fitted, color="darkred", linewidth=2, label="Fitted (continuous BMI)")
plt.xlabel("BMI")
plt.ylabel("Total Cholesterol")
plt.title("Continuous BMI Model with Raw Data")
plt.legend()
plt.savefig("bmi_continuous_fit.png", dpi=300, bbox_inches='tight')
plt.show()

#now lets do the categorical one

params = model_bmi_categorical.params

fitted_values = {
    "Underweight": params["const"] + params["Underweight"],
    "Normal": params["const"],
    "Overweight": params["const"] + params["Overweight"],
    "Obese": params["const"] + params["Obese"]
}

bounds = {
    "Underweight": (bmi_1d.min(), 18.5),
    "Normal": (18.5, 25.0),
    "Overweight": (25.0, 30.0),
    "Obese": (30.0, bmi_1d.max())
}

plt.figure(figsize=(8, 6))
plt.scatter(bmi_1d, chol_1d, alpha=0.4, color="steelblue", label="Raw data")

for category, (low, high) in bounds.items():
    plt.hlines(fitted_values[category], low, high, color="darkred", linewidth=3)

plt.xlabel("BMI")
plt.ylabel("Total Cholesterol")
plt.title("Categorical BMI Model with Raw Data")
plt.savefig("bmi_categorical_fit.png", dpi=300, bbox_inches='tight')
plt.show()

bmi_sq = bmi_1d ** 2
X_bmi_quad = np.column_stack((bmi_1d, bmi_sq))
X_bmi_quad_sm = sm.add_constant(X_bmi_quad)

model_bmi_quad = sm.OLS(chol_1d, X_bmi_quad_sm).fit()
print(model_bmi_quad.summary())

#now we are taking the best two models and turning them into a single "super" multi-linnear regression model

# Build the full combined design matrix
X_combined = np.column_stack((
    age_1d,                          # x1: age - original age column
    age_1d**2,                       #  x2: age^2 - here is the double polynomial age
    gender_1d,                       # x3: sex - same deal here w sex
    age_1d * gender_1d,              # x4: age x sex the multiplcation between the two
    (age_1d**2) * gender_1d,         # x5: age^2 x sex
    bmi_1d,                          # x6: BMI
    bmi_1d**2                        # x7: BMI^2
))
X_combined_sm = sm.add_constant(X_combined)

model_combined = sm.OLS(chol_1d, X_combined_sm).fit()
print(model_combined.summary())


#the next question is asking for crude vs. adjusted rates so we need less ronding
print(model_sm.params)
print(model_full.params)
print(model_quad.params)

#lets build the scatter plot to determine residuals vs. fitted vlaues

fitted_vals_combined = model_combined.fittedvalues
residuals_combined = model_combined.resid

plt.figure(figsize=(8, 6))
plt.scatter(fitted_vals_combined, residuals_combined, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Fitted Values (Combined Model)")
plt.savefig("residuals_combined.png", dpi=300, bbox_inches='tight')
plt.show()

#finally, we have to look at leverage and influence
#cooks method is used in class slides


influence = model_combined.get_influence()
cooks_d = influence.cooks_distance[0]

plt.figure(figsize=(8, 6))
plt.stem(cooks_d, markerfmt=",")
plt.xlabel("Observation Index")
plt.ylabel("Cook's Distance")
plt.title("Cook's Distance for Combined Model")
plt.axhline(y=4/len(cooks_d), color='red', linestyle='--', label=f"Threshold (4/n = {4/len(cooks_d):.4f})")
plt.legend()
plt.savefig("cooks_distance.png", dpi=300, bbox_inches='tight')
plt.show()

# Identify which specific observations exceed the threshold
threshold = 4/len(cooks_d)
high_influence_indices = np.where(cooks_d > threshold)[0]
print(f"Number of high-influence points: {len(high_influence_indices)}")
print(f"Indices: {high_influence_indices}")

threshold = 4/len(cooks_d)
high_influence_indices = np.where(cooks_d > threshold)[0]
print(f"Number of high-influence points: {len(high_influence_indices)}")
print(f"Indices: {high_influence_indices}")

# Show the top 10 highest Cook's Distance values specifically
top_10 = np.argsort(cooks_d)[-10:][::-1]
print("\nTop 10 highest Cook's Distance observations:")
for idx in top_10:
    print(f"Index {idx}: Cook's D = {cooks_d[idx]:.4f}")

#it does appear that actully more than 10 seem to have high leverage so lets take them out

# Take the 10 most influential observations
top_10_indices = np.argsort(cooks_d)[-10:]

# Drop these from your data
hw_1_dropped = hw_1_continued_data.drop(index=hw_1_continued_data.index[top_10_indices])

# Rebuild your predictors from the dropped dataset
age_dropped = hw_1_dropped["age"]
gender_dropped = hw_1_dropped["gender"]
bmi_dropped = hw_1_dropped["bmi"]
chol_dropped = hw_1_dropped["tc"]

X_combined_dropped = np.column_stack((
    age_dropped,
    age_dropped**2,
    gender_dropped,
    age_dropped * gender_dropped,
    (age_dropped**2) * gender_dropped,
    bmi_dropped,
    bmi_dropped**2
))
X_combined_dropped_sm = sm.add_constant(X_combined_dropped)

model_combined_dropped = sm.OLS(chol_dropped, X_combined_dropped_sm).fit()
print(model_combined_dropped.summary())


leverage = influence.hat_matrix_diag

# Common rule of thumb for high leverage: > 2*(p+1)/n, where p = number of predictors
p = 7  # number of predictors in your combined model
n = len(leverage)
leverage_threshold = 2*(p+1)/n

print(f"Leverage threshold: {leverage_threshold:.4f}")

# Check leverage specifically for your 10 flagged high-influence points
for idx in top_10_indices:
    print(f"Index {idx}: Leverage = {leverage[idx]:.4f}, Cook's D = {cooks_d[idx]:.4f}, Residual = {residuals_combined[idx]:.4f}")
