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
from statsmodels.gam.api import GLMGam, BSplines
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor

#lets get this new dataset going
df = pd.read_stata("/Users/dustinh/Downloads/framingham.dta")

df
df.columns

#lets fit our first logistic regression
#first step is using death as y values and BMI was our initial x value

logistic_model = smf.logit('death ~ bmi', data=df).fit()


print(logistic_model.summary())

#I want to visualize this model


# generate a range of BMI values to plot the smooth curve
bmi_range = np.linspace(df['bmi'].min(), df['bmi'].max(), 200)
pred_df = pd.DataFrame({'bmi': bmi_range})
pred_probs = logistic_model.predict(pred_df)

fig, ax = plt.subplots(figsize=(7, 5))

jitter = np.random.normal(0, 0.02, size=len(df))
ax.scatter(df['bmi'], df['death'] + jitter, alpha=0.3, s=15, label='Observed')

# predicted probability curve
ax.plot(bmi_range, pred_probs, color='black', linewidth=2, label='Predicted P(death)')

ax.set_xlabel('BMI')
ax.set_ylabel('P(death)')
ax.set_title('Logistic Regression: BMI vs. Mortality')
ax.legend()
plt.show()

#this question is a little tricky. It is asking for a 5 unit change in BMI
#also, for whatever reason my code seems to only show .975 CI? lets chagne that 
#looks like the cont_int() defaults at a .95 CI


coef = logistic_model.params['bmi']
ci_low, ci_high = logistic_model.conf_int().loc['bmi']

# scale to a 5-unit change
coef_5 = coef * 5
ci_low_5 = ci_low * 5
ci_high_5 = ci_high * 5

# exponentiate to get odds ratio
or_5 = np.exp(coef_5)
ci_5 = np.exp([ci_low_5, ci_high_5])

print(f"OR for 5-unit BMI increase: {or_5:.3f}")
print(f"95% CI: ({ci_5[0]:.3f}, {ci_5[1]:.3f})")

#now we are bringing in BMI squared

df['bmi_squared']= df['bmi'] ** 2

#now we are bringing in our BMI squared value to compare the results:

logistic_model_bmi_agesqared = smf.logit('death ~ bmi + bmi_squared', data=df).fit()

print(logistic_model_bmi_agesqared.summary())


# we we need a add both models together on a plot
bmi_range = np.linspace(df['bmi'].min(), df['bmi'].max(), 200)
pred_df = pd.DataFrame({'bmi': bmi_range, 'bmi_squared': bmi_range**2})

# get predicted probabilities from both models
pred_probs_linear = logistic_model.predict(pred_df)
pred_probs_quad = logistic_model_bmi_agesqared.predict(pred_df)  # adjust name if yours differs

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(bmi_range, pred_probs_linear, color='tab:blue', linestyle='-',
        linewidth=2, label='Linear (BMI only)')
ax.plot(bmi_range, pred_probs_quad, color='tab:orange', linestyle='--',
        linewidth=2, label='Quadratic (BMI + BMI²)')

ax.set_xlabel('BMI')
ax.set_ylabel('Predicted P(death)')
ax.set_title('Fitted Probability of Death: Linear vs. Quadratic BMI Model')
ax.legend()
plt.show()

#now we need to determine the impacts of 

params = logistic_model_bmi_agesqared.params  # use your actual model variable name

def log_odds(bmi):
    return params['Intercept'] + params['bmi']*bmi + params['bmi_squared']*bmi**2

# 20 to 25
log_or_20_25 = log_odds(25) - log_odds(20)
or_20_25 = np.exp(log_or_20_25)

# 25 to 30
log_or_30_35 = log_odds(35) - log_odds(30)
or_30_35 = np.exp(log_or_30_35)

print(f"OR for BMI 20 - 25: {or_20_25:.3f}")
print(f"OR for BMI 30 - 35: {or_30_35:.3f}")

#now we need to make an indicator variable for sex

# check unique values first to confirm coding
df['sex'].value_counts()

#now lets make an indicator variable for female to either be 1 or 0

df['Female'] = (df['sex'] == 2).astype(int)

#now lets me a box polot 

fig, ax = plt.subplots(figsize=(6, 5))

male_bmi = df.loc[df['Female'] == 0, 'bmi'].dropna()
female_bmi = df.loc[df['Female'] == 1, 'bmi'].dropna()

ax.boxplot([male_bmi, female_bmi], labels=['Male', 'Female'])
ax.set_ylabel('BMI')
ax.set_title('BMI Distribution by Sex')
plt.show()

#now we need to see if there is an association between sex and death


sex_logistic = smf.logit('death ~ bmi + Female', data=df).fit()




print(sex_logistic.summary())

#now we need to add the interaction term where my female is multiplied by bmi

interaction_model_bmi_sex = smf.logit('death ~ bmi * Female', data=df).fit()
print(interaction_model_bmi_sex.summary())


np.exp(0.0575) 

#now we are making three differnt models with three differnt versions of age

#Continuous age
model_age_continuous = smf.logit('death ~ age', data=df).fit()

#Age category treated as continuous/ordinal
model_agecat_ordinal = smf.logit('death ~ agecat', data=df).fit()

# 3. Age category treated as categorical/indicator
model_agecat_indicator = smf.logit('death ~ C(agecat)', data=df).fit()

df['agecat'].nunique()

df.groupby('agecat')['age'].agg(['min', 'max', 'count'])

#now lets plot
#model 1
age_range = np.linspace(df['age'].min(), df['age'].max(), 200)
pred_probs_cont = model_age_continuous.predict(pd.DataFrame({'age': age_range}))

#model 2
agecat_levels = sorted(df['agecat'].dropna().unique())
pred_df_cat = pd.DataFrame({'agecat': agecat_levels})

pred_probs_ordinal = model_agecat_ordinal.predict(pred_df_cat)
pred_probs_indicator = model_agecat_indicator.predict(pred_df_cat)

# use each category's mean age so points land in the right spot on the age x-axis
agecat_age_mean = df.groupby('agecat')['age'].mean().loc[agecat_levels]

fig, ax = plt.subplots(figsize=(8, 6))

# observed data, jittered so 0/1 points aren't all stacked
jitter = np.random.normal(0, 0.02, size=len(df))
ax.scatter(df['age'], df['death'] + jitter, alpha=0.15, s=10, color='gray', label='Observed (jittered)')

# three model predictions overlaid
ax.plot(age_range, pred_probs_cont, color='black', linewidth=2, label='Continuous age')
ax.plot(agecat_age_mean, pred_probs_ordinal, 'o-', color='tab:blue', markersize=8, linewidth=2, label='Age category (ordinal)')
ax.plot(agecat_age_mean, pred_probs_indicator, 's--', color='tab:orange', markersize=8, linewidth=2, label='Age category (indicator)')

ax.set_xlabel('Age')
ax.set_ylabel('Predicted P(death)')
ax.set_title('Comparing Age Specifications: Continuous vs. Ordinal vs. Indicator Category')
ax.legend()
plt.show()

print(model_age_continuous.aic, model_age_continuous.llf)
print(model_agecat_ordinal.aic, model_agecat_ordinal.llf)
print(model_agecat_indicator.aic, model_agecat_indicator.llf)

#ok now we are fitting a new model to determine if age * bmi is an effect modifer of death

#fist lets do our interaction
interaction_model_bmi_age = smf.logit('death ~ bmi * agecat', data=df).fit()

#now lets do no interaction
no_interaction_model_bmi_age = smf.logit('death ~ bmi + agecat', data=df).fit()

#now lets run summaries
interaction_model_bmi_age.summary()


no_interaction_model_bmi_age.summary()

#ok now we are brinigng int all together and building out models in incrasing complexity until we find the best one

#lets start w the most basic: death ~ bmi (already done)
logistic_model
logistic_model.summary()

#now we are adding continoues age 
model_bmi_age = smf.logit('death ~ bmi + age', data=df).fit()
model_bmi_age.summary()

#now we add the interaction between bmi and age

model_bmi_age_int = smf.logit('death ~ bmi * age', data=df).fit()
model_bmi_age_int.summary()

#now we add sex value

model_bmi_age_sex = smf.logit('death ~ bmi + age + Female', data=df).fit()
model_bmi_age_sex.summary()

#now we add control for age

model_full = smf.logit('death ~ bmi + age + Female + bmi:Female', data=df).fit()

model_bmi_age_sex.summary()

model_full.summary()

np.exp(-0.8488)

#building a new logistic regression with all models


full_model_all_variables = smf.logit('death ~ bmi + Female + age + cursmoke + cigpday + obese_normal + totchol + hyperten + diabetes + prevchd + prevmi + prevstrk', data=df).fit()

full_model_all_variables.summary()

#ok now we need to write code to see how much colinnearity there is between each coefficent
#thankfully statsmodels has a perfect little package for this


# use the exact design matrix statsmodels built from your formula
X = full_model_all_variables.model.exog
vif_data = pd.DataFrame({
    'Variable': full_model_all_variables.model.exog_names,
    'VIF': [variance_inflation_factor(X, i) for i in range(X.shape[1])]
})
print(vif_data)

#now we are going to do forward model select
#i am going to do this one by one to compare p values and AIC for practice
#in future versions, I think i will just use a for loop

null_model = smf.logit('death ~ 1', data=df).fit()
print(null_model.aic)

m_bmi = smf.logit('death ~ bmi', data=df).fit()
print(m_bmi.aic, m_bmi.pvalues['bmi'])


#actually, im realizing this is going to take too long. I think we should usea for loop


candidates = ['bmi', 'Female', 'age', 'cursmoke', 'cigpday', 'obese_normal',
              'totchol', 'hyperten', 'diabetes', 'prevchd', 'prevmi', 'prevstrk']

selected = []
remaining = candidates.copy()
current_aic = smf.logit('death ~ 1', data=df).fit(disp=0).aic

while remaining:
    aic_results = []
    for var in remaining:
        formula = 'death ~ ' + ' + '.join(selected + [var])
        model = smf.logit(formula, data=df).fit(disp=0)
        aic_results.append((model.aic, var))

    aic_results.sort()
    best_aic, best_var = aic_results[0]

    if best_aic < current_aic:
        selected.append(best_var)
        remaining.remove(best_var)
        current_aic = best_aic
        print(f"Added '{best_var}' AIC = {best_aic:.2f}")
    else:
        break

print("\nFinal selected variables (forward):", selected)
final_model_forward = smf.logit('death ~ ' + ' + '.join(selected), data=df).fit()
print(final_model_forward.summary())

#now lets do backwards using the for loop as well

selected = candidates.copy()  # start with all 12 variables
current_model = smf.logit('death ~ ' + ' + '.join(selected), data=df).fit(disp=0)
current_aic = current_model.aic

while len(selected) > 0:
    aic_results = []
    for var in selected:
        remaining_vars = [v for v in selected if v != var]
        formula = 'death ~ ' + ' + '.join(remaining_vars) if remaining_vars else 'death ~ 1'
        model = smf.logit(formula, data=df).fit(disp=0)
        aic_results.append((model.aic, var))

    aic_results.sort()
    best_aic, var_to_remove = aic_results[0]

    if best_aic < current_aic:
        selected.remove(var_to_remove)
        current_aic = best_aic
        print(f"Removed '{var_to_remove}' AIC = {best_aic:.2f}")
    else:
        break

print("\nFinal selected variables (backward):", selected)
final_model_backward = smf.logit('death ~ ' + ' + '.join(selected), data=df).fit()
print(final_model_backward.summary())

#ok now we are going to do stepeise
#seems like they are all going to have the same variables, p values and log likelihood

selected = candidates.copy()  # start with all 12 variables
current_model = smf.logit('death ~ ' + ' + '.join(selected), data=df).fit(disp=0)
current_aic = current_model.aic

while len(selected) > 0:
    aic_results = []
    for var in selected:
        remaining_vars = [v for v in selected if v != var]
        formula = 'death ~ ' + ' + '.join(remaining_vars) if remaining_vars else 'death ~ 1'
        model = smf.logit(formula, data=df).fit(disp=0)
        aic_results.append((model.aic, var))

    aic_results.sort()
    best_aic, var_to_remove = aic_results[0]

    if best_aic < current_aic:
        selected.remove(var_to_remove)
        current_aic = best_aic
        print(f"Removed '{var_to_remove}' AIC = {best_aic:.2f}")
    else:
        break

print("\nFinal selected variables (backward):", selected)
final_model_backward = smf.logit('death ~ ' + ' + '.join(selected), data=df).fit()
print(final_model_backward.summary())
