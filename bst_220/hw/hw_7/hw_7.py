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
from scipy import stats
from scipy.stats import chi2

#new data babyyyyy

#lets get this new dataset going

#new data babyyyyy
#this is a wierd new format so we are gonna have to break out some new commands
df = pd.read_csv(
    "/Users/dustinh/Downloads/fromelungcancer.txt",
    sep=r"\s+",
    skiprows=11,
    names=["smokedur", "cigpday", "cases", "manyears"],
    engine="python",
)
df


df


df

#lets visualize it

#quick exploratory plots to get a feel for the data before modeling

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

#cases vs smoking duration, colored by cigarettes/day
sc1 = axes[0].scatter(df["smokedur"], df["cases"], c=df["cigpday"], cmap="viridis")
axes[0].set_xlabel("Smoking duration (years)")
axes[0].set_ylabel("Cases")
plt.colorbar(sc1, ax=axes[0], label="Cigarettes/day")

#cases vs cigarettes/day, colored by smoking duration
sc2 = axes[1].scatter(df["cigpday"], df["cases"], c=df["smokedur"], cmap="viridis")
axes[1].set_xlabel("Cigarettes per day")
axes[1].set_ylabel("Cases")
plt.colorbar(sc2, ax=axes[1], label="Smoking duration (years)")

#distribution of follow-up time across strata
axes[2].hist(df["manyears"], bins=15, color="steelblue", edgecolor="white")
axes[2].set_xlabel("Man-years of follow-up")
axes[2].set_ylabel("Number of strata")

plt.tight_layout()
plt.show()

#firs we need to to do a basic poisson regression with cigsperday regressed on cases

pois1 = smf.glm(
    "cases ~ cigpday",
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df["manyears"]),
).fit()
print(pois1.summary())

#I wwan to get a visual on this initial regression


#visualizing the Poisson model: observed rates vs. the fitted rate curve

df["rate"] = df["cases"] / df["manyears"]

cig_grid = pd.DataFrame({"cigpday": np.linspace(df["cigpday"].min(), df["cigpday"].max(), 100)})
fitted_rate = np.exp(pois1.params["Intercept"] + pois1.params["cigpday"] * cig_grid["cigpday"])

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df["cigpday"], df["rate"], s=df["manyears"] / 50, alpha=0.5,
           label="Observed rate (point size = man-years)")
ax.plot(cig_grid["cigpday"], fitted_rate, color="red", lw=2, label="Fitted Poisson rate")
ax.set_xlabel("Cigarettes per day")
ax.set_ylabel("Lung cancer rate (cases per man-year)")
ax.set_title("Observed vs. Fitted Lung Cancer Incidence Rate by Smoking Intensity")
ax.legend()
plt.tight_layout()
plt.show()

#next step cases regressed on cigpday and smokedur, with offset for exposure time in manyears

pois2 = smf.glm(
    "cases ~ cigpday + smokedur",
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df["manyears"]),
).fit()
print(pois2.summary())

#leets try to take a peek

#visualizing the two-predictor Poisson model: fitted rate curves by cigpday, one per smokedur level

df["rate"] = df["cases"] / df["manyears"]

smokedur_levels = sorted(df["smokedur"].unique())
colors = plt.cm.viridis(np.linspace(0, 1, len(smokedur_levels)))

fig, ax = plt.subplots(figsize=(9, 6))
cig_grid = np.linspace(df["cigpday"].min(), df["cigpday"].max(), 100)

for sd, color in zip(smokedur_levels, colors):
    sub = df[df["smokedur"] == sd]
    ax.scatter(sub["cigpday"], sub["rate"], color=color, s=sub["manyears"] / 50, alpha=0.6)
    fitted = np.exp(pois2.params["Intercept"] + pois2.params["cigpday"] * cig_grid + pois2.params["smokedur"] * sd)
    ax.plot(cig_grid, fitted, color=color, lw=2, label=f"smokedur={sd}")

ax.set_xlabel("Cigarettes per day")
ax.set_ylabel("Lung cancer rate (cases per man-year)")
ax.set_title("Fitted Poisson Rates by Cigarettes/Day, Stratified by Smoking Duration")
ax.legend(title="Smoking duration (yrs)", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.show()

#now we are running effect modification between cigday and smoker

pois_int = smf.glm(
    "cases ~ cigpday * smokedur",
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df["manyears"])
).fit()

print(pois_int.summary())

#now we are adding a qudartic covariate to our model

#first we smake or cigpday squared and our smokedur squared

df["cigpday_sq"] = df["cigpday"] ** 2
df["smokedur_sq"] = df["smokedur"] ** 2

#now lets fit our model

pois_quad = smf.glm(
    "cases ~ cigpday + cigpday_sq + smokedur + smokedur_sq",
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df["manyears"])
).fit()

print(pois_quad.summary())

#since our models are nested we need to run a likelihood ratio test

lr_stat = -2 * (pois2.llf - pois_quad.llf)
df_diff = pois_quad.df_model - pois2.df_model
p_val = chi2.sf(lr_stat, df_diff)

print("LR statistic:", lr_stat)
print("df difference:", df_diff)
print("p-value:", p_val)

#now lets fit the categorical (indicator) form of both covariates, no interaction or quadratic terms

pois_cat = smf.glm(
    "cases ~ C(cigpday) + C(smokedur)",
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df["manyears"])
).fit()

print(pois_cat.summary())

#point estimate and CI for IRR of 20.4 vs 0 cigarettes/day, adjusting for categorical smokedur

beta = pois_cat.params["C(cigpday)[T.20.4]"]
irr = np.exp(beta)
ci = np.exp(pois_cat.conf_int().loc["C(cigpday)[T.20.4]"])

print("IRR (point estimate):", irr)
print("95% CI:", ci.values)

#point estimate and CI for IRR of 40.8 vs 20.4 cigarettes/day, adjusting for categorical smokedur
#neither level is the reference, so we need a contrast between the two non-reference coefficients

contrast = "C(cigpday)[T.40.8] - C(cigpday)[T.20.4] = 0"
t = pois_cat.t_test(contrast)

beta_diff = t.effect[0]
ci_diff = t.conf_int()[0]

irr_diff = np.exp(beta_diff)
ci_diff_exp = np.exp(ci_diff)

print("IRR:", irr_diff)
print("95% CI:", ci_diff_exp)

#checking for overdispersion in models B (quadratic) and C (categorical)
#number of unique covariate patterns relative to sample size

n_patterns = df[["cigpday", "smokedur"]].drop_duplicates().shape[0]
print("number of unique covariate patterns:", n_patterns, "out of n =", len(df))

#empirical dispersion check for each model since patterns are sparse (large # of patterns relative to n)

for name, m in [("Model B (quadratic)", pois_quad), ("Model C (categorical)", pois_cat)]:
    df_resid = m.df_resid
    pearson_chi2 = m.pearson_chi2
    deviance = m.deviance
    print(name)
    print("  df_resid:", df_resid)
    print("  Pearson chi2:", pearson_chi2, " / df =", pearson_chi2 / df_resid)
    print("  Deviance:", deviance, " / df =", deviance / df_resid)
    print()

#LRT comparing model B (quadratic) to model C (categorical) since B is nested in C

lr_stat_bc = -2 * (pois_quad.llf - pois_cat.llf)
df_diff_bc = pois_cat.df_model - pois_quad.df_model
p_val_bc = chi2.sf(lr_stat_bc, df_diff_bc)

print("LRT B vs C: stat =", lr_stat_bc, "df =", df_diff_bc, "p =", p_val_bc)
print("Model B AIC:", pois_quad.aic, "BIC:", pois_quad.bic)
print("Model C AIC:", pois_cat.aic, "BIC:", pois_cat.bic)

#by hand: IRR for 20 vs 0 cigarettes/day using model B (quadratic), adjusting for linear/quadratic smokedur

b1 = pois_quad.params["cigpday"]
b2 = pois_quad.params["cigpday_sq"]
cov_quad = pois_quad.cov_params()
var1 = cov_quad.loc["cigpday", "cigpday"]
var2 = cov_quad.loc["cigpday_sq", "cigpday_sq"]
cov12 = cov_quad.loc["cigpday", "cigpday_sq"]

x1, x0 = 20, 0
c1 = x1 - x0
c2 = x1**2 - x0**2

log_irr_20v0 = b1 * c1 + b2 * c2
irr_20v0 = np.exp(log_irr_20v0)

var_20v0 = c1**2 * var1 + c2**2 * var2 + 2 * c1 * c2 * cov12
se_20v0 = np.sqrt(var_20v0)

lower_20v0 = np.exp(log_irr_20v0 - 1.96 * se_20v0)
upper_20v0 = np.exp(log_irr_20v0 + 1.96 * se_20v0)

print("IRR (20 vs 0 cigpday):", irr_20v0)
print("95% CI:", lower_20v0, upper_20v0)

#check against code
t_20v0 = pois_quad.t_test("20*cigpday + 400*cigpday_sq = 0")
print(t_20v0)

#by hand: IRR for 40 vs 20 cigarettes/day using model B (quadratic), adjusting for linear/quadratic smokedur

x1, x0 = 40, 20
c1 = x1 - x0
c2 = x1**2 - x0**2

log_irr_40v20 = b1 * c1 + b2 * c2
irr_40v20 = np.exp(log_irr_40v20)

var_40v20 = c1**2 * var1 + c2**2 * var2 + 2 * c1 * c2 * cov12
se_40v20 = np.sqrt(var_40v20)

lower_40v20 = np.exp(log_irr_40v20 - 1.96 * se_40v20)
upper_40v20 = np.exp(log_irr_40v20 + 1.96 * se_40v20)

print("IRR (40 vs 20 cigpday):", irr_40v20)
print("95% CI:", lower_40v20, upper_40v20)

#check against code
t_40v20 = pois_quad.t_test("20*cigpday + 1200*cigpday_sq = 0")
print(t_40v20)
