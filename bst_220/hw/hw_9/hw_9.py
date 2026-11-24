##importing package

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
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

#lets get this new dataset going
df = pd.read_csv("/Users/dustinh/Downloads/Q1data.csv")

df

# If one row per time point: expand counts into one row per person (32 rows)
times = np.repeat(df["Time"], df["Dead"])
status = np.ones(len(times))   # 1 = event happened (no censoring)

kmf = KaplanMeierFitter()
kmf.fit(times, event_observed=status)
print(kmf.survival_function_)   # should match your S(t) column

kmf.plot_survival_function(ci_show=False, legend=False)
plt.xlabel("Time (days)")
plt.ylabel("S(t)")
plt.title("Estimated Survival Function")
plt.xlim(0, 15)
plt.savefig("survival_curve.png", dpi=300)
plt.show()

#now we need to calculate median, mean, percentile

print("Mean:  ", np.mean(times))
print("Median:", np.percentile(times, 50, method="inverted_cdf"))
print("25th:  ", np.percentile(times, 25, method="inverted_cdf"))
print("75th:  ", np.percentile(times, 75, method="inverted_cdf"))

#importing the second data set
df2 = pd.read_csv("/Users/dustinh/Downloads/Q2data.csv")
df2

time, event = df2["respdays"], df2["response"]

#there is a kaplanmeir class to bring in

kmf = KaplanMeierFitter()
kmf.fit(time, event_observed=event)

tab = kmf.event_table
tab = tab[tab["observed"] > 0].copy()
r, d = tab["at_risk"], tab["observed"]
greenwood = (d / (r * (r - d))).cumsum()   # infinite at the last time (r = d)
S = kmf.survival_function_.loc[tab.index, "KM_estimate"]
z = 1.96

v = np.sqrt(greenwood) / np.abs(np.log(S))
ll_lo = S ** np.exp(z * v)
ll_hi = S ** np.exp(-z * v)

se = S * np.sqrt(greenwood)
lin_lo = S - z * se
lin_hi = S + z * se

out = pd.DataFrame({"S(t)": S, "loglog_lower": ll_lo, "loglog_upper": ll_hi,
                    "linear_lower": lin_lo, "linear_upper": lin_hi}).round(3)

print(out)

#ok now it's time to plot new data

ax = kmf.plot_survival_function(ci_show=True, show_censors=True, legend=False)
ax.set_xlabel("Days since randomization")
ax.set_ylabel(r"$\hat{S}(t)$")
ax.set_title("Kaplan-Meier Estimate with 95% Confidence Interval (3-drug arm)")
ax.set_ylim(0, 1.05)
plt.savefig("km_curve_q2c.png", dpi=300, bbox_inches="tight")
plt.show()

print(kmf.confidence_interval_.round(3))

#we need to add some lines at 25 and 75 percentile
ax = kmf.plot_survival_function(ci_show=True, show_censors=True, legend=False)

# percentile lines: across from the y-axis, then down to the x-axis
for level, t in [(0.75, 22), (0.50, 90), (0.25, 180)]:
    ax.hlines(level, 0, t, color="gray", linestyle="--", linewidth=1)
    ax.vlines(t, 0, level, color="gray", linestyle="--", linewidth=1)

ax.set_xlabel("Days since randomization")
ax.set_ylabel(r"$\hat{S}(t)$")
ax.set_title("KM Estimate with 95% CI and Percentiles (3-drug arm)")
ax.set_ylim(0, 1.05)
plt.savefig("km_percentiles_q2d.png", dpi=300, bbox_inches="tight")
plt.show()

#now we need to pull exact percetnages for the next question

for t in [22, 90, 180]:
    print(t, round(kmf.survival_function_at_times(t).iloc[0], 3))

#now we are using our next dataset

df3 = pd.read_stata("/Users/dustinh/Downloads/MIstudy.dta")

df3

#fittin the cox proportional model

print(df3[["dthtime", "dthstat", "obese_ovwt"]].head())
print(df3["dthstat"].value_counts())   # should be 0s and 1s

#Q3A: simple (unadjusted) Cox model with obese_ovwt as the only covariate
cox_a = CoxPHFitter()
cox_a.fit(df3[["dthtime", "dthstat", "obese_ovwt"]],
          duration_col="dthtime", event_col="dthstat")
cox_a.print_summary()

#lets get the test statistics

# LR statt
lr = cox_a.log_likelihood_ratio_test()
print("LR:   ", round(lr.test_statistic, 3), " p =", lr.p_value)

# Score stat
from lifelines.statistics import logrank_test
g1, g0 = df3[df3["obese_ovwt"] == 1], df3[df3["obese_ovwt"] == 0]
score = logrank_test(g1["dthtime"], g0["dthtime"],
event_observed_A=g1["dthstat"], event_observed_B=g0["dthstat"])
print("Score:", round(score.test_statistic, 3), " p =", score.p_value)

#time to make another plot

dat3 = df3[["dthtime", "dthstat", "obese_ovwt"]].dropna()

fig, ax = plt.subplots(figsize=(7, 5))
for g, label in [(0, "Normal weight (obese_ovwt = 0)"),
                 (1, "Overweight/obese (obese_ovwt = 1)")]:
    sub = dat3[dat3["obese_ovwt"] == g]
    km = KaplanMeierFitter().fit(sub["dthtime"], event_observed=sub["dthstat"])
    s = km.survival_function_["KM_estimate"]
    s = s[(s > 0) & (s < 1) & (s.index > 0)]   # log(-log S) and log(t) need 0 < S < 1, t > 0
    ax.step(np.log(s.index), np.log(-np.log(s)), where="post", label=label)

ax.set_xlabel("log(t)  [t = months]")
ax.set_ylabel("log[-log(S(t))]")
ax.set_title("log(-log) Survival Plot by Overweight/Obese Status")
ax.legend()
plt.savefig("loglog_q3b.png", dpi=300, bbox_inches="tight")
plt.show()

#plot looks good ,let's evaluate the values

# same data and model as Part A
dat3 = df3[["dthtime", "dthstat", "obese_ovwt"]].dropna()
cox_a = CoxPHFitter()
cox_a.fit(dat3, duration_col="dthtime", event_col="dthstat")

# Part B(ii): weighted (scaled) Schoenfeld residual test
ph = proportional_hazard_test(cox_a, dat3, time_transform="rank")
ph.print_summary()

#ok now we are getting more complex with our models

cols = ["dthtime", "dthstat", "obese_ovwt", "age", "gender", "sysbp", "mitype"]
dat3c = df3[cols].dropna()

cox_c = CoxPHFitter()
cox_c.fit(dat3c, duration_col="dthtime", event_col="dthstat")
cox_c.print_summary(decimals=3)

print("Unadjusted (Part A):")
print(cox_a.summary.loc["obese_ovwt", ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].round(3))

print("\nAdjusted (Part C):")
print(cox_c.summary.loc["obese_ovwt", ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].round(3))

print("\nMean age by overweight/obese status:")
print(dat3c.groupby("obese_ovwt")["age"].mean().round(1))
