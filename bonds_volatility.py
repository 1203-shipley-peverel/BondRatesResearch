import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro
from scipy.stats import jarque_bera
from statsmodels.tsa import stattools

# Load the data from the Excel spreadsheet
data = pd.read_excel("peverelData.xlsx")

# Align rows
data = data.dropna()

# Create dataframes for dependent and independent variables
aX = data[['Q(t-1)', 'V(t)']]
ay = data[['Q(t)']]
bX = data[['S(t-1)', 'V(t)']]
by = data[['S(t)']]
cX = data[['S+Q(t-1)', 'V(t)']]
cy = data[['S+Q']]

# Create initial model
model_a = sm.OLS(ay, aX).fit()
model_b = sm.OLS(by, bX).fit()
model_c = sm.OLS(cy, cX).fit()

# Print initial model summary
print(model_a.summary())
print(model_b.summary())
print(model_c.summary())

#Create residual variables
residuals_a = model_a.resid
residuals_b = model_b.resid
residuals_c = model_c.resid

# Insert residuals into dataframe
data['residuals_a'] = residuals_a
data['residuals_b'] = residuals_b
data['residuals_c'] = residuals_c

# Create new independent variables with residuals included
aX_resid = data[['Q(t-1)', 'V(t)', 'residuals_a']]
bX_resid = data[['Q(t-1)', 'V(t)', 'residuals_b']]
cX_resid = data[['Q(t-1)', 'V(t)', 'residuals_c']]

# Create new model with residuals
model_a_resid = sm.OLS(ay, aX_resid).fit()
model_b_resid = sm.OLS(by, bX_resid).fit()
model_c_resid = sm.OLS(cy, cX_resid).fit()

# Print new model summary
print(model_a_resid.summary())
print(model_b_resid.summary())
print(model_c_resid.summary())

# Create residuals variables for new model
residuals_a = model_a_resid.resid
residuals_b = model_b_resid.resid
residuals_c = model_c_resid.resid

# Create variables for absolute values of residuals
abs_residuals_a = np.abs(residuals_a)
abs_residuals_b = np.abs(residuals_b)
abs_residuals_c = np.abs(residuals_c)

# Breaker
print("\n(a)\n")

# ACF plots
plot_acf(residuals_a, lags=40, title='A Autocorrelation')
plot_acf(abs_residuals_a, lags=40, title='A Absolute Autocorrelation')
plt.show()

#Ljung-Box Test
acf_resida, stat_resida, p_resida, = stattools.acf(residuals_a, fft = False, qstat = True)
acf_absresida, stat_avsresida, p_absresida, = stattools.acf(abs_residuals_a, fft = False, qstat = True)
print('Ljung-Box p-value for residuals, original values')
print('lag 10 = ', p_resida[9])
print('Ljung-Box p-value for residuals, original values')
print('lag 10 = ', p_absresida[9])

# Shapiro-Wilk test
shapiro_stat, shapiro_p = shapiro(residuals_a)
print(f"\nShapiro-Wilk Test Statistic: {shapiro_stat}")
print(f"Shapiro-Wilk Test p-value: {shapiro_p}")

# Interpret the Shapiro-Wilk test result
if shapiro_p > 0.05:
    print("The residuals are Gaussian (fail to reject null).")
else:
    print("The residuals are not Gaussian (reject null).")

# Jarque-Bera test
jarque_stat, jarque_p = jarque_bera(residuals_a)
print(f"\nJarque-Bera Test Statistic: {jarque_stat}")
print(f"Jarque-Bera Test p-value: {jarque_p}")

# QQ plot vs the normal law
sm.qqplot(residuals_a, line='s')
plt.title(f'QQ Plot vs Normal Law for a')
plt.show()

# Breaker
print("\n(b)\n")

# Repeat of lines 70-94 for b
plot_acf(residuals_b, lags=40, title='B Autocorrelation')
plot_acf(abs_residuals_b, lags=40, title='B Absolute Autocorrelation')
plt.show()

#Ljung-Box Test
acf_residb, stat_residb, p_residb, = stattools.acf(residuals_b, fft = False, qstat = True)
acf_absresidb, stat_avsresidb, p_absresidb, = stattools.acf(abs_residuals_b, fft = False, qstat = True)
print('Ljung-Box p-value for residuals, original values')
print('lag 10 = ', p_residb[9])
print('Ljung-Box p-value for residuals, original values')
print('lag 10 = ', p_absresidb[9])

shapiro_stat, shapiro_p = shapiro(residuals_b)
print(f"\nShapiro-Wilk Test Statistic: {shapiro_stat}")
print(f"Shapiro-Wilk Test p-value: {shapiro_p}")

if shapiro_p > 0.05:
    print("The residuals are Gaussian (fail to reject null).")
else:
    print("The residuals are not Gaussian (reject null).")

jarque_stat, jarque_p = jarque_bera(residuals_b)
print(f"\nJarque-Bera Test Statistic: {jarque_stat}")
print(f"Jarque-Bera Test p-value: {jarque_p}")

sm.qqplot(residuals_b, line='s')
plt.title(f'QQ Plot vs Normal Law for b')
plt.show()

# Breaker
print("\n(c)\n")

# Repeat of lines 70-94 for c
plot_acf(residuals_c, lags=40, title='C Autocorrelation')
plot_acf(abs_residuals_c, lags=40, title='C Absolute Autocorrelation')
plt.show()

#Ljung-Box Test
acf_residc, stat_residc, p_residc, = stattools.acf(residuals_c, fft = False, qstat = True)
acf_absresidc, stat_avsresidc, p_absresidc, = stattools.acf(abs_residuals_c, fft = False, qstat = True)
print('Ljung-Box p-value for residuals, original values')
print('lag 10 = ', p_residc[9])
print('Ljung-Box p-value for residuals, original values')
print('lag 10 = ', p_absresidc[9])

shapiro_stat, shapiro_p = shapiro(residuals_c)
print(f"\nShapiro-Wilk Test Statistic: {shapiro_stat}")
print(f"Shapiro-Wilk Test p-value: {shapiro_p}")

if shapiro_p > 0.05:
    print("The residuals are Gaussian (fail to reject null).")
else:
    print("The residuals are not Gaussian (reject null).")

jarque_stat, jarque_p = jarque_bera(residuals_c)
print(f"\nJarque-Bera Test Statistic: {jarque_stat}")
print(f"Jarque-Bera Test p-value: {jarque_p}")

sm.qqplot(residuals_c, line='s')
plt.title(f'QQ Plot vs Normal Law for c')
plt.show()