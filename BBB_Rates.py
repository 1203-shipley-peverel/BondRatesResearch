import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import shapiro, normaltest
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.api import AutoReg

# Load data from an Excel file
data = pd.read_excel('/mnt/c/Users/pevde/Documents/Research/BBB.xls')

# Perform linear regression of Mortgage Rates on AAA and BBB bonds
X = data[['AAA', 'BBB']]
X = sm.add_constant(X)  # Add a constant term (intercept)
y = data['MORTGAGE30US']

model = sm.OLS(y, X).fit()

# Print regression summary
print(model.summary())

# Calculate residuals
residuals = model.resid

# Calculate mean and standard deviation of residuals
residual_mean = residuals.mean()
residual_std = residuals.std()

# Print out residuals, mean, and stdev
print(residuals)
print(f"\nMean of Residuals: {residual_mean}")
print(f"Standard Deviation of Residuals: {residual_std}")

# Create and display the Q-Q plot
qqplot(residuals, line='s')
plt.title('Q-Q Plot of Residuals')

# Apply the Shapiro-Wilk test to assess normality
shapiro_stat, shapiro_p = shapiro(residuals)
print(f"\nShapiro-Wilk Test Statistic: {shapiro_stat}")
print(f"Shapiro-Wilk Test p-value: {shapiro_p}")

# Interpret the Shapiro-Wilk test result
if shapiro_p > 0.05:
    print("The residuals are Gaussian (fail to reject null).")
else:
    print("The residuals are not Gaussian (reject null).")

# Create and display the ACF plot of residuals
plot_acf(residuals, lags=40)
plt.title('ACF Plot of Residuals')

# D'Agostino's normality test
DAstat, DAp = normaltest(residuals)
print(f"\nStatistic=  {DAstat}\np={DAp}")
if DAp > 0.05:
    print('The residuals are Gaussian (fail to reject null)')
else:
    print('The residuals are not Gaussian (reject null)')

# Trains and tests the data
train_data = residuals.iloc[:int(0.8 * len(residuals))]
test_data = residuals.iloc[int(0.8 * len(residuals)):]

#Creates model
ar_results = AutoReg(train_data, lags=1).fit()

# Get AR(1) model summary
print(ar_results.summary())

# Calculate AR(1) residuals
ar_residuals = ar_results.resid

# Calculate mean and standard deviation of residuals
ar_residual_mean = ar_residuals.mean()
ar_residual_std = ar_residuals.std()

# Print out residuals, mean, and stdev
print(residuals)
print(f"\nMean of Residuals: {ar_residual_mean}")
print(f"Standard Deviation of Residuals: {ar_residual_std}")

# Create and display the Q-Q plot for AR(1) residuals
qqplot(ar_residuals, line='s')
plt.title('Q-Q Plot of AR(1) Residuals')

# Apply the Shapiro-Wilk test to AR(1) residuals
shapiro_stat_ar, shapiro_p_ar = shapiro(ar_residuals)
print(f"\nShapiro-Wilk Test Statistic (AR(1) Residuals): {shapiro_stat_ar}")
print(f"Shapiro-Wilk Test p-value (AR(1) Residuals): {shapiro_p_ar}")

# Interpret the Shapiro-Wilk test result for AR(1) residuals
if shapiro_p_ar > 0.05:
    print("The AR(1) residuals are Gaussian (fail to reject null).")
else:
    print("The AR(1) residuals are not Gaussian (reject null).")

# Create and display the ACF plot for AR(1) residuals
plot_acf(ar_residuals, lags=40)
plt.title('ACF Plot of AR(1) Residuals')

plt.show()