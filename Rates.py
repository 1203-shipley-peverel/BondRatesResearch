import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.gofplots import qqplot
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import normaltest

#Import data from Excel file to Python
data = pd.read_excel('/mnt/c/Users/pevde/Documents/Research/rates.xls')

#Trains and tests the data
train_data = data['Difference'][:len(data)]
test_data = data['Difference'][len(data):]

#Creates model
ar_model = AutoReg(train_data, lags=1).fit()

#Prints out model results
print(ar_model.summary())

# Calculate residuals
residuals = ar_model.resid

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

#D'Agostino's normality test
DAstat, DAp = normaltest(residuals)
print(f"\nStatistic=  {DAstat}\np={DAp}")
if DAp > 0.05:
    print('The residuals are Gaussian (fail to reject null)')
else:
    print('The residuals are not Gaussian (reject null)')

plt.show()