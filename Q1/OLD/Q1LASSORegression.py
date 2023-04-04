import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api
import statsmodels.formula.api as sm
import statsmodels
import scipy.stats as stats

def plot_rmses(lambdas, rmse_train, rmse_validation):
    fig = plt.figure(figsize=[20, 8])
    #ax = fig.add_subplot(2, 1, 1)
    #ax.plot(lambdas, rmse_train, label='Training RMSE')
    #ax.plot(lambdas, rmse_validation, label='Validation RMSE')
    #ax.legend();
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(lambdas[1:], rmse_train[1:], label='Training RMSE')
    ax.plot(lambdas[1:], rmse_validation[1:], label='Validation RMSE')
    ax.legend()

def standardise(data):
  """ Standardise/Normalise data to have zero mean and unit variance
  Args:
    data (np.array):
      data we want to standardise (usually covariates)
    Returns:
      Standardised data, mean of data, standard deviation of data
  """
  mu = np.mean(data, axis=0)
  sigma = np.std(data, axis=0)
  scaled = (data - mu) / sigma
  return scaled, mu, sigma

train = pd.read_csv('../Data/communities_train.csv')
val = pd.read_csv('../Data/communities_val.csv')
test = pd.read_csv('../Data/communities_test.csv')

X_train = train.iloc[:,0:-1]
Y_train = train.iloc[:,-1]
X_val = val.iloc[:,0:-1]
y_val = val.iloc[:,-1]
X_test = test.iloc[:,0:-1]
y_test = test.iloc[:,-1]

print(np.shape(X_train))
print(X_train)
X_train, Xtrain_mean, Xtrain_stdev = standardise(X_train)
X_val, Xval_mean, Xval_stdev = standardise(X_val)
X_test, Xtest_mean, Xtest_stdev = standardise(X_test)
print(X_train)

#Create model with Lambda = 1
str = "ViolentCrimesPerPop ~" + "+".join(X_train.columns)
model = sm.ols(formula=str, data=train).fit_regularized(alpha=0.02, L1_wt=1)

Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)
rmse_train = np.sqrt(np.mean((Y_train_pred - Y_train)**2))
rmse_test = np.sqrt(np.mean((Y_test_pred - y_test)**2))

# Plot Train vs Test
fig = plt.figure(figsize=[20, 8])
ax = fig.add_subplot(2, 1, 1)
ax.plot(np.arange(len(Y_train_pred)), Y_train_pred, label='Train Predicted')
ax.plot(np.arange(len(Y_train_pred)), Y_train, label='Actual')
ax.set_title(rmse_train)
ax.legend()
ax = fig.add_subplot(2, 1, 2)
ax.plot(np.arange(len(Y_test_pred)), Y_test_pred, label='Test Predicted')
ax.plot(np.arange(len(Y_test_pred)), y_test, label='Actual')
ax.set_title(rmse_test)
ax.legend()


lambdas = np.arange(0.0, 0.005, 0.0001)
rmse_train = []
rmse_validation = []
coeffs = []
for l in lambdas:
    trained_model_poly_lasso = statsmodels.api.OLS(Y_train, X_train).fit_regularized(alpha=l, L1_wt=1.0)
    coeffs.append(trained_model_poly_lasso.params)
    rmse_train.append(np.sqrt(np.mean((trained_model_poly_lasso.predict(X_train) - Y_train)**2)))
    rmse_validation.append(np.sqrt(np.mean((trained_model_poly_lasso.predict(X_val) - y_val)**2)))
    
plot_rmses(lambdas, rmse_train, rmse_validation)


plt.show()

#Useful Plot Function
def plot_diagnoistics(trained_model, X_train, Y_train, X_test, Y_test):

    fig = plt.figure(figsize=[20, 8])
    ax = fig.add_subplot(1, 2, 1)
    f = sm.qqplot(trained_model.resid, ax=ax)
    ax = fig.add_subplot(1, 2, 2)
    ax.hist(trained_model.resid, 50)

    Y_train_pred = trained_model.predict(X_train)
    Y_test_pred = trained_model.predict(X_test)
    rmse_train = np.sqrt(np.mean((Y_train_pred - Y_train)**2))
    rmse_test = np.sqrt(np.mean((Y_test_pred - Y_test)**2))

    fig = plt.figure(figsize=[20, 8])
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(np.arange(len(Y_train_pred)), Y_train_pred, label='Predicted')
    ax.plot(np.arange(len(Y_train_pred)), Y_train, label='Actual')
    ax.set_title(rmse_train)
    ax.legend()
    
    ax = fig.add_subplot(2, 1, 2)
    ax.plot(np.arange(len(Y_test_pred)), Y_test_pred, label='Predicted')
    ax.plot(np.arange(len(Y_test_pred)), Y_test, label='Actual')
    ax.set_title(rmse_test)
    ax.legend()