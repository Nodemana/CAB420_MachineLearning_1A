import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV

def eval_model(model, X_train, Y_train, X_test, Y_test, title):
    fig = plt.figure(figsize=[25, 8])
    ax = fig.add_subplot(1, 2, 1)
    fig.suptitle(title)
    conf = ConfusionMatrixDisplay.from_estimator(model, X_train, Y_train, normalize='true', ax=ax)
    pred = model.predict(X_train)
    conf.ax_.set_title('Training Set Performance: ' + str(sum(pred == Y_train)/len(Y_train)));
    ax = fig.add_subplot(1, 2, 2)
    conf = ConfusionMatrixDisplay.from_estimator(model, X_test, Y_test, normalize='true', ax=ax)
    pred = model.predict(X_test)
    conf.ax_.set_title('Test Set Performance: ' + str(sum(pred == Y_test)/len(Y_test)));

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

train = pd.read_csv('Data/training.csv')
val = pd.read_csv('Data/validation.csv')
test = pd.read_csv('Data/testing.csv')

X_train = train.iloc[:,1:]
Y_train = train.iloc[:,0]
X_val = val.iloc[:,1:]
Y_val = val.iloc[:,0]
X_test = test.iloc[:,1:]
Y_test = test.iloc[:,0]

# plot box plot for the data
fig = plt.figure(figsize=[25, 8])
ax = fig.add_subplot(1, 2, 1)
ax.boxplot(X_train)
ax.set_title('Raw Data')


# standardise data
X_train_std, mu_train_x, sigma_train_x = standardise(X_train)
X_val_std = (X_val - mu_train_x)/sigma_train_x #The issue is here, Data turns to NaN when standardised
X_test_std = (X_test - mu_train_x)/sigma_train_x


ax = fig.add_subplot(1, 2, 2)
ax.boxplot(X_train_std)
ax.set_title('Data after standardisation');


svm = SVC(C=10.0, kernel='rbf')
svm.fit(X_train_std, Y_train)
eval_model(svm, X_train_std, Y_train, X_test_std, Y_test, 'RBF')

svmp = SVC(C=1.0, kernel='poly')
svmp.fit(X_train, Y_train)
eval_model(svmp, X_train, Y_train, X_test, Y_test, 'Polynomial')

svms = SVC(C=1.0, kernel='sigmoid')
svms.fit(X_train, Y_train)
eval_model(svms, X_train, Y_train, X_test, Y_test, 'Sigmoid')

svml = SVC(C=1.0, kernel='linear')
svml.fit(X_train, Y_train)
eval_model(svml, X_train, Y_train, X_test, Y_test, 'Linear')
"""
param_grid = [
  {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']},
  {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001], 'degree': [3, 4, 5, 6], 'kernel': ['poly']},
  {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001], 'kernel': ['sigmoid']},
 ]
 """

#param_grid = [{'C': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 'gamma': [0.00008, 0.00009, 0.00011, 0.0001], 'kernel': ['rbf']},]
#param_grid = [{'C': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 'gamma': [0.00014, 0.00013, 0.00011, 0.00012], 'kernel': ['rbf']},]
param_grid = [{'C': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 'gamma': [0.00014, 0.00015, 0.00016, 0.00017, 0.00018, 0.00019], 'kernel': ['rbf']},]
svm = SVC()
grid_search = GridSearchCV(svm, param_grid)
grid_search.fit(X_train, Y_train)
grid_search.cv_results_

best_system = np.argmin(grid_search.cv_results_['rank_test_score'])
params = grid_search.cv_results_['params'][best_system]
print(params)
svm = SVC().set_params(**params)
svm.fit(X_train, Y_train)
eval_model(svm, X_train, Y_train, X_test, Y_test, 'Best Model: ' + str(svm.kernel))


plt.show()