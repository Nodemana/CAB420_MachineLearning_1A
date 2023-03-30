import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay
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

# standardise data
X_train_std, mu_train_x, sigma_train_x = standardise(X_train)
X_val_std = (X_val - mu_train_x)/sigma_train_x #The issue is here, Data turns to NaN when standardised
X_test_std = (X_test - mu_train_x)/sigma_train_x

cknn = KNeighborsClassifier(n_neighbors=10, weights='uniform')
cknn.fit(X_train, Y_train)
eval_model(cknn, X_train, Y_train, X_test, Y_test, "uniform")


cknnd = KNeighborsClassifier(n_neighbors=198, weights='distance')
cknnd.fit(X_train, Y_train)
eval_model(cknnd, X_train, Y_train, X_test, Y_test, 'distance')

cknn = KNeighborsClassifier()
params = {'n_neighbors' : list(range(1,198)), 'weights' : ['uniform', 'distance']}
rand_search = RandomizedSearchCV(cknn, params, n_iter=20)
rand_search.fit(X_train, Y_train)
print(rand_search.cv_results_)

best_system = np.argmin(rand_search.cv_results_['rank_test_score'])
params = rand_search.cv_results_['params'][best_system]
print(params)
cknn = KNeighborsClassifier().set_params(**params)
cknn.fit(X_train, Y_train)
eval_model(cknn, X_train, Y_train, X_test, Y_test, "Best model: " + str(params))
plt.show()