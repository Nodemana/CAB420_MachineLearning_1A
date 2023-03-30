import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api
import statsmodels.formula.api as sm
import statsmodels
import scipy.stats as stats

train = pd.read_csv('Data/communities_train.csv')
val = pd.read_csv('Data/communities_val.csv')
test = pd.read_csv('Data/communities_test.csv')

train.columns = [col.replace(' ', '') for col in train.columns] #pre-prosessing to get rid of the spaces around the words in the columns
val.columns = [col.replace(' ', '') for col in val.columns]
test.columns = [col.replace(' ', '') for col in test.columns]

X_train = train.iloc[:,0:-1]
Y_train = train.iloc[:,-1]
X_val = val.iloc[:,0:-1]
y_val = val.iloc[:,-1]
X_test = test.iloc[:,0:-1]
y_test = test.iloc[:,-1]



str = "ViolentCrimesPerPop ~" + "+".join(X_train.columns)
model = sm.ols(formula=str, data=train).fit()
print(model.summary())

predictions = model.predict(X_test)
results = pd.DataFrame({'y_test': y_test, 'predictions': predictions})
print(results)
f = statsmodels.api.qqplot(model.resid)

fig = plt.figure(figsize=[8, 8])
ax = fig.add_subplot(1, 1, 1)
ax.hist(model.resid, 20)
plt.show()
#plt.plot(results['y_test'], results['predictions'])
#plt.show()