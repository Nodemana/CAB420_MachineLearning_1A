import pandas as pd

train = pd.read_csv('../Data/communities_train.csv')
val = pd.read_csv('../Data/communities_val.csv')
test = pd.read_csv('../Data/communities_test.csv')

# Remove first and last letter from each column
train.columns = [col[1:-1] for col in train.columns]
val.columns = [col[1:-1] for col in val.columns]
test.columns = [col[1:-1] for col in test.columns]

#train.to_csv('../Data/communities_train.csv', index=False)
#val.to_csv('../Data/communities_val.csv', index=False)
#test.to_csv('../Data/communities_test.csv', index=False)