import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
train = pd.read_csv("../input/train_V2.csv")
test = pd.read_csv("../input/test_V2.csv")
submit = pd.read_csv('../input/sample_submission_V2.csv')
train.dtypes
train.isnull().sum()
train.fillna(-999, inplace=True)
test.fillna(-999,inplace=True)
y = train['winPlacePerc']
X = train.drop(['Id','groupId','matchId','winPlacePerc'], axis=1)
x_test = test.drop(['Id','groupId','matchId'], axis=1)
from sklearn.model_selection import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.7, random_state=1234)
categorical_features_indices = np.where(X.dtypes != np.float)[0]

model=CatBoostRegressor(iterations=200, depth=8, learning_rate=0.008, loss_function='RMSE')
model.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_validation, y_validation))
output = pd.DataFrame()
output['Id'] = submit['Id']
output['winPlacePerc'] = model.predict(x_test)
output.to_csv('output.csv', index=False)
