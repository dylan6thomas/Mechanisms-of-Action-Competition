
# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow import keras
from sklearn.utils import resample
from sklearn.naive_bayes import GaussianNB
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code]
train_features=pd.read_csv('train_features.csv')
train_targets=pd.read_csv('train_targets_scored.csv')
test_features=pd.read_csv('test_features.csv')
train_features=train_features.iloc[:,1:].values
test_features=test_features.iloc[:,1:].values

titles=pd.read_csv('train_targets_scored.csv')
titles=list(titles.columns)
labels=pd.read_csv('test_features.csv')
labels=list(labels.iloc[:,0].values)
print(len(labels))
targets={titles[0]:labels}
titles.pop(0)
print(len(titles))

# %% [code]
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0,2])],remainder='passthrough')
train_features=ct.fit_transform(train_features)
test_features=ct.transform(test_features)


# %% [code]
ss=StandardScaler()
train_features=ss.fit_transform(train_features)
test_features=ss.fit_transform(test_features)

# %% [code]
for col in range(len(titles)):
    train_target=train_targets.iloc[:,col+1].values
    target_majority,features_majority=train_target[train_target[:]==0],train_features[train_target[:]==0,:]
    target_minority,features_minority=train_target[train_target[:]==1],train_features[train_target[:]==1,:]
    scaled_target_minority=resample(target_minority,
                             replace=True,
                             n_samples=len(target_majority),
                             random_state=42)
    scaled_features_minority=resample(features_minority,
                             replace=True,
                             n_samples=len(target_majority),
                             random_state=42)
    scaled_train_target=pd.concat([pd.DataFrame(target_majority),pd.DataFrame(scaled_target_minority)])
    scaled_train_features=pd.concat([pd.DataFrame(features_majority),pd.DataFrame(scaled_features_minority)])
    print('Target',col+1)
    model=LogisticRegression()
    print('fitting...')
    model.fit(scaled_train_features,scaled_train_target)
    print(f1_score(scaled_train_target,model.predict(scaled_train_features).round()))
    pred=model.predict(test_features).round()

# %% [code]

    pred=list(pred)
    print(len(pred))
    targets[titles[col]]=pred

# %% [code]
output = pd.DataFrame(targets)
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

# %% [code]
