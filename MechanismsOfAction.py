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
import tensorflow_addons as tfa
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code]


titles=pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')
titles=list(titles.columns)
labels=pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
labels=list(labels.iloc[:,0].values)
print(len(labels))
targets={titles[0]:labels}
titles.pop(0)
print(len(titles))

rounds=10
for iteration in range(rounds):
    train_features=pd.read_csv('/kaggle/input/lish-moa/train_features.csv')
    train_targets=pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')
    test_features=pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
    train_features=train_features.iloc[:,1:].values
    test_features=test_features.iloc[:,1:].values
    # %% [code]
    ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0,2])],remainder='passthrough')
    train_features=ct.fit_transform(train_features)
    test_features=ct.transform(test_features)

    # %% [code]
    train_target=train_targets.iloc[:,1:].values
    train_features,train_target=shuffle(train_features,train_target)
#     train_features,val_train_features,train_target,val_train_target=train_test_split(train_features,train_target,test_size=0.3,random_state=42)

    ss=StandardScaler()
    train_features=ss.fit_transform(train_features)
    test_features=ss.transform(test_features)
#     val_train_features=ss.transform(val_train_features)
    train_features=ss.inverse_transform(train_features)
    test_features=ss.inverse_transform(test_features)
#     val_train_features=ss.inverse_transform(val_train_features)

    # pca=PCA(100)
    # train_features=pca.fit_transform(train_features)
    # test_features=pca.transform(test_features)
    # val_train_features=pca.transform(val_train_features)

#     cor_matrix=pd.DataFrame(train_features).corr().abs()
#     upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
#     to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 1.0)]
#     train_features = np.delete(train_features,to_drop, axis=1)
#     test_features = np.delete(test_features,to_drop, axis=1)
#     val_train_features = np.delete(val_train_features,to_drop, axis=1)
#     print(train_features.shape)
    num_rows,num_col=train_features.shape

    model=keras.models.Sequential([
        keras.layers.Input(num_col),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),
        tfa.layers.WeightNormalization(keras.layers.Dense(4096,activation='relu')),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.9),
        tfa.layers.WeightNormalization(keras.layers.Dense(2048,activation='relu')),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.7),
        tfa.layers.WeightNormalization(keras.layers.Dense(206,activation='sigmoid'))
        ])
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0004),metrics=keras.metrics.BinaryCrossentropy(),loss='binary_crossentropy')
    print('fitting...')
#     history=model.fit(train_features,train_target,epochs=50,validation_data=(val_train_features,val_train_target),verbose=1)
    history=model.fit(train_features,train_target,epochs=50)
    # print(f1_score(train_target,model.predict(train_features).round(),average='micro'))
    pred=model.predict(test_features)

#     acc=history.history['binary_crossentropy']
#     valAcc=history.history['val_binary_crossentropy']
#     loss=history.history['loss']
#     valLoss=history.history['val_loss']
#     print(valAcc[-1])

#     epochs=range(len(acc))

#     plt.plot(epochs,acc,'r',label='Training Accuracy')
#     plt.plot(epochs,valAcc,'b',label='Validation Accuracy')
#     plt.title('Training and Validation accuracy')
#     plt.legend(loc=0)
#     plt.figure()
#     plt.plot(epochs,loss,'r',label='Training Loss')
#     plt.plot(epochs,valLoss,'b',label='Validation Loss')
#     plt.title('Training and Validation loss')
#     plt.legend(loc=0)
#     plt.show()


    # %% [code]
    
    print(pred)
    if iteration==0:
        new_pred=pd.DataFrame(pred)
    else:
        new_pred.add(pd.DataFrame(pred))
new_pred.div(rounds)
for col in range(len(titles)):
    column=list(new_pred.values[:,col])
    targets[titles[col]]=column


# %% [code]

output=pd.DataFrame(targets)
print(output)
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

# %% [code]
