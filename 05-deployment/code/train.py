#!/usr/bin/env python
# coding: utf-8

import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# parameters

C = 1.0
n_splits = 5
output_file = f'model_C={C}.bin'


# data preparation

df = pd.read_csv('../../03-classification/data-week-3.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)


numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]

# training 

def train(df_train, y_train, C=1.0):
    preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False), categorical),
        ('num', 'passthrough', numerical)
    ]
)

    # Fit and transform the training data
    X_train = preprocessor.fit_transform(df_train[categorical + numerical])

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    return preprocessor, model


def predict(df, transformer, model):
    X = transformer.transform(df)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# validation

print(f'doing validation with C={C}')

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

fold = 0

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

    print(f'auc on fold {fold} is {auc}')
    fold = fold + 1


print('validation results:')
print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


# training the final model

print('training the final model')

preprocessor, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, preprocessor, model)

y_test = df_test.churn.values
auc = roc_auc_score(y_test, y_pred)

print(f'auc={auc}')


# Save the model

with open(output_file, 'wb') as f_out:
    pickle.dump((preprocessor, model), f_out)

print(f'the model is saved to {output_file}')
