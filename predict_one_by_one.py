__author__ = 'aifa'

import os
import pandas as pd
import numpy as np
import pylab as pl

from sklearn import clone

# note: these imports are incorrect in the example online!
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.forest import (RandomForestClassifier,RandomForestRegressor,
                                        ExtraTreesClassifier)


from sklearn import svm

import math as math
from sklearn.externals.six.moves import xrange
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVR

train = pd.read_csv("./transformed/train_transformed_all_with_use_last_flag.csv", header=0, encoding="UTF-8", error_bad_lines=False, sep=",", index_col=0)
test_2 = pd.read_csv("./transformed/test_v2_transformed_all_with_use_last_flag.csv", header=0, encoding="UTF-8", error_bad_lines=False, sep=",", index_col=0)

results_df = pd.DataFrame(index=test_2.index)
resultCol = pd.Series(index=test_2.index,dtype=object)

#try random forest on feature A
n_estimators = 100

rfcmodel = RandomForestClassifier(n_estimators=n_estimators)
rfrmodel = RandomForestRegressor(n_estimators=n_estimators)



#svr =  SVR(C=1.0, epsilon=0.2)
#C = 10.0  # SVM regularization parameter
#svc = svm.SVC(kernel='linear', C=C)
#rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C)
#poly_svc = svm.SVC(kernel='poly', degree=3, C=C)

def cleanup(df):

    #find columns that contain null values
    inds = pd.isnull(df).any(0).nonzero()

    df['car_value'] = df['car_value'].fillna('0')
    df['car_value']=df['car_value'].apply(lambda letter :  abs(ord(letter) - 96))
    df['state']=df['state'].apply(lambda letter :  abs(ord(letter[0]) - 96 + ord(letter[1]) - 96))

    #remove product feature
    #df=df.drop(['product'], axis=1)
    #df=df.drop(['state'], axis=1)
    #df=df.drop(['location'], axis=1)
    df=df.drop(['time'], axis=1)

    #impute the null values
    df['risk_factor'] = df['risk_factor'].fillna(0)
    df['C_previous'] = df['C_previous'].fillna(0)
    df['duration_previous'] = df['duration_previous'].fillna(0)
    df['location'] = df['location'].fillna(0)

    return df

def train_model(train_df, model, yName='A'):
    input_df = train_df.copy(deep=True)
    y = input_df[yName].values

    input_df=input_df.drop(['A_use_last'], axis=1)
    input_df=input_df.drop(['B_use_last'], axis=1)
    input_df=input_df.drop(['C_use_last'], axis=1)
    input_df=input_df.drop(['D_use_last'], axis=1)
    input_df=input_df.drop(['E_use_last'], axis=1)
    input_df=input_df.drop(['F_use_last'], axis=1)
    input_df=input_df.drop(['G_use_last'], axis=1)

    input_df=input_df.drop(['A'], axis=1)
    input_df=input_df.drop(['B'], axis=1)
    input_df=input_df.drop(['C'], axis=1)
    input_df=input_df.drop(['D'], axis=1)
    input_df=input_df.drop(['E'], axis=1)
    input_df=input_df.drop(['F'], axis=1)
    input_df=input_df.drop(['G'], axis=1)

    if (yName=='G'):
        input_df=input_df.drop(['C_previous'], axis=1)

    X = input_df.values

    trained_model = clone(model)
    trained_model = trained_model.fit(X, y)

    return trained_model


def predict(test_2_df, model, prodStr='A'):

    test_df = test_2_df.copy(deep=True)
    test_y = test_df[prodStr]

    test_df=test_df.drop(['A_use_last'])
    test_df=test_df.drop(['B_use_last'])
    test_df=test_df.drop(['C_use_last'])
    test_df=test_df.drop(['D_use_last'])
    test_df=test_df.drop(['E_use_last'])
    test_df=test_df.drop(['F_use_last'])
    test_df=test_df.drop(['G_use_last'])

    if prodStr=='total_offers':
        test_df=test_df.drop([prodStr])
    else:
        test_df=test_df.drop(['A'])
        test_df=test_df.drop(['B'])
        test_df=test_df.drop(['C'])
        test_df=test_df.drop(['D'])
        test_df=test_df.drop(['E'])
        test_df=test_df.drop(['F'])
        test_df=test_df.drop(['G'])

    if (prodStr=='G'):
        test_df=test_df.drop(['C_previous'])

    test_X = test_df.values

    prediction = model.predict(test_X)

    #print prodStr +":"+str(prediction)

    return prediction


train = cleanup(train)
A_model=train_model(train, rfcmodel)
B_model=train_model(train, rfcmodel, yName='B')
C_model=train_model(train, rfcmodel, yName='C')
D_model=train_model(train, rfcmodel, yName='D')
E_model=train_model(train, rfcmodel, yName='E')
F_model=train_model(train, rfcmodel, yName='F')
G_model=train_model(train, rfcmodel, yName='G')

A_uLast_model=train_model(train, rfcmodel,yName='A_use_last')
B_uLast_model=train_model(train, rfcmodel,yName='B_use_last')
C_uLast_model=train_model(train, rfcmodel,yName='C_use_last')
D_uLast_model=train_model(train, rfcmodel,yName='D_use_last')
E_uLast_model=train_model(train, rfcmodel,yName='E_use_last')
F_uLast_model=train_model(train, rfcmodel,yName='F_use_last')
G_uLast_model=train_model(train, rfcmodel,yName='G_use_last')

correct=0

meanOffers = train.total_offers.mean()
meanDurationOffers = train.duration_offers.mean()
meanPriceChange = train.price_change.mean()

mean_A_change = train.A_change.mean()
mean_B_change = train.B_change.mean()
mean_C_change = train.C_change.mean()
mean_D_change = train.D_change.mean()
mean_E_change = train.E_change.mean()
mean_F_change = train.F_change.mean()
mean_G_change = train.G_change.mean()

mean_A_var = train.A_var.mean()
mean_B_var = train.B_var.mean()
mean_C_var = train.C_var.mean()
mean_D_var = train.D_var.mean()
mean_E_var = train.E_var.mean()
mean_F_var = train.F_var.mean()
mean_G_var = train.G_var.mean()

test_2 = cleanup(test_2)

for key, cdf in test_2.iterrows():

    A_use_last= predict(cdf,A_uLast_model, prodStr='A_use_last')
    B_use_last= predict(cdf,B_uLast_model,prodStr='B_use_last')
    C_use_last= predict(cdf,C_uLast_model,prodStr='C_use_last')
    D_use_last= predict(cdf,D_uLast_model,prodStr='D_use_last')
    E_use_last= predict(cdf,E_uLast_model,prodStr='E_use_last')
    F_use_last= predict(cdf,F_uLast_model,prodStr='F_use_last')
    G_use_last= predict(cdf,G_uLast_model,prodStr='G_use_last')


    A= predict(cdf,A_model)
    B= predict(cdf,B_model,prodStr='B')
    C= predict(cdf,C_model,prodStr='C')
    D= predict(cdf,D_model,prodStr='D')
    E= predict(cdf,E_model,prodStr='E')
    F= predict(cdf,F_model,prodStr='F')
    G= predict(cdf,G_model,prodStr='G')

    if A_use_last==1:
        A[0]=cdf.A
    if B_use_last==1:
        B[0]=cdf.B
    if C_use_last==1:
        C[0]=cdf.C
    if D_use_last==1:
        D[0]=cdf.D
    if E_use_last==1:
        E[0]=cdf.E
    if F_use_last==1:
        F[0]=cdf.F
    if G_use_last==1:
        G[0]=cdf.G

    resultCol.loc[key]=str(A[0])+str(B[0])+str(C[0])+str(D[0])+str(E[0])+str(F[0])+str(G[0])

    if test_2.A.loc[key]==A and test_2.B.loc[key]==B and test_2.C.loc[key]==C and test_2.D.loc[key]==D\
        and test_2.E.loc[key]==E and test_2.F.loc[key]==F and test_2.G.loc[key]==G:
        correct += 1


res = float(correct)/float(len(test_2))
print(len(test_2))
print correct
print str(res)

results_df["plan"]=resultCol
results_df.to_csv("results.csv")


