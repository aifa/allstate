__author__ = 'aifa'

import pandas as pd
import numpy as np
import pylab as pl

from sklearn import clone

# note: these imports are incorrect in the example online!
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.forest import (RandomForestClassifier,RandomForestRegressor,
                                        ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.cross_validation import StratifiedKFold
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

lr = LogisticRegression()

#svr =  SVR(C=1.0, epsilon=0.2)
#C = 10.0  # SVM regularization parameter
#svc = svm.SVC(kernel='linear', C=C)
#rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C)
#poly_svc = svm.SVC(kernel='poly', degree=3, C=C)

def filter_features(train_df, model, prodStr='A'):

    input_df = train_df.copy(deep=True)
    #input_df = cleanup(input_df)
    y = input_df[prodStr].values
    input_df=input_df.drop([prodStr], axis=1)
    print len(input_df.columns)
    X = input_df.values
    selector = RFE(model, step=1)
    selector = selector.fit(X, y)

#    rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(y, 2))
#    rfecv.fit(X, y)
#    print("Optimal number of features : %d" % rfecv.n_features_)

    idx=0
    for i in input_df.columns:
        if (selector.support_[idx]==False):
            input_df=input_df.drop([i], axis=1)
        idx += 1

    return input_df


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

    df=df.drop(['A_last'], axis=1)
    df=df.drop(['B_last'], axis=1)
    df=df.drop(['C_last'], axis=1)
    df=df.drop(['D_last'], axis=1)
    df=df.drop(['E_last'], axis=1)
    df=df.drop(['F_last'], axis=1)
    input_df=df.drop(['G_last'], axis=1)


    #impute the null values
    df['risk_factor'] = df['risk_factor'].fillna(0)
    df['C_previous'] = df['C_previous'].fillna(0)
    df['duration_previous'] = df['duration_previous'].fillna(0)
    df['location'] = df['location'].fillna(0)

    return df

def train_model(train_df, select_model, model, yName='A'):
    input_df = train_df.copy(deep=True)
    y = input_df[yName].values

    #input_df=filter_features(input_df, select_model, yName)
    if '_use_last' in yName:
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

    skf=StratifiedKFold(y, n_folds=3 )

    best_model=None
    for train_index, test_index in skf:
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = clone(model)
        clf = clf.fit(X_train, y_train)

        print yName +":"+str(clf.score(X_test,y_test))

        if best_model==None or best_model.score(X_test,y_test)<clf.score(X_test, y_test):
            best_model=clf

    trained_model = best_model

    return trained_model


def predict(test_2_df, select_model, model, prodStr='A'):

    test_df = test_2_df.copy(deep=True)
    test_y = test_df[prodStr]
    #test_df=filter_features(test_df, select_model, prodStr)
    if '_use_last' in prodStr:
        test_df=test_df.drop(['A_use_last'], axis=1)
        test_df=test_df.drop(['B_use_last'], axis=1)
        test_df=test_df.drop(['C_use_last'], axis=1)
        test_df=test_df.drop(['D_use_last'], axis=1)
        test_df=test_df.drop(['E_use_last'], axis=1)
        test_df=test_df.drop(['F_use_last'], axis=1)
        test_df=test_df.drop(['G_use_last'], axis=1)

    if prodStr=='total_offers':
        test_df=test_df.drop([prodStr], axis=1)
    else:
        test_df=test_df.drop(['A'], axis=1)
        test_df=test_df.drop(['B'], axis=1)
        test_df=test_df.drop(['C'], axis=1)
        test_df=test_df.drop(['D'], axis=1)
        test_df=test_df.drop(['E'], axis=1)
        test_df=test_df.drop(['F'], axis=1)
        test_df=test_df.drop(['G'], axis=1)

    if (prodStr=='G'):
        test_df=test_df.drop(['C_previous'], axis=1)


    test_X = test_df.values

    prediction = model.predict(test_X)

    #print prodStr +":"+str(prediction)

    return prediction


train = cleanup(train)

A_uLast_model=train_model(train, lr, rfcmodel,yName='A_use_last')
B_uLast_model=train_model(train, lr, rfcmodel,yName='B_use_last')
C_uLast_model=train_model(train, lr, rfcmodel,yName='C_use_last')
D_uLast_model=train_model(train, lr, rfcmodel,yName='D_use_last')
E_uLast_model=train_model(train, lr, rfcmodel,yName='E_use_last')
F_uLast_model=train_model(train, lr, rfcmodel,yName='F_use_last')
G_uLast_model=train_model(train, lr, rfcmodel,yName='G_use_last')



#meanOffers = train.total_offers.mean()
#meanDurationOffers = train.duration_offers.mean()
#meanPriceChange = train.price_change.mean()

#mean_A_change = train.A_change.mean()
#mean_B_change = train.B_change.mean()
#mean_C_change = train.C_change.mean()
#mean_D_change = train.D_change.mean()
#mean_E_change = train.E_change.mean()
#mean_F_change = train.F_change.mean()
#mean_G_change = train.G_change.mean()

mean_A_var = train.A_var.mean()
mean_B_var = train.B_var.mean()
mean_C_var = train.C_var.mean()
mean_D_var = train.D_var.mean()
mean_E_var = train.E_var.mean()
mean_F_var = train.F_var.mean()
mean_G_var = train.G_var.mean()

test_2 = cleanup(test_2)

A_use_last= predict(test_2,lr, A_uLast_model, prodStr='A_use_last')
B_use_last= predict(test_2,lr, B_uLast_model,prodStr='B_use_last')
C_use_last= predict(test_2,lr, C_uLast_model,prodStr='C_use_last')
D_use_last= predict(test_2,lr, D_uLast_model,prodStr='D_use_last')
E_use_last= predict(test_2,lr, E_uLast_model,prodStr='E_use_last')
F_use_last= predict(test_2,lr, F_uLast_model,prodStr='F_use_last')
G_use_last= predict(test_2,lr, G_uLast_model,prodStr='G_use_last')

test_2["A_use_last"] = pd.Series(data=A_use_last, index=test_2.index)
test_2["B_use_last"] = pd.Series(data=B_use_last, index=test_2.index)
test_2["C_use_last"] = pd.Series(data=C_use_last, index=test_2.index)
test_2["D_use_last"] = pd.Series(data=D_use_last, index=test_2.index)
test_2["E_use_last"] = pd.Series(data=E_use_last, index=test_2.index)
test_2["F_use_last"] = pd.Series(data=F_use_last, index=test_2.index)
test_2["G_use_last"] = pd.Series(data=G_use_last, index=test_2.index)

A_change_model=train_model(train, lr, rfcmodel, yName='A_change')
B_change_model=train_model(train, lr, rfcmodel, yName='B_change')
C_change_model=train_model(train, lr, rfcmodel, yName='C_change')
D_change_model=train_model(train, lr, rfcmodel, yName='D_change')
E_change_model=train_model(train, lr, rfcmodel, yName='E_change')
F_change_model=train_model(train, lr, rfcmodel, yName='F_change')
G_change_model=train_model(train, lr, rfcmodel, yName='G_change')

A_model=train_model(train, lr, rfcmodel)
B_model=train_model(train, lr, rfcmodel, yName='B')
C_model=train_model(train, lr, rfcmodel, yName='C')
D_model=train_model(train, lr, rfcmodel, yName='D')
E_model=train_model(train, lr, rfcmodel, yName='E')
F_model=train_model(train, lr, rfcmodel, yName='F')
G_model=train_model(train, lr, rfcmodel, yName='G')


A_change= predict(test_2,lr, A_change_model,prodStr='A_change')
B_change= predict(test_2,lr, B_change_model,prodStr='B_change')
C_change= predict(test_2,lr, C_change_model,prodStr='C_change')
D_change= predict(test_2,lr, D_change_model,prodStr='D_change')
E_change= predict(test_2,lr, E_change_model,prodStr='E_change')
F_change= predict(test_2,lr, F_change_model,prodStr='F_change')
G_change= predict(test_2,lr, G_change_model,prodStr='G_change')

A= predict(test_2,lr, A_model)
B= predict(test_2,lr, B_model,prodStr='B')
C= predict(test_2,lr, C_model,prodStr='C')
D= predict(test_2,lr, D_model,prodStr='D')
E= predict(test_2,lr, E_model,prodStr='E')
F= predict(test_2,lr, F_model,prodStr='F')
G= predict(test_2,lr, G_model,prodStr='G')

index = 0
correct=0
for i in range(0,np.size(A)):
    #strA=str(test_2.A.iat[i])
    #strB=str(test_2.B.iat[i])
    #strC=str(test_2.C.iat[i])
    #strD=str(test_2.D.iat[i])
    #strE=str(test_2.E.iat[i])
    #strF=str(test_2.F.iat[i])
    #strG=str(test_2.G.iat[i])


    #if test_2.A_change.iat[i]<A_change[i]:
    strA=str(A[i])
    #if test_2.B_change.iat[i]<B_change[i]:
    strB=str(B[i])
    #if test_2.C_change.iat[i]<C_change[i]:
    strC=str(C[i])
    #if test_2.D_change.iat[i]<D_change[i]:
    strD=str(D[i])
    #if test_2.E_change.iat[i]<E_change[i]:
    strE=str(E[i])
    #if test_2.F_change.iat[i]<F_change[i]:
    strF=str(F[i])
    #if test_2.G_change.iat[i]<G_change[i]:
    strG=str(G[i])

    resultCol.iat[i]=strA+strB+strC+strD+strE+strF+strG

    #Calculate number of correct predictions
    if test_2.A.iat[i]==float(strA) and test_2.B.iat[i]==float(strB) and test_2.C.iat[i]==float(strC) and test_2.D.iat[i]==float(strD)\
        and test_2.E.iat[i]==float(strE) and test_2.F.iat[i]==float(strF) and test_2.G.iat[i]==float(strG):
        correct += 1

res = float(correct)/float(len(test_2))
print(len(test_2))
print correct
print str(res)

results_df["plan"]=resultCol
results_df.to_csv("results_fail.csv")

