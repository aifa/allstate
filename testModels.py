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

lr = LogisticRegression()

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

def filter_features(train_df, model, prodStr='A'):

    input_df = train_df.copy(deep=True)
    input_df = cleanup(input_df)
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

def predict(train_df, test_2_df, select_model, model, prodStr='A'):

    input_df = train_df.copy(deep=True)
    test_df = test_2_df.copy(deep=True)

    input_df = cleanup(input_df)
    test_df = cleanup(test_df)

    y = input_df[prodStr].values

    if "_use_last" in prodStr:
        input_df=input_df.drop(['A_use_last'], axis=1)
        input_df=input_df.drop(['B_use_last'], axis=1)
        input_df=input_df.drop(['C_use_last'], axis=1)
        input_df=input_df.drop(['D_use_last'], axis=1)
        input_df=input_df.drop(['E_use_last'], axis=1)
        input_df=input_df.drop(['F_use_last'], axis=1)
        input_df=input_df.drop(['G_use_last'], axis=1)

    if prodStr=='total_offers':
        input_df=input_df.drop([prodStr], axis=1)
    else:
        input_df=input_df.drop(['A'], axis=1)
        input_df=input_df.drop(['B'], axis=1)
        input_df=input_df.drop(['C'], axis=1)
        input_df=input_df.drop(['D'], axis=1)
        input_df=input_df.drop(['E'], axis=1)
        input_df=input_df.drop(['F'], axis=1)
        input_df=input_df.drop(['G'], axis=1)
    if (prodStr=='G'):
        input_df=input_df.drop(['C_previous'], axis=1)


  #  input_df=filter_features(input_df, select_model, prodStr)

    X = input_df.values

    test_y = test_df[prodStr].values

    if "_use_last" in prodStr:
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

    test_df=filter_features(test_df, select_model, prodStr)

    test_X = test_df.values

    # Shuffle
    idx = np.arange(X.shape[0])
    np.random.seed(13)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]


    clf = clone(model)
    clf = clf.fit(X, y)

    prediction = clf.predict(test_X)

    #print prodStr +":"+str(clf.score(test_X,test_y))

    #z=[tree==test_y]

    return prediction

#A_uLast=predict(train, test_2, rfcmodel,prodStr='A_use_last')
#B_uLast=predict(train, test_2, rfcmodel,prodStr='B_use_last')
#C_uLast=predict(train, test_2, rfcmodel,prodStr='C_use_last')
#D_uLast=predict(train, test_2, rfcmodel,prodStr='D_use_last')
#E_uLast=predict(train, test_2, rfcmodel,prodStr='E_use_last')
#F_uLast=predict(train, test_2, rfcmodel,prodStr='F_use_last')
#G_uLast=predict(train, test_2, rfcmodel,prodStr='G_use_last')
#A_change=predict(train, test_2,rfcmodel,prodStr='A_change')
#B_change=predict(train, test_2,rfcmodel,prodStr='B_change')
#C_change=predict(train, test_2,rfcmodel,prodStr='C_change')
#D_change=predict(train, test_2,rfcmodel,prodStr='D_change')
#E_change=predict(train, test_2,rfcmodel,prodStr='E_change')
#F_change=predict(train, test_2,rfcmodel,prodStr='F_change')
#G_change=predict(train, test_2,rfcmodel,prodStr='G_change')
#A_var=predict(train, test_2, rfrmodel,prodStr='A_var')
#B_var=predict(train, test_2, rfrmodel,prodStr='B_var')
#C_var=predict(train, test_2, rfrmodel,prodStr='C_var')
#D_var=predict(train, test_2, rfrmodel,prodStr='D_var')
#E_var=predict(train, test_2, rfrmodel,prodStr='E_var')
#F_var=predict(train, test_2, rfrmodel,prodStr='F_var')
#G_var=predict(train, test_2, rfrmodel,prodStr='G_var')
#offers=predict(train, test_2, rfrmodel,prodStr='total_offers')
#price_change=predict(train, test_2, rfrmodel,prodStr='price_change')
#price=predict(train, test_2, rfrmodel,prodStr='cost')


#for i in range(0,np.size(A_uLast)):
   # test_2.A_use_last.iloc[i]=A_uLast[i]
    #test_2.B_use_last.iloc[i]=B_uLast[i]
    #test_2.C_use_last.iloc[i]=C_uLast[i]
    #test_2.D_use_last.iloc[i]=D_uLast[i]
    #test_2.E_use_last.iloc[i]=E_uLast[i]
    #test_2.F_use_last.iloc[i]=F_uLast[i]
    #test_2.G_use_last.iloc[i]=G_uLast[i]

   # test_2.A_change.iloc[i]=A_change[i]
   # test_2.B_change.iloc[i]=B_change[i]
   # test_2.C_change.iloc[i]=C_change[i]
   # test_2.D_change.iloc[i]=D_change[i]
   # test_2.E_change.iloc[i]=E_change[i]
   # test_2.F_change.iloc[i]=G_change[i]
   # test_2.G_change.iloc[i]=G_change[i]

   # test_2.A_var.iloc[i]=A_var[i]
   # test_2.B_var.iloc[i]=B_var[i]
   # test_2.C_var.iloc[i]=C_var[i]
   # test_2.D_var.iloc[i]=D_var[i]
   # test_2.E_var.iloc[i]=E_var[i]
   # test_2.F_var.iloc[i]=F_var[i]
   # test_2.G_var.iloc[i]=G_var[i]

   # test_2.total_offers.iloc[i]=offers[i]
   # test_2.price_change.iloc[i]=price_change[i]
   # test_2.cost.iloc[i]=price[i]

A= predict(train, test_2,rfcmodel)
B= predict(train, test_2,rfcmodel,prodStr='B')
C= predict(train, test_2,rfcmodel,prodStr='C')
D= predict(train, test_2,rfcmodel,prodStr='D')
E= predict(train, test_2,rfcmodel,prodStr='E')
F= predict(train, test_2,rfcmodel,prodStr='F')
G= predict(train, test_2,rfcmodel,prodStr='G')


correct=0

for i in range(0,np.size(A)):
   # strA=str(test_2.A.iat[i])
   # strB=str(test_2.B.iat[i])
   # strC=str(test_2.C.iat[i])
   # strD=str(test_2.D.iat[i])
   # strE=str(test_2.E.iat[i])
   # strF=str(test_2.F.iat[i])
   # strG=str(test_2.G.iat[i])


   # if test_2.A_change.iat[i]/test_2.total_offers.iat[i]<A_change[i]/offers[i]:
    strA=str(A[i])
   # if test_2.B_change.iat[i]/test_2.total_offers.iat[i]<B_change[i]/offers[i]:
    strB=str(B[i])
   # if test_2.C_change.iat[i]/test_2.total_offers.iat[i]<C_change[i]/offers[i]:
    strC=str(C[i])
   # if test_2.D_change.iat[i]/test_2.total_offers.iat[i]<D_change[i]/offers[i]:
    strD=str(D[i])
   # if test_2.E_change.iat[i]/test_2.total_offers.iat[i]<E_change[i]/offers[i]:
    strE=str(E[i])
   # if test_2.F_change.iat[i]/test_2.total_offers.iat[i]<F_change[i]/offers[i]:
    strF=str(F[i])
   # if test_2.G_change.iat[i]/test_2.total_offers.iat[i]<G_change[i]/offers[i]:
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
results_df.to_csv("results_one.csv")

