__author__ = 'aifa'

import pandas as pd
import numpy as np
import pylab as pl

from sklearn import clone

# note: these imports are incorrect in the example online!
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.forest import (RandomForestClassifier, RandomForestRegressor,
                                        ExtraTreesClassifier)

from sklearn.linear_model import LogisticRegression

from sklearn import cross_validation

from sklearn.cross_validation import StratifiedKFold

from sklearn import datasets, linear_model

from sklearn import svm

from sklearn.feature_selection import RFE, RFECV

from sklearn.feature_selection import f_regression

from sklearn.preprocessing import OneHotEncoder


from sklearn.decomposition import PCA

import math as math
from sklearn.externals.six.moves import xrange
from sklearn.tree import DecisionTreeClassifier


from sklearn.svm import SVR

train = pd.read_csv("./transformed/train_transformed_all_with_use_last_flag.csv", header=0, encoding="UTF-8", error_bad_lines=False, sep=",", index_col=0)
test_2 = pd.read_csv("./transformed/test_v2_transformed_all_with_use_last_flag.csv", header=0, encoding="UTF-8", error_bad_lines=False, sep=",", index_col=0)

results_df = pd.DataFrame(index=test_2.index)
resultCol = pd.Series(index=test_2.index,dtype=object)

#try random forest on feature A
n_estimators = 30

lr = LogisticRegression()
rfcmodel = RandomForestClassifier(n_estimators=n_estimators)
rfrmodel = RandomForestRegressor(n_estimators=n_estimators)
regr = linear_model.LinearRegression()
dtc = DecisionTreeClassifier(random_state=0)

svr =  SVR(C=1.0, epsilon=0.2)
C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C)
#rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C)
#poly_svc = svm.SVC(kernel='poly', degree=3, C=C)

def cleanup(df):

    enc = OneHotEncoder()
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


def predict(train_df, select_model, predict_model, prodStr='A'):

    input_df = train_df.copy(deep=True)

    input_df = cleanup(input_df)

    y = input_df[prodStr].values
    #input_df=input_df.drop(prodStr, axis=1)
    input_df=input_df.drop(['A'], axis=1)
    input_df=input_df.drop(['B'], axis=1)
    input_df=input_df.drop(['C'], axis=1)
    input_df=input_df.drop(['D'], axis=1)
    input_df=input_df.drop(['E'], axis=1)
    input_df=input_df.drop(['F'], axis=1)
    input_df=input_df.drop(['G'], axis=1)

    input_df=input_df.drop(['A_last'], axis=1)
    input_df=input_df.drop(['B_last'], axis=1)
    input_df=input_df.drop(['C_last'], axis=1)
    input_df=input_df.drop(['D_last'], axis=1)
    input_df=input_df.drop(['E_last'], axis=1)
    input_df=input_df.drop(['F_last'], axis=1)
    input_df=input_df.drop(['G_last'], axis=1)


    if (prodStr=='A' or prodStr=='G'):
        input_df=input_df.drop(['C_previous'], axis=1)
    #input_df=input_df.drop(['A_init'], axis=1)
    #input_df=input_df.drop(['B_init'], axis=1)
    #input_df=input_df.drop(['C_init'], axis=1)
    #input_df=input_df.drop(['D_init'], axis=1)
    #input_df=input_df.drop(['E_init'], axis=1)
    #input_df=input_df.drop(['F_init'], axis=1)
    #input_df=input_df.drop(['G_init'], axis=1)
    #input_df=input_df.drop(['A_var'], axis=1)
    #input_df=input_df.drop(['B_var'], axis=1)
    #input_df=input_df.drop(['C_var'], axis=1)
    #input_df=input_df.drop(['D_var'], axis=1)
    #input_df=input_df.drop(['E_var'], axis=1)
    #input_df=input_df.drop(['F_var'], axis=1)
    #input_df=input_df.drop(['G_var'], axis=1)

    #input_df=filter_features(train_df, select_model, prodStr)
    #print input_df.columns
    X = input_df.values

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=0)

    skf=StratifiedKFold(y_train, n_folds=3 )

    best_model=None
    for train_index, test_index in skf:
        print("TRAIN:", train_index, "TEST:", test_index)
        X_strain, X_stest = X[train_index], X[test_index]
        y_strain, y_stest = y[train_index], y[test_index]

        clf = clone(predict_model)
        clf = clf.fit(X_strain, y_strain)

        print prodStr +":"+str(clf.score(X_stest,y_stest))

        if best_model==None or best_model.score(X_test,y_test)<clf.score(X_test, y_test):
            best_model=clf

   # X_train = X_train - np.mean(X_train, axis=0)
   # pca = PCA(n_components=2)
   # X_train = pca.fit(X_train).transform(X_train) ## do the fit, then transform down

   # X_test = X_test - np.mean(X_test, axis=0)
   # pca = PCA(n_components=2)
   # X_test = pca.fit(X_test).transform(X_test) ## do the fit, then transform down


    # Shuffle
    #idx = np.arange(X.shape[0])
    #np.random.seed(13)
    #np.random.shuffle(idx)
    #X = X[idx]
    #y = y[idx]
    res = clf.predict(X_test)

    print prodStr +":"+str(clf.score(X_test,y_test))

    return res, y_test, X_test


#A_uLast=predict(train, regr, rfcmodel,prodStr='A_use_last')
#B_uLast=predict(train, regr, rfcmodel,prodStr='B_use_last')
#C_uLast=predict(train, regr, rfcmodel,prodStr='C_use_last')
#D_uLast=predict(train, regr, rfcmodel,prodStr='D_use_last')
#E_uLast=predict(train, regr, rfcmodel,prodStr='E_use_last')
#F_uLast=predict(train, regr, rfcmodel,prodStr='F_use_last')
#G_uLast=predict(train, regr, rfcmodel,prodStr='G_use_last')

#A_change=predict(train, regr, rfcmodel,prodStr='A_change')
#B_change=predict(train, regr, rfcmodel,prodStr='B_change')
#C_change=predict(train, regr, rfcmodel,prodStr='C_change')
#D_change=predict(train, regr, rfcmodel,prodStr='D_change')
#E_change=predict(train, regr, rfcmodel,prodStr='E_change')
#F_change=predict(train, regr, rfcmodel,prodStr='F_change')
#G_change=predict(train, regr, rfcmodel,prodStr='G_change')
#A_var=predict(train, regr, rfrmodel,prodStr='A_var')
#B_var=predict(train, regr, rfrmodel,prodStr='B_var')
#C_var=predict(train, regr, rfrmodel,prodStr='C_var')
#D_var=predict(train, regr, rfrmodel,prodStr='D_var')
#E_var=predict(train, regr, rfrmodel,prodStr='E_var')
#F_var=predict(train, regr, rfrmodel,prodStr='F_var')
#G_var=predict(train, regr, rfrmodel,prodStr='G_var')
#offers=predict(train,regr, rfrmodel,prodStr='total_offers')
#price_change=predict(train,regr, rfrmodel,prodStr='price_change')
#price=predict(train,regr, rfrmodel,prodStr='cost')


A, A_test, X_test= predict(train, lr,rfcmodel)
B, B_test, X_test= predict(train, lr,rfcmodel,prodStr='B')
C, C_test, X_test= predict(train, lr,rfcmodel,prodStr='C')
D, D_test, X_test= predict(train, lr,rfcmodel,prodStr='D')
E, E_test, X_test= predict(train, lr,rfcmodel,prodStr='E')
F, F_test, X_test= predict(train, lr,rfcmodel,prodStr='F')
G, G_test, X_test= predict(train, lr,rfcmodel,prodStr='G')

correct=0

for i in range(0,np.size(A)):
#    strA=str(A_test[i])
#    strB=str(B_test[i])
#    strC=str(C_test[i])
#    strD=str(D_test[i])
#    strE=str(E_test[i])
#    strF=str(F_test[i])
#    strG=str(G_test[i])

#    if X_test[[i,33]]<A_change[i][0]:
    strA=str(A[i])
#    if X_test[[i,34]]<B_change[i]:
    strB=str(B[i])
#    if X_test[[i,35]]<C_change[i]:
    strC=str(C[i])
#    if X_test[[i,36]]<D_change[i]:
    strD=str(D[i])
#    if X_test[[i,37]]<E_change[i]:
    strE=str(E[i])
#    if X_test[[i,38]]<F_change[i]:
    strF=str(F[i])
#    if X_test[[i,39]]<G_change[i]:
    strG=str(G[i])

    resultCol.iat[i]=strA+strB+strC+strD+strE+strF+strG

    #Calculate number of correct predictions
    if A_test[i]==float(strA) and B_test[i]==float(strB) and C_test[i]==float(strC) and D_test[i]==float(strD)\
        and E_test[i]==float(strE) and F_test[i]==float(strF) and G_test[i]==float(strG):
        correct += 1

res = float(correct)/float(len(X_test))
#print(len(test_2))
print correct
print str(res)

#results_df["plan"]=resultCol
#results_df.to_csv("results.csv")

