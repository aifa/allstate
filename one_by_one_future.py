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

from sklearn.linear_model import LogisticRegression
from sklearn import svm

import math as math
from sklearn.externals.six.moves import xrange
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVR
from sklearn.cross_validation import StratifiedKFold

train_sets={}
train_1 = pd.read_csv("./future/train_all_split_1.csv", header=0, encoding="UTF-8", error_bad_lines=False, sep=",", index_col=0)
train_sets[1]=train_1
train_2 = pd.read_csv("./future/train_all_split_2.csv", header=0, encoding="UTF-8", error_bad_lines=False, sep=",", index_col=0)
train_sets[2]=train_2
train_3 = pd.read_csv("./future/train_all_split_3.csv", header=0, encoding="UTF-8", error_bad_lines=False, sep=",", index_col=0)
train_sets[3]=train_3
train_4 = pd.read_csv("./future/train_all_split_4.csv", header=0, encoding="UTF-8", error_bad_lines=False, sep=",", index_col=0)
train_sets[4]=train_4
train_5 = pd.read_csv("./future/train_all_split_5.csv", header=0, encoding="UTF-8", error_bad_lines=False, sep=",", index_col=0)
train_sets[5]=train_5
train_6 = pd.read_csv("./future/train_all_split_6.csv", header=0, encoding="UTF-8", error_bad_lines=False, sep=",", index_col=0)
train_sets[6]=train_6
train_7 = pd.read_csv("./future/train_all_split_7.csv", header=0, encoding="UTF-8", error_bad_lines=False, sep=",", index_col=0)
train_sets[7]=train_7
train_8 = pd.read_csv("./future/train_all_split_8.csv", header=0, encoding="UTF-8", error_bad_lines=False, sep=",", index_col=0)
train_sets[8]=train_8
train_9 = pd.read_csv("./future/train_all_split_9.csv", header=0, encoding="UTF-8", error_bad_lines=False, sep=",", index_col=0)
train_sets[9]=train_9
train_10 = pd.read_csv("./future/train_all_split_10.csv", header=0, encoding="UTF-8", error_bad_lines=False, sep=",", index_col=0)
train_sets[10]=train_10
train_11 = pd.read_csv("./future/train_all_split_11.csv", header=0, encoding="UTF-8", error_bad_lines=False, sep=",", index_col=0)
train_sets[11]=train_11
train_12 = pd.read_csv("./future/train_all_split_12.csv", header=0, encoding="UTF-8", error_bad_lines=False, sep=",", index_col=0)
train_sets[12]=train_12

test_2 = pd.read_csv("./future/test_v2_all_future.csv", header=0, encoding="UTF-8", error_bad_lines=False, sep=",", index_col=0)


results_df = pd.DataFrame(index=test_2.index)
resultCol = pd.Series(index=test_2.index,dtype=object)

n_estimators = 30

rfcModel = RandomForestClassifier(n_estimators=n_estimators, max_features=None)
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

    df['A_var'] = df['A_var'].fillna(0)
    df['B_var'] = df['B_var'].fillna(0)
    df['C_var'] = df['C_var'].fillna(0)
    df['D_var'] = df['D_var'].fillna(0)
    df['E_var'] = df['E_var'].fillna(0)
    df['F_var'] = df['F_var'].fillna(0)
    df['G_var'] = df['G_var'].fillna(0)
    df['cost_var'] = df['cost_var'].fillna(0)
    #drop any other rows that might contain null values (at this point only lines with empty key cols should be removed from the training sets)
    df=df.dropna()

    return df

def cleanup_test(df):

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

    df['A_var'] = df['A_var'].fillna(0)
    df['B_var'] = df['B_var'].fillna(0)
    df['C_var'] = df['C_var'].fillna(0)
    df['D_var'] = df['D_var'].fillna(0)
    df['E_var'] = df['E_var'].fillna(0)
    df['F_var'] = df['F_var'].fillna(0)
    df['G_var'] = df['G_var'].fillna(0)
    df['cost_var'] = df['cost_var'].fillna(0)

    return df

def train_model(train_df, select_model, model, yName='A_purchase'):
    input_df = train_df.copy(deep=True)
    y = input_df[yName].values

    input_df=input_df.drop(['A_purchase'], axis=1)
    input_df=input_df.drop(['B_purchase'], axis=1)
    input_df=input_df.drop(['C_purchase'], axis=1)
    input_df=input_df.drop(['D_purchase'], axis=1)
    input_df=input_df.drop(['E_purchase'], axis=1)
    input_df=input_df.drop(['F_purchase'], axis=1)
    input_df=input_df.drop(['G_purchase'], axis=1)

    if (yName=='G'):
        input_df=input_df.drop(['C_previous'], axis=1)


    X = input_df.values

    skf=StratifiedKFold(y, n_folds=3 )

    best_model=None
    for train_index, test_index in skf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = clone(model)
        clf = clf.fit(X_train, y_train)

        print yName +":"+str(clf.score(X_test,y_test))

        if best_model==None or best_model.score(X_test,y_test)<clf.score(X_test, y_test):
            best_model=clf

    trained_model = best_model

    return trained_model

def predict(test_2_df, model, prodStr='A_purchase'):

    test_df = test_2_df.copy(deep=True)
    test_y = test_df[prodStr]

    if prodStr=='total_offers':
        test_df=test_df.drop([prodStr])
    else:
        test_df=test_df.drop(['A_purchase'])
        test_df=test_df.drop(['B_purchase'])
        test_df=test_df.drop(['C_purchase'])
        test_df=test_df.drop(['D_purchase'])
        test_df=test_df.drop(['E_purchase'])
        test_df=test_df.drop(['F_purchase'])
        test_df=test_df.drop(['G_purchase'])

    if (prodStr=='G'):
        test_df=test_df.drop(['C_previous'])

    test_X = test_df.values

    prediction = model.predict(test_X)

    #print prodStr +":"+str(prediction)

    return prediction


modelADict={}
modelBDict={}
modelCDict={}
modelDDict={}
modelEDict={}
modelFDict={}
modelGDict={}

for i in range(1,13):
    train = cleanup(train_sets[i])
    modelADict[i] = train_model(train, lr, rfcModel)
    modelBDict[i] = train_model(train, lr, rfcModel, yName='B_purchase')
    modelCDict[i] = train_model(train, lr, rfcModel, yName='C_purchase')
    modelDDict[i] = train_model(train, lr, rfcModel, yName='D_purchase')
    modelEDict[i] = train_model(train, lr, rfcModel, yName='E_purchase')
    modelFDict[i] = train_model(train, lr, rfcModel, yName='F_purchase')
    modelGDict[i] = train_model(train, lr, rfcModel, yName='G_purchase')

correct=0
test_2 = cleanup_test(test_2)

for key, cdf in test_2.iterrows():
    print test_2.total_offers.loc[key]
    A= predict(cdf,modelADict[test_2.total_offers.loc[key]])
    B= predict(cdf,modelBDict[test_2.total_offers.loc[key]],prodStr='B_purchase')
    C= predict(cdf,modelCDict[test_2.total_offers.loc[key]],prodStr='C_purchase')
    D= predict(cdf,modelDDict[test_2.total_offers.loc[key]],prodStr='D_purchase')
    E= predict(cdf,modelEDict[test_2.total_offers.loc[key]],prodStr='E_purchase')
    F= predict(cdf,modelFDict[test_2.total_offers.loc[key]],prodStr='F_purchase')
    G= predict(cdf,modelGDict[test_2.total_offers.loc[key]],prodStr='G_purchase')

    resultCol.loc[key]=str(int(A))+str(int(B))+str(int(C))+str(int(D))+str(int(E))+str(int(F))+str(int(G))

    if test_2.A.loc[key]==A and test_2.B.loc[key]==B and test_2.C.loc[key]==C and test_2.D.loc[key]==D\
        and test_2.E.loc[key]==E and test_2.F.loc[key]==F and test_2.G.loc[key]==G:
        correct += 1


res = float(correct)/float(len(test_2))
print(len(test_2))
print correct
print str(res)

results_df["plan"]=resultCol
results_df.to_csv("results.csv")
