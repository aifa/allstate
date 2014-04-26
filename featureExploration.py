__author__ = 'aifa'

import pandas as pd
import numpy as np

import pylab as pl

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.ensemble.forest import (RandomForestClassifier,
                                        ExtraTreesClassifier)

train = pd.read_csv("./transformed/train_transformed_all.csv", header=0, encoding="UTF-8", error_bad_lines=False, sep=",", index_col=0)

def cleanup(df):

    #find columns that contain null values
    inds = pd.isnull(df).any(0).nonzero()

    df['car_value'] = df['car_value'].fillna('0')
    df['car_value']=df['car_value'].apply(lambda letter :  ord(letter) - 96)
    df['state']=df['state'].apply(lambda letter :  ord(letter[0]) - 96 + ord(letter[1]) - 96)

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


def scree_plot(train_df, prodStr='A'):

    input_df = train_df.copy(deep=True)
    input_df = cleanup(input_df)

    y = input_df[prodStr].values
    input_df=input_df.drop([prodStr], axis=1)
    X = input_df.values

    # center data (important for dim reduction)
    X = X - np.mean(X, axis=0)

    # get covariance matrix
    #np.cov(X).shape            # this has the wrong dimensions
    #np.cov(X, rowvar=0).shape    # this is good

    X_cov = np.cov(X, rowvar=0)

    # eigenvalue decomp
    X_eig = np.linalg.eig(X_cov)
    X_egval = X_eig[0]

    # pct of variance explained by each principal component
    pcts = [k/sum(X_egval) for k in X_egval]

    plt.plot(pcts)
    plt.xlabel('principal cmpts')
    plt.ylabel('pct variance explained')
    plt.title('iris scree plot')
    plt.show()

    pca_plot(input_df, y)

def pca_plot(X_df, y, n_components=3):

    target_names = X_df.columns
    print target_names

    X=X_df.values
    pca = PCA(n_components)
    X_decomp = pca.fit(X).transform(X) ## do the fit, then transform down


    #try random forest on feature A
    n_estimators = 100

    rfcmodel = RandomForestClassifier(n_estimators=n_estimators)

    rfcmodel


    print len(X_decomp)
    print 'explained variance ratio (first two components):', \
        pca.explained_variance_ratio_

    print 'components in PCA (these are our :', \
        pca.components_

    pl.figure()
    for c, i, target_name in zip("rgb", [0, 1, 2, 3, 4, 5, 6, 7, 8,9 , 10], target_names):
        pl.scatter(X_decomp[y == i, 0], X_decomp[y == i, 1], c=c, label=target_name)
    pl.legend()
    pl.title('feature PCA')
    pl.show()


scree_plot(train, "A")