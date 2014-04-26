__author__ = 'aifa'

import pandas as pd
import numpy as np
import dateutil.parser as dtparser
import datetime as dt

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    # cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dType = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dType)

    m = n / arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in xrange(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out





def transform(fileName):

    input_df = pd.read_csv(fileName, header=0, encoding="UTF-8", error_bad_lines=False, sep=",", index_col=0)

    #print train_df.head()

    print input_df.columns

    #create all available product option combinations
    A = (0, 1, 2)
    B = (0, 1)
    C = (1, 2, 3, 4)
    D = (1, 2, 3)
    E = (0, 1)
    F = (0, 1, 2, 3)
    G = (1, 2, 3, 4)

    #produce an array with all the possible options combinations
    combos = cartesian((A, B, C, D, E, F, G))
    #print (list(train_df['A'].values),list(train_df['B'].values),list(train_df['C'].values),list(train_df['D'].values),
    #       list(train_df['E'].values),list(train_df['F'].values),list(train_df['G'].values))[1][1]
    #print train_df.values[:,17:24].searchsorted(combos)
    #print combos
    #print combos.size
    def comboCompare(row, values):
        return (row[0] == values[0] and row[1] == values[1] and row[2] == values[2] and row[3] == values[3] and row[4] ==
            values[4] and row[5] == values[5] and row[6] == values[6]) == True


    def findCombo(row, cCombos=combos):
        currentCombo = (row.values[16:23])
        return np.where(np.apply_along_axis(comboCompare, 1, cCombos, currentCombo) == True)[0][0]

    train_df = input_df.copy(deep=True)
    #add product combo column
    #train_df["product"] = train_df.apply(findCombo, axis=1)

    #checkpoint to backup changes
    train_df.to_csv(fileName+"_checkpoint_all_with_use_last_flag.csv")

    #create an dictionary that groups rows per customer id
    customerDict = {}
    for row in train_df.iterrows():
        if not customerDict.has_key(row[0]):
            customerDict[row[0]] = train_df[train_df.index == row[0]].copy()

    #create an empty data frame with a unique entry per customer id
    trans_df = pd.DataFrame(index=train_df.index, columns=train_df.columns.tolist())
    trans_df = trans_df.groupby(level=0).last()
    #define extra cols as series
    initPriceCol = pd.Series(index=trans_df.index)
    initACol = pd.Series(index=trans_df.index)
    initBCol = pd.Series(index=trans_df.index)
    initCCol = pd.Series(index=trans_df.index)
    initDCol = pd.Series(index=trans_df.index)
    initECol = pd.Series(index=trans_df.index)
    initFCol = pd.Series(index=trans_df.index)
    initGCol = pd.Series(index=trans_df.index)

    lastACol = pd.Series(index=trans_df.index)
    lastBCol = pd.Series(index=trans_df.index)
    lastCCol = pd.Series(index=trans_df.index)
    lastDCol = pd.Series(index=trans_df.index)
    lastECol = pd.Series(index=trans_df.index)
    lastFCol = pd.Series(index=trans_df.index)
    lastGCol = pd.Series(index=trans_df.index)
    #prodChangeCol = pd.Series(index=trans_df.index)
    AchangeCol = pd.Series(index=trans_df.index)
    BchangeCol = pd.Series(index=trans_df.index)
    CchangeCol = pd.Series(index=trans_df.index)
    DchangeCol = pd.Series(index=trans_df.index)
    EchangeCol = pd.Series(index=trans_df.index)
    FchangeCol = pd.Series(index=trans_df.index)
    GchangeCol = pd.Series(index=trans_df.index)

    AvarCol = pd.Series(index=trans_df.index)
    BvarCol = pd.Series(index=trans_df.index)
    CvarCol = pd.Series(index=trans_df.index)
    DvarCol = pd.Series(index=trans_df.index)
    EvarCol = pd.Series(index=trans_df.index)
    FvarCol = pd.Series(index=trans_df.index)
    GvarCol = pd.Series(index=trans_df.index)

    AuseLastCol = pd.Series(index=trans_df.index)
    BuseLastCol = pd.Series(index=trans_df.index)
    CuseLastCol = pd.Series(index=trans_df.index)
    DuseLastCol = pd.Series(index=trans_df.index)
    EuseLastCol = pd.Series(index=trans_df.index)
    FuseLastCol = pd.Series(index=trans_df.index)
    GuseLastCol = pd.Series(index=trans_df.index)

    durationCol = pd.Series(index=trans_df.index)
    totalOffersCol = pd.Series(index=trans_df.index)
    priceChangeCol = pd.Series(index=trans_df.index)

    ####
    #print(train_df.dtypes)

    #for index_val, sub_df in train_df.groupby(level=0):
    #    first_row = sub_df['Colname1', 'Colname2'].iloc[0, :]

    #   (sub_df.product.diff() != 0).fillna(False).sum()


    #for key, cDf in customerDict.iteritems():

    ####

    for key, cDf in customerDict.iteritems():
        #print "transforming product:" + str(key)

        index = 0
        currentProduct = -1
        #record how many times has the product changed
        productChange = -1
        initialPrice = -1
        finalPrice = 0
        startTime=None
        endTime=None
        startDay=0
        endDay=0
        durationHours=-1
        initialProduct=-1

        #price change between first offer and final product
        priceChange=-1
        totalRows=len(cDf.index)


        for cRow in cDf.values:
            recordType = int(cRow[1])
           # product = int(cRow[24])
            price = int(cRow[23])
            time=dtparser.parse(cRow[3])
            day=int(cRow[2])
            #first offer
            if index == 0:
                initialPrice = price
                startTime=time
                startDay=day
                #initialProduct=product


           # if product != currentProduct:
           #     currentProduct = product
           #     productChange += 1

#            if recordType == 0:
#                pass
            # bought product or last available offer
            if recordType == 1 or index == totalRows-1:
                trans_df.loc[key] = cRow
                endDay=day
                endTime=time
                finalPrice = price
                priceChange=finalPrice-initialPrice
                days=endDay-startDay
                if days>0:
                    durationHours=days*24
                else:
                    durationHours = endTime.hour-startTime.hour

            if recordType != 1 and index == totalRows-1: #test set only
                lastACol.loc[key] = cDf['A'].iloc[index]
                lastBCol.loc[key] = cDf['B'].iloc[index]
                lastCCol.loc[key] = cDf['C'].iloc[index]
                lastDCol.loc[key] = cDf['D'].iloc[index]
                lastECol.loc[key] = cDf['E'].iloc[index]
                lastFCol.loc[key] = cDf['F'].iloc[index]
                lastGCol.loc[key] = cDf['G'].iloc[index]

            if recordType == 1 and index == totalRows-1: #Training set only
                lastACol.loc[key] = cDf['A'].iloc[index-1]
                lastBCol.loc[key] = cDf['B'].iloc[index-1]
                lastCCol.loc[key] = cDf['C'].iloc[index-1]
                lastDCol.loc[key] = cDf['D'].iloc[index-1]
                lastECol.loc[key] = cDf['E'].iloc[index-1]
                lastFCol.loc[key] = cDf['F'].iloc[index-1]
                lastGCol.loc[key] = cDf['G'].iloc[index-1]

                if cDf['A'].iloc[index] == cDf['A'].iloc[index-1]:
                    AuseLastCol.loc[key] = 1
                else:
                    AuseLastCol.loc[key] = 0
                if cDf['B'].iloc[index] == cDf['B'].iloc[index-1]:
                    BuseLastCol.loc[key] = 1
                else:
                    BuseLastCol.loc[key] = 0
                if cDf['C'].iloc[index] == cDf['C'].iloc[index-1]:
                    CuseLastCol.loc[key] = 1
                else:
                    CuseLastCol.loc[key] = 0
                if cDf['D'].iloc[index] == cDf['D'].iloc[index-1]:
                    DuseLastCol.loc[key] = 1
                else:
                    DuseLastCol.loc[key] = 0
                if cDf['E'].iloc[index] == cDf['E'].iloc[index-1]:
                    EuseLastCol.loc[key] = 1
                else:
                    EuseLastCol.loc[key] = 0
                if cDf['F'].iloc[index] == cDf['F'].iloc[index-1]:
                    FuseLastCol.loc[key] = 1
                else:
                    FuseLastCol.loc[key] = 0
                if cDf['G'].iloc[index] == cDf['G'].iloc[index-1]:
                    GuseLastCol.loc[key] = 1
                else:
                    GuseLastCol.loc[key] = 0


            index += 1

        initPriceCol.loc[key] = initialPrice
        #initProductCol.loc[key] = initialProduct
        #prodChangeCol.loc[key] = productChange
        durationCol.loc[key] = durationHours
        totalOffersCol.loc[key] = totalRows
        priceChangeCol.loc[key] = cDf.cost.var()
        AchangeCol.loc[key]=(cDf.A.diff() != 0).fillna(False).sum()
        BchangeCol.loc[key]=(cDf.B.diff() != 0).fillna(False).sum()
        CchangeCol.loc[key]=(cDf.C.diff() != 0).fillna(False).sum()
        DchangeCol.loc[key]=(cDf.D.diff() != 0).fillna(False).sum()
        EchangeCol.loc[key]=(cDf.E.diff() != 0).fillna(False).sum()
        FchangeCol.loc[key]=(cDf.F.diff() != 0).fillna(False).sum()
        GchangeCol.loc[key]=(cDf.G.diff() != 0).fillna(False).sum()
        AvarCol.loc[key]=cDf.A.var()
        BvarCol.loc[key]=cDf.B.var()
        CvarCol.loc[key]=cDf.C.var()
        DvarCol.loc[key]=cDf.D.var()
        EvarCol.loc[key]=cDf.E.var()
        FvarCol.loc[key]=cDf.F.var()
        GvarCol.loc[key]=cDf.G.var()
        initACol.loc[key]= cDf['A'].iloc[0]
        initBCol.loc[key]= cDf['B'].iloc[0]
        initCCol.loc[key]= cDf['C'].iloc[0]
        initDCol.loc[key]= cDf['D'].iloc[0]
        initECol.loc[key]= cDf['E'].iloc[0]
        initFCol.loc[key]= cDf['F'].iloc[0]
        initGCol.loc[key]= cDf['G'].iloc[0]

    trans_df["init_price"] = initPriceCol
    #trans_df["init_product"] = initProductCol
    #trans_df["prod_change"] = prodChangeCol
    trans_df["duration_offers"] = durationCol
    trans_df["total_offers"] = totalOffersCol
    trans_df["price_change"] = priceChangeCol

    trans_df["A_init"] = initACol
    trans_df["B_init"] = initBCol
    trans_df["C_init"] = initCCol
    trans_df["D_init"] = initDCol
    trans_df["E_init"] = initECol
    trans_df["F_init"] = initFCol
    trans_df["G_init"] = initGCol

    trans_df["A_last"] = lastACol
    trans_df["B_last"] = lastBCol
    trans_df["C_last"] = lastCCol
    trans_df["D_last"] = lastDCol
    trans_df["E_last"] = lastECol
    trans_df["F_last"] = lastFCol
    trans_df["G_last"] = lastGCol

    trans_df["A_change"] = AchangeCol
    trans_df["B_change"] = BchangeCol
    trans_df["C_change"] = CchangeCol
    trans_df["D_change"] = DchangeCol
    trans_df["E_change"] = EchangeCol
    trans_df["F_change"] = FchangeCol
    trans_df["G_change"] = GchangeCol

    trans_df["A_var"] = AvarCol
    trans_df["B_var"] = BvarCol
    trans_df["C_var"] = CvarCol
    trans_df["D_var"] = DvarCol
    trans_df["E_var"] = EvarCol
    trans_df["F_var"] = FvarCol
    trans_df["G_var"] = GvarCol

    trans_df["A_use_last"] = AuseLastCol
    trans_df["B_use_last"] = BuseLastCol
    trans_df["C_use_last"] = CuseLastCol
    trans_df["D_use_last"] = DuseLastCol
    trans_df["E_use_last"] = EuseLastCol
    trans_df["F_use_last"] = FuseLastCol
    trans_df["G_use_last"] = GuseLastCol

    trans_df.to_csv(fileName + "_transformed_all_with_use_last_flag.csv")


#transform("test_v2.csv")
transform("train.csv")