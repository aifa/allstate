__author__ = 'aifa'

import pandas as pd
import numpy as np
import dateutil.parser as dtparser
import datetime as dt


def transform(fileName):

    input_df = pd.read_csv(fileName, header=0, encoding="UTF-8", error_bad_lines=False, sep=",", index_col=0)

    #print train_df.head()

    print input_df.columns

    train_df = input_df.copy(deep=True)

    ####
    #print(train_df.dtypes)

    #for index_val, sub_df in train_df.groupby(level=0):
    #    first_row = sub_df['Colname1', 'Colname2'].iloc[0, :]

    #   (sub_df.product.diff() != 0).fillna(False).sum()


    #for key, cDf in customerDict.iteritems():

    ####

    #find max number of offers and create a dataset per rowNumber
    maxRows=0
    trainingDict = {}
    for key, cDf in train_df.groupby(level=0):
        totalRows=len(cDf[cDf['record_type']==0])
        if totalRows>maxRows:
            maxRows=totalRows

    print maxRows

    #Create a dictionary holding a data frame per number of rows.

    for tRow in range(0, maxRows):

        trainingDict[tRow]= train_df.groupby(level=0).last()
        #define extra cols as series

        trainingDict[tRow]["duration_offers"] = pd.Series(index=trainingDict[tRow].index)
        trainingDict[tRow]["total_offers"] = pd.Series(index=trainingDict[tRow].index)

        trainingDict[tRow]["A_init"] = pd.Series(index=trainingDict[tRow].index)
        trainingDict[tRow]["B_init"] = pd.Series(index=trainingDict[tRow].index)
        trainingDict[tRow]["C_init"] = pd.Series(index=trainingDict[tRow].index)
        trainingDict[tRow]["D_init"] = pd.Series(index=trainingDict[tRow].index)
        trainingDict[tRow]["E_init"] = pd.Series(index=trainingDict[tRow].index)
        trainingDict[tRow]["F_init"] = pd.Series(index=trainingDict[tRow].index)
        trainingDict[tRow]["G_init"] = pd.Series(index=trainingDict[tRow].index)

        trainingDict[tRow]["A_purchase"] = pd.Series(index=trainingDict[tRow].index)
        trainingDict[tRow]["B_purchase"] = pd.Series(index=trainingDict[tRow].index)
        trainingDict[tRow]["C_purchase"] = pd.Series(index=trainingDict[tRow].index)
        trainingDict[tRow]["D_purchase"] = pd.Series(index=trainingDict[tRow].index)
        trainingDict[tRow]["E_purchase"] = pd.Series(index=trainingDict[tRow].index)
        trainingDict[tRow]["F_purchase"] = pd.Series(index=trainingDict[tRow].index)
        trainingDict[tRow]["G_purchase"] = pd.Series(index=trainingDict[tRow].index)

        trainingDict[tRow]["A_change"] = pd.Series(index=trainingDict[tRow].index)
        trainingDict[tRow]["B_change"] = pd.Series(index=trainingDict[tRow].index)
        trainingDict[tRow]["C_change"] = pd.Series(index=trainingDict[tRow].index)
        trainingDict[tRow]["D_change"] = pd.Series(index=trainingDict[tRow].index)
        trainingDict[tRow]["E_change"] = pd.Series(index=trainingDict[tRow].index)
        trainingDict[tRow]["F_change"] = pd.Series(index=trainingDict[tRow].index)
        trainingDict[tRow]["G_change"] = pd.Series(index=trainingDict[tRow].index)

        trainingDict[tRow]["A_var"] = pd.Series(index=trainingDict[tRow].index)
        trainingDict[tRow]["B_var"] = pd.Series(index=trainingDict[tRow].index)
        trainingDict[tRow]["C_var"] = pd.Series(index=trainingDict[tRow].index)
        trainingDict[tRow]["D_var"] = pd.Series(index=trainingDict[tRow].index)
        trainingDict[tRow]["E_var"] = pd.Series(index=trainingDict[tRow].index)
        trainingDict[tRow]["F_var"] = pd.Series(index=trainingDict[tRow].index)
        trainingDict[tRow]["G_var"] = pd.Series(index=trainingDict[tRow].index)

        trainingDict[tRow]["init_cost"] = pd.Series(index=trainingDict[tRow].index)
        trainingDict[tRow]["cost_var"] = pd.Series(index=trainingDict[tRow].index)

    print 'start'
    for key, cDf in train_df.groupby(level=0):
        print "transforming customer:" + str(key)

        #record how many times has the product changed

        initialPrice = cDf['cost'].iloc[0]
        initialA = cDf['A'].iloc[0]
        initialB = cDf['B'].iloc[0]
        initialC = cDf['C'].iloc[0]
        initialD = cDf['D'].iloc[0]
        initialE = cDf['E'].iloc[0]
        initialF = cDf['F'].iloc[0]
        initialG = cDf['G'].iloc[0]

        initialTime=dtparser.parse(cDf['time'].iloc[0])
        initialDay=int(cDf['day'].iloc[0])

        i = 0

        for index, row in cDf.iterrows():
            if row['record_type']==0:
                trainingDict[i]['shopping_pt'].loc[key]=row['shopping_pt']
                trainingDict[i]['record_type'].loc[key]=row['record_type']
                trainingDict[i]['day'].loc[key]=row['day']
                trainingDict[i]['time'].loc[key]=row['time']
                trainingDict[i]['state'].loc[key]=row['state']
                trainingDict[i]['location'].loc[key]=row['location']
                trainingDict[i]['group_size'].loc[key]=row['group_size']
                trainingDict[i]['homeowner'].loc[key]=row['homeowner']
                trainingDict[i]['car_age'].loc[key]=row['car_age']
                trainingDict[i]['car_value'].loc[key]=row['car_value']
                trainingDict[i]['risk_factor'].loc[key] =row['risk_factor']
                trainingDict[i]['age_oldest'].loc[key] =row['age_oldest']
                trainingDict[i]['age_youngest'].loc[key]=row['age_youngest']
                trainingDict[i]['married_couple'].loc[key] =row['married_couple']
                trainingDict[i]['C_previous'].loc[key]=row['C_previous']
                trainingDict[i]['duration_previous'].loc[key]=row['duration_previous']
                trainingDict[i]['cost'].loc[key]=row['cost']

                trainingDict[i]["A_change"].loc[key]=(cDf.A.iloc[0:i].diff() != 0).fillna(False).sum()
                trainingDict[i]["B_change"].loc[key]=(cDf.B.iloc[0:i].diff() != 0).fillna(False).sum()
                trainingDict[i]["C_change"].loc[key]=(cDf.C.iloc[0:i].diff() != 0).fillna(False).sum()
                trainingDict[i]["D_change"].loc[key]=(cDf.D.iloc[0:i].diff() != 0).fillna(False).sum()
                trainingDict[i]["E_change"].loc[key]=(cDf.E.iloc[0:i].diff() != 0).fillna(False).sum()
                trainingDict[i]["F_change"].loc[key]=(cDf.F.iloc[0:i].diff() != 0).fillna(False).sum()
                trainingDict[i]["G_change"].loc[key]=(cDf.G.iloc[0:i].diff() != 0).fillna(False).sum()

                trainingDict[i]["A_var"].loc[key]=np.var(cDf.A.iloc[0:i])
                trainingDict[i]["B_var"].loc[key]=np.var(cDf.B.iloc[0:i])
                trainingDict[i]["C_var"].loc[key]=np.var(cDf.C.iloc[0:i])
                trainingDict[i]["D_var"].loc[key]=np.var(cDf.D.iloc[0:i])
                trainingDict[i]["E_var"].loc[key]=np.var(cDf.E.iloc[0:i])
                trainingDict[i]["F_var"].loc[key]=np.var(cDf.F.iloc[0:i])
                trainingDict[i]["G_var"].loc[key]=np.var(cDf.G.iloc[0:i])

                trainingDict[i]["A_init"].loc[key]= initialA
                trainingDict[i]["B_init"].loc[key]= initialB
                trainingDict[i]["C_init"].loc[key]= initialC
                trainingDict[i]["D_init"].loc[key]= initialD
                trainingDict[i]["E_init"].loc[key]= initialE
                trainingDict[i]["F_init"].loc[key]= initialF
                trainingDict[i]["G_init"].loc[key]= initialG

                trainingDict[i]["A"].loc[key]= row['A']
                trainingDict[i]["B"].loc[key]= row['B']
                trainingDict[i]["C"].loc[key]= row['C']
                trainingDict[i]["D"].loc[key]= row['D']
                trainingDict[i]["E"].loc[key]= row['E']
                trainingDict[i]["F"].loc[key]= row['F']
                trainingDict[i]["G"].loc[key]= row['G']

                trainingDict[i]["A_purchase"].loc[key]= cDf[cDf['record_type']==1].loc[key]['A']
                trainingDict[i]["B_purchase"].loc[key]= cDf[cDf['record_type']==1].loc[key]['B']
                trainingDict[i]["C_purchase"].loc[key]= cDf[cDf['record_type']==1].loc[key]['C']
                trainingDict[i]["D_purchase"].loc[key]= cDf[cDf['record_type']==1].loc[key]['D']
                trainingDict[i]["E_purchase"].loc[key]= cDf[cDf['record_type']==1].loc[key]['E']
                trainingDict[i]["F_purchase"].loc[key]= cDf[cDf['record_type']==1].loc[key]['F']
                trainingDict[i]["G_purchase"].loc[key]= cDf[cDf['record_type']==1].loc[key]['G']

                trainingDict[i]["init_cost"].loc[key] = initialPrice
                trainingDict[i]["cost_var"].loc[key]=np.var(cDf.cost.iloc[0:i])

                trainingDict[i]["total_offers"] = i+1

                days=int(row['day'])-initialDay
                if days>0:
                    durationHours=days*24
                else:
                    durationHours = dtparser.parse(row['time']).hour-initialTime.hour
                trainingDict[i]["duration_offers"].loc[key]=durationHours
                i += 1

    #print in different files
    for tRow in range(0, maxRows):
        trainingDict[tRow].to_csv(fileName + "_all_split_"+str(tRow+1)+".csv")


transform("train.csv")
