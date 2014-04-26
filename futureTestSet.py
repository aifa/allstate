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

    trans_df = train_df.groupby(level=0).last()

    trans_df["duration_offers"] = pd.Series(index=trans_df.index)
    trans_df["total_offers"] = pd.Series(index=trans_df.index)

    trans_df["A_init"] = pd.Series(index=trans_df.index)
    trans_df["B_init"] = pd.Series(index=trans_df.index)
    trans_df["C_init"] = pd.Series(index=trans_df.index)
    trans_df["D_init"] = pd.Series(index=trans_df.index)
    trans_df["E_init"] = pd.Series(index=trans_df.index)
    trans_df["F_init"] = pd.Series(index=trans_df.index)
    trans_df["G_init"] = pd.Series(index=trans_df.index)

    trans_df["A_purchase"] = pd.Series(index=trans_df.index)
    trans_df["B_purchase"] = pd.Series(index=trans_df.index)
    trans_df["C_purchase"] = pd.Series(index=trans_df.index)
    trans_df["D_purchase"] = pd.Series(index=trans_df.index)
    trans_df["E_purchase"] = pd.Series(index=trans_df.index)
    trans_df["F_purchase"] = pd.Series(index=trans_df.index)
    trans_df["G_purchase"] = pd.Series(index=trans_df.index)

    trans_df["A_change"] = pd.Series(index=trans_df.index)
    trans_df["B_change"] = pd.Series(index=trans_df.index)
    trans_df["C_change"] = pd.Series(index=trans_df.index)
    trans_df["D_change"] = pd.Series(index=trans_df.index)
    trans_df["E_change"] = pd.Series(index=trans_df.index)
    trans_df["F_change"] = pd.Series(index=trans_df.index)
    trans_df["G_change"] = pd.Series(index=trans_df.index)

    trans_df["A_var"] = pd.Series(index=trans_df.index)
    trans_df["B_var"] = pd.Series(index=trans_df.index)
    trans_df["C_var"] = pd.Series(index=trans_df.index)
    trans_df["D_var"] = pd.Series(index=trans_df.index)
    trans_df["E_var"] = pd.Series(index=trans_df.index)
    trans_df["F_var"] = pd.Series(index=trans_df.index)
    trans_df["G_var"] = pd.Series(index=trans_df.index)

    trans_df["init_cost"] = pd.Series(index=trans_df.index)
    trans_df["cost_var"] = pd.Series(index=trans_df.index)

    print 'start'
    for key, cDf in train_df.groupby(level=0):
        #print "transforming product:" + str(key)

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

        totalRows=len(cDf.index)

        i = totalRows-1

        row = cDf.iloc[i]
        trans_df['shopping_pt'].loc[key]=row['shopping_pt']
        trans_df['record_type'].loc[key]=row['record_type']
        trans_df['day'].loc[key]=row['day']
        trans_df['time'].loc[key]=row['time']
        trans_df['state'].loc[key]=row['state']
        trans_df['location'].loc[key]=row['location']
        trans_df['group_size'].loc[key]=row['group_size']
        trans_df['homeowner'].loc[key]=row['homeowner']
        trans_df['car_age'].loc[key]=row['car_age']
        trans_df['car_value'].loc[key]=row['car_value']
        trans_df['risk_factor'].loc[key] =row['risk_factor']
        trans_df['age_oldest'].loc[key] =row['age_oldest']
        trans_df['age_youngest'].loc[key]=row['age_youngest']
        trans_df['married_couple'].loc[key] =row['married_couple']
        trans_df['C_previous'].loc[key]=row['C_previous']
        trans_df['duration_previous'].loc[key]=row['duration_previous']
        trans_df['cost'].loc[key]=row['cost']
        trans_df["A"].loc[key]= row['A']
        trans_df["B"].loc[key]= row['B']
        trans_df["C"].loc[key]= row['C']
        trans_df["D"].loc[key]= row['D']
        trans_df["E"].loc[key]= row['E']
        trans_df["F"].loc[key]= row['F']
        trans_df["G"].loc[key]= row['G']

        trans_df["A_change"].loc[key]=(cDf.A.iloc[0:i].diff() != 0).fillna(False).sum()
        trans_df["B_change"].loc[key]=(cDf.B.iloc[0:i].diff() != 0).fillna(False).sum()
        trans_df["C_change"].loc[key]=(cDf.C.iloc[0:i].diff() != 0).fillna(False).sum()
        trans_df["D_change"].loc[key]=(cDf.D.iloc[0:i].diff() != 0).fillna(False).sum()
        trans_df["E_change"].loc[key]=(cDf.E.iloc[0:i].diff() != 0).fillna(False).sum()
        trans_df["F_change"].loc[key]=(cDf.F.iloc[0:i].diff() != 0).fillna(False).sum()
        trans_df["G_change"].loc[key]=(cDf.G.iloc[0:i].diff() != 0).fillna(False).sum()

        trans_df["A_var"].loc[key]=np.var(cDf.A.iloc[0:i])
        trans_df["B_var"].loc[key]=np.var(cDf.B.iloc[0:i])
        trans_df["C_var"].loc[key]=np.var(cDf.C.iloc[0:i])
        trans_df["D_var"].loc[key]=np.var(cDf.D.iloc[0:i])
        trans_df["E_var"].loc[key]=np.var(cDf.E.iloc[0:i])
        trans_df["F_var"].loc[key]=np.var(cDf.F.iloc[0:i])
        trans_df["G_var"].loc[key]=np.var(cDf.G.iloc[0:i])

        trans_df["A_init"].loc[key]= initialA
        trans_df["B_init"].loc[key]= initialB
        trans_df["C_init"].loc[key]= initialC
        trans_df["D_init"].loc[key]= initialD
        trans_df["E_init"].loc[key]= initialE
        trans_df["F_init"].loc[key]= initialF
        trans_df["G_init"].loc[key]= initialG

#        trans_df["A_purchase"].loc[key]= cDf[cDf['record_type']==1].loc[key]['A']
#        trans_df["B_purchase"].loc[key]= cDf[cDf['record_type']==1].loc[key]['B']
#        trans_df["C_purchase"].loc[key]= cDf[cDf['record_type']==1].loc[key]['C']
#        trans_df["D_purchase"].loc[key]= cDf[cDf['record_type']==1].loc[key]['D']
#        trans_df["E_purchase"].loc[key]= cDf[cDf['record_type']==1].loc[key]['E']
#        trans_df["F_purchase"].loc[key]= cDf[cDf['record_type']==1].loc[key]['F']
#        trans_df["G_purchase"].loc[key]= cDf[cDf['record_type']==1].loc[key]['G']

        trans_df["init_cost"].loc[key] = initialPrice
        trans_df["cost_var"].loc[key]=np.var(cDf.cost.iloc[0:i])

        trans_df["total_offers"].loc[key] = totalRows

        days=int(row['day'])-initialDay
        if days>0:
            durationHours=days*24
        else:
            durationHours = dtparser.parse(row['time']).hour-initialTime.hour
        trans_df["duration_offers"].loc[key]=durationHours

    #print in different files
    trans_df.to_csv(fileName + "_all_future_.csv")

transform("test_v2.csv")