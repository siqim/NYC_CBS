# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 13:18:56 2019

@author: msq96
"""


import numpy as np
import pandas as pd
import seaborn as sns


nyc_taxi = pd.read_csv('../data/yellow_tripdata_2016-05.csv')
#nyc_taxi[['lpep_pickup_datetime', 'Lpep_dropoff_datetime']] = nyc_taxi[['lpep_pickup_datetime', 'Lpep_dropoff_datetime']].astype('datetime64[ns]')
nyc_taxi[['tpep_pickup_datetime', 'tpep_dropoff_datetime']] = nyc_taxi[['tpep_pickup_datetime', 'tpep_dropoff_datetime']].astype('datetime64[ns]')



#nyc_taxi['hr'] = nyc_taxi['Lpep_dropoff_datetime'].apply(lambda x: x.hour)
#nyc_taxi['day'] = nyc_taxi['Lpep_dropoff_datetime'].apply(lambda x: x.day)

nyc_taxi['hr'] = nyc_taxi['tpep_dropoff_datetime'].apply(lambda x: x.hour)
nyc_taxi['day'] = nyc_taxi['tpep_dropoff_datetime'].apply(lambda x: x.day)


nyc_taxi['cnt'] = 1

#nyc_taxi['weekday'] = nyc_taxi['Lpep_dropoff_datetime'].apply(lambda x: x.weekday())
nyc_taxi['weekday'] = nyc_taxi['tpep_dropoff_datetime'].apply(lambda x: x.weekday())


nyc_taxi = nyc_taxi[nyc_taxi['day']<=30]
nyc_taxi = nyc_taxi[(nyc_taxi['weekday'] <=5) & (nyc_taxi['weekday']>=1)]

cnt = nyc_taxi[['hr', 'day', 'cnt']].groupby(['hr', 'day']).sum()

df = {}
for i in range(24):
    temp = cnt.loc[i].values.reshape(-1)
    if len(temp) !=21:
        temp = np.append(temp, np.median(temp))

    temp[np.argmin(temp)] =  np.median(temp)

    df[i] = temp


df = pd.DataFrame(data=df)

sns.swarmplot(data=df)
sns.boxplot(data=df)
