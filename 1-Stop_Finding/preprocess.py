# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 22:56:15 2019

@author: msq96
"""


import utm
import pickle
import pandas as pd


print('Reading CSV...')
green = pd.read_csv('../data/green_tripdata_2016-05.csv')
yellow = pd.read_csv('../data/yellow_tripdata_2016-05.csv')

green['type'] = 'green'
yellow['type'] = 'yellow'

new_green = green[['VendorID', 'lpep_pickup_datetime', 'Lpep_dropoff_datetime', 'Pickup_longitude', 'Pickup_latitude',
                   'Dropoff_longitude', 'Dropoff_latitude', 'Trip_distance', 'Total_amount', 'type']]

new_yellow = yellow[['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'pickup_longitude', 'pickup_latitude',
                     'dropoff_longitude', 'dropoff_latitude', 'trip_distance', 'total_amount', 'type']]

new_green = new_green.rename(index=str, columns={'lpep_pickup_datetime': 'pickup_datetime', 'Lpep_dropoff_datetime': 'dropoff_datetime',
                                                 'Pickup_longitude': 'pickup_longitude', 'Pickup_latitude': 'pickup_latitude',
                                                 'Dropoff_longitude': 'dropoff_longitude', 'Dropoff_latitude': 'dropoff_latitude',
                                                 'Trip_distance': 'trip_distance', 'Total_amount': 'total_amount'})

new_yellow = new_yellow.rename(index=str, columns={'tpep_pickup_datetime': 'pickup_datetime', 'tpep_dropoff_datetime': 'dropoff_datetime'})

print('Concating data frames...')
res = pd.concat([new_yellow, new_green])

res[['pickup_datetime', 'dropoff_datetime']] = res[['pickup_datetime', 'dropoff_datetime']].astype('datetime64[ns]')

res['drop_day'] = res['dropoff_datetime'].apply(lambda x:x.day)
res['drop_hour'] = res['dropoff_datetime'].apply(lambda x:x.hour)

res['pick_day'] = res['pickup_datetime'].apply(lambda x:x.day)
res['pick_hour'] = res['pickup_datetime'].apply(lambda x:x.hour)


print('Projecting coordinates...')
res['raw_coords'] = res[['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']]\
                .apply(lambda x: [utm.from_latlon(x[0], x[1]), utm.from_latlon(x[2], x[3])], axis=1)

non_outlier = res['raw_coords'].map(lambda x: x[0][-1] + x[1][-1]) == 'TT' # To make sure all coords are in zone T.

res['coords'] = res['raw_coords'].apply(lambda x: list(x[0][:2]) + list(x[1][:2]))

res = res[non_outlier]

print('Saving file...')
with open('../data/yg_tripdata_2016-05.pickle', 'wb') as f:
    pickle.dump(res, f)
