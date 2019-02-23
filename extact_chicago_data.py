# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 11:29:49 2019

@author: msq96
"""

import pandas as pd
from sodapy import Socrata


socrata_dataset_identifier = 'wrvz-psew'
client = Socrata("data.cityofchicago.org",
                  "..",
                  username="..",
                  password="..")

results = client.get(socrata_dataset_identifier,
                     where="trip_start_timestamp>='2017-06-1T00:00:00.000' and trip_start_timestamp<'2017-07-01T00:00:00.000'",
                     limit=2000000)

df = pd.DataFrame.from_records(results)
df.to_csv('chicago_18_month_data.csv')
