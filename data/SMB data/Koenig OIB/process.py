# -*- coding: utf-8 -*-
"""
Created on %(date)s
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Your original loop with modifications
col_needed = ['start_date', 'end_date', 'start_year', 'end_year', 'latitude',
              'longitude', 'elevation', 'notes', 'smb',
              'error', 'name', 'method_key']
df_list = []
plt.close('all')
path_to_folder= 'C:/Users/bav/OneDrive - GEUS/Data/OIB Lora Koenig/koenig_snow_layers/data_formatted'

for filename in os.listdir(path_to_folder):
    if filename.endswith('.csv'):
        print(filename)
        df = pd.read_csv(path_to_folder+'/' + filename)
        df = df.rename(columns= {'thickness_m_we': 'smb'})
        df = df.loc[df.longitude>-180,:]

        # filtering averages where the smb_std > 0.015
        # resampled_df = resampled_df.loc[resampled_df.smb_std<0.015,:]

        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date'] = pd.to_datetime(df['end_date'])

        df['start_year'] = df['start_date'].dt.year
        df['end_year'] = df['end_date'].dt.year

        df['name'] = 'Koenig_OIB_' + df['timestamp'].astype(str).str[:10].str.replace('-','') +'_'+ df.index.astype(str)
        df['method_key'] = 9
        df['notes'] = ''

        df['elevation'] = np.nan
        df['error'] = 0.01 #np.maximum(df['smb_std'],0.01)
        df['latitude'] = np.round(df['latitude'],6)
        df['longitude'] = np.round(df['longitude'],6)
        df['smb'] = np.round(df['smb'],2)
        df['error'] = np.round(df['error'],2)

        df = df.drop(columns = ['thickness_m','timestamp'])

        # df['method'] = 'airborne radar'
        # df['reference_short'] =  'Koenig et al. (2016)'
        # df['reference'] = 'Koenig, L. S., Ivanoff, A., et al.: Annual Greenland accumulation rates (2009–2012) from airborne snow radar, The Cryosphere, 10, 1739–1752, https://doi.org/10.5194/tc-10-1739-2016, 2016.'
        # df['method'] = ''
        # df['reference_short'] =  ''
        # df['reference'] = ''

        # Keep only necessary columns
        for v in col_needed:
            assert v in df.columns, f'{v} is missing'
        df_list.append(df[col_needed])
        # df[col_needed].to_csv(
        #     'data_formatted_' + filename.replace('_stacked_filtered', ''), index=None)

# Concatenate and save the final dataframe
pd.concat(df_list).reset_index(drop=True).to_csv('data_formatted.csv', index=None)
