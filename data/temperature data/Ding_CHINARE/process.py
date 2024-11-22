import pandas as pd
import numpy as np
needed_cols=['method_key', 'timestamp', 'latitude', 'longitude', 'elevation', 'depth', 'open_time', 'duration', 'temperature', 'error']
df = pd.read_excel('Firn Temperature along CHINARE.xls')
df.columns = ['name', 'latitude', 'longitude', 'elevation', 'temperature',
       'year', 'point of CHINARE']

df['depth']=10
df['duration'] = 1
df['timestamp'] = pd.to_datetime('1994-01-01')
msk = df.year.str.contains('-')
msk=(msk==True)
df.loc[msk,'timestamp'] = pd.to_datetime(df.loc[msk,'year'].str.split('-').str[1]+'-01-01')
df.loc[msk,'duration'] = (df.loc[msk,'year'].str.split('-').str[1].astype(float) - df.loc[msk,'year'].str.split('-').str[0].astype(float))*365
# df['timestamp']  = pd.to_datetime(df.year.str.split('-'))

df['reference'] =    'Ding et al.: Distribution of δ18O in surface snow along a transect from Zhongshan Station to Dome A, East Antarctica, doi: 10.1007/s11434-010-3179-3, 2010'
df['reference_short'] =    'Ding et al. (2010)'
df['error'] = np.nan
df['open_time'] = np.nan
df['method_key'] = np.nan
df['method'] = 'not available'

df[needed_cols].to_csv('data_formatted.csv')
