import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
col_needed = ['profile',  'reference_short',
                       'reference', 'method_key','method', 'date', 'timestamp', 'latitude',
                       'longitude', 'elevation', 'start_depth', 'stop_depth',
                       'midpoint', 'density', 'error']

# def process():
# %%

df_list = []
plot = True


df = pd.read_csv('D47_Verfaillie.csv',skiprows=3, sep=';',decimal=',',encoding='ansi')
df.columns = ['midpoint', 'density']
df['error'] = np.nan
df['density'] = df.density*1000
df['start_depth'] = df['midpoint'].shift(fill_value=0) + (df['midpoint'] - df['midpoint'].shift(fill_value=0)) / 2
df['stop_depth'] = df['midpoint'] + (df['midpoint'] - df['start_depth'])
df.loc[0, 'start_depth'] = 0

df['profile'] = 'D47'
df['reference_short'] = 'Arnaud et al. (1998)'
df['reference'] = 'Arnaud L, Lipenkov V, Barnola JM, Gay M, Duval P. Modelling of the densification of polar firn: characterization of the snow–firn transition. Annals of Glaciology. 1998;26:39-44. doi:10.3189/1998AoG26-1-39-44 '

df['method_key'] = 4
df['method'] = "ice or firn core section"
df['timestamp'] = pd.to_datetime('1989-01-10')
# print(file,df_meta.iloc[10,0],df_meta.iloc[10,1], (f'{y}-{m}-{d}'))

df['date'] = [int(v) for v in df.timestamp.dt.date.astype(str).str.replace('-','')]

df['latitude'] =  -67.3833333
df['longitude'] = 138.71666666666667
df['elevation'] = 1548

if plot:
    plt.plot(df["midpoint"], df["density"],marker='o', label='D47')
    plt.legend()
    plt.xlabel('Depth (m)')
    plt.ylabel('Density (kg m-3)')

for v in col_needed:
    assert v in df.columns, f'{v} is missing'
df[col_needed].to_csv('data_formatted.csv', index=None)
#     #%%

# if __name__ == "__main__":
#     process()
