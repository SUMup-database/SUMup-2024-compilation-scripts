import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
col_needed = ['profile',  'reference_short', 
                       'reference', 'method_key','method', 'date', 'timestamp', 'latitude', 
                       'longitude', 'elevation', 'start_depth', 'stop_depth',
                       'midpoint', 'density', 'error']
       
def process():
    df_meta = pd.read_excel('firn_cores_2021.xlsx')
    list_add = []
    for c in df_meta.core:
        print(c)
        df = pd.read_excel('firn_cores_2021.xlsx', sheet_name = c)
        df.columns = [v.replace(' ','_').replace('(','').replace(')','') for v in df.columns]
        df =  df.dropna(subset='density')
        df_formatted = pd.DataFrame()
        if c in ['FS5_20m', 'FS5_5m']:
            df_formatted['start_depth'] = np.append(df.depth_cm.iloc[0],
                                                    df.loc[df.density.diff().bfill()!=0,'depth_cm'].values) /100
        else:
            df_formatted['start_depth'] = df.loc[df.density.diff().bfill()!=0,'depth_cm'].values /100
        df_formatted['stop_depth'] = np.append(df.loc[df.density.diff().shift(-1).ffill()!=0,'depth_cm'].values,
                                               df.depth_cm.values[-1])/100
        msk = (df_formatted['start_depth'] == df_formatted['stop_depth'])
        df_formatted.loc[msk, 'start_depth'] = df_formatted.loc[msk, 'stop_depth'].values - 0.1
        df_formatted.loc[~msk, 'stop_depth'] = df_formatted.loc[~msk, 'stop_depth'].values + 0.01
        if c in ['FS5_20m', 'FS5_5m']:
            df_formatted['density'] = np.append(df.density.iloc[0],
                                                    df.loc[df.density.diff().bfill()!=0,'density'].values)
        else:
            df_formatted['density'] = df.loc[df.density.diff().bfill()!=0,'density'].values
        
        df_formatted['midpoint'] = df_formatted.start_depth + (df_formatted.stop_depth - df_formatted.start_depth)/2
        
        if c == 'FS4_20m': df_formatted['profile_key'] = 2603
        if c == 'FS4_5m': df_formatted['profile_key'] = 2604
        if c == 'FS5_20m': df_formatted['profile_key'] = 2605
        if c == 'FS5_5m': df_formatted['profile_key'] = 2606
        if c == 'FS2_12m': df_formatted['profile_key'] = 2607
        df_formatted['profile'] = c
        df_formatted['reference_key'] = 238
        df_formatted['reference_short'] = 'Clerx et al. (2022)'
        df_formatted['method_key'] = 4
        df_formatted['method'] = 'ice or firn core section'
        df_formatted['timestamp'] = df_meta.loc[df_meta.core==c, 
                                                'datetime cored (UTC)'].dt.strftime('%Y-%m-%d').values[0]
        df_formatted['date'] = int(df_meta.loc[df_meta.core==c, 
                                               'datetime cored (UTC)'].dt.strftime('%Y%m%d').values[0])
        df_formatted['latitude'] = df_meta.loc[df_meta.core==c, 'N'].values[0]
        df_formatted['longitude'] = df_meta.loc[df_meta.core==c, 'E'].values[0]
        df_formatted['elevation'] = df_meta.loc[df_meta.core==c, 'Z'].values[0]
        df_formatted['error'] = np.nan
        
        df_formatted['reference'] = 'Clerx, N., Machguth, H., Tedstone, A., Jullien, N., Wever, N., Weingartner, R., and Roessler, O.: In situ measurements of meltwater flow through snow and firn in the accumulation zone of the SW Greenland Ice Sheet, The Cryosphere, 16, 4379–4401, https://doi.org/10.5194/tc-16-4379-2022, 2022. Data: Clerx, N., Machguth, H., Tedstone, A., Jullien, N., Wever, N., Weingartner, R., and Roessler, O. (2022). DATASET: In situ measurements of meltwater flow through snow and firn in the accumulation zone of the SW Greenland Ice Sheet [Data set]. In The Cryosphere. Zenodo. https://doi.org/10.5281/zenodo.7119818'
        # sumup_index_conflict = check_conflicts(df_sumup, df_formatted,
        #                                         var=['profile_key', 'profile', 'date', 'start_depth', 'density'])
        list_add.append(df_formatted)


    df_add = pd.concat(list_add)
    # df_add=df_add.loc[df_add.notnull()]
            
    for v in col_needed:
        assert v in df_add.columns, f'{v} is missing'
    df_add[col_needed].to_csv('data_formatted.csv', index=None)

if __name__ == "__main__":
    process()
    
    
    
