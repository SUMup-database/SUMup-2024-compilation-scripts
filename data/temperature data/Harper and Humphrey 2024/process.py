

import os
import re
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

needed_cols = [ 'timestamp', 'latitude', 'longitude', 'elevation', 'depth',
                'temperature', 'error', 'name', 'method', 'reference',
                'reference_short', 'duration']


def stack_multidepth_df(df_in, temp_var, depth_var):
    print('    -> converting to multi-depth xarray')
    df_in = df_in.rename(columns={'date':'timestamp'})
       
    # checking the time variable is no less than hourly
    # dif_time = df_in.time.diff()
    # if len(dif_time[dif_time<pd.Timedelta(minutes=50)])>0:
    #     print('found time steps under 1 h')
    #     print(df_in.time[dif_time<pd.Timedelta(minutes=50)])
    # df_in.time = df_in.time.dt.round('H')

    df_in = df_in.set_index(['name', 'timestamp', 'reference_short','latitude', 
                             'longitude', 'elevation', 'note', 'reference'])

    # some filtering
    df_in = df_in.dropna(subset=temp_var, how='all')
    for v in temp_var:
        df_in.loc[df_in[v] > 1, v] = np.nan
        df_in.loc[df_in[v] < -70, v] = np.nan
    df_in = df_in.loc[~df_in[temp_var].isnull().all(axis=1),:]
    
    df_stack = df_in[temp_var].rename(columns=dict(zip(temp_var, 
         range(1,len(temp_var)+1)))).stack(future_stack=True).to_frame(name='temperature').reset_index()
    df_stack['depth'] = df_in[depth_var].rename(columns=dict(zip(depth_var, 
             range(1,len(depth_var)+1)))).stack(future_stack=True).to_frame(name='depth').values

    return df_stack

def plot_string_dataframe(df_stack, filename):
    df_stack = df_stack.rename(columns={'date':'timestamp',
                                        'temperatureObserved':'temperature',
                                        'depthOfTemperatureObservation':'depth'})

    f=plt.figure(figsize=(9,9))
    sc=plt.scatter(df_stack.timestamp,
                -df_stack.depth,
                12, df_stack.temperature)
    plt.ylim(-df_stack.depth.max(), 0)
    plt.title(df_stack.name.unique().item())
    plt.colorbar(sc)
    f.savefig('./'+filename+'.png', dpi=400)
    plt.show()

df_all = pd.DataFrame()
# all GC-Net stations
metadata = {
    'CP': {
        'file': '2019_CP_Temp_30m.csv',
        'latitude': 69 + 52.595 / 60,
        'longitude': -(47 + 0.652 / 60),
        'elevation': 1951
    },
    
    'T2m': {
        'file': '2019_T2m_Temp_32m.csv',
        'latitude': 69+45.227/60,
        'longitude': -(47  + 54.727 / 60),
        'elevation': np.nan
    },
    'T2': {
        'file': '2019_T2_Temp_32m.csv',
        'latitude':69+45.415/60,
        'longitude': -(47+52.817/60),
        'elevation': np.nan
    },
    'T3': {
        'file': '2019_T3_Temp_32m.csv',
        'latitude': 69 + 46.902 / 60,
        'longitude': -(47 + 40.102 / 60),
        'elevation': 1779
    },
    
    'T4': {
        'file': '2019_T4_Temp_32m.csv',
        'latitude': 69 + 49.203 / 60,
        'longitude': -(47 + 27.027 / 60),
        'elevation': 1951
    }
}

plot = False

for site in metadata.keys():   
    
    df_aws = pd.read_csv(metadata[site]['file'], skiprows=5, encoding='ansi')
    df_aws.columns=    df_aws.columns.str.lower()
    df_aws['timestamp'] = pd.to_datetime(df_aws.date, utc=True)
    df_aws = df_aws.set_index('timestamp')
    print('   '+site)
    df_aws.columns = ['Date'] + [f'temp_{col}' for col in df_aws.columns[1:]]
    
    for col in df_aws.columns[1:]:
        depth_value = col.split('_')[1]
        try:
            depth_value = float(depth_value)
        except ValueError:
            pass  # Handle or skip non-numeric depth values if necessary
        df_aws[f'depth_{depth_value}'] = depth_value
    
    temp_var = [v for v in df_aws.columns if 'temp' in v]
    depth_var = [v for v in df_aws.columns if 'depth' in v]
    df_aws.index = df_aws.index.rename('date')

    df_aws = df_aws[temp_var+depth_var]
    df_aws = df_aws.resample('D').mean()
    df_aws["latitude"] = metadata[site]['latitude']
    df_aws["longitude"] = metadata[site]['longitude']
    df_aws["elevation"] = metadata[site]['elevation']
    df_aws['name'] = site
    df_aws["note"] = ""       

    df_aws["reference"] = " Saito, J., Harper, J., & Humphrey, N. (2024). Uptake and transfer of heat within the firn layer of Greenland Ice Sheet's percolation zone. Journal of Geophysical Research: Earth Surface, 129, e2024JF007667. https://doi.org/10.1029/2024JF007667 . Data: Joel Harper, & Neil Humphrey. (2024). Firn temperature-time series to 30 meter depth at five sites along the west Expéditions Glaciologiques Internationales au Groenland (EGIG) line, Greenland summer of 2019. Arctic Data Center. doi:10.18739/A2JM23H8X."
    df_aws["reference_short"] = "Saito et al. (2024); Harper and Humphrey (2024)"
    df_stack = stack_multidepth_df(df_aws.reset_index(), temp_var, depth_var)
    df_stack["method"] = "thermistor string"
    df_stack["error"] = 0.1
    df_stack["duration"] = 0.5

    if plot:
        plot_string_dataframe(df_stack,  filename= 'Harper and Humphrey'+site)

    df_all = pd.concat((df_all, df_stack), ignore_index=True)
    
df_all.to_csv('data_formatted.csv')
