    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
col_needed = ['method_key','start_date', 'end_date', 'start_year','end_year', 'latitude', 
              'longitude', 'elevation', 'notes', 'smb', 'error', 'name', 
              'reference', 'method', 'reference_short']
def process():
    plt.close('all')    
    df_promice_meta = pd.read_csv('AWS_metadata.csv').set_index('stid')
    list_add = []
    for station in ['KAN_M', 'QAS_M', 'QAS_U','TAS_A','THU_U2']:
        file = 'SF_'+station+'.txt'
    
        df_sf = pd.read_csv(file, sep='\s+')
        df_sf[df_sf==-999] = np.nan
        df_sf['time'] = pd.to_datetime(df_sf[['Year','Month','Day']])
        df_sf = df_sf.set_index('time')
        
        df_smooth = df_sf[['SWE(cmWeq)']].rolling('7D',center=True).median().resample('7D').asfreq()
        df_smooth['error'] = df_sf[['SWE(cmWeq)']].rolling('7D',center=True).std().resample('7D').asfreq()
        fig = plt.figure()
        ax=plt.gca()
        df_sf['SWE(cmWeq)'].plot(ax=ax, marker='o')
        df_sf['SWE(cmWeq)'].rolling('7D',center=True).median().resample('7D').asfreq().plot(ax=ax, marker='o')
        plt.title(station)
        plt.ylabel('Snow accumulation (m w.e.)')    
        
        df_smooth.loc[df_smooth['SWE(cmWeq)']<0.2]  = np.nan
        df_smooth['SWE(cmWeq)'] = df_smooth['SWE(cmWeq)'].diff()
        df_smooth['smb'] = df_smooth['SWE(cmWeq)']/100
        
        (df_smooth['smb'] *100).plot(ax=ax)
        
        df_smooth =df_smooth.reset_index()
        
        start_date = df_smooth.time.values[0:-1]
        df_smooth = df_smooth.iloc[1:]
        
        df_smooth['end_date'] = df_smooth.time
        df_smooth['start_date'] = start_date
        df_smooth =df_smooth.loc[df_smooth.smb.notnull()]
    
    
        df_smooth['start_year'] = df_smooth['start_date'].dt.year
        df_smooth['end_year'] = df_smooth['end_date'].dt.year
        df_smooth['name'] = station+'_snowfox'
        df_smooth['reference'] = "Fausto, Robert S., 2021, Snow-water equivalent of snowpacks, https://doi.org/10.22008/FK2/B5KVJV, GEUS Dataverse, V2 "
        df_smooth['reference_short'] = "Fausto (2021)"
        df_smooth['latitude'] = df_promice_meta.loc[station].lat_last_known
        df_smooth['longitude'] = df_promice_meta.loc[station].lon_last_known
        df_smooth['elevation'] = df_promice_meta.loc[station].alt_last_known
        df_smooth['notes'] = ''
        df_smooth['method'] = 'snowfox'
        df_smooth['method_key'] = 14
        list_add.append(df_smooth)
    
    df_add = pd.concat(list_add, ignore_index=True)
        
    for v in col_needed:
        assert v in df_add.columns, f'{v} is missing'
    df_add[col_needed].to_csv('data_formatted.csv', index=None)

if __name__ == "__main__":
    process()
    