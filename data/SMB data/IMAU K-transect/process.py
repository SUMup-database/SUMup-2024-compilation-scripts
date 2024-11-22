import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
col_needed = ['method_key','start_date', 'end_date', 'start_year','end_year', 'latitude', 
              'longitude', 'elevation', 'notes', 'smb', 'error', 'name', 
              'reference', 'method', 'reference_short']
def process():
    plt.close('all')    
    df_meta = pd.read_csv('meta.csv')
    df_meta.site = df_meta.site.str.replace('Site-','AWS')
    df_meta=df_meta.set_index('site')
    list_add = []
    for id in ['5','6']:
        print('K-transect','S'+id)
        df_all = pd.DataFrame()
        for yr in range(2003,2016):
            yr = str(yr)
            if not os.path.isfile('Smeets et al. 2022/K-transect_AWS'+id+'_'+yr+'.tab'):
                continue
            df = pd.read_csv('Smeets et al. 2022/K-transect_AWS'+id+'_'+yr+'.tab',
                             skiprows=44, sep='\t')
            
            try:
                df = df[['Date/Time',  'Height rel [m]']]
            except:
                try:
                    df = pd.read_csv('Smeets et al. 2022/K-transect_AWS'+id+'_'+yr+'.tab',
                                     skiprows=45, sep='\t')
                    df = df[['Date/Time',  'Height rel [m]']]
                except:
                    df = pd.read_csv('Smeets et al. 2022/K-transect_AWS'+id+'_'+yr+'.tab',
                                     skiprows=43, sep='\t')
                    df = df[['Date/Time',  'Height rel [m]']]
    
                
            df.columns = ['end_date','z_boom']
            df.end_date = pd.to_datetime(df.end_date)
            df = df.set_index('end_date')
            df['z_height'] = df.z_boom.max() - df.z_boom.ffill(limit=24*2)
            df_all = pd.concat((df_all,df))
    
        plt.figure()
        shifts = df_all['z_height'].diff()
        shifts.loc[shifts.abs()<2] = 0
        df_all['z_height'] = df_all['z_height']-shifts.cumsum()
        ablation =df_all.z_height.rolling('D', center=True).median().resample('D').asfreq().diff()
    
        ablation_season = (df_all.z_height.rolling('30D', center=True).median()
                           .resample('D').asfreq().diff())
        ablation_season.loc[ablation_season>-0.01] = 0
        ablation_season.loc[ablation_season.index.month.isin([10, 11,12,1,2,3,4])] = 0
        df_all.z_height.plot(ax=plt.gca())
        df_all.z_height.rolling('D', center=True).median().resample('D').asfreq().plot(ax=plt.gca())
        (ablation_season*100-1).plot(ax=plt.gca(), marker='x',c='turquoise')
        ablation_season = ablation_season.rolling('14D', center=True).mean()
        (ablation_season*100-1).plot(ax=plt.gca(), marker='d',c='tab:blue')
        ablation.plot(ax=plt.gca(), marker='.', c='gray')
    
        ablation.loc[ablation>-0.002] = np.nan
        ablation.loc[ablation<-0.3] = np.nan
        ablation.loc[ablation_season>-0.005] = np.nan
        ablation.plot(ax=plt.gca(), marker='o',c='tab:green')
        plt.title('S'+id)
        plt.xlim('2003','2022')
        plt.show()
    
        ablation = ablation.to_frame(name='smb') * 917 / 1000
        ablation = ablation.reset_index()
        ablation['start_date'] = pd.to_datetime(ablation.end_date) - pd.Timedelta('1 days')
        ablation['reference'] = "Paul C. J. P. Smeets, Peter Kuipers Munneke, Dirk van As, Michiel R. van den Broeke, Wim Boot, Hans Oerlemans, Henk Snellen, Carleen H. Reijmer & Roderik S. W. van de Wal (2018) The K-transect in west Greenland: Automatic weather station data (1993–2016), Arctic, Antarctic, and Alpine Research, 50:1, DOI: 10.1080/15230430.2017.1420954. Data: Smeets, PCJP et al. (2022): Automatic weather station data collected from 2003 to 2021 at the Greenland ice sheet along the K-transect, West-Greenland [dataset publication series]. PANGAEA, https://doi.org/10.1594/PANGAEA.947483"
        ablation['reference_short'] = "Smeets et al. (2018, 2022)"
        ablation['name'] = 'AWS'+id
        ablation['latitude'] = df_meta.loc[ablation.name].lat.values
        ablation['longitude'] = df_meta.loc[ablation.name].lon.values
        ablation['elevation'] = df_meta.loc[ablation.name].elev.values
        ablation['notes'] = ''
        ablation['method'] = 'surface height ranging'
        ablation['method_key'] = 14
        
        ablation['error'] = np.nan
        ablation['end_year'] = ablation.end_date.dt.year
        ablation['start_year'] =  ablation.start_date.dt.year
    
        ablation = ablation.loc[ablation.smb.notnull()]

        list_add.append(ablation)
    
    df_add = pd.concat(list_add, ignore_index=True)
        
    for v in col_needed:
        assert v in df_add.columns, f'{v} is missing'
    df_add[col_needed].to_csv('data_formatted.csv',index=None)

if __name__ == "__main__":
    process()
    