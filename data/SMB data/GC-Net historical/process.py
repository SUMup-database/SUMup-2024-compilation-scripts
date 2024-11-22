import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nead
col_needed = ['method_key','start_date', 'end_date', 'start_year','end_year', 'latitude', 
              'longitude', 'elevation', 'notes', 'smb', 'error', 'name', 
              'reference', 'method', 'reference_short']
def process():
    plt.close('all')    
    df_meta = pd.read_csv('GC-Net_location.csv')
    df_meta.columns = df_meta.columns.str.lower()
    msk = df_meta.name.str.contains('JAR') | \
        df_meta.name.str.contains('SMS') | \
            df_meta.name.str.contains('SwissCamp')
    df_meta = df_meta.loc[msk,:]
    
    list_add=[]
    for name in df_meta.name[:-1]:
        print(name)
        df_all = nead.read(''+name+'.csv').to_dataframe()
        if 'HS_combined' not in df_all.columns:
            df_all = df_all[
            [ 'timestamp', 'HS1','latitude', 'longitude', 'elevation']]
            df_all.columns =  [ 'timestamp', 'HS_combined','latitude', 'longitude', 'elevation']
        else:
            df_all = df_all[
            [ 'timestamp', 'HS_combined','latitude', 'longitude', 'elevation']]
            
        df_all['end_date'] = pd.to_datetime(df_all.timestamp)
        df_all = df_all.set_index('end_date').drop(columns=['timestamp'])
    
        plt.figure()
        
        ablation =df_all.HS_combined.rolling('D', center=True).median().resample('D').asfreq().diff()
    
        ablation_season = (df_all.HS_combined.rolling('30D', center=True).median()
                           .resample('D').asfreq().diff())
        ablation_season.loc[ablation_season>-0.01] = 0
        ablation_season.loc[ablation_season.index.month.isin([10, 11,12,1,2,3,4])] = 0
        df_all.HS_combined.plot(ax=plt.gca())
        df_all.HS_combined.rolling('D', center=True).median().resample('D').asfreq().plot(ax=plt.gca())
        (ablation_season*100-1).plot(ax=plt.gca(), marker='x',c='turquoise')
        ablation_season = ablation_season.rolling('14D', center=True).mean()
        (ablation_season*100-1).plot(ax=plt.gca(), marker='d',c='tab:blue')
        ablation.plot(ax=plt.gca(), marker='.', c='gray')
    
        ablation.loc[ablation>-0.002] = np.nan
        ablation.loc[ablation<-0.3] = np.nan
        ablation.loc[ablation_season>-0.005] = np.nan
        ablation.plot(ax=plt.gca(), marker='o',c='tab:green')
        plt.title(name)
        plt.show()
    
        ablation = ablation.to_frame(name='smb') * 917 / 1000
        ablation = ablation.reset_index()
        ablation['start_date'] = pd.to_datetime(ablation.end_date) - pd.Timedelta('1 days')
    
        ablation["reference"] = "Steffen, K., Box, J.E. and Abdalati, W., 1996. Greenland climate network: GC-Net. US Army Cold Regions Reattach and Engineering (CRREL), CRREL Special Report, pp.98-103. and Steffen, K. and J. Box: Surface climatology of the Greenland ice sheet: Greenland Climate Network 1995-1999, J. Geophys. Res., 106, 33,951-33,972, 2001 and Steffen, K., Vandecrux, B., Houtz, D., Abdalati, W., Bayou, N., Box, J., Colgan, L., Espona Pernas, L., Griessinger, N., Haas-Artho, D., Heilig, A., Hubert, A., Iosifescu Enescu, I., Johnson-Amin, N., Karlsson, N. B., Kurup, R., McGrath, D., Cullen, N. J., Naderpour, R., Pederson, A. Ø., Perren, B., Philipps, T., Plattner, G.K., Proksch, M., Revheim, M. K., Særrelse, M., Schneebli, M., Sampson, K., Starkweather, S., Steffen, S., Stroeve, J., Watler, B., Winton, Ø. A., Zwally, J., Ahlstrøm, A.: GC-Net Level 1 automated weather station data, https://doi.org/10.22008/FK2/VVXGUT, GEUS Dataverse, V2, 2023. and Vandecrux, B., Box, J.E., Ahlstrøm, A.P., Andersen, S.B., Bayou, N., Colgan, W.T., Cullen, N.J., Fausto, R.S., Haas-Artho, D., Heilig, A., Houtz, D.A., How, P., Iosifescu Enescu , I., Karlsson, N.B., Kurup Buchholz, R., Mankoff, K.D., McGrath, D., Molotch, N.P., Perren, B., Revheim, M.K., Rutishauser, A., Sampson, K., Schneebeli, M., Starkweather, S., Steffen, S., Weber, J., Wright, P.J., Zwally, J., Steffen, K.: The historical Greenland Climate Network (GC-Net) curated and augmented Level 1 dataset, Submitted to ESSD, 2023"
        ablation["reference_short"] = "Historical GC-Net: Steffen et al. (1996, 2001, 2023); Vandecrux et al. (2023)"
      
        ablation['name'] = name
        coord = df_all[['latitude','longitude','elevation']].resample('D').mean()
        ablation['latitude'] = coord.latitude.values
        ablation['longitude'] = coord.longitude.values
        ablation['elevation'] = coord.elevation.values
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
    