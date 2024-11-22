

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
    sc=plt.scatter(df_stack.time,
                -df_stack.depth,
                12, df_stack.temperature)
    plt.ylim(-df_stack.depth.max(), 0)
    plt.title(df_stack.site.unique().item())
    plt.colorbar(sc)
    f.savefig('./'+filename+'.png', dpi=400)
    plt.show()


# all GC-Net stations
from nead import read
np.seterr(invalid="ignore")
df_info = pd.read_csv('./GC-Net_location.csv', skipinitialspace=True)
df_info = df_info.loc[df_info.Northing>0,:]

plot = False
df_gcn = pd.DataFrame()
temp_var = ['TS1','TS2', 'TS3', 'TS4', 'TS5', 'TS6', 'TS7', 'TS8', 'TS9', 'TS10']
depth_var = ['DTS1', 'DTS2','DTS3', 'DTS4', 'DTS5', 'DTS6', 'DTS7', 'DTS8', 'DTS9', 'DTS10']
for ID, site in zip(df_info.ID, df_info.Name):   
    if not os.path.exists('./L1-daily/'+site.replace(' ','')+'_daily.csv'):
        continue
        print('skipping',site,'because already in merged PROMICE/GC-Net data')
    df_aws = read('./L1-daily/'+site.replace(' ','')+'_daily.csv').to_dataframe()
    df_aws['timestamp'] = pd.to_datetime(df_aws.timestamp, utc=True)
    df_aws = df_aws.set_index('timestamp')
    print('   '+site)
    if 'DTS1' not in df_aws.columns:
        print('    No temperature data')
        continue        
    if df_aws[temp_var].isnull().all().all():
        print('    No temperature data')
        continue

    df_aws.index = df_aws.index.rename('date')

    df_aws = df_aws[temp_var+depth_var+['TS_10m']]
    df_aws["latitude"] = df_info.loc[df_info.Name==site, "Northing"].values[0]
    df_aws["longitude"] = df_info.loc[df_info.Name==site, "Easting"].values[0]
    df_aws["elevation"] = df_info.loc[df_info.Name==site, "Elevationm"].values[0]
    df_aws['name'] = site
    df_aws["note"] = ""

    # filtering
    for v in temp_var+['TS_10m']:
        df_aws.loc[df_aws[v] > 0.1, v] = np.nan
        df_aws.loc[df_aws[v] < -70, v] = np.nan
        

    # df_aws["reference"] = "Steffen, K., Box, J.E. and Abdalati, W., 1996. Greenland climate network: GC-Net. US Army Cold Regions Reattach and Engineering (CRREL), CRREL Special Report, pp.98-103. and Steffen, K. and J. Box: Surface climatology of the Greenland ice sheet: Greenland Climate Network 1995-1999, J. Geophys. Res., 106, 33,951-33,972, 2001 and Steffen, K., Vandecrux, B., Houtz, D., Abdalati, W., Bayou, N., Box, J., Colgan, L., Espona Pernas, L., Griessinger, N., Haas-Artho, D., Heilig, A., Hubert, A., Iosifescu Enescu, I., Johnson-Amin, N., Karlsson, N. B., Kurup, R., McGrath, D., Cullen, N. J., Naderpour, R., Pederson, A. Ø., Perren, B., Philipps, T., Plattner, G.K., Proksch, M., Revheim, M. K., Særrelse, M., Schneebli, M., Sampson, K., Starkweather, S., Steffen, S., Stroeve, J., Watler, B., Winton, Ø. A., Zwally, J., Ahlstrøm, A.: GC-Net Level 1 automated weather station data, https://doi.org/10.22008/FK2/VVXGUT, GEUS Dataverse, V2, 2023. and Vandecrux, B., Box, J.E., Ahlstrøm, A.P., Andersen, S.B., Bayou, N., Colgan, W.T., Cullen, N.J., Fausto, R.S., Haas-Artho, D., Heilig, A., Houtz, D.A., How, P., Iosifescu Enescu , I., Karlsson, N.B., Kurup Buchholz, R., Mankoff, K.D., McGrath, D., Molotch, N.P., Perren, B., Revheim, M.K., Rutishauser, A., Sampson, K., Schneebeli, M., Starkweather, S., Steffen, S., Weber, J., Wright, P.J., Zwally, J., Steffen, K.: The historical Greenland Climate Network (GC-Net) curated and augmented Level 1 dataset, Submitted to ESSD, 2023"
    # df_aws["reference_short"] = "Historical GC-Net: Steffen et al. (1996, 2001, 2023); Vandecrux et al. (2023)"
    df_aws["reference"] = ""
    df_aws["reference_short"] = "Historical GC-Net: Steffen et al. (1996, 2001, 2023); Vandecrux et al. (2023)"
    df_stack = stack_multidepth_df(df_aws.reset_index(), temp_var, depth_var)
    df_stack["method"] = "type-T thermocouples"
    df_stack["error"] = np.nan
    df_stack["duration"] = 1

    if plot:
        plot_string_dataframe(df_stack,  filename= 'GC-Net historical '+site)

    df_gcn = pd.concat((df_gcn, df_stack), ignore_index=True)
    

# saving full-res data
df_all = df_gcn.copy()

# %% Summit string 2005-2009
print('Summit string 2005-2009')
df = pd.read_csv('./Summit Snow Thermistors/2007-2009/t_hour.dat', sep=r'\s+', header=None)
df.columns = 'id;year;day;hour_min;t_1;t_2;t_3;t_4;t_5;t_6;t_7;t_8;t_9;t_10;t_11;t_12;t_13;t_14'.split(';')
df[df==-6999] = np.nan

df.loc[df.hour_min==2400,'day'] = df.loc[df.hour_min==2400,'day']+1
df.loc[df.hour_min==2400,'hour_min'] = 0
df.loc[df.day==367,'year'] = df.loc[df.day==367,'year']+1
df.loc[df.day==367,'day'] = 1

df['hour'] = np.trunc(df.hour_min/100)
df['minute'] = df.hour_min - df.hour*100
df['timestamp'] = pd.to_datetime(df.year*100000+df.day*100+df.hour, format='%Y%j%H', utc=True, errors='coerce')
df = df.set_index('timestamp').drop(columns=['year','hour','minute','id','day', 'hour_min'])


df3 = pd.read_csv('./Summit Snow Thermistors/snow.dat', header = None, sep=r'\s+')
df3.columns = 'year;day;hour_min;t_3;t_4;t_5;t_6;t_7;t_8;t_9;t_10;t_11;t_12;t_13;t_14'.split(';')
df3[df3==-6999] = np.nan

df3.loc[df3.hour_min==2400,'day'] = df3.loc[df3.hour_min==2400,'day']+1
df3.loc[df3.hour_min==2400,'hour_min'] = 0
df3.loc[df3.day==367,'year'] = df3.loc[df3.day==367,'year']+1
df3.loc[df3.day==367,'day'] = 1

df3['hour'] = np.trunc(df3.hour_min/100)
df3['minute'] = df3.hour_min - df3.hour*100
df3['timestamp'] = pd.to_datetime(df3.year*100000+df3.day*100+df3.hour, format='%Y%j%H', utc=True, errors='coerce')
df3 = df3.set_index('timestamp').drop(columns=['year','hour','minute','day', 'hour_min'])
df3[['t_1','t_2']] = np.nan

df = pd.concat((df, df3))
df = df.resample('D').mean()

col_temp = [v for v in df.columns if 't_' in v]
# df[col_temp].plot()

for i in range(1, len(col_temp)+1):
    df['depth_'+str(i)] = np.nan
col_depth = [v for v in df.columns if 'depth_' in v]

# adding the three dates
surface = [4.35, 5.00, 5.57]
depths =  np.array([[2.15, 2.65, 3.15, 3.65, 4.15, 4.65, 
                     4.65,  4.85, 5.35, 6.35, 9.35, 11.85, 14.35, 19.35],
           [2.75, 3.40, 3.90, 4.40, 4.90, 5.40, 5.40, 5.60, 
            6.40, 7.40, 10.4, 12.60, 15.4, 20.40],
           [np.nan, np.nan, 0.50, 2.00, 3.32, 4.97 ,5.97,  
            6.17, 6.97, 7.97, 10.97, 13.17, 16.17, 21.17]])

for i, date in enumerate(['2007-04-07', '2008-07-25', '2009-05-19']):
    tmp = df.iloc[0:1,:]*np.nan
    tmp.index = [pd.to_datetime(date, utc=True)]
    tmp['surface_height'] = surface[i]
    tmp[col_depth] = depths[i,:]
    df = pd.concat((tmp, df))
df = df.sort_index()
df.surface_height = df.surface_height.interpolate().values - df.surface_height.iloc[0]
tmp = df.surface_height.diff()
tmp.loc[tmp==0] = 0.62/365/24
df.surface_height = tmp.cumsum().values
df.loc[df[col_depth[-1]].notnull(), col_depth] = df.loc[df[col_depth[-1]].notnull(), col_depth] - np.repeat(np.expand_dims(df.loc[df[col_depth[-1]].notnull(), 'surface_height'].values,1), len(col_depth), axis=1)
df[col_depth] = df[col_depth].ffill().values

df.loc[:'2008',col_depth]  = df.loc[:'2008',col_depth].bfill() 
for col in col_depth:
    df[col] = df[col] + df['surface_height']
    
df = df[col_temp+col_depth].reset_index().rename(columns={'index':'timestamp'})

site = 'Summit'
df["latitude"] = df_info.loc[df_info["Name"] == site, "Northing"].values[0]
df["longitude"] = df_info.loc[df_info["Name"] == site, "Easting"].values[0]
df["elevation"] = df_info.loc[df_info["Name"] == site, "Elevationm"].values[0]
df["reference"] = "GC-Net unpublished"
df["reference_short"] = "GC-Net unpublished"
df['name'] = site
df["note"] = ''
    
# saving full-res
df_stack = stack_multidepth_df(df, col_temp, col_depth)
df_stack["duration"] = 1
df_stack['error'] = np.nan
df_stack["method"] = "T107 thermistors"
if plot:
    plot_string_dataframe(df_stack, filename= 'GC-Net unpublished '+site)
df_all = pd.concat((df_stack[needed_cols], df_all[needed_cols]),ignore_index=True)

# %% Summit string 2000-2002
print('Summit string 2000-2002')
df = pd.read_csv('./Summit Snow Thermistors/2000-2002 thermistor string/2002_sun_00_01_raw.dat', header = None)
df.columns = 'id;day;hour_min;TS1;TS2;TS3;TS4;TS5;TS6;TS7;TS8;TS9;TS10;TS11;TS12;TS13;TS14;TS15;TS16'.split(';')
df['year'] = 2000
df.loc[4781:,'year'] = 2001
df2 = pd.read_csv('./Summit Snow Thermistors/2000-2002 thermistor string/2002_sum_01_02_raw.dat', header = None)
df2.columns = 'id;day;hour_min;TS1;TS2;TS3;TS4;TS5;TS6;TS7;TS8;TS9;TS10;TS11;TS12;TS13;TS14;TS15;TS16'.split(';')
df2['year'] = 2001
df2.loc[4901:,'year'] = 2002
df = pd.concat((df, df2))

df[df==-6999] = np.nan

df.loc[df.hour_min==2400,'day'] = df.loc[df.hour_min==2400,'day']+1
df.loc[df.hour_min==2400,'hour_min'] = 0
df.loc[df.day==367,'year'] = df.loc[df.day==367,'year']+1
df.loc[df.day==367,'day'] = 1

df['hour'] = np.trunc(df.hour_min/100)
df['minute'] = df.hour_min - df.hour*100
df['date'] = pd.to_datetime(df.year*100000+df.day*100+df.hour, format='%Y%j%H',utc=True, errors='coerce')
df = df.set_index('date')  #.drop(columns=['year','hour','minute','id','day', 'hour_min'])

col_temp = [v for v in df.columns if 'TS' in v]

a=-9.0763671; b=0.704343; c=0.00919; d=0.000137; e=0.00000116676; f=0.00000000400674

# calibration coefficients from ice bath at Summit 2000
coef = [0.361 , 0.228, 0.036, 0.170, 0.170, 0.228, -0.022, 0.361, 0.036, 0.036, 0.228, 0.170, 0.323, 0.132, 0.323, 0.266]

#       convert UUB thermister reading to temperature      
df[col_temp]=a+b*df[col_temp]+c*(df[col_temp]**2)+d*(df[col_temp]**3)+e*(df[col_temp]**4)+f*(df[col_temp]**5)

# use the calibration coefficient from Summit 2000 ice bath

for i, col in enumerate(col_temp):
    df[col] = df[col] + coef[i]
    df.loc[df[col]>-5,col] = np.nan

depths = ['10m', '9m', '8m', '7m', '6m', '5m', '4m', '3m', '2.5m', '2m', '1.5m',
       '1m', '0.75m', '0.5m', '0.3m', '0.1m']
for col, col2 in zip(depths, col_temp):   
    df['DTS'+col2.replace('TS','')]=np.nan
    df['DTS'+col2.replace('TS','')] = float(col.replace('m','')) +  np.arange(len(df.index)) * 0.62/365/24
df = df.sort_index()
df.loc['2001-06-11':'2001-12-15', 'DTS2'] = np.nan
col_depth = ['DTS'+str(i) for i in range(1,17)]

df = df[col_temp+col_depth].resample('D').mean()
df = df.reset_index()
site = 'Summit'
df["latitude"] = df_info.loc[df_info["Name"] == site, "Northing"].values[0]
df["longitude"] = df_info.loc[df_info["Name"] == site, "Easting"].values[0]
df["elevation"] = df_info.loc[df_info["Name"] == site, "Elevationm"].values[0]
df["reference"] = "GC-Net unpublished"
df["reference_short"] = "GC-Net unpublished"
df['name'] = site
df["note"] = ''

df_stack = stack_multidepth_df(df, col_temp, col_depth)
df_stack["duration"] = 1
df_stack["error"] = np.nan
df_stack["method"] = "T107 thermistors"

if plot:
    plot_string_dataframe(df_stack, filename= 'GC-Net unpublished '+site)
df_all = pd.concat((df_stack[needed_cols], df_all[needed_cols]),ignore_index=True)

# %% Swiss Camp TENT thermistor
print('Swiss Camp TENT thermistor')
import os
list_file = os.listdir('./Swiss Camp TENT thermistor')
list_file = [f for f in list_file if f.lower().endswith('.dat')]
# 2000_ICETEMP.DAT "l=channel3 +3.5 C"
df_swc = pd.DataFrame()
for f in list_file:
    print('   '+f)
    df = pd.read_csv('./Swiss Camp TENT thermistor/'+f, 
                     header=None,
                     sep=r'\s+')
    df = df.apply(pd.to_numeric)
    df.columns = ['doy'] + ['TS'+str(i) for i in df.columns[1:].values]
    year = float(f[:4])
    df['year'] = year
    
    if (any(df.doy.diff()<0))>0:
        for i, ind in enumerate(df.loc[df.doy.diff()<0,:].index.values):
            df.loc[slice(ind), 'year'] = year - (len(df.loc[df.doy.diff()<0,:].index.values)-i)
    df_swc = pd.concat((df_swc, df), ignore_index=True)
    
df_swc['timestamp'] = pd.to_datetime(df_swc.year*1000+df_swc.doy, format='%Y%j', utc=True, errors='coerce')
df_swc = df_swc.set_index('timestamp').resample('D').mean()

col_temp = [v for v in df_swc.columns if 'TS' in v]
col_depth = ['DTS'+str(i) for i in range(1,len(col_temp)+1)]
for col in col_temp:
    df_swc.loc[df_swc[col]<-17,col] = np.nan
    df_swc.loc[df_swc[col]>10,col] = np.nan

if plot: df_swc[col_temp].plot()

# Installation depth
depth_ini = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0.1, 0.75, 0.5]
for i,d in enumerate(depth_ini):
    df_swc[col_depth[i]] = d

df_swc = df_swc.loc[df_swc[col_temp].notnull().any(axis=1),:]

df_swc = df_swc.drop(columns=['doy','year']).reset_index()
site = 'Swiss Camp'
df_swc["latitude"] = df_info.loc[df_info["Name"] == site, "Northing"].values[0]
df_swc["longitude"] = df_info.loc[df_info["Name"] == site, "Easting"].values[0]
df_swc["elevation"] = df_info.loc[df_info["Name"] == site, "Elevationm"].values[0]
df_swc["reference"] = "GC-Net unpublished"
df_swc["reference_short"] = "GC-Net unpublished"
df_swc['name'] = site
df_swc["note"] = ''   
df_stack = stack_multidepth_df(df_swc, col_temp, col_depth)
df_stack["method"] = "UUB thermistors"
df_stack["duration"] = 1
df_stack["error"] = np.nan

if plot:
    plot_string_dataframe(df_stack, filename= 'GC-Net unpublished '+site)
df_all = pd.concat((df_stack[needed_cols], df_all[needed_cols]),ignore_index=True)

df_all['error'] = np.nan
df_all=df_all.loc[
    (df_all.temperature.notnull() &
     df_all.latitude.notnull() &
     df_all.depth.notnull()), :]
df_all.to_csv('data_formatted.csv')
