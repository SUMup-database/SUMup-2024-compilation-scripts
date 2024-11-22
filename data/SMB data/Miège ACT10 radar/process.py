import pandas as pd
import numpy as np

df_raw = pd.read_excel('act10_gpr_accum_rate.xls')
df_raw.columns = [v.lower() for v in df_raw.columns]
df_list = []
start_year = [1992.2, 1995.6, 1999.7, 2002.2,2006.2, 2008.5, 1992.2]
end_year = [1995.6, 1999.7, 2002.2,2006.2, 2008.5, 2009.3, 2008.5]
for i, col in enumerate([ 'period:92_95',
       'period:95_99', 'period:99_02', 'period:02_06', 'period:06_08',
       'period:08_09', 'period:92_08']):
    df = df_raw[['latitude', 'longitude','elevation',col]]
    df.columns = ['latitude', 'longitude','elevation','smb']

    x=start_year[i]
    df['start_date']=pd.to_datetime(f"{int(x)}-01-01") + pd.to_timedelta(round((x % 1) * 365), unit='D')
    x=end_year[i]
    df['end_date']=pd.to_datetime(f"{int(x)}-01-01") + pd.to_timedelta(round((x % 1) * 365), unit='D')

    df_list.append(df)
df = pd.concat(df_list)

df['start_year']=df['start_date'].dt.year
df['end_year']=df['end_date'].dt.year
df['error'] = 0.23
df['notes'] = ''
df['name'] = 'ACT2010 Radar profile'
df['method'] = 'radar isochrones'

df['reference_short'] = 'Miège et al. (2013, 2014a)'
df['reference'] = 'Miège, C., Forster, R. R., Box, J. E., Burgess, E. W., McConnell, J. R., Pasteris, D. R., & Spikes, V. B. (2013). Southeast Greenland high accumulation rates derived from firn cores and ground-penetrating radar. Annals of Glaciology, 54(63), 322–332. doi:10.3189/2013AoG63A358. Data: Clement Miege, Richard R Forster, Jason E Box, Evan W Burgess, Joe R McConnell, Daniel R Pasteris, & Vandy B Spikes. (2014a). SE Greenland snow accumulation rates from GPR and 3 firn cores. Arctic Data Center. doi:10.18739/A2ST7DX47.'

df.to_csv('data_formatted.csv',index=None)
