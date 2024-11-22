import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def stack_multidepth_df(df_in, temp_var, depth_var):
    print('    -> converting to multi-depth xarray')
    df_in = df_in.rename(columns={'date':'time'})

    # checking the time variable is no less than hourly
    # dif_time = df_in.time.diff()
    # if len(dif_time[dif_time<pd.Timedelta(minutes=50)])>0:
    #     print('found time steps under 1 h')
    #     print(df_in.time[dif_time<pd.Timedelta(minutes=50)])
    # df_in.time = df_in.time.dt.round('h')

    df_in = df_in.set_index(['site', 'time', 'reference_short','latitude',
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
    df_stack = df_stack.rename(columns={'date':'time',
                                        'temperatureObserved':'temperature',
                                        'depthOfTemperatureObservation':'depth'})

    f=plt.figure(figsize=(9,9))
    sc=plt.scatter(df_stack.time,
                -df_stack.depth,
                12, df_stack.temperature)
    plt.ylim(-df_stack.depth.max(), 0)
    plt.title(df_stack.site.unique().item())
    plt.colorbar(sc)
    f.savefig(filename+'.png', dpi=400)
    plt.show()

needed_cols=['method_key', 'timestamp', 'latitude', 'longitude', 'elevation', 'depth', 'open_time', 'duration', 'temperature', 'error', 'reference', 'reference_short','name','method']


df = pd.read_csv(r'DATA/SIGMA_AWS_SiteA_2012-2020_Lv1_3.csv')
df = df[['date','st1', 'st2', 'st3', 'st4', 'st5', 'st6','st1_depth', 'st2_depth', 'st3_depth', 'st4_depth', 'st5_depth', 'st6_depth']]
df[df==-9999] = np.nan
df['time'] = pd.to_datetime(df.date)
df = df.drop(columns='date').set_index('time')
df = df.resample('D').mean().reset_index()
df['site'] = 'SIGMA-A'
df ['latitude'] = 78.052
df ['longitude'] = -67.628
df ['elevation'] = np.nan
df ['note'] = ''
df ['reference_short'] = 'Nishimura et al. (2023)'
df ['reference'] = 'Nishimura, M., Aoki, T., Niwano, M., Matoba, S., Tanikawa, T., Yamasaki, T., Yamaguchi, S., and Fujita, K.: Quality-controlled meteorological datasets from SIGMA automatic weather stations in northwest Greenland, 2012–2020, Earth Syst. Sci. Data, 15, 5207–5226, https://doi.org/10.5194/essd-15-5207-2023, 2023. Data: Nishimura, M., T. Aoki, M. Niwano, S. Matoba, T. Tanikawa, S. Yamaguchi, T. Yamasaki, A. Tsushima, K. Fujita, Y. Iizuka, Y. Kurosaki, 2023, Quality-controlled datasets of Automatic Weather Station (AWS) at SIGMA-A site from 2012 to 2020: Level 1.3, 2.00, Arctic Data archive System (ADS), Japan, http://doi.org/10.17592/001.2022041303'

df_stack = stack_multidepth_df(df, ['st1', 'st2', 'st3', 'st4', 'st5', 'st6' ],
               ['st1_depth', 'st2_depth', 'st3_depth', 'st4_depth', 'st5_depth', 'st6_depth'])
df_stack = df_stack.loc[df_stack.depth>0,:]
df_stack['depth'] = df_stack.depth/100
df_stack = df_stack.loc[df_stack.depth.notnull(),:]
df_stack = df_stack.loc[df_stack.temperature.notnull(),:]
plot_string_dataframe(df_stack, 'SIGMA-A')

df_stack ['error'] = 0.15
df_stack ['method'] = 'Thermistor'
df_stack ['method_key'] = 4
df_stack ['open_time'] = np.nan
df_stack['duration'] = 24
df_stack = df_stack.rename(columns={'time':'timestamp', 'site':'name'})
df_stack[needed_cols].to_csv('data_formatted.csv', index=None)
