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
plot = False
if plot:
    plt.figure()
df_ref = pd.read_csv('reference.txt').set_index('name')
for file in os.listdir('.'):
    if not file.endswith('.csv'):
        continue

    if file == 'data_formatted.csv':
        continue

    df_meta = pd.read_csv(file, sep=',',encoding='ansi', header=None).iloc[:12,:]
    df = pd.read_csv(file, sep=',',skiprows=12,encoding='ansi')

    if 'Firn Density [g/cm**3]' in df.columns:
        df['Firn Density [kg/m**3]'] = df['Firn Density [g/cm**3]']*1000

    if 'Top depth (cm)' in df.columns:
        df['Top depth (m)'] = df['Top depth (cm)']/100
        df['Bottom depth (m)'] = df['Bottom depth (cm)']/100
    df = df.rename(columns={'Top depth (m)':'start_depth',
                       'Bottom depth (m)':'stop_depth',
                       'Firn Density [kg/m**3]': 'density'})
    df['error'] = np.nan
    df['midpoint'] = (df.stop_depth + df.start_depth)/2

    df['profile'] = df_meta.iloc[0,1]
    df['reference_short'] = df_ref.loc[file.replace('.csv','')].values[0]
    df['reference'] = df_meta.iloc[1,1].replace('\n','')
    # print('\''+file.replace('.csv','')+'\':',
    #       df_meta.iloc[1,1])

    df['method_key'] = 4
    df['method'] = "ice or firn core section"
    timestamp= df_meta.iloc[10,1]
    if '-' in timestamp:
        timestamp = timestamp.replace(' ','').split('-')[1]
    d, m, y  = timestamp.split('/')
    if int(m)>12:
        tmp = m
        m = d
        d = tmp

    df['timestamp'] = pd.to_datetime(f'{y}-{m}-{d}')
    # print(file,df_meta.iloc[10,0],df_meta.iloc[10,1], (f'{y}-{m}-{d}'))

    df['date'] = [int(v) for v in df.timestamp.dt.date.astype(str).str.replace('-','')]

    df['latitude'] = float(df_meta.iloc[4,1])
    df['longitude'] = float(df_meta.iloc[5,1])
    df['elevation'] = float(df_meta.iloc[6,1])

    df_list.append(df[col_needed])

    if plot:
        plt.plot(df["midpoint"], df["density"],marker='o', label=file)

if plot:
    plt.legend()
    plt.show()
    plt.xlabel('Depth (m)')
    plt.ylabel('Density (kg m-3)')

df_add = pd.concat(df_list,ignore_index=True)
# df_add=df_add.loc[df_add.notnull()]

for v in col_needed:
    assert v in df_add.columns, f'{v} is missing'
df_add[col_needed].to_csv('data_formatted.csv', index=None)
#     #%%

# if __name__ == "__main__":
#     process()
