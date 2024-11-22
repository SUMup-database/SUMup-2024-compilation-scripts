import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
col_needed = ['profile',  'reference_short',
                       'reference', 'method_key','method', 'date', 'timestamp', 'latitude',
                       'longitude', 'elevation', 'start_depth', 'stop_depth',
                       'midpoint', 'density', 'error']

def process():
    # %%
    df_coords = pd.read_csv('lonlatdis.dat',sep=';',header=None)
    df_coords.columns = ['name','longitude','latitude','distance','?']
    df_list = []
    plot = True
    for year in range(2004,2025):
        if plot:
            plt.figure()
        for name in df_coords.name:
            if not os.path.isfile(f'density/{year}/{name}.csv'):
                print(f'Could not find density/{year}/{name}.csv')
                continue


            df = pd.read_csv(f'density/{year}/{name}.csv', sep=';',header=None)
            if len(df.columns)==4:
                df.columns = ['stop_depth','density_a','density_b','?']
            elif len(df.columns)==3:
                df.columns = ['stop_depth','density_a','density_b']
            else:
                df.columns = ['stop_depth','density_a','density_b','?','?']

            df['stop_depth'] = df.stop_depth/100
            df['start_depth'] = df.stop_depth.shift(1).fillna(0)
            df['density'] = df[['density_a', 'density_b']].mean(axis=1)*1000
            df['error'] = df[['density_a', 'density_b']].std(axis=1)*1000
            df['midpoint'] = (df.stop_depth + df.start_depth)/2

            df['profile'] = name+'_'+str(year)
            df['reference_short'] = 'GlacioClim-SAMBA (2024)'
            df['reference'] = 'Favier, V.: GacioClim-SAMBA dataset (2024)'

            df['method_key'] = 6
            df['method'] = "Density cutter – size unknown"

            df['timestamp'] = pd.to_datetime(str(year)+'-01-15')
            df['date'] = [int(v) for v in df.timestamp.dt.date.astype(str).str.replace('-','')]

            df['latitude'] = df_coords.loc[df_coords.name==name,'latitude'].values[0]
            df['longitude'] = df_coords.loc[df_coords.name==name,'longitude'].values[0]
            df['elevation'] = np.nan

            df_list.append(df[col_needed])

            if plot:
                plt.plot(df["midpoint"], df["density"],marker='o', label=name+' '+str(year))

        if plot:
            # plt.legend()
            plt.show()
            plt.xlabel('Depth (m)')
            plt.ylabel('Density (kg m-3)')

    df_add = pd.concat(df_list,ignore_index=True)
    # df_add=df_add.loc[df_add.notnull()]

    for v in col_needed:
        assert v in df_add.columns, f'{v} is missing'
    df_add[col_needed].to_csv('data_formatted.csv', index=None)
    #%%

if __name__ == "__main__":
    process()
