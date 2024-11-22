import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

col_needed = ['profile', 'reference_short', 'reference', 'method_key', 'method',
              'date', 'timestamp', 'latitude', 'longitude', 'elevation',
              'start_depth', 'stop_depth', 'midpoint', 'density', 'error']

# Define constants
reference = ('Verfaillie, D., Fily, M., Le Meur, E., Magand, O., Jourdain, B., Arnaud, L., and Favier, V.: '
             'Snow accumulation variability derived from radar and firn core data along a 600 km transect in '
             'Adelie Land, East Antarctic plateau, The Cryosphere, 6, 1345–1358, https://doi.org/10.5194/tc-6-1345-2012, 2012.')
reference_short = 'Verfaillie et al. (2012)'

def process():
    # %%
    df_list=[]
    for file in ['DomeC-Verfaillie.xls', 'D57-Verfaillie.xls']:
        if file == 'D57-Verfaillie.xls':
            df_all = pd.read_excel(file, sheet_name=0,header=None)
            header = df_all.iloc[0, :].copy()
            header[:4] = np.array(['D57', pd.NA, pd.NA,'tous'])

        else:
            df_all = pd.read_excel(file, sheet_name=0, skiprows=3)
            header = df_all.iloc[0, :]
        df_all.columns=df_all.iloc[1,:].str.lower()
        df_all = df_all.iloc[4:,:]

        # Group the columns based on non-NaN values
        grouped_column_numbers = []
        current_group = []

        for i, col in enumerate(header):
            if pd.notna(col):
                if current_group:
                    grouped_column_numbers.append(current_group)
                current_group = [i]
            else:
                current_group.append(i)

        if current_group:
            grouped_column_numbers.append(current_group)

        plt.figure()
        # Iterating through grouped columns
        for column_numbers in grouped_column_numbers:
            name_group = header.iloc[column_numbers[0]]
            if name_group[:5] == 'tous':
                break
            df = df_all.iloc[:, column_numbers]
            df=df.loc[df.notnull().any(axis=1),:]
            df=df.loc[:,df.notnull().any(axis=0)]
            print( header.iloc[column_numbers[0]])


            df = df.rename(columns={'profondeur haute (cm)':'start_depth_cm',
              'profondeur basse (cm)':'stop_depth_cm',
              'densité brute':'density_gcm-1',
              'profondeur moyenne (m)':'midpoint',
              'profondeur (m)':'midpoint',
              'mean depth':'midpoint',
              'densité corrigée' :'density_gcm-1',
              'density' :'density_gcm-1',
            'profondeur(m)':'midpoint',
            'densité (g/cm3)':'density_gcm-1',
            'profondeur\n(cm)': 'midpoint_cm',
            'densité\n(g/cm3)':'density_gcm-1',
            'depth':'start_depth',
            'density max':'density_gcm-1',
             'density min':'density_gcm-1_min',
              'mean depth':'midpoint',
              'density max':'density_gcm-1',
              'profondeur basse':'stop_depth',
              'profondeur haute':'start_depth',
              'profondeur moy':'midpoint',
              'densité':'density_gcm-1',
             'profondeur':'midpoint',
             'profondeur moyenne ':'midpoint',})
            df = df.loc[:, ~df.columns.duplicated()]

            if name_group=='Firetracc 1999':
                df['stop_depth'] = df['start_depth']+df['length']/100
                df = df.loc[df['density_gcm-1'].notnull(), :]


            for v in ['midpoint','start_depth','stop_depth']:
                if v not in df.columns:
                    df[v] = pd.NA
                if v+'_cm' not in df.columns:
                    df[v+'_cm'] = pd.NA
                if df[v].isnull().all():
                    df[v] = df[v+'_cm']/100


            if  df['midpoint'].notnull().any() & \
                (df['stop_depth'].isnull().all() & df['start_depth'].isnull().all()):
                # print('caluclating start/end depth')
                df['stop_depth'] = df['stop_depth'].fillna(
                    (df['midpoint'] + df['midpoint'].shift(-1)) / 2
                    )
                df['start_depth'] = df['start_depth'].fillna(
                    df['stop_depth'].shift(1).fillna(df['midpoint'] - df['stop_depth'].diff() / 2)
                    )
            else:
                # print('caluclating midpoint depth')
                df['midpoint'] = (df['start_depth'] + df['stop_depth'] )/2
            if 'density_gcm-1' in df.columns:
                df['density'] = df['density_gcm-1']*1000
            df['profile'] = name_group
            df['reference_short'] = reference_short
            df['reference'] = reference
            df['method_key'] = 4
            df['method'] = "ice or firn core section"
            df['error'] = np.nan
            df['latitude'] = -75.100000
            df['longitude'] = 123.35
            df['elevation'] = 3233
            if name_group == 'D57':
                df['latitude'] = -68.183333
                df['longitude'] = 137.55
                df['elevation'] = np.nan
                df['timestamp'] = pd.to_datetime('1981-01-15')
                df['reference_short'] = 'Raynaud and Barnola (1985)'
                df['reference'] = 'Raynaud, D., Barnola, J. An Antarctic ice core reveals atmospheric CO2 variations over the past few centuries. Nature 315, 309–311 (1985). https://doi.org/10.1038/315309a0'
            else:
                if '_' in name_group:
                    df['timestamp'] = pd.to_datetime(name_group.split('_')[1].split('_')[0]+'-01-15')
                else:
                    df['timestamp'] = pd.to_datetime('2009-01-15')

            df['date'] = [int(v) for v in df.timestamp.dt.date.astype(str).str.replace('-','')]
            plt.plot(df['density'], -df['midpoint'],marker='.',ls='None', label=name_group)

            df_list.append(df)
        plt.legend()
    df_add = pd.concat(df_list,ignore_index=True)
    # df_add=df_add.loc[df_add.notnull()]

    for v in col_needed:
        assert v in df_add.columns, f'{v} is missing'
    df_add[col_needed].to_csv('data_formatted.csv', index=None)
#%%
if __name__ == "__main__":
    process()
