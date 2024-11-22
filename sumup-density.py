# -*- coding: utf-8 -*-
"""
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""
import pandas as pd
import numpy as np
import requests
import os
import matplotlib.pyplot as plt
from zipfile import ZipFile
from lib.utilities import resolve_name_keys, resolve_reference_keys

plt.close('all')
# loading and preparing the data
try:
    df_sumup = pd.concat((pd.read_csv('data/SUMup 2023/density/SUMup_2023_density_greenland.csv'),
                          pd.read_csv('data/SUMup 2023/density/SUMup_2023_density_antarctica.csv')))
except:
    print('Downloading SUMup 2023 density file')
    url = 'https://arcticdata.io/metacat/d1/mn/v2/object/urn%3Auuid%3A2fb5502d-f98f-43f4-a265-085d800232de'
    r = requests.get(url, allow_redirects=True)
    open('data/SUMup 2023/SUMup_2023_density_csv.zip', 'wb').write(r.content)
    with ZipFile('data/SUMup 2023/SUMup_2023_density_csv.zip', 'r') as zObject:
        zObject.extractall( path='data/SUMup 2023/density/')
    df_sumup = pd.concat((pd.read_csv('data/SUMup 2023/density/SUMup_2023_density_greenland.csv'),
                          pd.read_csv('data/SUMup 2023/density/SUMup_2023_density_antarctica.csv')))

df_sumup['timestamp'] = pd.to_datetime(df_sumup.timestamp)
df_methods = pd.read_csv('data/SUMup 2023/density/SUMup_2023_density_methods.tsv',
                         sep='\t').set_index('key')
df_profiles = pd.read_csv('data/SUMup 2023/density/SUMup_2023_density_profile_names.tsv',
                         sep='\t').set_index('key')
df_references = pd.read_csv('data/SUMup 2023/density/SUMup_2023_density_references.tsv',
                         sep='\t').set_index('key')

# % creating a metadata frame
# that contains the important information of all unique locations
df_sumup['profile'] = df_profiles.loc[df_sumup.profile_key].profile.values
df_sumup['method'] = df_methods.loc[df_sumup.method_key].method.values
df_sumup['reference'] = df_references.loc[df_sumup.reference_key].reference.values
df_sumup['reference_short'] = df_references.loc[df_sumup.reference_key].reference_short.values
df_sumup[df_sumup==-9999] = np.nan

if df_sumup.latitude.isnull().sum() | (df_sumup.latitude==-9999).sum(): print(wtf)

len_sumup_2023 = df_sumup.shape[0]
print('{:,.0f}'.format(df_sumup.shape[0]).replace(',',' '),
      'density observations in SUMup 2023')
print('from', len(df_sumup.reference_short.unique()), 'sources')
print('representing', len(df_sumup.reference.unique()), 'references')
print('{:,.0f}'.format(df_sumup.loc[df_sumup.latitude>0].shape[0]).replace(',',' '), 'in Greenland')
print('{:,.0f}'.format(df_sumup.loc[df_sumup.latitude<0].shape[0]).replace(',',' '), 'in Antarctica')

# Variables needed
necessary_variables = ['profile_key','profile', 'reference_key', 'reference_short',
                       'reference', 'method_key','method', 'date', 'timestamp', 'latitude',
                       'longitude', 'elevation', 'start_depth', 'stop_depth',
                       'midpoint', 'density', 'error']

# removing Clerx et al. 2022 data:
df_sumup = df_sumup.loc[df_sumup.reference_short != 'Clerx et al. (2022)', :]

# removing GEUS snowpit and firn core database
df_sumup = df_sumup.loc[df_sumup.reference_key != 202, :]

# removing outlier in df_sumup.loc[df_sumup.reference_short=='Kameda et al. (1995)',:]
# df_sumup = df_sumup.loc[df_sumup.density>0,:]

# sorting out Fourteau cores
for key in range(244,247):
    # plt.figure()
    # df_sumup.loc[df_sumup.reference_key==key , :].plot(y='midpoint', ax = plt.gca())

    tmp = df_sumup.loc[df_sumup.reference_key==key , :]
    index_saved = tmp.index
    tmp = tmp.sort_values(by='midpoint')
    tmp.index = index_saved
    tmp['start_depth'] = tmp.midpoint - 0.05/2
    tmp['stop_depth'] = tmp.midpoint + 0.05/2
    df_sumup.loc[df_sumup.reference_key==key , :] = tmp.values
    # df_sumup.loc[df_sumup.reference_key==key , :].plot(y='midpoint', ax = plt.gca())

# %% (re)loading data from Clerx et al. 2022 data:
print('\nClerx et al. 2022')
df = pd.read_csv("data/density data/Clerx data/data_formatted.csv")
df['profile_key'] = resolve_name_keys(df, df_sumup, v="profile", df_existing=df_profiles)
df['reference_key'] = 238 # resolve_reference_keys(df, df_sumup)
# check_duplicates(df, df_sumup, verbose = True, plot=True, tol=0.6)
df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)

# %% GEUS snow pit and firn core dataset
print('loading GEUS snow pit and firn core dataset')
df = pd.read_csv("data/density data/GEUS snowpit and firn core dataset/data_formatted.csv")
df['profile_key'] = resolve_name_keys(df, df_sumup, v="profile", df_existing=df_profiles)
df['reference_key'] = 202 # resolve_reference_keys(df, df_sumup)
# check_duplicates(df, df_sumup, verbose = True, plot=True, tol=0.6)
df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)

# %% Siple Dome 1996-1997 data
print('loading Lamorey (2003)')
df = pd.read_csv("data/density data/Lamorey2003/data_formatted.csv")
df['profile_key'] = resolve_name_keys(df, df_sumup, v="profile", df_existing=df_profiles)
df['reference_key'] = resolve_reference_keys(df, df_sumup)
# check_duplicates(df, df_sumup, verbose = True, plot=True, tol=0.6)
df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)

# %% RICE data
print('loading Winstrup et al. (2019) RICE ice core')
df = pd.read_csv("data/density data/Winstrup2019/data_formatted.csv")
df['profile_key'] = resolve_name_keys(df, df_sumup, v="profile", df_existing=df_profiles)
df['reference_key'] = resolve_reference_keys(df, df_sumup)
# check_duplicates(df, df_sumup, verbose = True, plot=True, tol=0.6)
df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)

# %% Harper 2023 EGIG data
print('loading Harper 2023 EGIG')
df = pd.read_csv("data/density data/Harper 2023 EGIG/data_formatted.csv")
df['profile_key'] = resolve_name_keys(df, df_sumup, v="profile", df_existing=df_profiles)
df['reference_key'] = resolve_reference_keys(df, df_sumup)
# check_duplicates(df, df_sumup, verbose = True, plot=True, tol=0.6)
df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)

# %% EastGRIP data
print('loading EastGRIP ice core')
df = pd.read_csv("data/density data/Rasmussen EastGRIP/data_formatted.csv")
df['profile_key'] = resolve_name_keys(df, df_sumup, v="profile", df_existing=df_profiles)
df['reference_key'] = resolve_reference_keys(df, df_sumup)
# check_duplicates(df, df_sumup, verbose = True, plot=True, tol=0.6)
df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)

# %% Picard 2022 data
print('loading Picard 2022')
df = pd.read_csv("data/density data/Picard2022/data_formatted.csv")
df['profile_key'] = resolve_name_keys(df, df_sumup, v="profile", df_existing=df_profiles)
df['reference_key'] = resolve_reference_keys(df, df_sumup)
# check_duplicates(df, df_sumup, verbose = True, plot=True, tol=0.6)
df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)

# %% Winski 2019 data
print('loading Winski 2019')
df = pd.read_csv("data/density data/Winski2019/data_formatted.csv")
df['profile_key'] = resolve_name_keys(df, df_sumup, v="profile", df_existing=df_profiles)
df['reference_key'] = resolve_reference_keys(df, df_sumup)
# check_duplicates(df, df_sumup, verbose = True, plot=True, tol=0.6)
df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)

# %% SAMBA data
print('loading SAMBA 2024')
df = pd.read_csv("data/density data/SAMBA_density_Favier_2024/data_formatted.csv")
df['profile_key'] = resolve_name_keys(df, df_sumup, v="profile", df_existing=df_profiles)
df['reference_key'] = resolve_reference_keys(df, df_sumup)
# check_duplicates(df, df_sumup, verbose = True, plot=True, tol=0.6)
df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)

# %% Brazilian data
print('loading Brazilian data')
df = pd.read_csv("data/density data/Brazilian firn and ice cores - UFRGS/data_formatted.csv")
df['profile_key'] = resolve_name_keys(df, df_sumup, v="profile", df_existing=df_profiles)
df['reference_key'] = resolve_reference_keys(df, df_sumup)
# check_duplicates(df, df_sumup, verbose = True, plot=True, tol=0.6)
df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)

# %% Verfaillie2012 data
print('loading Verfaillie 2012 data')
df = pd.read_csv("data/density data/Verfaillie2012/data_formatted.csv")
df['profile_key'] = resolve_name_keys(df, df_sumup, v="profile", df_existing=df_profiles)
df['reference_key'] = resolve_reference_keys(df, df_sumup)
# check_duplicates(df, df_sumup, verbose = True, plot=True, tol=0.6)
df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)

# %% Arnaud1998 data
print('loading Arnaud1998 data')
df = pd.read_csv("data/density data/Arnaud1998/data_formatted.csv")
df['profile_key'] = resolve_name_keys(df, df_sumup, v="profile", df_existing=df_profiles)
df['reference_key'] = resolve_reference_keys(df, df_sumup)
# check_duplicates(df, df_sumup, verbose = True, plot=True, tol=0.6)
df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)

# %% Quality tests (units and missing fields)
# tests to know if some units have been missinterpreted
df_ref_short = df_sumup[['reference_key','reference_short']].drop_duplicates().set_index('reference_key')

wrong_density_unit = (df_sumup.groupby('reference_key').density.mean()<1)
assert wrong_density_unit.sum()==0, print('Ref. that have wrong density unit:', df_ref_short.loc[wrong_density_unit.index[wrong_density_unit]])

wrong_density_unit = (df_sumup.groupby('reference_key').density.mean()>2000)
assert wrong_density_unit.sum()==0, print('Ref. that have wrong density unit:', df_ref_short.loc[wrong_density_unit.index[wrong_density_unit]])

wrong_density = (df_sumup.density>1000)
if wrong_density.sum()==0:
    print('Warning: Densities above 1000 kg m-3')
    print(
        df_sumup.loc[wrong_density,
                     ['midpoint','density','profile','reference_short']
                     ].to_markdown())

# wrong_density = (df_sumup.density<0)
# assert wrong_density.sum()==0, print('Densities < 0',
#                      df_sumup.loc[wrong_density,
#                                   ['midpoint','density','profile','reference_short']].to_markdown())

# calculating midpoint for the reference that don't have it
print(df_sumup.midpoint.isnull().sum(), 'measurements missing midpoint')
df_sumup.loc[df_sumup.midpoint.isnull(), 'midpoint'] = \
    df_sumup.loc[df_sumup.midpoint.isnull(), 'start_depth'] \
        + (df_sumup.loc[df_sumup.midpoint.isnull(), 'start_depth'] \
           -df_sumup.loc[df_sumup.midpoint.isnull(), 'stop_depth'])/2

# looking for depth added as cm instead of m
df_p = df_sumup[['profile_key','profile']].drop_duplicates().set_index('profile_key')
profile_keys = df_sumup.loc[(df_sumup.midpoint > 100) & (df_sumup.density < 700),'profile_key'].drop_duplicates().values
plt.close('all')
for p in profile_keys:
    print(p)
    plt.figure()
    df_sumup.loc[df_sumup.profile_key==p,:].plot(ax=plt.gca(),
                                             x='density',
                                             y='midpoint',
                                             marker='o',
                                             label=df_p.loc[p].values[0])

# missing latitude
# assert not (df_sumup.latitude.isnull().sum() | (df_sumup.latitude==-9999).sum()), "some profile_key missing latitude"
# List profiles with missing latitude
missing_latitude_profiles = df_sumup[df_sumup.latitude.isnull() | (df_sumup.latitude == -9999)]['profile_key'].unique()

# Print unique profiles with missing latitude
if len(missing_latitude_profiles)>0:
    print("Profiles missing latitude:",
          df_sumup[df_sumup.latitude.isnull() | (df_sumup.latitude == -9999)][
              ['profile', 'reference_short']].drop_duplicates())

# Remove entries with missing latitude
df_sumup_cleaned = df_sumup[~df_sumup['profile_key'].isin(missing_latitude_profiles)]


# positive longitudes in greenland
if len(df_sumup.loc[(df_sumup.latitude>0)&(df_sumup.longitude>0), 'profile'])>0:
    print(df_sumup.loc[(df_sumup.latitude>0)&(df_sumup.longitude>0), 'longitude'].drop_duplicates())
assert len(df_sumup.loc[(df_sumup.latitude>0)&(df_sumup.longitude>0), 'profile'])==0, "some Greenland measurement has positive longitudes"
# df_sumup.loc[(df_sumup.latitude>0)&(df_sumup.longitude>0), 'longitude'] = -df_sumup.loc[(df_sumup.latitude>0)&(df_sumup.longitude>0), 'longitude']

# missing profile_key keys
assert not df_sumup.profile_key.isnull().any(), "some profile_key have NaN as profile_key key"
if df_sumup.profile_key.isnull().any():
    print(df_sumup.loc[df_sumup.profile_key.isnull(), ['profile_key', 'profile','reference_short']].drop_duplicates())

# missing time stamp
if df_sumup['timestamp'].isnull().any():
    print('Missing time stamp for:')
    print(df_sumup.loc[df_sumup['timestamp'].isnull()|(df_sumup['timestamp']=='Na'), ['profile','reference_short']].drop_duplicates())
    print('Ignoring entry')
    df_sumup = df_sumup.loc[
        df_sumup['timestamp'].notnull()&(df_sumup['timestamp']!='Na'),:]

# checking duplicate reference
tmp = df_sumup[['reference_key','reference']].drop_duplicates()
if tmp.reference_key.duplicated().any():
    print('\n====> Found two references for same reference key')
    dup_ref = tmp.loc[tmp.reference_key.duplicated()]
    for ref in dup_ref.reference_key.values:
        doubled_ref = df_sumup.loc[df_sumup.reference_key == ref,
                           ['reference_key','reference']].drop_duplicates()
        if doubled_ref.iloc[0,1].replace(' ','').lower() == doubled_ref.iloc[1,1].replace(' ','').lower():
            df_sumup.loc[df_sumup.reference_key == ref, 'reference'] = doubled_ref.iloc[0,1]
            print('Merging\n', doubled_ref.iloc[1,1],'\ninto\n', doubled_ref.iloc[0,1],'\n')
        else:
            print(wtf)


df_meta = df_sumup[['profile_key','method_key','method']].drop_duplicates()
for ind in df_meta.index[df_meta.index.duplicated()]:
    if len(df_meta.loc[ind,'method'])>1:
        print(wtf)
        print('\n> found profile_key with multiple methods')
        print(df_meta.loc[ind,['profile', 'method', 'reference_short']])
        print('renaming method for this profile_key')
        df_meta.loc[ind, 'method'] = ' or '.join(df_meta.loc[ind, 'method'].tolist())
print('=== Finished ===')

# %% writing to file
# CSV format
from lib.write import (round_and_format, write_reference_csv, write_method_csv,
                       write_profile_csv, write_data_to_csv,
                       write_density_to_netcdf)
df_sumup = round_and_format(df_sumup)

write_reference_csv(df_sumup,
        'SUMup 2024 beta/SUMup_2024_density_csv/SUMup_2024_density_references.tsv')
write_method_csv(df_sumup,
        'SUMup 2024 beta/SUMup_2024_density_csv/SUMup_2024_density_methods.tsv')
write_profile_csv(df_sumup,
        'SUMup 2024 beta/SUMup_2024_density_csv/SUMup_2024_density_profile_names.tsv')

write_data_to_csv(df_sumup,
            csv_folder='SUMup 2024 beta/SUMup_2024_density_csv',
            filename='SUMup_2024_density',
            write_variables= ['profile_key', 'reference_key', 'method_key',
                  'timestamp','latitude', 'longitude', 'elevation', 'start_depth',
                  'stop_depth', 'midpoint', 'density', 'error'])

# netcdf format
df_sumup[['elevation']] =  df_sumup[['elevation']].replace('','-9999').astype(int)
df_sumup[df_sumup==-9999] = np.nan

write_density_to_netcdf(df_sumup.loc[df_sumup.latitude>0, :],
                        'SUMup 2024 beta/SUMup_2024_density_greenland.nc')
write_density_to_netcdf(df_sumup.loc[df_sumup.latitude<0, :],
                        'SUMup 2024 beta/SUMup_2024_density_antarctica.nc')

#%% creating tables for ReadMe file
from lib.plot import plot_dataset_composition, plot_map
from lib.write import write_dataset_composition_table, write_location_file, create_kmz

df_sumup['timestamp'] = pd.to_datetime(df_sumup.timestamp,utc=True)

df_meta = pd.DataFrame()
df_meta['total'] = [df_sumup.shape[0]]
df_meta['added'] = df_sumup.shape[0]-len_sumup_2023
df_meta['nr_references'] = str(len(df_sumup.reference.unique()))
df_meta['greenland'] = df_sumup.loc[df_sumup.latitude>0].shape[0]
df_meta['antarctica'] = df_sumup.loc[df_sumup.latitude<0].shape[0]
df_meta.index.name='index'
df_meta.to_csv('doc/ReadMe_2024_src/tables/density_meta.csv')

print('{:,.0f}'.format(df_sumup.shape[0]).replace(',',' ') +\
      ' density observations in SUMup 2024')
print('{:,.0f}'.format(df_sumup.shape[0]-len_sumup_2023).replace(',',' ') +\
      ' more than in SUMup 2023')
print('from '+ str(len(df_sumup.reference_short.unique())) + ' sources')
print('representing '+ str(len(df_sumup.reference.unique()))+' references')

print('{:,.0f}'.format(df_sumup.loc[df_sumup.latitude>0].shape[0]).replace(',',' ')+' observations in Greenland')
print('{:,.0f}'.format(df_sumup.loc[df_sumup.latitude<0].shape[0]).replace(',',' ')+' observations in Antarctica')


plot_dataset_composition(df_sumup.loc[df_sumup.latitude>0],
        'doc/ReadMe_2024_src/figures/density_dataset_composition_greenland.png')
plot_map(df_sumup.loc[df_sumup.latitude>0,['latitude','longitude']].drop_duplicates(),
         'doc/ReadMe_2024_src/figures/density_map_greenland.png',
         area='greenland')

plot_dataset_composition(df_sumup.loc[df_sumup.latitude<0],
        'doc/ReadMe_2024_src/figures/density_dataset_composition_antarctica.png')

plot_map(df_sumup.loc[df_sumup.latitude<0,['latitude','longitude']].drop_duplicates(),
         'doc/ReadMe_2024_src/figures/density_map_antarctica.png',
         area='antarctica')

write_dataset_composition_table(df_sumup.loc[df_sumup.latitude>0],
                                'doc/ReadMe_2024_src/tables/composition_density_greenland.csv')

write_dataset_composition_table(df_sumup.loc[df_sumup.latitude<0],
                                'doc/ReadMe_2024_src/tables/composition_density_antarctica.csv')

# print('writing out measurement locations')
# write_location_file(df_sumup.loc[df_sumup.latitude>0,:],
#                     'doc/GIS/SUMup_2024_density_location_greenland.csv')

# write_location_file(df_sumup.loc[df_sumup.latitude<0, :],
#                     'doc/GIS/SUMup_2024_density_location_antarctica.csv')

# create_kmz(df_sumup, output_prefix="doc/GIS/SUMup_2024_density")
