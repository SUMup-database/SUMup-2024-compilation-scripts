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
from zipfile import ZipFile
from lib.utilities import resolve_name_keys, resolve_reference_keys
from sumup_lib import add_vandecrux_full_res, stack_multidepth_df, plot_string_dataframe

# loading data
try:
    df_sumup = pd.concat((pd.read_csv('data/SUMup 2023/temperature/SUMup_2023_temperature_greenland.csv'),
                          pd.read_csv('data/SUMup 2023/temperature/SUMup_2023_temperature_antarctica.csv')))
except:
    print('Downloading SUMup 2023 temperature file')
    url = 'https://arcticdata.io/metacat/d1/mn/v2/object/urn%3Auuid%3A179ccaf3-79e4-4cc7-a35a-6ef5058118f3'
    r = requests.get(url, allow_redirects=True)
    open('data/SUMup 2023/SUMup_2023_temperature_csv.zip', 'wb').write(r.content)
    with ZipFile('data/SUMup 2023/SUMup_2023_temperature_csv.zip', 'r') as zObject:
        zObject.extractall( path='data/SUMup 2023/temperature/')
    df_sumup = pd.concat((pd.read_csv('data/SUMup 2023/temperature/SUMup_2023_temperature_greenland.csv'),
                          pd.read_csv('data/SUMup 2023/temperature/SUMup_2023_temperature_antarctica.csv')))

df_sumup['timestamp'] = pd.to_datetime(df_sumup.timestamp)
df_methods = pd.read_csv('data/SUMup 2023/temperature/SUMup_2023_temperature_methods.tsv',
                         sep='\t').set_index('key')
df_names = pd.read_csv('data/SUMup 2023/temperature/SUMup_2023_temperature_names.tsv',
                         sep='\t').set_index('key')
df_references = pd.read_csv('data/SUMup 2023/temperature/SUMup_2023_temperature_references.tsv',
                         sep='\t').set_index('key')

# % creating a metadata frame
# that contains the important information of all unique locations
df_sumup['name'] = df_names.loc[df_sumup.name_key].name.values
df_sumup['method'] = df_methods.loc[df_sumup.method_key].method.values
df_sumup['reference'] = df_references.loc[df_sumup.reference_key].reference.values
df_sumup['reference_short'] = df_references.loc[df_sumup.reference_key].reference_short.values

df_sumup[df_sumup==-9999] = np.nan

# correcting an error in SUMup 2023:
# removing Picard et al. 2022 temperatures
# ind = df_sumup.loc[df_sumup.reference_short=='Picard et al. (2022)', 'reference_key'].unique().item()
df_sumup = df_sumup.loc[df_sumup.reference_key != 82, :]
len_sumup_2022 = df_sumup.shape[0]
print(df_sumup.shape[0], 'temperature observations currently in SUMup from', len(df_sumup.reference_key.unique()), 'sources')
print(df_sumup.loc[df_sumup.latitude>0].shape[0], 'in Greenland')
print(df_sumup.loc[df_sumup.latitude<0].shape[0], 'in Antarctica')

needed_cols = ['name_key', 'reference_key', 'method_key', 'timestamp',
               'latitude', 'longitude', 'elevation', 'depth',
                'temperature', 'error', 'name', 'method', 'reference',
                'reference_short', 'duration']

# %%  loading PROMICE / GC-Net
print('   PROMICE / GC-Net')
df = pd.read_csv('data/temperature data/PROMICE/data_formatted.csv')
df["method"] = "thermistor and thermocouple strings"
df["error"] = np.nan
# all_sites_df["name"] = all_sites_df["site"]
df["duration"] = 1
df['reference_short'] = "PROMICE/GC-Net: Fausto et al. (2021); How et al. (2022)"

df['reference'] = 'Fausto, R. S., van As, D., Mankoff, K. D., Vandecrux, B., Citterio, M., Ahlstrøm, A. P., Andersen, S. B., Colgan, W., Karlsson, N. B., Kjeldsen, K. K., Korsgaard, N. J., Larsen, S. H., Nielsen, S., Pedersen, A. Ø., Shields, C. L., Solgaard, A. M., and Box, J. E.: Programme for Monitoring of the Greenland Ice Sheet (PROMICE) automatic weather station data, Earth Syst. Sci. Data, 13, 3819–3845, https://doi.org/10.5194/essd-13-3819-2021 , 2021. and How, P., Ahlstrøm, A.P., Andersen, S.B., Box, J.E., Citterio, M., Colgan, W.T., Fausto, R., Karlsson, N.B., Jakobsen, J., Larsen, S.H., Mankoff, K.D., Pedersen, A.Ø., Rutishauser, A., Shields, C.L., Solgaard, A.M., van As, D., Vandecrux, B., Wright, P.J., PROMICE and GC-Net automated weather station data in Greenland, https://doi.org/10.22008/FK2/IW73UU, GEUS Dataverse, 2022.'
df['reference_key'] = 23
df['name_key'] = resolve_name_keys(df, df_sumup)
df['method_key'] = resolve_name_keys(df, df_sumup, v="method")
df_sumup = pd.concat((df_sumup, df[needed_cols]), ignore_index=True)

# %% loading historical GC-Net
print('    Historical GC-Net')
df = pd.read_csv('data/temperature data/GC-Net historical/data_formatted.csv', low_memory=False)


df.loc[df.reference_short == "Historical GC-Net: Steffen et al. (1996, 2001, 2023); Vandecrux et al. (2023)",
       'reference'] = "Steffen, K., Box, J.E. and Abdalati, W., 1996. Greenland climate network: GC-Net. US Army Cold Regions Reattach and Engineering (CRREL), CRREL Special Report, pp.98-103. and Steffen, K. and J. Box: Surface climatology of the Greenland ice sheet: Greenland Climate Network 1995-1999, J. Geophys. Res., 106, 33,951-33,972, 2001 and Steffen, K., Vandecrux, B., Houtz, D., Abdalati, W., Bayou, N., Box, J., Colgan, L., Espona Pernas, L., Griessinger, N., Haas-Artho, D., Heilig, A., Hubert, A., Iosifescu Enescu, I., Johnson-Amin, N., Karlsson, N. B., Kurup, R., McGrath, D., Cullen, N. J., Naderpour, R., Pederson, A. Ø., Perren, B., Philipps, T., Plattner, G.K., Proksch, M., Revheim, M. K., Særrelse, M., Schneebli, M., Sampson, K., Starkweather, S., Steffen, S., Stroeve, J., Watler, B., Winton, Ø. A., Zwally, J., Ahlstrøm, A.: GC-Net Level 1 automated weather station data, https://doi.org/10.22008/FK2/VVXGUT, GEUS Dataverse, V2, 2023. and Vandecrux, B., Box, J. E., Ahlstrøm, A. P., Andersen, S. B., Bayou, N., Colgan, W. T., Cullen, N. J., Fausto, R. S., Haas-Artho, D., Heilig, A., Houtz, D. A., How, P., Iosifescu Enescu, I., Karlsson, N. B., Kurup Buchholz, R., Mankoff, K. D., McGrath, D., Molotch, N. P., Perren, B., Revheim, M. K., Rutishauser, A., Sampson, K., Schneebeli, M., Starkweather, S., Steffen, S., Weber, J., Wright, P. J., Zwally, H. J., and Steffen, K.: The historical Greenland Climate Network (GC-Net) curated and augmented level-1 dataset, Earth Syst. Sci. Data, 15, 5467–5489, https://doi.org/10.5194/essd-15-5467-2023, 2023."
df['reference_key'] = 24
df['name_key'] = resolve_name_keys(df, df_sumup)
df['method_key'] = resolve_name_keys(df, df_sumup, v="method")

df_sumup = pd.concat((df_sumup, df[needed_cols]), ignore_index=True)

# %% loading FA data
print('    FA data')
df = pd.read_csv('data/temperature data/Miege firn aquifer/data_formatted.csv', low_memory=False)
df['reference_key'] = resolve_reference_keys(df, df_sumup)
df['name_key'] = resolve_name_keys(df, df_sumup)
df['method_key'] = resolve_name_keys(df, df_sumup, v="method")
df_sumup = pd.concat((df_sumup, df[needed_cols]), ignore_index=True)

# %% loading Humphrey temperature data
print('    Humphrey temperature')
df = pd.read_csv('data/temperature data/Humphrey string/data_formatted.csv', low_memory=False)
df['reference_key'] = resolve_reference_keys(df, df_sumup)
df['name_key'] = resolve_name_keys(df, df_sumup)
df['method_key'] = resolve_name_keys(df, df_sumup, v="method")
df_sumup = pd.concat((df_sumup, df[needed_cols]), ignore_index=True)

# %% Camp Century Climate
# del df_stack, df, sonic_df, df3, df_surf, depth, depth_label, temp_label, i, filepath, sites
print("   Camp Century data")
df = pd.read_csv('data/temperature data/Camp Century Climate/data_formatted.csv', low_memory=False)
df['reference_key'] = resolve_reference_keys(df, df_sumup)
df['name_key'] = resolve_name_keys(df, df_sumup)
df['method_key'] = resolve_name_keys(df, df_sumup, v="method")
df_sumup = pd.concat((df_sumup,  df[needed_cols] ), ignore_index=True )

# %% Hills 2018
print("   Hills 2018 data")
df = pd.read_csv('data/temperature data/Hills/data_formatted.csv', low_memory=False)
df['reference_key'] = resolve_reference_keys(df, df_sumup)
df['name_key'] = resolve_name_keys(df, df_sumup)
df['method_key'] = resolve_name_keys(df, df_sumup, v="method")
df_sumup = pd.concat((df_sumup,  df[needed_cols] ), ignore_index=True )

# %% Adding non-interpolated Vandecrux data (Daily means for continuous measurements)
print('## Vandecrux et al. compilation')
df_vdx = add_vandecrux_full_res(plot=False)
df_vdx = df_vdx.rename(columns={'date':'timestamp',
                                'temperatureObserved':'temperature',
                                'depthOfTemperatureObservation':'depth',
                                'site':'name',
                                'durationMeasured':'duration'})
df_vdx['reference_key'] = resolve_reference_keys(df_vdx, df_sumup)
df_vdx['name_key'] = resolve_name_keys(df_vdx, df_sumup)
df_vdx['method_key'] = resolve_name_keys(df_vdx, df_sumup, v="method")

df_sumup = pd.concat((df_sumup, df_vdx[needed_cols]), ignore_index=True)

# print('Checking conflicts:\n')
# sumup_index_conflict = check_conflicts(df_sumup, df_vdx, var=['name', 'depth','temperature'])
# print('\noverwriting conflicting data in SUMup (checked by bav)\n')
# msk = ~df_sumup.index.isin(sumup_index_conflict)

# %% Sigma-A
print('loading SIGMA-A')
df = pd.read_csv('data/temperature data/SIGMA-A/data_formatted.csv', low_memory=False)
df['reference_key'] = resolve_reference_keys(df, df_sumup)
df['name_key'] = resolve_name_keys(df, df_sumup)
df['method_key'] = resolve_name_keys(df, df_sumup, v="method")
df_sumup = pd.concat((df_sumup,  df[needed_cols] ), ignore_index=True )

# %%  loading Humphrey temperature data
print('    Harper and Humphrey 2023 temperature')
df = pd.read_csv('data/temperature data/Harper and Humphrey 2023/data_formatted.csv', low_memory=False)
df['reference_key'] = resolve_reference_keys(df, df_sumup)
df['name_key'] = resolve_name_keys(df, df_sumup)
df['method_key'] = resolve_name_keys(df, df_sumup, v="method")
df_sumup = pd.concat((df_sumup, df[needed_cols]), ignore_index=True)

print('    Harper and Humphrey 2024 temperature')
df = pd.read_csv('data/temperature data/Harper and Humphrey 2024/data_formatted.csv', low_memory=False)
df['reference_key'] = resolve_reference_keys(df, df_sumup)
df['name_key'] = resolve_name_keys(df, df_sumup)
df['method_key'] = resolve_name_keys(df, df_sumup, v="method")
df_sumup = pd.concat((df_sumup, df[needed_cols]), ignore_index=True)

# fixing some mistakes:
for var in ['thermistor','thermistors','thermistor string','custom thermistors']:
    df_sumup.loc[df_sumup.method==var, 'method'] = 'Thermistor'

df_sumup.loc[df_sumup.method.isnull(), 'method'] = 'NA'
df_sumup.loc[df_sumup.method=='not_reported', 'method'] = 'NA'
df_sumup.loc[df_sumup.method=='digital Thermarray system from RST©', 'method'] = 'RST ThermArray'

# looking for redundant references
df_sumup.loc[df_sumup.reference.str.startswith('Miller'),'reference_short'] = 'Miller et al. (2020)'
df_sumup.loc[df_sumup.reference.str.startswith('Miller'),'reference'] = 'Miller, O., Solomon, D.K., Miège, C., Koenig, L., Forster, R., Schmerr, N., Ligtenberg, S.R., Legchenko, A., Voss, C.I., Montgomery, L. and McConnell, J.R., 2020. Hydrology of a perennial firn aquifer in Southeast Greenland: an overview driven by field data. Water Resources Research, 56(8), p.e2019WR026348. Dataset doi:10.18739/A2R785P5W'

df_sumup.loc[df_sumup.reference.str.startswith('Clausen HB and Stauffer B (1988)'),
             'reference_short'] = 'Clausen and Stauffer (1988)'

df_sumup.loc[df_sumup.reference.str.startswith('Charalampidis'),
             'reference_short'] = 'Charalampidis et al. (2016, 2022)'

# %% Removing duplicate obs, reference and renaming method
# renaming some redundant references
from lib.check import (check_coordinates, check_missing_key,
                       check_duplicate_key, check_duplicate_reference,
                       check_missing_timestamp)

df_sumup = check_coordinates(df_sumup)
df_sumup = check_missing_timestamp(df_sumup)
df_sumup = check_missing_key(df_sumup, var='method')
df_sumup = check_missing_key(df_sumup, var='name')
df_sumup = check_missing_key(df_sumup, var='reference')

df_sumup = check_duplicate_key(df_sumup, var='method')
df_sumup = check_duplicate_reference(df_sumup)

# numeric check on variables:
for var_int in ['elevation','open_time', 'duration']:
    # if there was something that is not a number, then we shift it to 'notes'
    # df_sumup.loc[pd.to_numeric(df_sumup[var_int], errors='coerce').isnull(), 'notes'] = \
    #     df_sumup.loc[pd.to_numeric(df_sumup[var_int], errors='coerce').isnull(), var_int]
    df_sumup.loc[pd.to_numeric(df_sumup[var_int], errors='coerce').isnull(), var_int] = -9999
    df_sumup[var_int] = pd.to_numeric(df_sumup[var_int], errors='coerce').round(0).astype(int)

print(df_sumup.shape[0], 'temperature observations after merging from', len(df_sumup.reference.unique()), 'sources')
print(df_sumup.loc[df_sumup.latitude>0].shape[0], 'in Greenland')
print(df_sumup.loc[df_sumup.latitude<0].shape[0], 'in Antarctica')

# %% writin CSV files
from lib.write import (round_and_format, write_reference_csv, write_method_csv,
                       write_names_csv, write_data_to_csv,
                       write_temperature_to_netcdf)

df_sumup = round_and_format(df_sumup)

csv_folder='SUMup 2024 beta/SUMup_2024_temperature_csv'
write_reference_csv(df_sumup, f'{csv_folder}/SUMup_2024_temperature_references.tsv')
write_method_csv(df_sumup, f'{csv_folder}/SUMup_2024_temperature_methods.tsv')
write_names_csv(df_sumup, f'{csv_folder}/SUMup_2024_temperature_names.tsv')

var_to_csv = ['name_key', 'reference_key', 'method_key', 'timestamp', 'latitude',
        'longitude', 'elevation', 'depth', 'open_time', 'duration',
        'temperature', 'error']
write_data_to_csv(df_sumup.loc[df_sumup.latitude>0, var_to_csv],
                  csv_folder,
            filename='SUMup_2024_temperature',
            write_variables= var_to_csv)


import shutil
shutil.make_archive('SUMup 2024 beta/SUMup_2024_temperature_csv',
                    'zip', 'SUMup 2024 beta/SUMup_2024_temperature_csv')

#  netcdf format
df_sumup[['elevation','open_time', 'duration']] = \
    df_sumup[['elevation','open_time', 'duration']].replace('','-9999').astype(int)

write_temperature_to_netcdf(df_sumup.loc[df_sumup.latitude>0, :],
                            'SUMup 2024 beta/SUMup_2024_temperature_greenland.nc')
write_temperature_to_netcdf(df_sumup.loc[df_sumup.latitude<0, :],
                            'SUMup 2024 beta/SUMup_2024_temperature_antarctica.nc')
#%% producing files for ReadMe file
from lib.plot import plot_dataset_composition, plot_map
from lib.write import write_dataset_composition_table,write_location_file,create_kmz

df_meta = pd.DataFrame()
df_meta['total'] = [df_sumup.shape[0]]
df_meta['added'] = df_sumup.shape[0]-len_sumup_2022
df_meta['nr_references'] = str(len(df_sumup.reference.unique()))
df_meta['greenland'] = df_sumup.loc[df_sumup.latitude>0].shape[0]
df_meta['antarctica'] = df_sumup.loc[df_sumup.latitude<0].shape[0]
df_meta.index.name='index'

df_meta.to_csv('doc/ReadMe_2024_src/tables/temperature_meta.csv')
print('  ')
print('{:,.0f}'.format(df_sumup.shape[0]).replace(',',' ') +\
      ' temperature observations in SUMup 2024')
print('{:,.0f}'.format(df_sumup.shape[0]-len_sumup_2022).replace(',',' ') +\
      ' more than in SUMup 2022')
print('from '+ str(len(df_sumup.reference_short.unique())) + ' sources')
print('representing '+ str(len(df_sumup.reference.unique()))+' references')

print('{:,.0f}'.format(df_sumup.loc[df_sumup.latitude>0].shape[0]).replace(',',' ')+' observations in Greenland')
print('{:,.0f}'.format(df_sumup.loc[df_sumup.latitude<0].shape[0]).replace(',',' ')+' observations in Antarctica')


plot_dataset_composition(df_sumup.loc[df_sumup.latitude>0], 'doc/ReadMe_2024_src/figures/temperature_dataset_composition_greenland.png')

plot_map(df_sumup.loc[df_sumup.latitude>0,['latitude','longitude']].drop_duplicates(),
         'doc/ReadMe_2024_src/figures/temperature_map_greenland.png',
         area='greenland')

plot_dataset_composition(df_sumup.loc[df_sumup.latitude<0], 'doc/ReadMe_2024_src/figures/temperature_dataset_composition_antarctica.png')

plot_map(df_sumup.loc[df_sumup.latitude<0,['latitude','longitude']].drop_duplicates(),
         'doc/ReadMe_2024_src/figures/temperature_map_antarctica.png',
         area='antarctica')


write_dataset_composition_table(df_sumup.loc[df_sumup.latitude>0],
                'doc/ReadMe_2024_src/tables/composition_temperature_greenland.csv')

write_dataset_composition_table(df_sumup.loc[df_sumup.latitude<0],
                'doc/ReadMe_2024_src/tables/composition_temperature_antarctica.csv')

# print('writing out measurement locations')
# write_location_file(df_sumup.loc[df_sumup.latitude>0,:],
#                     'doc/GIS/SUMup_2024_temperature_location_greenland.csv')

# write_location_file(df_sumup.loc[df_sumup.latitude<0, :],
#                     'doc/GIS/SUMup_2024_temperature_location_antarctica.csv')

# create_kmz(df_sumup, output_prefix="doc/GIS/SUMup_2024_temperature")
