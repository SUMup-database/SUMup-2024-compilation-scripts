# -*- coding: utf-8 -*-
"""
Created on %(date)s
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Columns needed in the final output
col_needed = ['method_key', 'start_date', 'end_date', 'start_year', 'end_year',
              'latitude', 'longitude', 'elevation', 'notes', 'smb', 'error',
              'name', 'reference', 'method', 'reference_short']

# def process():
# %%
# Set to True if you want to plot the data
plot = False

list_add = []
df_ref=pd.read_csv('../AntSMB/references.csv')

# Load the Excel file
file_path = "Ant2k_RegionalComposites_withSourcesAndData_Thomas_2017_Oct.xlsx"

# Load the sheets into DataFrames
df_ice_core = pd.read_excel(file_path, sheet_name="Original data",skiprows=6)
df_metadata_annual = pd.read_excel(file_path, sheet_name="Data Sources",skiprows=4)
df_metadata_annual=df_metadata_annual.loc[df_metadata_annual.Latitude.notnull(),:].reset_index(drop=True)

df_metadata = pd.read_excel('cp-13-1609-2017-supplement.xlsx')
df_metadata.columns = ['Site id', 'name', 'latitude','longitude','elevation',
                       'Min year\n(CE)', 'Max year\n(CE)',
       'Resolution (years)', 'Included', 'reference', 'URL']

print('name in sheet | name in metadata',)

for i, name1 in enumerate(df_ice_core.columns[1:]):
    name2 = df_metadata_annual['Site Name'][i]
    print(name1,'|',name2)
    df = pd.DataFrame({
        'end_year': df_ice_core['Year'],
        'smb': df_ice_core[name1].astype(float) / 1000,  # Convert SMB from kg m-3 a-1 to appropriate m w.e. a-1
        'start_year': df_ice_core['Year']
    }).dropna(subset=['smb'])  # Drop rows without SMB values

    # Extract metadata for this dataset ID
    metadata = df_metadata_annual[df_metadata_annual['Site Name'] == name2]
    df['name'] = name1
    df['latitude'] = metadata['Latitude'].astype(float).values[0]
    df['longitude'] = metadata['Longitude'].astype(str).str.replace('−','-').astype(float).values[0]
    df['elevation'] = metadata['Elevation (m)'].astype(float).values[0]
    ref=metadata['Publication Reference'].values[0]
    if (ref.strip() == df_ref.short_ref.str.strip()).any():
        if isinstance(df_ref.loc[df_ref.short_ref==ref, 'reference'].values[0], str):
            df['reference_short'] = df_ref.loc[df_ref.short_ref==ref,  'reference_short'].values[0]
            array=df_ref.loc[df_ref.short_ref==ref, ['reference', 'Unnamed: 3', 'Unnamed: 4']].values[0]
            df['reference'] = '. '.join(np.where(pd.isna(array), '', array))

        else:
            df['reference_short'] = metadata['Publication Reference'].values[0]
            df['reference'] =  metadata['Publication Reference'].values[0]
    else:
        df['reference_short'] = metadata['Publication Reference'].values[0]
        df['reference'] =  metadata['Publication Reference'].values[0]

    # doi = (metadata['Data Source or URL'].iloc[0].strip().replace('doi:','')
    #        .replace('https://doi.org/','').replace('https://doi.org/','')
    #        .replace('. 2000','').replace('https://doi.pangae.de/',''))
    # if doi == 'Antarctica 2k':
    #     doi = '10.5194/cp-13-1491-2017'
    # if doi == '10.1594/PANGAEA.407657':
    #     (df_metadata.name==name1).any()
    # if not doi.startswith('10'):
    #     print(doi, 'not DOI')
    # else:
    #     bibtex = get_bibtex_from_doi(doi)

    #     if bibtex:
    #         short_reference = parse_short_reference(bibtex)
    #         long_reference = parse_long_reference(bibtex)

    #         print("Short reference:", short_reference)
    #         print("Long reference:", long_reference)
    #     else:
    #         print(f"Error: Unable to retrieve BibTeX for DOI {doi}")
    #     # %%
    df['error'] = np.nan
    df[['start_date', 'end_date']] = np.nan
    df['notes'] = ''
    df['method'] = 'ice core'
    df['method_key'] = 3

    # Append df to list (list_add in the original script)
    list_add.append(df)


# Concatenating all dataframes from both multi-year and annual SMB compilations
df_add = pd.concat(list_add)

df_add = df_add.loc[df_add.smb.notnull() & df_add.latitude.notnull(), :]

# Ensuring all needed columns are present
for v in col_needed:
    assert v in df_add.columns, f'{v} is missing'

df_add['reference'] = df_add['reference'].str.strip()
df_add['reference_short'] = df_add['reference_short'].str.strip()


df_ref = pd.read_csv('../AntSMB/references.csv')  # Reference CSV with 'reference_short' and 'reference' columns
df_ref['reference'] = df_ref[['reference', 'Unnamed: 3', 'Unnamed: 4']].apply(lambda x: ' and '.join(x.dropna().astype(str)), axis=1)
df_ref['short_ref'] = df_ref['short_ref'].str.strip()
df_ref['reference_short'] = df_ref['reference_short'].str.strip()

# Loop through the DataFrame and update references
for _, (ref_short, ref) in df_add[['reference_short','reference']].drop_duplicates().iterrows():
    if ref_short == ref:
        idx = df_add.reference==ref


    # Check if the short reference exists in the reference DataFrame
    match = df_ref[df_ref['short_ref'] == ref]
    if not match.empty:
        df_add.loc[idx, 'reference'] = match['reference'].values[0]
        df_add.loc[idx, 'reference_short'] = match['reference_short'].values[0]
df_add.loc[df_add.reference.str.strip()=='', 'reference'] = df_add.loc[df_add.reference.str.strip()=='', 'reference_short']


this_ref = "Thomas et al. 2017"
df_add.loc[df_add.reference_short != this_ref, 'reference'] =  df_add.loc[df_add.reference_short != this_ref, 'reference'].values + ". As in: Thomas, E. R., van Wessem, J. M., Roberts, J., Isaksson, E., Schlosser, E., Fudge, T. J., Vallelonga, P., Medley, B., Lenaerts, J., Bertler, N., van den Broeke, M. R., Dixon, D. A., Frezzotti, M., Stenni, B., Curran, M., and Ekaykin, A. A.: Regional Antarctic snow accumulation over the past 1000 years, Clim. Past, 13, 1491–1513, https://doi.org/10.5194/cp-13-1491-2017, 2017."
df_add.loc[df_add.reference_short != this_ref, 'reference_short'] = \
    df_add.loc[df_add.reference_short != this_ref,
               'reference_short'].values + " as in Thomas et al. (2017)"


skip_list = [#'FB97DML05', 'BerknerR25 45.716W', 'Derwael Ice Rise', # other cores nearby but looks very different
             'FB9804','FB9805',  'FB9807', 'FB9808', 'FB9809',
 'FB9810', 'FB9811', 'FB9812', 'FB9813','FB9814','FB9816', 'FB9817', 'ITASE-02-7',
 'ITASE-02-4',  'THW2010', 'PIG2010','CWA-A', 'CWA-D', 'SDM-94',
 'UP-C', 'RIDS-A','RIDS-B', 'RIDS-C','ITASE-99-1','DIV2010',
 'B31 3.43W ',  'B33 6.4983E', 'B32 0.00667E', 'ITASE-00-1','ITASE-00-4','ITASE-00-5',
 'ITASE-01-1','ITASE-01-2','ITASE-01-3','ITASE-01-4', 'ITASE-01-5',
 'BerknerR25 45.716W', 'WD05A 112.125W', 'WD05Q 112.086W']

skip_list = [v.strip() for v in skip_list]
df_add['name'] = df_add.name.str.strip()

# -> corrected according to https://doi.org/10.3189/172756404781813961
df_add.loc[df_add.name == 'GV2 145.2631 E',
             ['latitude','longitude','elevation']] = [ -71.71,  145.2631, 2143.  ]
df_add.loc[df_add.name == 'D66 136.9352E',
             ['latitude','longitude','elevation']] = [ -68.94,  136.9352, 2333.  ]
df_add.loc[df_add.name == 'WD05A 112.125W',
             ['latitude','longitude','elevation']] = [-79.46,    -112.13, 1759,]
df_add.loc[df_add.name == 'RICE',
             ['latitude','longitude']] = [ -79.36, -161.64]
df_add.loc[df_add.name == 'Beethoven',
             ['latitude','longitude']] = [ -71.9, -74.6]
df_add.loc[df_add.name == 'FB96DML02',
             ['latitude','longitude']] = [  -74.968333,  3.9185]


# %%
if 'df_sumup' in locals():
    def find_nearby_entries(row, df_sumup):
        lat_range = (df_sumup.latitude >= row.latitude - 0.02) \
            & (df_sumup.latitude <= row.latitude + 0.02)
        lon_range = (df_sumup.longitude >= row.longitude - 0.02) \
            & (df_sumup.longitude <= row.longitude + 0.02)
        return df_sumup[lat_range & lon_range][
            ['name','latitude','longitude','reference_short','reference','method']
            ].drop_duplicates()

    # Iterate through df_add and find corresponding entries in df_sumup
    df_meta = df_add.loc[df_add.name.isin(skip_list),
                         ['name','latitude','longitude','reference_short',
                          'reference',]].drop_duplicates().reset_index(drop=True)

    plt.close('all')
# df_add.loc[df_add.name.str.contains('Derwael'),'name'].drop_duplicates()
# names = df_sumup.loc[df_sumup.name.astype(str).str.contains('WDC05A'),'name'].drop_duplicates()
# for n in names:
#     print(df_sumup.loc[df_sumup.name==n, ['latitude','longitude','elevation']].drop_duplicates())

# df_add.loc[df_add.name=='BerknerR25 45.716W',['latitude','longitude','method']].drop_duplicates()

    print('\tin Ant2k (name, lat, lon)\t\t\tpreviously in SUMup (name, lat, lon)')
    # [v for v in skip_list if v not in df_meta.name.values]
    for _, row in df_meta.iterrows():
        nearby_entries = find_nearby_entries(row, df_sumup)

        if nearby_entries.empty:
            print('=== missing ===')
            print(row['name'])
            print('======')
        else:
            # Plot profiles from df_add and df_sumup
            mask1 = (df_add.name == row['name']) & \
                (df_add.latitude == row.latitude) & \
                    (df_add.longitude == row.longitude)

            plt.figure()
            plt.plot(df_add.loc[mask1,'start_year'], df_add.loc[mask1,'smb'],
                     marker='d',ls ='None',
                     label=row['name']+' '+row['reference_short']+' in Ant2k')

            for _, near_row in nearby_entries.iterrows():
                if 'radar' in near_row.method:
                    continue
                mask2 = (df_sumup.name.astype(str) == str(near_row['name'])) & \
                    (df_sumup.latitude == near_row.latitude) & \
                        (df_sumup.longitude == near_row.longitude)
                sumup_match = df_sumup.loc[mask2,:]
                plt.plot(sumup_match['start_year'], sumup_match['smb'],
                         marker='.',ls ='None',
                         label=str(sumup_match['name'].iloc[0])+' '+sumup_match['reference_short'].iloc[0]+' in SUMup')
                print(row['name'], row['latitude'], row['longitude'],'\t\t',sumup_match['name'].iloc[0],
                      sumup_match['latitude'].iloc[0],
                      sumup_match['longitude'].iloc[0])
            plt.legend()
            plt.xlabel("Depth")
            plt.ylabel("Density")
            plt.show()

    # In Ant2k ITASE-01-2 has the coordinates of ITASE-02-1
        # ITASE-01-2 -82.001 -110.0082
        # -> should be -77.8436°, -102.9103°
    # DML16C98_13 and FB97DML10 have interpolated values
    # check -73.9743 126.5808

    # %%
df_add = df_add.loc[~df_add.name.isin( skip_list )]


df_add[col_needed].to_csv('data_formatted.csv', index=None)
# # %%
# if __name__ == "__main__":
#     process()
