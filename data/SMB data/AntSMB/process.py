import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# Columns needed in the final output
col_needed = ['method_key', 'start_date', 'end_date', 'start_year', 'end_year',
              'latitude', 'longitude', 'elevation', 'notes', 'smb', 'error',
              'name', 'reference', 'method', 'reference_short']

# def process():

# Set to True if you want to plot the data
plot = False

# Process Multi-year mean SMB data
print('reading AntSMB_ Multi-year-mean SMB compilation.xlsx')
df_metadata_multi = pd.read_excel('AntSMB_ Multi-year-mean SMB compilation.xlsx',
                                  sheet_name='Multi-year mean SMB database')


df_multi = pd.DataFrame()
# Filling in the metadata for each dataset
df_multi['name'] = df_metadata_multi['Geo_siteName']
df_multi['latitude'] = df_metadata_multi['Geo_latitude']
df_multi['longitude'] = df_metadata_multi['Geo_longitude']
df_multi['elevation'] = df_metadata_multi['Geo_elevation']
df_multi['reference_short'] = df_metadata_multi['References']
df_multi['reference'] = 'Short ref'  # Customize as needed

df_multi['error'] = np.nan
df_multi['start_year'] = df_metadata_multi['MinYear']
df_multi['end_year'] = df_metadata_multi['MaxYear']
df_multi['smb'] = df_metadata_multi['SMB (kg m-2 yr-1)']/1000 * (df_multi['end_year'] - df_multi['start_year'])
df_multi[['start_date', 'end_date']] = np.nan
df_multi['notes'] = 'Multi-year mean SMB data'
df_multi['error'] = np.nan

# Combining method and dating method for each entry
df_multi['method'] = df_metadata_multi['Method'] + ', ' + df_metadata_multi['Dating_Method']
df_multi['method_key'] = 12  # Adjust the method key as required

list_add = [df_multi]

# Load metadata
df_metadata_annual = pd.read_excel('AntSMB_Annual SMB compilation.xlsx', sheet_name='Site_information')

# Process Ice core or stake farm records
print('reading ice cores or stake farms from AntSMB_Annual SMB compilation.xlsx')

sheet = 'Ice core or stake farm records'
df_ice_core = pd.read_excel('AntSMB_Annual SMB compilation.xlsx', sheet_name=sheet)
df_ice_core.columns = ['Year']+df_ice_core.columns[1:].to_list()
df_ice_core['Year'] = np.arange(2014,2014-len(df_ice_core['Year']),-1)
# Loop through each column (DatasetName ID)
for col in df_ice_core.columns[1:]:
    dataset_id = col
    df = pd.DataFrame({
        'end_year': df_ice_core['Year'],
        'smb': df_ice_core[dataset_id] / 1000,  # Convert SMB to appropriate m w.e.
        'start_year': df_ice_core['Year']
    }).dropna(subset=['smb'])  # Drop rows without SMB values
    # df['smb'] = df.smb / (df.end_year - df.start_year)

    # Extract metadata for this dataset ID
    metadata = df_metadata_annual[df_metadata_annual['Dataset Name ID'] == dataset_id].iloc[0]
    df['name'] = metadata['Geo_siteName']
    df['latitude'] = metadata['Geo_latitude']
    df['longitude'] = metadata['Geo_longitude']
    df['elevation'] = metadata['Geo_elevation']
    df['reference_short'] = metadata['Reference']
    df['reference'] = 'Short ref'

    df['error'] = np.nan
    df[['start_date', 'end_date']] = np.nan
    df['notes'] = 'Annual SMB data - Ice core or stake farm records'

    # Assign method and method_key
    df['method'] = 'ice core measurements'
    df['method_key'] = 3

    # Append df to list (list_add in the original script)
    list_add.append(df)

# Process Stake measurements
sheet = 'Stake measurements'
df_stake = pd.read_excel('AntSMB_Annual SMB compilation.xlsx', sheet_name=sheet)
print('reading stake measurements from AntSMB_Annual SMB compilation.xlsx')

# Loop through each row (site)
for i, row in df_stake.iterrows():
    site = row.values[0]
    df = pd.DataFrame({
        'end_year': df_stake.columns[1:],  # Years are the columns
        'smb': row.values[1:] / 1000,  # Convert SMB to appropriate m w.e.
        'start_year': df_stake.columns[1:]
    }).dropna(subset=['smb'])  # Drop rows without SMB values

    # Extract metadata for this site from the metadata file
    metadata = df_metadata_annual[df_metadata_annual['Dataset Name ID'] == site].iloc[0]
    df['name'] = metadata['Geo_siteName']
    df['latitude'] = metadata['Geo_latitude']
    df['longitude'] = metadata['Geo_longitude']
    df['elevation'] = metadata['Geo_elevation']
    df['reference_short'] = metadata['Reference']
    df['reference'] = 'Short ref'

    df['error'] = np.nan
    df[['start_date', 'end_date']] = np.nan
    df['notes'] = 'Annual SMB data - Stake measurements'

    # Assign method and method_key
    df['method'] = 'stake measurements'
    df['method_key'] = 4

    # Append df to list (list_add in the original script)
    list_add.append(df)


# Concatenating all dataframes from both multi-year and annual SMB compilations
df_add = pd.concat(list_add)

# Ensuring all needed columns are present
for v in col_needed:
    assert v in df_add.columns, f'{v} is missing'

# %% updating method to readable string
met = {'STR': 'stratigraphy',
       'ECM': 'conductivity',
       'TRI': 'tritium',
       'ISO': 'dO18',
       'DEP': 'dielectric profiling',
       'CFA': 'chemistry',
       'VOH': 'volcanic horizon',
       'FIP':'anthropogenic radioactivity',
        'NAR': 'natural radioactivity'}

df_add.method.loc[df_add.method.astype(str).str.contains('/+')] = df_add.method.loc[df_add.method.astype(str).str.contains('/+')]+' dating'
df_add.method = (df_add.method
 .str.replace('Ice core/snow pit','firn or ice core')
 .str.replace('+', ' and ')
 .str.replace('GPR', 'ground-based radar')
 .str.replace('ice core measurements', 'firn or ice core'))

for k in met.keys():
    df_add.method = df_add.method.str.replace(k,met[k])
    df_add.method = df_add.method.str.replace('  ',' ')


# %% checking for missing SMB and correcting some entries
print("missing SMB:")
print(df_add.loc[df_add.smb.isnull(),['start_year','end_year','name','reference_short']])
df_add =df_add.loc[df_add.smb.notnull(),:]

# disambiguation of two profiles at H72. The one that needs to be ignored because already in Ant2k
df_add.loc[
    (df_add.name.astype(str).str.strip()=='H72') & \
     df_add.reference_short.str.startswith('Nishio'), 'name'] = 'H72 41.08333E'

# -> corrected according to  https://doi.org/10.1029/2003JD004065
# but https://doi.org/10.5194/tc-2016-27 has LGB65 -71.8500 77.9200 1850 probably an error
df_add.loc[df_add.name.astype(str).str.contains('LGB65'),
             ['latitude','longitude','elevation']] = [-70.83527, 77.07472, 1850]

df_add.loc[df_add.name == 'FB96DML02',
             ['latitude','longitude']] = [ -74.968333, 3.918500]

df_add['name'] = df_add.name.str.strip()

# %% listing duplicates

skip_list = ['105km', 'Km200', '200km', 'B04', 'FB9804', 'FB9812', 'FB9805', 'FB9813',
     'FB9814', 'FB9811', 'FB9817', 'FB9807', 'FB9808', 'FB9806', 'FB9810', 'Hercules',
     'south pole', 'WAIS2014', 'Berkener', 'DIV2010', 'SEAT10-6', 'DML73C05_03',
     'BER01C90_01', 'Gomez', 'THW2010', 'PIG2010', 'WDC05A', 'SEAT10-1', 'SEAT10-3',
     'SEAT10-4', 'SEAT10-5', 'FB9803', 'FRI09C90_90', 'FRI17C90_231', 'FRI21C90_HWF',
     'FRI11C90_235', 'FRI15C90_131', 'FRI29C95_10', 'FRI25C95_14', 'FRI27C95_12',
     'FRI23C95_16', 'FRI28C95_11', 'FRI33C95_06', 'FRI10C90_136', 'FRI32C95_07',
     'FRI14C90_336', 'FRI16C90_230', 'FRI19C90_05', 'FRI18C90_330', 'FRI20C90_06',
     'FRI12C90_236', 'FRI38C95_04', 'FRI35C95_01', 'NM02C02_02', 'BER02C90_02',
     '200 km', 'DML83S05_18', 'DML81S05_16', 'DML79S05_14', 'DML87S05_22', 'DML88S05_23',
     'DML86S05_21', 'DML90S05_25', 'DML89S05_24', 'DML80S05_15', 'DML84S05_19',
     'DML76S05_11', 'DML85S05_20', 'NM02C89_01', 'NM03C98_01', 'DML68C04_03', 'DML67C04_02',
     'DML74C05_04', 'DML71C05_01', 'DML72C05_02', 'FB9815', 'FB97DML06', 'FB97DML04',
     'Upstream-C', 'Siple Dome94', 'CWA-D', 'Ross ice drainage system C', 'RIDS- B',
     'RIDS-A', 'ITASE-02-6', 'ITASE-02-4', 'ITASE-02-1', 'ITASE-01-6', 'ITASE-01-5',
     'ITASE-01-3', 'ITASE-01-1', 'ITASE-00-4', 'ITASE-00-2', 'ITASE-00-1', 'ITASE-99-1',
     'WD05Q', 'Bryan Coast', 'FB0189', 'FB0704', 'FB0702', 'FB96DML01', 'FB96DML02',
     'FB97DML03', 'FB97DML04', 'FB97DML06', 'FB97DML07', 'FB97DML08', 'FB97DML09',
     'FB97DML10', 'FB9803', 'FB9815', 'FB9816', 'Berkner Island', 'Gomez', 'Dyer Plateau',
     'Bruce Plateau', 'Beethoven', 'Ferrigno', 'ITASE-00-3', 'Hercules Névé', 'RICE',
     'FB9802', 'H72 41.08333E',  'R1', 'Derwael Ice Rise IC12',
     'LGB65', # has multiple reference in AntSMB but all should be removed
     'James Ross Island',
]
skip_list = [v.strip() for v in skip_list]

skip_list_with_ref = [   ['GV2','Frezzotti et al., 2004; 2007',],
                      ['Vostok', 'Ekaykin et al., 2014'],
                      ['S20', 'Isaksson'],
                      ['Dome Fuji', 'Kameda, 2008'],
              ['GV5','Frezzotti et al., 2004; 2007',],
              ['GV7', 'Frezzotti et al., 2004; 2014',],
              ['GV6', 'McCrae',],
              ['GV7', 'McCrae',],
              ['B33', 'Oerter'],
              ['B33', 'Graf'],
              ['B32', 'Graf'],
              ['B32', 'Oerter'],
              ['B31', 'Oerter'],
              ['B31', 'Graf'],
              ['B25', 'Mulvaney'],
              ['B39', 'Fernandoy'],
              ['D66', 'Magand et al., 2004',],
              ['D66','Frezzotti et al., 2004; 2013',],
              ['Law Dome', 'Roberts'],
              ['S100', 'Kaczmarska'],
              ['S100-DML', 'Kaczmarska'],
              ]

# absent from AntAWS
# ['SPOLE 0EW', 'vostok-vrs', 'LGB65 77.07472E', 'GV2 145.2631 E', 'JRI (Aristarain et al., 2004)', 'Bryan coast', 'WD05A 112.125W', 'WD05Q 112.086W', 'WAIS 2014', 'Talos Dome 159.07575E', 'GV7 158.8624E', 'GV5 158.5369 E', 'Fimbulisen s20', 'Ekstroem B04', 'Derwael Ice Rise']

# 'GV6','Dome F' couldn't find which duplicate


# %% Checking that the listed names are indeed duplicates
# In this part we take each entry of the skip_list and
# 1) look if there is a SMB record nearby in SUMup
# 2) if there's nothing nearby, then maybe it is beause of improper coordinates
#     so we look at mathcing names in AntSMB and SUMup
if 'df_sumup' in locals():
    plot = True
    # for multiyear record, it's easier to look at accumulation rates rather than SMB
    df_add['acc'] = df_add['smb']/np.maximum(1, df_add['end_year']-df_add['start_year'])
    df_sumup['acc'] = df_sumup['smb']/np.maximum(1, df_sumup['end_year']-df_sumup['start_year'])

    def find_nearby_entries(row, df_sumup):
        lat_range = (df_sumup.latitude >= row.latitude - 0.1) \
            & (df_sumup.latitude <= row.latitude + 0.1)
        lon_range = (df_sumup.longitude >= row.longitude - 0.1) \
            & (df_sumup.longitude <= row.longitude + 0.1)
        return df_sumup[lat_range & lon_range][
            ['name','latitude','longitude','reference_short','reference','method']
            ].drop_duplicates()

    # Iterate through df_add and find corresponding entries in df_sumup
    df_meta = df_add.loc[df_add.name.isin(skip_list),
                         ['name','latitude','longitude','reference_short',
                          'reference',]].drop_duplicates()
    df_meta = df_meta.sort_values(by='name').reset_index(drop=True)

    plt.close('all')

    print('\tin AntSMB (name, lat, lon)\t\t\tpreviously in SUMup (name, lat, lon)')

    for _, row in df_meta.iterrows():
        nearby_entries = find_nearby_entries(row, df_sumup)

        if nearby_entries.empty:
            print('=== missing ===')
            print(row['name'])
            pattern = row['name']
            # pattern='LGB65'

            print('finding a name in AntSMB')
            names = df_add.loc[df_add.name.astype(str).str.contains(pattern),
                               'name'].drop_duplicates()
            match_list = []
            for n in names:
                match_list.append(
                    df_add.loc[df_add.name==n,
                            ['latitude','longitude','name','reference_short']
                            ].drop_duplicates()
                            )
            print(pd.concat(match_list).to_markdown(index=None))

            print('finding a name in SUMup')
            match_list = []
            for n in names:
                match_list.append(
                    df_sumup.loc[df_sumup.name==n,
                            ['latitude','longitude','name','reference_short']
                            ].drop_duplicates()
                            )
            print(pd.concat(match_list).to_markdown(index=None))

            print('======')
        else:
            # Plot profiles from df_add and df_sumup
            mask1 = (df_add.name == row['name']) & \
                (df_add.latitude == row.latitude) & \
                    (df_add.longitude == row.longitude)
            if plot:
                plt.figure()
                plt.plot(df_add.loc[mask1,'start_year'], df_add.loc[mask1,'acc'],
                         marker='d',ls ='None',
                         label=row['name']+' '+row['reference_short']+' in AntSMB')

            for _, near_row in nearby_entries.iterrows():
                if 'radar' in near_row.method:
                    continue
                mask2 = (df_sumup.name.astype(str) == str(near_row['name'])) & \
                    (df_sumup.latitude == near_row.latitude) & \
                        (df_sumup.longitude == near_row.longitude)
                sumup_match = df_sumup.loc[mask2,:]
                if plot:
                    plt.plot(sumup_match['start_year'], sumup_match['acc'],
                             marker='.',ls ='None',
                             label=str(sumup_match['name'].iloc[0]) \
                                 +' '+sumup_match['reference_short'].iloc[0] \
                                     +' in SUMup')

                print(row['name'], row['latitude'], row['longitude'],'\t\t',sumup_match['name'].iloc[0],
                      sumup_match['latitude'].iloc[0],
                      sumup_match['longitude'].iloc[0])
            if plot:
                plt.legend()
                plt.xlabel("Depth")
                plt.ylabel("Accumulation rate (m w.e. a-1)")

# %% Removing duplicates
print('## Removing based on name')
match_list=[]
for name in skip_list:
    mask = df_add.name.str.strip()==name.strip()
    match_list.append(df_add.loc[mask,
                                 ['name','latitude','longitude','reference_short']
                                 ].drop_duplicates())
    df_add = df_add.loc[~mask, :]
print(pd.concat(match_list).to_markdown(index=None))
print('## Removing based on name+ref')
match_list=[]
for row in skip_list_with_ref:
    mask = (df_add.name==row[0]) & df_add.reference_short.str.contains(row[1], regex=False)
    match_list.append(df_add.loc[mask,
                                 ['name','latitude','longitude','reference_short']
                                 ].drop_duplicates())
    df_add = df_add.loc[~mask, :]
print(pd.concat(match_list).to_markdown(index=None))


df_add = df_add.loc[~df_add.reference_short.str.startswith('Spikes et al., 2005')]
df_add = df_add.loc[~df_add.reference_short.str.startswith('Verfaillie et al., 2012')]


# %% updating references based on references.csv
df_ref = pd.read_csv('references.csv')  # Reference CSV with 'reference_short' and 'reference' columns
df_ref['reference'] = df_ref[['reference', 'Unnamed: 3', 'Unnamed: 4']].apply(lambda x: ' and '.join(x.dropna().astype(str)), axis=1)
df_ref['short_ref'] = df_ref['short_ref'].str.strip()
df_ref['reference_short'] = df_ref['reference_short'].str.strip()
df_ref = df_ref.set_index('short_ref')
df_ref = df_ref[~df_ref.index.duplicated(keep='first')]

df_add['reference_short'] = df_add.reference_short.str.strip()

ref_with_no_match = np.unique([v for v in df_add.reference_short.values if v not in df_ref.index.values])
if len(ref_with_no_match):
    print(wtf)

# Loop through the DataFrame and update references
for short_ref in df_add['reference_short'].drop_duplicates():
    mask = df_add.reference_short==short_ref
    if df_ref.loc[short_ref, 'reference']=='':
        df_ref.loc[short_ref, 'reference'] = df_ref.loc[short_ref, 'reference_short']
    df_add.loc[mask, 'reference'] = df_ref.loc[short_ref, 'reference']
    df_add.loc[mask, 'reference_short'] = df_ref.loc[short_ref, 'reference_short']

if (df_add.reference.str.strip()=='').any():
    # df_add.loc[(df_add.reference.str.strip()==''),'reference_short'].iloc[0]
    print(wtf)
if (df_add.reference_short.str.contains('study')).any():
    print(wtf)

for this_ref in ["This study", "Wang et al. 2021"]:
    df_add.loc[df_add.reference_short == this_ref, 'reference'] =  "Wang, Y., Ding, M., Reijmer, C. H., Smeets, P. C. J. P., Hou, S., and Xiao, C.: The AntSMB dataset: a comprehensive compilation of surface mass balance field observations over the Antarctic Ice Sheet, Earth Syst. Sci. Data, 13, 3057–3074, https://doi.org/10.5194/essd-13-3057-2021, 2021. "
    df_add.loc[df_add.reference_short == this_ref, 'reference_short'] =  "Wang et al. (2021)"


mask = df_add.reference_short==df_add.reference
print('References still missing:')
print(df_add.loc[mask,'reference_short'].drop_duplicates().values)

# %% Printing to file
print('New entries:')
values = df_add.name.drop_duplicates().values
for i in range(0, len(values), 15):
    print(*values[i:i+15])

# adding mention "as in"
this_ref =  "Wang et al., 2021"
df_add.loc[df_add.reference_short != this_ref, 'reference'] =  df_add.loc[df_add.reference_short != this_ref, 'reference'].values + ". As in: Wang, Y., Ding, M., Reijmer, C. H., Smeets, P. C. J. P., Hou, S., and Xiao, C.: The AntSMB dataset: a comprehensive compilation of surface mass balance field observations over the Antarctic Ice Sheet, Earth Syst. Sci. Data, 13, 3057–3074, https://doi.org/10.5194/essd-13-3057-2021, 2021. "
df_add.loc[df_add.reference_short != this_ref, 'reference_short'] =  df_add.loc[df_add.reference_short != this_ref, 'reference_short'].values + " as in Wang et al. (2021)"

df_add[col_needed].to_csv('data_formatted.csv', index=None)

# %% Finding more duplicates
if 'df_sumup' in locals():
    #  finding by coordinates
    plot_duplicates = False
    if plot_duplicates:
        name_to_skip = []
        not_duplicates =['SS9801', 'South Pole', 'filchner-ronne ice shelf firn core D136',
         'filchner-ronne ice shelf firn core D230', 'filchner-ronne ice shelf firn core D236',
         'filchner-ronne ice shelf firn core D330', 'filchner-ronne ice shelf firn core D336',
         'filchner-ronne ice shelf firn core BAS site 5', 'filchner-ronne ice shelf firn core BAS site 6',
         '42 Pentagon STA network', '11-550',
         'F4-2009', 'berkner island north dome firn core BER01C90_01',
         'berkner island south dome firn core BER02C90_02', 'LT940',]

        for name, lat, lon in df_add[['name', 'latitude', 'longitude']].drop_duplicates().itertuples(index=False):
            if isinstance(name,float):
                continue
            if name in not_duplicates:
                continue
            # Check for matching lat/lon within 0.001 in df_sumup
            match = df_sumup[(df_sumup['latitude'].between(lat - 0.005, lat + 0.005)) &
                              (df_sumup['longitude'].between(lon - 0.005, lon + 0.005))]
            # print(name,lat,lon)
            if not match.empty:
                matching_name_in_sumup = match['name'].values[0]  # Get the first matching name
                if isinstance(matching_name_in_sumup,float):
                    df_sumup.loc[df_sumup.name_key==match['name_key'].values[0] , 'name'] = name+'?'
                    matching_name_in_sumup = name+'?'
                if isinstance(name,float):
                    continue
                    df_add.loc[(df_add.latitude == lat) & (df_add.longitude == lon), 'name'] = matching_name_in_sumup+'?'
                    name = matching_name_in_sumup+'?'
                    if matching_name_in_sumup == 'iSTAR Traverse':
                        continue
                    if name == 'iSTAR Traverse?':
                        continue
                print(name, matching_name_in_sumup)
                name_to_skip.append(name)
                print(match[['latitude','longitude']].drop_duplicates().values)  # Get the first matching name
                print(lat, lon)  # Get the first matching name
                plt.figure(figsize=(10, 6))
                plt.scatter(df_add.loc[df_add.name == name, 'start_year'],
                            df_add.loc[df_add.name == name,'smb'], color='blue',marker='d', label='AntSMB: '+name, s=50)
                plt.scatter(df_sumup.loc[df_sumup.name == matching_name_in_sumup, 'start_year'],
                            df_sumup.loc[df_sumup.name == matching_name_in_sumup,'smb'],
                            color='tab:orange', label='SUMup: '+matching_name_in_sumup, s=20)
                plt.xlabel('Start Year')
                plt.ylabel('SMB (kg m² yr⁻¹)')
                plt.legend()
                plt.grid(True)
                plt.show()

    # finding by name
    plot_duplicates = False
    if plot_duplicates:
        df_add['reference_short'] = df_add['reference_short'] + 'as in Wang et al. (2021)'
        df_add['reference'] = df_add['reference'] + '. Compiled in Wang, Y., Ding, M., Reijmer, C. H., Smeets, P. C. J. P., Hou, S., and Xiao, C.: The AntSMB dataset: a comprehensive compilation of surface mass balance field observations over the Antarctic Ice Sheet, Earth Syst. Sci. Data, 13, 3057–3074, https://doi.org/10.5194/essd-13-3057-2021, 2021.'
        names_antsmb = df_add.name.drop_duplicates().str.strip()
        names_sumup = df_sumup.loc[df_sumup.name.notnull(),:].name.drop_duplicates().str.strip()
        skip_list = []
        for name in names_antsmb.values:
            if name in names_sumup.values:
                same_length= len(df_add.loc[df_add.name==name,:].set_index('start_year').smb) == len(df_sumup.loc[df_sumup.name==name,:].set_index('start_year').smb)
                same_value = False
                if same_length:
                    same_value = (df_add.loc[df_add.name==name,:].set_index('start_year').smb - \
                                  df_sumup.loc[df_sumup.name==name,:].set_index('start_year').smb).mean()<0.001
                if same_length and same_value:
                    print(name, 'already in SUMup, skipping')
                    if ((df_add.loc[df_add.name==name,'latitude'].iloc[0] - \
                          df_sumup.loc[df_sumup.name==name,'latitude'].iloc[0])<0.005) & \
                                  ((df_add.loc[df_add.name==name,'longitude'].iloc[0] - \
                                      df_sumup.loc[df_sumup.name==name,'longitude'].iloc[0])<0.005):
                        skip_list.append(name)
                    else:
                        print(wtf)

                else:
                    print(name, 'different from what is in SUMup')

                f = plt.figure()
                df_add.loc[df_add.name==name,:].set_index('start_year').smb.plot(label='AntSMB: '+name,
                                                                         ls='None',marker='d',)
                df_sumup.loc[df_sumup.name==name,:].set_index('start_year').smb.plot(
                    ls='None',marker='.',label='SumUP: '+name)
                plt.legend()
                plt.grid()
                plt.show()
