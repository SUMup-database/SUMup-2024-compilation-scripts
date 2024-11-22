# -*- coding: utf-8 -*-
"""
SUMup compilation script

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
import matplotlib.pyplot as plt
from sumup_lib import resolve_reference_keys, resolve_name_keys

# loading 2023 data
try:
    df_sumup = pd.concat((pd.read_csv('data/SUMup 2023/SMB/SUMup_2023_SMB_antarctica.csv', low_memory=False),
                          pd.read_csv('data/SUMup 2023/SMB/SUMup_2023_SMB_greenland.csv', low_memory=False)))
except:
    # 2023 data too heavy for GitHub, it needs to be downloaded and stored locally
    print('Downloading SUMup 2023 accumulation file')
    url = 'https://arcticdata.io/metacat/d1/mn/v2/object/urn%3Auuid%3Ad7055e41-25d6-425a-9548-e48fccf1b5a1'
    r = requests.get(url, allow_redirects=True)
    open('data/SUMup 2023/SUMup_2023_SMB_csv.zip', 'wb').write(r.content)
    with ZipFile('data/SUMup 2023/SUMup_2023_SMB_csv.zip', 'r') as zObject:
        zObject.extractall( path='data/SUMup 2023/SMB/')
    df_sumup = pd.concat((pd.read_csv('data/SUMup 2023/SMB/SUMup_2023_SMB_greenland.csv'),
                          pd.read_csv('data/SUMup 2023/SMB/SUMup_2023_SMB_antarctica.csv')),
                         ignore_index=True)
    df_sumup = df_sumup.reset_index(drop=True)

df_sumup[df_sumup==-9999] = np.nan

df_ref = pd.read_csv('data/SUMup 2023/SMB/SUMup_2023_SMB_references.tsv', sep='\t')
df_ref = df_ref.set_index('key')

df_names = pd.read_csv('data/SUMup 2023/SMB/SUMup_2023_SMB_names.tsv', sep='\t')
df_names = df_names.set_index('key')

df_methods = pd.read_csv('data/SUMup 2023/SMB/SUMup_2023_SMB_methods.tsv', sep='\t')
df_methods = df_methods.set_index('key')

df_sumup['name'] = np.nan
df_sumup.loc[df_sumup.name_key.notnull(),
             'name'] = df_names.loc[df_sumup.loc[df_sumup.name_key.notnull(),
                              'name_key'] , 'name'].values
df_sumup['reference'] = df_ref.loc[df_sumup.reference_key, 'reference'].values
df_sumup['reference_short'] = df_ref.loc[df_sumup.reference_key, 'reference_short'].values
df_sumup['method'] = df_methods.loc[df_sumup.method_key, 'method'].values

len_sumup_2023 = df_sumup.shape[0]

print(df_sumup.shape[0], 'SMB observations currently in SUMup from', len(df_sumup.reference.unique()), 'sources')
print(df_sumup.loc[df_sumup.latitude>0].shape[0], 'in Greenland')
print(df_sumup.loc[df_sumup.latitude<0].shape[0], 'in Antarctica')

has_duplicated_index = df_sumup.index.duplicated().any()
print("Duplicated indices exist:" if has_duplicated_index else "No duplicated indices.")
df_sumup.loc[df_sumup.index.duplicated(),'reference_short'].drop_duplicates()

col_needed = ['name_key', 'reference_key', 'method_key', 'start_date',
       'end_date', 'start_year','end_year', 'latitude', 'longitude', 'elevation',
       'notes', 'smb', 'error', 'name',
       'reference', 'method', 'reference_short']

len_start = df_sumup.shape[0]


# %% correcting existing data
# correcting reference on Verfaillie 2012:
mask = df_sumup.reference.astype(str).str.startswith('Verfaillie, D., Fily, M., Le Meur, E., Magand, O., Jourdain, B., Arnaud, L., and Favier, V.: Snow accumulation variability derived from radar and firn core data along a 600 km transect in Adelie Land, East Antarctic plateau')
df_sumup.loc[mask, 'reference_short'] = 'Verfaillie et al. (2012)'

# correcting reference and m ice/water eq. for Karlsson 2016
mask = df_sumup["reference_short"] == 'Karlsson et al. (2016)'
df_sumup.loc[mask, "smb"] = df_sumup.loc[mask, "smb"] * 0.917

df_sumup.loc[mask, "renference"] = "Karlsson, NB et al. (2016a): Accumulation rates during 1311-2011 CE in North Central Greenland derived from air-borne radar data. Frontiers in Earth Science, 4(97), 18 pp, https://doi.org/10.3389/feart.2016.00097. Data: Karlsson, Nanna Bjørnholt; Eisen, Olaf; Dahl-Jensen, Dorthe; Freitag, Johannes; Kipfstuhl, Sepp; Lewis, Cameron; Nielsen, Lisbeth T; Paden, John D; Winter, Anna; Wilhelms, Frank (2016b): Accumulation rates. PANGAEA, https://doi.org/10.1594/PANGAEA.868447"
df_sumup.loc[mask, "renference_short"] = 'Karlsson et al. (2016a, 2016b)'

# correcting data previously added as accumulation rates back to actual accumulation
mask = (df_sumup.end_year-df_sumup.start_year)>1
mask2 = ~df_sumup.reference.str.contains('Machguth')
ref_list = df_sumup.reference.loc[mask&mask2].drop_duplicates()
print('Adjusting following references from accumulation rates to smb')
print('\n\n'.join(ref_list))
mask3 = np.isin(df_sumup.reference, ref_list) & mask


df_sumup.loc[mask3,'smb'] = df_sumup.loc[mask3,'smb'] * \
                        (df_sumup.loc[mask3,'end_year'] - \
                         df_sumup.loc[mask3,'start_year'])

# Removing smb data prior to year 1000 from SP19 core
mask = df_sumup.reference.str.startswith('Winski, D. A., Fudge, T. J., Ferris, D. G., Osterberg, E. C., Fegyveresi')
mask2= df_sumup.start_year.isnull()

df_sumup = df_sumup.loc[~(mask&mask2), :]

# Removing uncorrected values from Philippe et al. (2016) ice core
msk = df_sumup.reference_short=='Philippe et al. (2016)'
df_sumup=df_sumup.loc[~msk, :]
# the corrected profile is part of AntSMB

# %% Re-adding ACT2010
# ACT2010 profiles have been added with not enough digits in the lat/lon,
# leading to points having multiple smb for the same locations
mask = (df_sumup.name == 'ACT2010 Radar profile')
df_sumup = df_sumup.loc[~mask, :]
# print('removing', mask.sum())

df = pd.read_csv('data/SMB data/Miège ACT10 radar/data_formatted.csv')
df['name_key'] = 69
df['method_key'] = resolve_name_keys(df, df_sumup, v='method')
df['reference_key'] = 186
# print('adding back', len(df))
# plt.figure()
# df.plot(x='longitude',y='latitude',marker='o', ls='None',ax=plt.gca())
# df_sumup.loc[mask, :].plot(x='longitude',y='latitude',marker='o', ls='None',ax=plt.gca())
df_sumup = pd.concat((df_sumup, df[col_needed]), ignore_index=True)

# fixing reference for the ACCT10 cores
msk_miege_cores = df_sumup.name.astype(str).isin(['ACT10-A', 'ACT10-B', 'ACT10-C', 'ACT-11b', 'ACT-11c', 'ACT-11d'])
df_sumup.loc[msk_miege_cores, 'reference_short'] = 'Miège et al. (2013, 2014b)'
df_sumup.loc[msk_miege_cores, 'reference'] = 'Miège, C., Forster, R. R., Box, J. E., Burgess, E. W., McConnell, J. R., Pasteris, D. R., & Spikes, V. B. (2013). Southeast Greenland high accumulation rates derived from firn cores and ground-penetrating radar. Annals of Glaciology, 54(63), 322–332. doi:10.3189/2013AoG63A358. Data: Clement Miege, Richard R Forster, Jason E Box, Evan W Burgess, Joe R McConnell, Daniel R Pasteris, & Vandy B Spikes. (2014b). Snow accumulation rates in SE Greenland from firn cores. Arctic Data Center. doi:10.18739/A2P26Q419.'


# %% Re-adding PARCA cores
# removing first
mask = df_sumup.reference == 'Mosley-Thompson, E., J.R. McConnell, R.C. Bales, Z. Li, P-N. Lin, K. Steffen, L.G. Thompson, R. Edwards, and D. Bathke. (2001)Local to Regional-Scale Variability of Greenland Accumulation from PARCA cores. Journal of Geophysical Research (Atmospheres), 106 (D24), 33,839-33,851.'
# print("Removing")
# print(df_sumup.loc[mask,:].name.drop_duplicates().values)
# print('from')
# print(df_sumup.loc[mask,['reference_short','reference']].drop_duplicates().values)
# print('')
df_sumup = df_sumup.loc[~mask,:]

mask = np.isin(df_sumup.name, ['S.Domea','S.Domeb']) & (df_sumup.reference=='Mosley-Thompson, E., McConnell, J. R., Bales, R. C., Li, Z., Lin, P.-N., Steffen, K., Thompson, L. G., Edwards, R., and Bathke, D. (2001), Local to regional-scale variability of annual net accumulation on the Greenland ice sheet from PARCA cores, J. Geophys. Res., 106(D24), 33839–33851, doi:10.1029/2001JD900067.')
# print("Removing")
# print(df_sumup.loc[mask,:].name.drop_duplicates().values)
# print('from')
# print(df_sumup.loc[mask,['reference_short','reference']].drop_duplicates().values)
# print('')
df_sumup = df_sumup.loc[~mask,:]

print('(re)adding PARCA cores')
df = pd.read_csv('data/SMB data/PARCA/data_formatted.csv')
df['name_key'] = resolve_name_keys(df, df_sumup, df_existing=df_names)
df['reference_key'] = resolve_reference_keys(df, df_sumup)
df['method_key'] = resolve_name_keys(df, df_sumup,  v='method')
df_sumup = pd.concat((df_sumup, df[col_needed]), ignore_index=True)


# %% re-adding Bolzan and Strobel data
print('(re)adding Bolzan and Strobel data')
df_sumup = df_sumup.loc[~np.isin(df_sumup.reference_key,
                                 np.arange(12,21)),:]
df_sumup.loc[np.isin(df_sumup.reference_key,
                                 np.arange(12,21)),['name','reference_key','reference']].drop_duplicates().to_markdown()


df = pd.read_csv('data/SMB data/bolzan_strobel/data_formatted.csv')
df['name_key'] = resolve_name_keys(df, df_sumup, df_existing=df_names)
df['method_key'] = resolve_name_keys(df, df_sumup, v='method')
df_sumup = pd.concat((df_sumup, df[col_needed]), ignore_index=True)

# %% Fixing Dibb and Fahnstock (2001)
df_sumup_22 = pd.read_csv('data/SMB data/SUMup 2022/SUMup_accumulation_2022.csv')
df_sumup_22.columns = df_sumup_22.columns.str.lower()
df_sumup_22 = df_sumup_22.rename(columns={'accumulation':'smb',
                                    'citation': 'reference',
                                    'radar_horizontal_resolution': 'notes'})

df_sumup_22[df_sumup_22==-9999] = np.nan
df_sumup_22=df_sumup_22.loc[df_sumup_22.reference==21,:]
df_sumup_22['start_date'] = pd.to_datetime(df_sumup_22.timestamp)
df_sumup_22['end_date'] =  df_sumup_22['start_date'] + pd.offsets.MonthEnd(0)

mask = (df_sumup.reference_key == 21)
df_sumup.loc[mask, 'start_date'] = df_sumup_22['start_date'].values
df_sumup.loc[mask, 'end_date'] = df_sumup_22['end_date'].values
df_sumup.loc[mask, 'smb'] = df_sumup_22['smb'].values
df_sumup.loc[mask, 'error'] = df_sumup_22['error'].values
df_sumup.loc[mask, 'reference_short'] = "Dibb and Fahnestock (2004)"

# %% GEUS SWE database
# removing Box compilation which had bad dates and duplicates
mask = np.isin(df_sumup.reference_short, ['GEUS unpublished',
                                  'Burgress et al. (2010)',
                                  'Schaller et al. (2016)',
                                  'Hermann et al. (2018)',
                                  'Niwano et al. (2020)',
                                  'Kjær et al. (2021)']) & \
    (df_sumup.reference != 'Kjær, Helle Astrid; Hauge, Lisa Lolk; Simonsen, Marius; Yoldi, Zurine; Koldtoft, Iben; Hörhold, Maria; Freitag, Johannes; Kipfstuhl, Sepp; Svensson, Anders M; Vallelonga, Paul T (2021): Accumulation of snow as determined by the summer hydrogen peroxide peak measured using the LISA box for several sites in northern Greenland. PANGAEA, https://doi.org/10.1594/PANGAEA.935333, ')
print("Removing")
print(df_sumup.loc[mask,:].name.drop_duplicates().values)
print('from')
print(df_sumup.loc[mask,['reference_short','reference']].drop_duplicates().values)
print('')
df_sumup = df_sumup.loc[~mask,:]


print('GEUS SWE database')
df = pd.read_csv('data/SMB data/GEUS SWE database/data_formatted.csv')
df['name_key'] = resolve_name_keys(df, df_sumup, df_existing=df_names)
df['reference_key'] = resolve_reference_keys(df, df_sumup)
df['method_key'] = resolve_name_keys(df, df_sumup, v='method')
df_sumup = pd.concat((df_sumup, df[col_needed]), ignore_index=True)

# %% NSIDC files
print('NSIDC Crete Milcent')
df = pd.read_csv('data/SMB data/NSIDC Crete Milcent/data_formatted.csv')
df['name_key'] = resolve_name_keys(df, df_sumup, df_existing=df_names)
df['reference_key'] = resolve_reference_keys(df, df_sumup)
df['method_key'] = resolve_name_keys(df, df_sumup, v='method')
df_sumup = pd.concat((df_sumup, df[col_needed]), ignore_index=True)

# %% PROMICE daily ablation
# plt.close('all')
# path_dir = 'data/SMB data/PROMICE/'
# skip_list = ['QAS_Uv3', 'QAS_Lv3', 'NUK_K','KPC_Uv3','KPC_U',
#              'KAN_U', 'MIT', 'ZAK_L','ZAK_Uv3',
#              'THU_U2','SWC', 'LYN_T','LYN_L',
#              ]
# plot = False
# for f in os.listdir(path_dir):
#     if f.replace('.csv','') in skip_list:
#         os.remove(path_dir + f)
#         continue
#     print(f)
#     df = pd.read_csv(path_dir+f)
#     df['time'] = pd.to_datetime(df.time)
#     df = df.set_index('time')
#     diff = df.z_surf_combined.diff()
#     diff = diff.loc[diff<diff.quantile(0.98)]
#     # diff = diff.resample('W').sum()

#     smb = diff.copy().to_frame(name='smb') * 900/1000
#     smb['year'] = smb.index.year
#     smb['time'] = smb.index.values
#     smb['start_date'] = smb.time - pd.Timedelta(days=1)
#     smb['end_date'] = smb.time
#     smb = smb.reset_index(drop=True)

#     smb_y = smb.groupby('year').smb.sum().to_frame().reset_index(drop=True)
#     smb_y['start_date'] = smb.groupby('year').time.first().values
#     smb_y['end_date'] = smb.groupby('year').time.last().values
#     smb_y = smb_y.reset_index(drop=True)

#     df_new = pd.concat((smb[['start_date','end_date','smb']],
#                     smb_y.loc[smb_y.start_date.dt.year>2020,
#                               ['start_date','end_date','smb']]), ignore_index=True)

#     df_new['start_year'] = smb_y.start_date.dt.year
#     df_new['end_year'] = smb_y.end_date.dt.year
#     df_new['latitude'] = df.gps_lat.mean()  #smb.groupby('year').time.latitude().values
#     df_new['longitude'] = df.gps_lon.mean()  #smb.groupby('year').time.longitude().values
#     df_new['elevation'] = df.gps_alt.mean()  # smb.groupby('year').time.elevation().values
#     df_new['name'] = f.replace('.csv','')
#     df_new['method'] = 13
#     df_new['notes'] = ''
#     df_new['error'] = np.nan
#     df_new['method'] = 'pressure transducer in ablation hose'
#     df_new['reference_short'] = 'PROMICE (2023)'
#     df_new['reference'] = 'How, P.; Abermann, J.; Ahlstrøm, A.P.; Andersen, S.B.; Box, J. E.; Citterio, M.; Colgan, W.T.; Fausto. R.S.; Karlsson, N.B.; Jakobsen, J.; Langley, K.; Larsen, S.H.; Mankoff, K.D.; Pedersen, A.Ø.; Rutishauser, A.; Shield, C.L.; Solgaard, A.M.; van As, D.; Vandecrux, B.; Wright, P.J., 2022, "PROMICE and GC-Net automated weather station data in Greenland", https://doi.org/10.22008/FK2/IW73UU, GEUS Dataverse, V9'
#     df_new['name'] = resolve_name_keys(df_new, df_sumup)
#     df_new['reference_key'] = resolve_reference_keys(df_new, df_sumup)

#     if plot:
#         fig, ax = plt.subplots(2,1,sharex=True, figsize=(10,10))
#         # df.z_surf_combined.plot(ax=ax[0])
#         # (df.gps_alt-df.gps_alt.iloc[0]).plot(marker='o')
#         smb.set_index('start_date').smb.cumsum().plot(ax=ax[0], marker='o')
#         smb_y.set_index('start_date').smb.cumsum().plot(ax=ax[0], drawstyle="steps-post")
#         smb_y.set_index('end_date').smb.cumsum().plot(ax=ax[0], drawstyle="steps-post")
#         ax[1].set_ylabel('cumulated SMB')
#         smb_y.set_index('start_date').smb.plot(ax=ax[1], drawstyle="steps-mid")
#         ax[1].set_ylabel('annual SMB')
#         plt.suptitle(df.site.unique()[0])

#     # df_candidates = check_duplicates(df_new, df_sumup,tol = 0.1)
#     df_sumup = pd.concat((df_sumup, df_new[col_needed]), ignore_index=True)
df_sumup['start_date'] = pd.to_datetime(df_sumup.start_date, utc=True).dt.tz_localize(None)
df_sumup['end_date'] = pd.to_datetime(df_sumup.end_date, utc=True).dt.tz_localize(None)
msk = (df_sumup.reference_short=='PROMICE (2023)') & \
    df_sumup.end_date.notnull() & (df_sumup.end_date.dt.year != df_sumup.end_year)
df_sumup.loc[msk, 'end_year'] = df_sumup.loc[msk, 'end_date'].dt.year
df_sumup.loc[msk, 'start_year'] = df_sumup.loc[msk, 'start_date'].dt.year

# %% Leverett Galcier ablation
print('Adding Leverett catchment')
df = pd.read_csv('data/SMB data/Leverett catchment/data_formatted.csv')

df['reference_key'] = resolve_reference_keys(df, df_sumup)
df['name_key'] = resolve_name_keys(df, df_sumup)
df['method_key'] = resolve_name_keys(df, df_sumup, v='method')
df_sumup = pd.concat((df_sumup, df[col_needed]), ignore_index=True)

# %% loading SnowFox
print('Adding SnowFox data')
df = pd.read_csv('data/SMB data/SnowFox_GEUS/data_formatted.csv')
df['reference_key'] = resolve_reference_keys(df, df_sumup)
df['name_key'] = resolve_name_keys(df, df_sumup)
df['method_key'] = resolve_name_keys(df, df_sumup, v='method')
df_sumup = pd.concat((df_sumup, df[col_needed]), ignore_index=True)

# %% loading IMAU K-transect
print('Adding IMAU K-transect')
df = pd.read_csv('data/SMB data/IMAU K-transect/data_formatted.csv')
df['reference_key'] = resolve_reference_keys(df, df_sumup)
df['name_key'] = resolve_name_keys(df, df_sumup)
df['method_key'] = resolve_name_keys(df, df_sumup, v='method')
df_sumup = pd.concat((df_sumup, df[col_needed]), ignore_index=True)

# Note: vandewal_2012 is already part of Machguth et al. 2016
# file = 'data/SMB data/IMAU K-transect/vandewal_2012.tab'
# df = pd.read_csv(file,sep='\t', skiprows=25)
# df.columns = ['name', 'start_year', 'end_year', 'smb', 'Comment']

# %% loading GC-Net historical
print('Adding GC-Net historical')
df = pd.read_csv('data/SMB data/GC-Net historical/data_formatted.csv')
df['reference_key'] = resolve_reference_keys(df, df_sumup)
df['name_key'] = resolve_name_keys(df, df_sumup)
df['method_key'] = resolve_name_keys(df, df_sumup, v='method')
df_sumup = pd.concat((df_sumup, df[col_needed]), ignore_index=True)

# %% loading Ant2k
print('Adding Ant2k')
df = pd.read_csv('data/SMB data/Ant2k Thomas/data_formatted.csv')

df['reference_key'] = resolve_reference_keys(df, df_sumup)
df['name_key'] = resolve_name_keys(df, df_sumup)
df['method_key'] = resolve_name_keys(df, df_sumup, v='method')
df_sumup = pd.concat((df_sumup, df[col_needed]), ignore_index=True)

# %% loading AntSMB
print('Adding AntSMB')
df = pd.read_csv('data/SMB data/AntSMB/data_formatted.csv', low_memory=False)
# check low smb in sumup
# B38 GC01 -71.16218333333333 -6.69885
# check coordinates:
# FB97DML07 DML07C98_31 -75.583333 -3.433333
# SS9801 FB9803 -74.85 -8.5

# check 'F5-2009','F6-2009','F3-2009','F2-2009','F1-2009'
# from Wagenbach
# 'berkner island south dome firn core BER02C90_02',
# 'berkner island south dome firn core BER02C90_02',
# 'berkner island north dome firn core BER01C90_01'

df['reference_key'] = resolve_reference_keys(df, df_sumup)
df['name_key'] = resolve_name_keys(df, df_sumup)
df['method_key'] = resolve_name_keys(df, df_sumup, v='method')
df_sumup = pd.concat((df_sumup, df[col_needed]), ignore_index=True)

# %% loading Koenig OIB accumulation
print('Adding Koenig OIB accumulation')
df = pd.read_csv('data/SMB data/Koenig OIB/data_formatted.csv')
df['method'] = 'airborne radar'
df['reference_short'] =  'Koenig et al. (2016)'
df['reference'] = 'Koenig, L. S., Ivanoff, A., et al.: Annual Greenland accumulation rates (2009–2012) from airborne snow radar, The Cryosphere, 10, 1739–1752, https://doi.org/10.5194/tc-10-1739-2016, 2016.'

df['reference_key'] = resolve_reference_keys(df, df_sumup)
df['name_key'] = df_sumup.name_key.max() + 1 + df.index.values
df['method_key'] = resolve_name_keys(df, df_sumup, v='method')
df_sumup = pd.concat((df_sumup, df[col_needed]), ignore_index=True)

# %% loading amundsen-scott-south-pole-station
print('Adding amundsen-scott-south-pole-station')
df = pd.read_csv('data/SMB data/amundsen-scott-south-pole-station-climatology-data-1957-present-ongoing-4rvtod/data_formatted.csv')
df['reference_key'] = resolve_reference_keys(df, df_sumup)
df['name_key'] = resolve_name_keys(df, df_sumup)
df['method_key'] = resolve_name_keys(df, df_sumup, v='method')
df_sumup = pd.concat((df_sumup, df[col_needed]), ignore_index=True)

# %% checking duplicates
    # df_sumup=df_sumup.loc[df_sumup.latitude>0,:]
    # df_sumup=df_sumup.loc[~df_sumup.method_key.isin([2,9,10,13]),:]
    # df_sumup['lat_round'] = np.round(df_sumup.latitude)
    # df_sumup['lon_round'] = np.round(df_sumup.longitude)
    # cols=['lat_round', 'lon_round', 'end_year','smb']
    # msk = df_sumup[cols].duplicated() & df_sumup.smb!=0
    # tmp = df_sumup.loc[msk,cols].drop_duplicates()
    # for y, smb in zip(tmp.end_year, tmp.smb):
    #     if (df_sumup.loc[(df_sumup.end_year == y) & (df_sumup.smb==smb),
    #                  'method_key']!=2).sum()<2: continue
    #     print(df_sumup.loc[(df_sumup.end_year == y) & (df_sumup.smb==smb),
    #                  ['start_year', 'end_year', 'latitude', 'longitude', 'smb', 'name','method_key']
    #                  ].to_markdown(index=False),'\n')

# %% checking file format
from lib.check import check_coordinates, check_missing_key, check_duplicate_key, check_duplicate_reference

df_sumup = check_coordinates(df_sumup)
df_sumup = check_missing_key(df_sumup, var='method')
df_sumup = check_missing_key(df_sumup, var='name')
df_sumup = check_missing_key(df_sumup, var='reference')

df_sumup = check_duplicate_key(df_sumup, var='method')
df_sumup = check_duplicate_reference(df_sumup)

# checking inconsistent reference
df_references = df_sumup[['reference_key','reference_short','reference']].drop_duplicates()
df_references.columns = ['key','reference_short','reference']
df_references = df_references.set_index('key')
for ref in df_references.reference.loc[df_references.index.duplicated()]:
    print('\nFound reference key with multiple references:')
    print(df_references.loc[df_references.reference == ref, :].drop_duplicates().to_markdown())

# checking for inconsitent dates
df_sumup['start_date'] = pd.to_datetime(df_sumup.start_date, utc=True).dt.tz_localize(None)
df_sumup['end_date'] = pd.to_datetime(df_sumup.end_date, utc=True).dt.tz_localize(None)
msk = df_sumup.start_date.notnull() & (df_sumup.start_date.dt.year != df_sumup.start_year)
if msk.any():
    print('start_date and start_year mismatch for:')
    print(df_sumup.loc[msk, ['name','start_date','end_date','start_year','end_year',  'reference_short',]].to_markdown())
    print(wtf)

msk = df_sumup.end_date.notnull() & (df_sumup.end_date.dt.year != df_sumup.end_year)
if msk.any():
    print('start_date and start_year mismatch for:')
    print(df_sumup.loc[msk, ['name','end_date','end_year', 'reference_short',]].to_markdown())
    print(wtf)

# making sure keys are well assigned
# df_ref_new = df_sumup[['reference_key', 'reference','reference_short']].drop_duplicates()
# df_ref_new.columns = ['key', 'reference', 'reference_short']
# df_ref_new['key'] = np.arange(1,len(df_ref_new)+1)
# df_ref_new = df_ref_new.set_index('key')
# df_sumup.reference = df_ref_new.reset_index().set_index('reference').loc[df_sumup.reference].values

df_sumup.loc[df_sumup.method =='firn core', 'method'] = 'firn or ice core'
df_method_new = pd.DataFrame(df_sumup.method.unique())
df_method_new.columns = ['method']
df_method_new.index = df_method_new.index+1
df_method_new.index.name = 'key'
df_sumup.method_key = df_method_new.reset_index().set_index('method').loc[df_sumup.method].values

df_name_new = pd.DataFrame(df_sumup.name.unique())
df_name_new.columns = ['name']
df_name_new['key'] = np.arange(1,len(df_name_new)+1)
df_name_new = df_name_new.set_index('key')
df_sumup['name_key'] = df_name_new.reset_index().set_index('name').loc[df_sumup.name].values

print(' ======== Finished ============')
print('{:,.0f}'.format(df_sumup.shape[0]).replace(',',' ') +\
      ' observations currently in new dataset')
print('{:,.0f}'.format(df_sumup.shape[0] - len_start).replace(',',' ') +\
      ' new observations')
# print('Checking conflicts')
# sumup_index_conflict = check_conflicts(df_sumup, df_vdx)

# print('\noverwriting conflicting data in SUMup (checked by bav)\n')
# msk = ~df_sumup.index.isin(sumup_index_conflict)
# df_sumup = pd.concat((df_sumup.loc[msk,:], df_vdx), ignore_index=True)


# looking for redundant references
# tmp = df_sumup.reference.unique()

print(df_sumup.shape[0], 'accumulation observations after merging from',
      len(df_sumup.reference.unique()), 'sources')
print(df_sumup.loc[df_sumup.latitude>0].shape[0], 'in Greenland')
print(df_sumup.loc[df_sumup.latitude<0].shape[0], 'in Antarctica')

# %% writing to file
# CSV format
from lib.write import (round_and_format, write_reference_csv, write_method_csv,
                       write_names_csv, write_data_to_csv,
                       write_smb_to_netcdf)
df_sumup = round_and_format(df_sumup)

write_reference_csv(df_sumup,
        'SUMup 2024 beta/SUMup_2024_smb_csv/SUMup_2024_SMB_references.tsv')
write_method_csv(df_sumup,
        'SUMup 2024 beta/SUMup_2024_SMB_csv/SUMup_2024_SMB_methods.tsv')
write_names_csv(df_sumup,
        'SUMup 2024 beta/SUMup_2024_SMB_csv/SUMup_2024_SMB_names.tsv')

write_data_to_csv(df_sumup,
            csv_folder='SUMup 2024 beta/SUMup_2024_SMB_csv',
            filename='SUMup_2024_smb',
            write_variables= ['name_key', 'reference_key', 'method_key', 'start_date', 'end_date',
                                'start_year', 'end_year','latitude', 'longitude',
                                'elevation',  'smb',  'error', 'notes'])

# netcdf format
for var in ['elevation','start_year','end_year']:
    df_sumup[[var]] =  df_sumup[[var]].replace('','-9999').astype(int)
    df_sumup[df_sumup==-9999] = np.nan

write_smb_to_netcdf(df_sumup.loc[df_sumup.latitude>0, :],
                        'SUMup 2024 beta/SUMup_2024_SMB_greenland.nc')
write_smb_to_netcdf(df_sumup.loc[df_sumup.latitude<0, :],
                        'SUMup 2024 beta/SUMup_2024_SMB_antarctica.nc')

#%% updating ReadMe file
from lib.plot import plot_dataset_composition, plot_map
from lib.write import write_dataset_composition_table, write_location_file, create_kmz

df_meta = pd.DataFrame()
df_meta['total'] = [df_sumup.shape[0]]
df_meta['added'] = df_sumup.shape[0]-len_sumup_2023
df_meta['nr_references'] = str(len(df_sumup.reference.unique()))
df_meta['greenland'] = df_sumup.loc[df_sumup.latitude>0].shape[0]
df_meta['antarctica'] = df_sumup.loc[df_sumup.latitude<0].shape[0]
df_meta.index.name='index'
df_meta.to_csv('doc/ReadMe_2024_src/tables/SMB_meta.csv')

print('{:,.0f}'.format(df_sumup.shape[0]).replace(',',' ') +\
      ' SMB observations in SUMup 2024')
print('{:,.0f}'.format(df_sumup.shape[0]-len_sumup_2023).replace(',',' ') +\
      ' more than in SUMup 2023')
print('from '+ str(len(df_sumup.reference_short.unique())) + ' sources')
print('representing '+ str(len(df_sumup.reference.unique()))+' references')

print('{:,.0f}'.format(df_sumup.loc[df_sumup.latitude>0].shape[0]).replace(',',' ')+' observations in Greenland')
print('{:,.0f}'.format(df_sumup.loc[df_sumup.latitude<0].shape[0]).replace(',',' ')+' observations in Antarctica')

plot_dataset_composition(df_sumup.loc[df_sumup.latitude>0],
        'doc/ReadMe_2024_src/figures/SMB_dataset_composition_greenland.png')
plot_map(df_sumup.loc[df_sumup.latitude>0,['latitude','longitude','method']].drop_duplicates(),
         'doc/ReadMe_2024_src/figures/SMB_map_greenland.png',
         area='greenland')

plot_dataset_composition(df_sumup.loc[df_sumup.latitude<0],
        'doc/ReadMe_2024_src/figures/SMB_dataset_composition_antarctica.png')

plot_map(df_sumup.loc[df_sumup.latitude<0,['latitude','longitude','method']].drop_duplicates(),
         'doc/ReadMe_2024_src/figures/SMB_map_antarctica.png',
         area='antarctica')


write_dataset_composition_table(df_sumup.loc[df_sumup.latitude>0],
                'doc/ReadMe_2024_src/tables/composition_SMB_greenland.csv')

write_dataset_composition_table(df_sumup.loc[df_sumup.latitude<0],
                'doc/ReadMe_2024_src/tables/composition_SMB_antarctica.csv',)
#
# print('writing out measurement locations')
# write_location_file(df_sumup.loc[df_sumup.latitude>0,:],
#                     'doc/GIS/SUMup_2024_smb_location_greenland.csv')

# write_location_file(df_sumup.loc[df_sumup.latitude<0, :],
#                     'doc/GIS/SUMup_2024_SMB_location_antarctica.csv')

# create_kmz(df_sumup.loc[~ df_sumup.method.astype(str).str.contains('radar'),:],
#            output_prefix="doc/GIS/SUMup_2024_smb")
