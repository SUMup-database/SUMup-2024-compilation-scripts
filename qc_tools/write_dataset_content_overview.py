# -*- coding: utf-8 -*-
"""
Created on %(date)s
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""

df_result = df_sumup.groupby(['latitude', 'longitude', 'profile','timestamp',  'reference_short', 'reference']).agg(
    profile_depth_m=('midpoint', 'max')
).reset_index()

# Join with the unique values
df_unique = df_sumup[['latitude', 'longitude', 'profile', 'timestamp', 'reference_short', 'reference']].drop_duplicates()
df_unique = df_unique.merge(df_result, on=['latitude', 'longitude', 'profile', 'timestamp', 'reference_short', 'reference'], how='left')
df_unique.timestamp = df_unique.timestamp.dt.date
df_unique = df_unique[['latitude', 'longitude', 'profile', 'timestamp', 'profile_depth_m', 'reference_short', 'reference']]
df_unique.to_csv('all_density_profiles.tsv', sep='\t', index=None)

# %% temperature

df_result = df_sumup.groupby(['latitude', 'longitude', 'name', 'reference_short', 'reference']).agg(
    start_date=('timestamp', 'min'),
    end_date=('timestamp', 'max')
).reset_index()

# Calculate timespan as the difference between start_date and end_date
# df_result['timespan'] = (df_result['end_date'] - df_result['start_date']).dt.days

# Merge with unique values and format timestamp columns as dates
df_unique = df_sumup[['latitude', 'longitude', 'name', 'reference_short', 'reference']].drop_duplicates()
df_unique = df_unique.merge(df_result[['latitude', 'longitude', 'name', 'reference_short', 'reference', 'start_date', 'end_date']],
                            on=['latitude', 'longitude', 'name', 'reference_short', 'reference'],
                            how='left')

# Format dates
df_unique['start_date'] = df_unique['start_date'].dt.date
df_unique['end_date'] = df_unique['end_date'].dt.date

# Select columns and save to file
df_unique = df_unique[['latitude', 'longitude', 'name', 'start_date', 'end_date', 'reference_short', 'reference']].sort_values(by='latitude', ascending=False)
df_unique.to_csv('all_temperatures.tsv', sep='\t', index=None)

# %% smb

df_sumup[['latitude','longitude']] = df_sumup[['latitude','longitude']].round(4)
df_sumup_radar = df_sumup.loc[df_sumup.method.astype(str).str.contains('radar'),:]
df_condensed = df_sumup_no_radar.groupby(['reference_short', 'reference']).agg(
    start_year=('start_year', 'min'),
    end_year=('end_year', 'max'),
    number_values=('smb', 'count'),
    latitude=('latitude', 'mean'),
    longitude=('longitude', 'mean'),
).reset_index()
df_condensed[['latitude','longitude']] = df_condensed[['latitude','longitude']].round(4)
df_condensed['name'] = 'radar profile'
df_condensed_1 = df_condensed[['latitude', 'longitude', 'name','start_year','end_year','number_values','reference_short', 'reference']]
# Save the condensed table to a file

df_sumup_no_radar = df_sumup.loc[~ df_sumup.method.astype(str).str.contains('radar'),:]
df_condensed = df_sumup_no_radar.groupby(['latitude', 'longitude', 'name', 'reference_short', 'reference']).agg(
    start_year=('start_year', 'min'),
    end_year=('end_year', 'max'),
    number_values=('smb', 'count')
).reset_index()
df_condensed = df_condensed[['latitude', 'longitude', 'name','start_year','end_year','number_values','reference_short', 'reference']]

# Save the condensed table to a file
pd.concat((df_condensed,df_condensed_1)).sort_values(by='latitude', ascending=False).to_csv('all_smb.tsv', sep='\t', index=False)
