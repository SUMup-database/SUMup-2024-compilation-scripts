# -*- coding: utf-8 -*-
"""
Created on %(date)s
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""
def find_multiyear(df):
    df1 = df.loc[df.latitude>0,:].copy()
    # df1 = df_sumup.loc[df_sumup.reference == 'Mosley-Thompson, E., J.R. McConnell, R.C. Bales, Z. Li, P-N. Lin, K. Steffen, L.G. Thompson, R. Edwards, and D. Bathke. (2001)Local to Regional-Scale Variability of Greenland Accumulation from PARCA cores. Journal of Geophysical Research (Atmospheres), 106 (D24), 33,839-33,851.',:]

    df1 = df1.sort_values(by=['name_key', 'reference_key', 'method_key', 'latitude', 'longitude', 'start_year', 'end_year'])
    df1['count_end_year'] = df1.end_year #set up column to count end years in grouping
    df1['count_smb'] = df1.smb #count smbs

    #find measurements with single and multi-year rows
    gr = df1.groupby(['name_key', 'reference_key', 'method_key', 'latitude', 'longitude','start_year']).agg({'end_year': 'unique', 'count_end_year': 'nunique'})
    result = gr[gr['count_end_year'] > 1].sort_values(by='count_end_year')

    #find 'duplicate' measurements with different smbs
    gr1 = df1.groupby(['name_key', 'reference_key', 'method_key', 'latitude', 'longitude','start_year', 'end_year']).agg({'smb':list, 'count_smb': 'count'})
    result1 = gr1[gr1['count_smb'] > 1].sort_values(by='count_smb')

    return result, result1
result, result1 = find_multiyear(df_sumup)

# %%
# Sort the DataFrame
df1 = df_sumup.loc[df_sumup.latitude>0,:].copy()

df1 = df1.sort_values(by=['name_key', 'reference_key', 'method_key', 'latitude', 'longitude', 'start_year', 'end_year'])

# Find unique rows based on ['name_key', 'reference_key', 'method_key', 'latitude', 'longitude', 'start_year']
unique_start_year = df1.drop_duplicates(subset=['name_key', 'reference_key', 'method_key', 'latitude', 'longitude', 'start_year'])

# Aggregate unique 'end_year' and count them for multi-year rows
multi_year_rows = []
for _, row in unique_start_year.iterrows():
    subset = df1[(df1['name_key'] == row['name_key']) &
                 (df1['reference_key'] == row['reference_key']) &
                 (df1['method_key'] == row['method_key']) &
                 (df1['latitude'] == row['latitude']) &
                 (df1['longitude'] == row['longitude']) &
                 (df1['start_year'] == row['start_year'])]
    end_years = subset['end_year'].unique()
    if len(end_years) > 1:
        multi_year_rows.append({'name_key': row['name_key'],
                                'reference_key': row['reference_key'],
                                'method_key': row['method_key'],
                                'latitude': row['latitude'],
                                'longitude': row['longitude'],
                                'start_year': row['start_year'],
                                'end_years': end_years,
                                'count_end_year': len(end_years)})

result = pd.DataFrame(multi_year_rows).sort_values(by='count_end_year')


result1 = find_duplicates_differentsmb
duplicate_smb_rows = []
unique_rows = df1.drop_duplicates(subset=['name_key', 'reference_key', 'method_key', 'latitude', 'longitude', 'start_year', 'end_year'])
for _, row in unique_rows.iterrows():
    subset = df1[(df1['name_key'] == row['name_key']) &
                 (df1['reference_key'] == row['reference_key']) &
                 (df1['method_key'] == row['method_key']) &
                 (df1['latitude'] == row['latitude']) &
                 (df1['longitude'] == row['longitude']) &
                 (df1['start_year'] == row['start_year']) &
                 (df1['end_year'] == row['end_year'])]
    smbs = subset['smb'].tolist()
    if len(smbs) > 1:
        duplicate_smb_rows.append({'name_key': row['name_key'],
                                   'reference_key': row['reference_key'],
                                   'method_key': row['method_key'],
                                   'latitude': row['latitude'],
                                   'longitude': row['longitude'],
                                   'start_year': row['start_year'],
                                   'end_year': row['end_year'],
                                   'smbs': smbs,
                                   'count_smb': len(smbs)})

result1 = pd.DataFrame(duplicate_smb_rows).sort_values(by='count_smb')



#%%
for name in df1.name.drop_duplicates():
    print(name)
    if df1.loc[df1.name==name, ['start_year','end_year']].duplicated().sum()>0:
        print('DUPLICATED')
    plt.figure()
    tmp = df1.loc[df1.name==name, :]
    tmp.plot(x='start_year', y='smb', marker='o',ls='None')
    plt.title(name)
    plt.show()


#%%

condi = SUMup_raw[['start_year','end_year','latitude','longitude','elevation','smb','error']].duplicated()
print("Nb of duplicate rows:",condi.sum())
# SUMup_raw.drop_duplicates(['start_year','end_year','latitude','longitude','elevation','smb','error'], inplace=True)
# condi = SUMup_raw[['start_year','end_year','latitude','longitude','elevation','smb','error']].duplicated()
# print("Nb of duplicate rows after change:",condi.sum())


# %%

msk = df_sumup.start_date.isnull()
condi = df_sumup[['start_year','end_year','latitude','longitude','elevation','smb','error']].duplicated(keep=False) & msk
ref_list = df_sumup.loc[condi,['smb','name','reference_short']].reference_short.drop_duplicates().values
for ref in ref_list:
    print('***')
    print(df_sumup.loc[condi&(df_sumup.reference_short==ref),
           ['start_year','end_year','smb','name','reference_short']].to_markdown())

# print("Nb of duplicate rows:",condi.sum())
