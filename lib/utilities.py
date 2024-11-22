# -*- coding: utf-8 -*-
"""
@author: bav@geus.dk

"""
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt


def resolve_reference_keys(df, df_sumup):
    '''
    Compares a dataframes 'reference_full' field  to the references that are already in
    SUMup. If a reference is already in SUMup (df_sumup), then it's reference_key
    is reused. If it is a new reference, then a reference_key is created.

    '''
    df_ref = df_sumup[['reference_key','reference']].drop_duplicates().set_index('reference')
    df_ref_to_add = (df.reference.drop_duplicates()
                        .reset_index(drop=True)
                        .reset_index(drop=False)
                        .set_index('reference')
                        )
    count=1
    for r in df_ref_to_add.index:
        tmp = df_ref.index.str.contains(r, regex=False)
        if tmp.any():
            # reference already in df_sumup, reusing key
            if len(df_ref.loc[tmp].reference_key.values)>1:
                print('\n', r, 'matching with')
                print(df_ref.loc[tmp].reference_key)
                print('using first one')
            df_ref_to_add.loc[r, 'index'] = df_ref.loc[tmp].reference_key.values[0]
        else:
            # new reference, creating new key
            df_ref_to_add.loc[r, 'index'] = df_sumup.reference_key.max() + count
            count = count+1
    return df_ref_to_add.loc[df.reference].values

def remove_and_print_duplicates(df, df_sumup):
    '''
    Check for duplicates based on 'name' and 'reference_short' between two DataFrames.
    Print min and max 'start_year' for matching rows and remove these rows from df.

    Input:
        df:        DataFrame with new data
        df_sumup:  DataFrame with existing data

    Output:
        DataFrame with duplicates removed from df
    '''
    # Create a dictionary to index df by 'name' and 'reference_short'
    df_index = df.set_index(['name', 'reference_short']).index
    df_dict = {}
    for idx in df_index:
        if idx not in df_dict:
            df_dict[idx] = []
        df_dict[idx].append(idx)

    # Initialize lists to store indices to remove and results
    to_remove = []

    for name, reference_short in df_sumup[['name', 'reference_short']].drop_duplicates().values:
        idx = (name, reference_short)
        if idx in df_dict:
            # Extract matching rows from df_sumup
            subset_new = df[(df['name'] == name) & (df['reference_short'] == reference_short)]
            subset_existing = df_sumup[(df_sumup['name'] == name) & (df_sumup['reference_short'] == reference_short)]

            # Calculate min and max of 'start_year'
            min_start_year_new = subset_new['start_year'].min()
            max_start_year_new = subset_new['start_year'].max()
            min_start_year_existing = subset_existing['start_year'].min()
            max_start_year_existing = subset_existing['start_year'].max()

            # Print the results
            print(f"Name: {name}, Reference Short: {reference_short}")
            print(f"New Data - Start Year Min: {min_start_year_new}, Max: {max_start_year_new}")
            print(f"Existing Data - Start Year Min: {min_start_year_existing}, Max: {max_start_year_existing}")
            print('')

            # Collect indices to remove
            to_remove.extend(df_dict[idx])

    # Remove duplicates from df using the collected indices
    df_cleaned = df[~df.index.isin(to_remove)]

    return df_cleaned

def check_duplicates(df_stack, df_sumup, verbose=True, plot=True, tol=0.3):
    '''
    Finding potential duplicates in new SMB data compared to what's in SUMup
    Input:
        df_stack:   dataframe with potentially multiple new SMB measurements to be
                    added to SUMup
        df_sumup:   dataframe with SUMup observations
        verbose:    if True print the candidates-duplicates
        plot:       if True print the candidates-duplicates
        tol:        tolerance, in degrees, accepted for coordinates of potential
                    duplicates compared to the location of new data
    Output:
        A dataframe containing the potential duplicates' name, reference_short,
        latitude, longitude and elevation
    '''
    df_all_candidates = pd.DataFrame()
    for p in df_stack.name.unique():
        df = df_stack.loc[df_stack.name == p, :]
        if len(df)==0:
            print('skipping',p)
            continue

        # looking for obs within defined tolerance
        msk1 = abs(df_sumup.latitude - df.latitude.iloc[0]) < tol
        msk2 = abs(df_sumup.longitude - df.longitude.iloc[0]) < tol
        # looking for cores or snow pits and not radar profiles
        msk3 = ~df_sumup.method.isin([2, 6, 7])
        msk = msk1 & msk2 & msk3

        if msk.any():
            # looping through the different observation groups (as defined by name)
            df_all_candidates_group = pd.DataFrame()  # Temporary DataFrame for current group
            for p_dup in df_sumup.loc[msk, 'name'].unique():
                # extracting potential duplicate from SUMup
                df_sumup_candidate_dupl = df_sumup.loc[df_sumup.name == p_dup, :]
                # making a last check that the new data (in df) and the candidate
                # duplicate ends more or less at the same year
                if abs(df_sumup_candidate_dupl.end_year.max() - df.end_year.max()) > 3:
                    continue

                # append that candidate to the list of candidates
                df_all_candidates_group = pd.concat((
                    df_all_candidates_group,
                    df_sumup_candidate_dupl[
                        ['name', 'reference_short', 'latitude', 'longitude', 'elevation']
                        ].drop_duplicates()))

            if len(df_all_candidates_group) >= 1:
                df_all_candidates = pd.concat((df_all_candidates, df_all_candidates_group))

                if verbose:
                    print('')
                    print(df_all_candidates_group.values)
                    print('might be the same as')
                    print(df[['name','reference_short', 'latitude', 'longitude']].drop_duplicates().values)
                    if df.name.iloc[0] == 'sdo2':
                        print('sdo2 has same coordinates as S.Domea and b but',
                              'different time range')

                if plot:
                    plt.figure()
                    plt.plot(df.end_year.values, df.smb.values,
                             marker='o', ls='None',
                             label=df.name.iloc[0] + ' '+df.reference_short.iloc[0])
                    for p_dup in df_all_candidates_group.name.unique():
                        plt.plot(df_sumup.loc[df_sumup.name == p_dup, 'end_year'].values,
                                 df_sumup.loc[df_sumup.name == p_dup, 'smb'].values,
                                 marker='^', ls='None', alpha=0.7,
                                 label=(df_sumup.loc[df_sumup.name == p_dup,
                                                    'name'].iloc[0]
                                        + ' ' +
                                        df_sumup.loc[df_sumup.name == p_dup,
                                                       'reference_short'].iloc[0]))
                    plt.title(df.name.iloc[0])
                    plt.legend()
                    plt.show()  # Ensure plot is displayed

    return df_all_candidates


def resolve_name_keys(df, df_sumup, v='name', df_existing=None):
    '''
    Creates name_keys for all the unique 'name' in df.
    '''
    df_names_to_add = (df[v]
                        .drop_duplicates()
                        .reset_index(drop=True)
                        .reset_index(drop=False)
                        .set_index(v)
                        )
    df_names_to_add.columns = [v+'_key']

    if df_existing is not None:
        df_existing = df_existing.reset_index().set_index(v)
        df_existing = df_existing[~df_existing.index.duplicated(keep='first')]
    else:
        df_existing = (df_sumup[[v,v+'_key']]
                            .drop_duplicates()
                            .set_index(v)
                            )
        df_existing.columns = ['key']
    for p in df_names_to_add.index:
        if p in df_existing.index:
            df_names_to_add.loc[p, v+'_key'] = df_existing.loc[p, 'key']
            print('reusing old key', df_existing.loc[p, 'key'], 'for',p)
        else:
            df_names_to_add.loc[p, v+'_key'] = df_names_to_add.loc[p, v+'_key'] + 1 + df_sumup[v+'_key'].max()

    return df_names_to_add.loc[df[v]].values


# import cartopy
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature


def parse_short_reference(df_in, verbose=False):
    df_sumup = df_in.copy()
    abc = 'bcdefghijklmnopqrstuvwxyz'
    df_sumup['reference'] = df_sumup.reference_full.astype(str)
    all_refs = np.array([['reference_short', 'reference']])
    for ref in df_sumup.reference.unique():
        if (',' not in ref) and (' ' not in ref):
            print(ref, 'left as is')
            ref_short = ref
        else:
            year = re.findall(r'\d{4}', ref)
            if len(year) > 0:
                year = year[0]
            else:
                year = ''

            # first guess
            name = ref.lstrip().split(',')[0].split(' ')[0]

            # some exceptions
            if name == 'U.S.': name = 'US_Army'
            if name == 'SIMBA': name = 'SIMBA: Lewis'
            if name == 'Paul': name = 'Smeets'
            if name == 'K.': name = 'Wegener'
            if name in ['van', 'Van']: name = ref.lstrip().split(',')[0].replace(' ',' ')
            ref_short = name + ' et al. ('+ year+')'
            if name == 'US': ref_short = 'US ITASE: Mayewski and Dixon ('+ year+')'
            if name == 'Satellite-Era': ref_short = 'SEAT11: Brucker and Koenig ('+ year+')'

        count = 0
        while ref_short in all_refs[:,0]:
            if count == 0:
                ref_short = ref_short[:-1] + 'b)'
                # tmp = all_refs[-1][0]
                # all_refs[-1] = tmp[:-1] + 'a)'
                count = count + 1
            else:
                ref_short = ref_short[:-2] + abc[count] +')'
                count = count + 1
        if verbose: print(ref_short)
        all_refs = np.vstack([all_refs, [ref_short, ref]])
    df_ref = pd.DataFrame(all_refs[1:,:], columns=all_refs[0,:])
    return df_ref


def check_conflicts(df_sumup, df_new, var=['name', 'depth','temperature'],
                    verbose=1):
    coords_sumup = df_sumup[['latitude','longitude']].drop_duplicates()
    coords_new = df_new[['latitude','longitude']].drop_duplicates()
    diff_lat = np.abs(coords_sumup.latitude.values - coords_new.latitude.values[:, np.newaxis])
    diff_lon = np.abs(coords_sumup.longitude.values - coords_new.longitude.values[:, np.newaxis])

    potential_duplicates = np.where( (diff_lat < 0.01) & (diff_lon < 0.01))
    for k in range(len(potential_duplicates[0])):
        i = potential_duplicates[0][k]
        j = potential_duplicates[1][k]
        tmp = pd.DataFrame()
        tmp['in SUMup']= df_sumup.loc[coords_sumup.iloc[j,:].name,:].T
        tmp['in new dataset'] = df_new.loc[coords_new.iloc[i,:].name,:].T
        if np.round(tmp.loc['date','in SUMup']/10000) == np.round(tmp.loc['date','in new dataset']/10000):
            if verbose:
                print('\nPotential duplicate found:\n')

                print(tmp.loc[var])
                print('reference in SUMup:')
                print(tmp.loc['reference_full', 'in SUMup'])
                print('reference in new dataset:')
                print(tmp.loc['reference_full', 'in new dataset'])
            return [tmp.loc[var[0], 'in SUMup']]
    return []
