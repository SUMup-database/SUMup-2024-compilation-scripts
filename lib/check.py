# -*- coding: utf-8 -*-
"""
Created on %(date)s
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

def check_missing_timestamp(df_sumup):
    msk = (df_sumup.timestamp.isnull())|(df_sumup.timestamp=='NA')|(df_sumup.timestamp=='Na')
    missing_timestamp = df_sumup.loc[msk, :]
    if len(missing_timestamp)>0:
        print('Missing time stamp on:')
        print(missing_timestamp[['reference_short','depth','temperature']].drop_duplicates())
        print('removing')
        df_sumup = df_sumup.loc[~msk,:]
    return df_sumup

def check_missing_key(df_sumup, var='method'):
    """
    Checks for missing keys in the specified column of the DataFrame.

    Args:
        df_sumup (pd.DataFrame): The DataFrame containing the data to be checked.
        var (str): The variable name prefix to check for missing keys (default is 'method').

    Returns:
        pd.DataFrame: The updated DataFrame with missing keys assigned as 'NA' for 'method'.

    Prints:
        The names and references for entries missing the specified key.
    """
    missing_key = ((df_sumup[var+'_key'] == -9999) | df_sumup[var+'_key'].isnull())
    if missing_key.any():
        print(f'\nMissing {var} key for')
        print(df_sumup.loc[missing_key, ['name', 'reference_short']].drop_duplicates().set_index('name'))
        print('')
        if var == 'method':
            df_sumup.loc[missing_key, 'method_key'] = -9999
            df_sumup.loc[missing_key, 'method'] = "NA"
            print('assigning "NA"')
        else:
            raise ValueError(f"variable '{var}' did not pass check_missing_key.")
    return df_sumup


def check_duplicate_key(df_sumup, var='method'):
    """
    Checks for duplicate keys in the specified column and keeps only the first occurrence.

    Args:
        df_sumup (pd.DataFrame): The DataFrame to check for duplicate keys.
        var (str): The variable name prefix to check for duplicates (default is 'method').

    Returns:
        pd.DataFrame: The updated DataFrame with duplicate keys handled.

    Prints:
        Each duplicate value and the kept key for the duplicate.
    """
    df_unique = df_sumup[[var+'_key', var]].drop_duplicates()
    dup = df_unique.loc[df_unique[var].duplicated()]
    for d in np.unique(dup[var]):
        print(f'\n{var} "{d}" has multiple keys:')
        print(df_unique.loc[df_unique[var] == d].set_index(f'{var}_key'))
        print(f'only keeping the first: {df_unique.loc[df_unique[var] == d, var + "_key"].iloc[0]}')

        df_sumup.loc[df_sumup[var] == d, var+'_key'] = df_unique.loc[df_unique[var] == d, var+'_key'].iloc[0]
    return df_sumup


def check_duplicate_reference(df_sumup):
    """
    Checks for duplicate references with different short versions and standardizes to the first occurrence.

    Args:
        df_sumup (pd.DataFrame): The DataFrame containing references and their short forms.

    Returns:
        pd.DataFrame: The updated DataFrame with standardized reference short names.

    Prints:
        Each duplicate reference with different short names, showing the chosen version.
    """
    df_ref_new = df_sumup[['reference', 'reference_short']].drop_duplicates()
    while len(df_ref_new.loc[df_ref_new.reference.duplicated(), 'reference']) > 0:
        for dup_ref in df_ref_new.loc[df_ref_new.reference.duplicated(), 'reference']:
            print('\nOne reference has two different short versions:')
            print(dup_ref)
            print(df_ref_new.loc[df_ref_new.reference == dup_ref, 'reference_short'].values)
            print('    Keeping the first one')
            df_sumup.loc[df_sumup.reference == dup_ref,
                         'reference_short'] = df_ref_new.loc[
                             df_ref_new.reference == dup_ref, 'reference_short'].values[0]
        df_ref_new = df_sumup[['reference', 'reference_short']].drop_duplicates()
    df_ref_new = df_ref_new.reset_index(drop=True)

    df_sumup['reference_key'] = (df_ref_new.reset_index()
                                 .set_index('reference')
                                 .loc[df_sumup.reference, :])['index'].values + 1

    # checking duplicate reference
    tmp = df_sumup[['reference_key','reference']].drop_duplicates()
    if tmp.reference_key.duplicated().any():
        print('\n====> Found two references for same reference key')
        dup_ref = tmp.loc[tmp.reference_key.duplicated()]
        for ref in dup_ref.reference_key.values:
            doubled_ref = df_sumup.loc[df_sumup.reference_key == ref,
                               ['reference_key','reference']].drop_duplicates()
            # if doubled_ref.iloc[0,1].replace(' ','').lower() == doubled_ref.iloc[1,1].replace(' ','').lower():
            df_sumup.loc[df_sumup.reference_key == ref, 'reference'] = doubled_ref.iloc[0,1]
            print('Merging\n', doubled_ref.iloc[1,1],'\ninto\n', doubled_ref.iloc[0,1],'\n')
            # else:
            #     print(wtf)
    return df_sumup


def check_coordinates(df_sumup):
    """
    Validates and corrects coordinates (latitude and longitude) in the DataFrame.

    Args:
        df_sumup (pd.DataFrame): The DataFrame containing latitude and longitude columns.

    Returns:
        pd.DataFrame: The updated DataFrame with corrected coordinates.

    Prints:
        Warnings and details for entries with missing, invalid, or out-of-range coordinates.
    """
    # Correct positive longitude for entries in the southern hemisphere
    df_sumup.loc[(df_sumup.latitude > 0) & (df_sumup.longitude > 0), 'longitude'] = -df_sumup.loc[
        (df_sumup.latitude > 0) & (df_sumup.longitude > 0), 'longitude']

    # Check for missing latitude
    if df_sumup.latitude.astype(float).isnull().any():
        print('Missing latitude for:')
        print(df_sumup.loc[df_sumup.latitude.astype(float).isnull(), ['name', 'reference_short']])
        print('Removing from compilation\n')
        df_sumup = df_sumup.loc[df_sumup.latitude.notnull(), :]

    # Check for missing longitude
    if df_sumup.longitude.astype(float).isnull().any():
        print('Missing longitude for:')
        print(df_sumup.loc[df_sumup.longitude.astype(float).isnull(), ['name', 'reference_short']])
        print('Removing from compilation\n')
        df_sumup = df_sumup.loc[df_sumup.longitude.notnull(), :]

    # Validate latitude range
    if (df_sumup.latitude.astype(float) > 90).any():
        print('Error: Latitude values exceed 90 degrees:')
        print(df_sumup.loc[(df_sumup.latitude.astype(float) > 90), ['name', 'latitude', 'reference_short']])
        raise ValueError("Latitude exceeds the valid range of -90 to 90 degrees.")

    if (df_sumup.latitude.astype(float) < -90).any():
        print('Error: Latitude values below -90 degrees:')
        print(df_sumup.loc[(df_sumup.latitude.astype(float) < -90), ['name', 'latitude', 'reference_short']])
        raise ValueError("Latitude is below the valid range of -90 to 90 degrees.")

    # Validate longitude range
    if (df_sumup.longitude.astype(float) > 360).any():
        print('Error: Longitude values exceed 360 degrees:')
        print(df_sumup.loc[(df_sumup.longitude.astype(float) > 360), ['name', 'longitude', 'reference_short']])
        raise ValueError("Longitude exceeds the valid range of -180 to 180 degrees.")

    if (df_sumup.longitude.astype(float) < -180).any():
        print('Error: Longitude values below -180 degrees:')
        print(df_sumup.loc[(df_sumup.longitude.astype(float) < -180), ['name', 'longitude', 'reference_short']])
        print('Removing entries with invalid longitudes\n')
        df_sumup = df_sumup.loc[df_sumup.longitude.astype(float) >= -180, :]
    return df_sumup
