import pandas as pd
import xarray as xr
import numpy as np
import shutil
import re

def round_and_format(df_sumup):

    # formatting keys
    df_sumup.reference_key = df_sumup.reference_key.astype(int)

    df_sumup.loc[df_sumup.method.isnull(),'method'] = 'NA'
    df_sumup.loc[df_sumup.method=='','method'] = 'NA'
    df_sumup.loc[df_sumup.method_key.isnull(),'method'] = 'NA'
    df_sumup.loc[df_sumup.method_key == -9999,'method'] = 'NA'
    df_sumup.loc[df_sumup.method_key.isnull(),'method_key'] = -9999
    for var in ['method','reference','reference_short']:
        df_sumup[var ] = df_sumup[var].astype(str)

    # formatting timestamp
    if 'timestamp' in df_sumup.columns:
        df_sumup['timestamp'] = df_sumup.timestamp.astype(str).str.split(' ').str[0].str.split('T').str[0]
        try:
            df_sumup['timestamp'] = pd.to_datetime(df_sumup.timestamp, format='mixed')
        except:
            df_sumup['timestamp'] = pd.to_datetime(df_sumup.timestamp, utc=True)

    #formattin coordinates
    # all longitude should be ranging from -180 to 180
    df_sumup['longitude'] = (df_sumup.longitude + 180) % 360 - 180

    df_sumup.latitude = df_sumup.latitude.astype(float).round(6)
    df_sumup.longitude = df_sumup.longitude.astype(float).round(6)

    df_sumup.loc[df_sumup.elevation.isnull(), 'elevation'] = -9999
    df_sumup.loc[ df_sumup.elevation=='', 'elevation'] = -9999
    df_sumup['elevation'] = df_sumup.elevation.astype(int)

    # making sure error is numeric
    df_sumup['error'] = pd.to_numeric(df_sumup.error, errors='coerce')
    df_sumup.loc[df_sumup.error == -9999, 'error'] = np.nan


    # density variables
    if 'density' in df_sumup.columns:
        df_sumup.profile_key = df_sumup.profile_key.astype(int)
        df_sumup.loc[df_sumup.profile.isnull(),'profile'] = 'NA'
        df_sumup.loc[df_sumup.profile == '','profile'] = 'NA'
        df_sumup['profile'] = df_sumup.profile.astype(str)

        df_sumup.start_depth = df_sumup.start_depth.astype(float).round(4)
        df_sumup.stop_depth = df_sumup.stop_depth.astype(float).round(4)
        df_sumup.midpoint = df_sumup.midpoint.astype(float).round(4)

        for v in ['start_depth', 'stop_depth', 'midpoint', 'error']:
            df_sumup.loc[df_sumup[v]==-9999, v] = np.nan

        df_sumup.density = df_sumup.density.astype(float).round(3)

    # temperature variables
    if 'temperature' in df_sumup.columns:
        for var_int in ['elevation','open_time', 'duration']:
            df_sumup.loc[df_sumup[var_int].isnull(), var_int] = -9999
            df_sumup.loc[df_sumup[var_int]=='', var_int] = -9999
            # if there was something that is not a number, then we shift it to 'notes'
            df_sumup.loc[pd.to_numeric(df_sumup[var_int], errors='coerce').isnull(), 'notes'] = \
                df_sumup.loc[pd.to_numeric(df_sumup[var_int], errors='coerce').isnull(), var_int]
            df_sumup.loc[pd.to_numeric(df_sumup[var_int], errors='coerce').isnull(), var_int] = -9999
            df_sumup[var_int] = pd.to_numeric(df_sumup[var_int], errors='coerce').round(0).astype(int)
            df_sumup[var_int] = df_sumup[var_int].astype(str).replace('-9999','')

        df_sumup.loc[df_sumup.name.isnull(),'name'] = 'NA'
        df_sumup['name'] = df_sumup.name.astype(str)

        df_sumup.temperature = df_sumup.temperature.round(3)

    # smb variables
    if 'smb' in df_sumup.columns:
        df_sumup['smb'] = df_sumup.smb.astype(float).round(6)
        df_sumup['start_date'] = pd.to_datetime(df_sumup.start_date, utc=True)
        df_sumup['end_date'] = pd.to_datetime(df_sumup.end_date, utc=True)

        msk = df_sumup.start_year.isnull() & df_sumup.start_date.notnull()
        df_sumup.loc[msk, 'start_year'] = df_sumup.loc[msk, 'start_date'].dt.year
        df_sumup.loc[df_sumup.start_year.isnull(), 'start_year'] = -9999
        df_sumup.loc[ df_sumup.start_year=='', 'start_year'] = -9999
        df_sumup['start_year'] = df_sumup.start_year.astype(int)

        msk = df_sumup.end_year.isnull() & df_sumup.end_date.notnull()
        df_sumup.loc[msk, 'end_year'] = df_sumup.loc[msk, 'end_date'].dt.year
        df_sumup.loc[df_sumup.end_year.isnull(), 'end_year'] = -9999
        df_sumup.loc[ df_sumup.end_year=='', 'end_year'] = -9999
        df_sumup['end_year'] = df_sumup.end_year.round(0).astype(int)

        df_sumup.loc[df_sumup.name.isnull(),'name'] = 'NA'
        df_sumup['name'] = df_sumup.name.astype(str)

    return df_sumup

# %% CSV functions
def write_reference_csv(df_sumup, filename):
    df_ref_new = df_sumup[['reference_key','reference','reference_short']].drop_duplicates()
    df_ref_new.columns = ['key', 'reference','reference_short']
    df_ref_new = df_ref_new.set_index('key').sort_index()
    df_ref_new.to_csv(filename, sep='\t')

def write_method_csv(df_sumup, filename):
    df_method_new = df_sumup[['method_key','method']].drop_duplicates()
    df_method_new.columns = ['key', 'method']
    df_method_new.loc[-9999, 'method'] = 'Not available'
    df_method_new = df_method_new.dropna()
    df_method_new = df_method_new.set_index('key').sort_index()
    df_method_new.index = df_method_new.index.astype(int)
    df_method_new.to_csv(filename, sep='\t')

def write_profile_csv(df_sumup, filename):
    df_profiles = df_sumup[['profile_key','profile']].drop_duplicates()
    df_profiles.columns = ['key', 'profile']
    df_profiles = df_profiles.set_index('key').sort_index()
    df_profiles.to_csv(filename, sep='\t')

def write_names_csv(df_sumup, filename):
    df_name_new = pd.DataFrame(df_sumup.name.unique())
    df_name_new.columns = ['name']
    df_name_new['key'] = np.arange(1,len(df_name_new)+1)
    df_name_new = df_name_new.set_index('key')
    df_name_new.to_csv(filename, sep='\t')

def write_data_to_csv(df_sumup, csv_folder, filename, write_variables):
    for var in ['elevation','start_year','end_year']:
        if var in df_sumup.columns:
            df_sumup[var] = df_sumup[var].astype(str).replace('-9999','')

    for var in ['timestamp','start_date','end_date']:
        if var in df_sumup.columns:
            df_sumup[var] = pd.to_datetime(df_sumup[var]).dt.strftime('%Y-%m-%d').values

    df_sumup.loc[df_sumup.latitude>0, write_variables].to_csv(
        f'{csv_folder}/{filename}_greenland.csv', index=None)
    df_sumup.loc[df_sumup.latitude<0, write_variables].to_csv(
        f'{csv_folder}/{filename}_antarctica.csv', index=None)

    shutil.make_archive(csv_folder,
                        'zip', csv_folder)



def write_dataset_composition_table(df_in, filename):
    df_sumup = df_in.copy()
    df_sumup['coeff'] = 1
    if 'start_year' in df_sumup.columns:
        df_summary =(df_sumup.groupby('reference_short')
                          .apply(lambda x: x.start_year.min())
                          .reset_index(name='start_year'))
        df_summary['end_year'] =(df_sumup.groupby('reference_short')
                            .apply(lambda x: x.end_year.max())
                            .reset_index(name='end_year')).end_year
    else:
        df_summary =(df_sumup.groupby('reference_short')
                          .apply(lambda x: pd.to_datetime(x.timestamp).dt.year.min())
                          .reset_index(name='start_year'))
        df_summary['end_year'] =(df_sumup.groupby('reference_short')
                            .apply(lambda x: pd.to_datetime(x.timestamp).dt.year.max())
                            .reset_index(name='end_year')).end_year
    df_summary['num_measurements'] = (df_sumup.groupby('reference_short')
                              .apply(lambda x: x.coeff.sum())
                              .reset_index(name='num_measurements')).num_measurements
    # df_summary['reference_key'] = (df_sumup.groupby('reference_short')
    #                           .reference_key.unique().apply(list)
    #                           .astype(str).str.replace("[","").str.replace("]","")
    #                           .reset_index(name='reference_keys')).reference_keys

    df_summary.sort_values('reference_short').to_csv(filename,index=None)


def write_location_file(df_sumup, path_out):
    if 'timestamp' in df_sumup.columns:
        v_time = ['timestamp', 'timestamp']
    else:
        v_time = ['start_year','end_year']

    if 'profile' in df_sumup.columns:
        v_names = ['profile_key','profile']
    else:
        v_names = ['name_key','name']

    tmp = df_sumup[
            v_names + ['latitude', 'longitude','reference_key',
                       'method_key'] + list(dict.fromkeys(v_time))
            ].groupby(['latitude','longitude'])

    df_loc = pd.DataFrame()

    for v in  v_names + ['reference_key', 'method_key']:
        df_loc['list_of_'+v+'s'] = tmp[v].unique().apply(list)

    df_loc['timestamp_min'] = tmp[v_time[0]].min()
    df_loc['timestamp_max'] = tmp[v_time[1]].max()
    df_loc['num_measurements'] = tmp['method_key'].count()

    for v in df_loc.columns:
        df_loc[v] = (df_loc[v].astype(str)
                    .str.replace('[','')
                    .str.replace(']','')
                    .str.replace(' 00:00:00+00:00',''))
        if 'key' in v:
            df_loc[v] = (df_loc[v].astype(str)
                        .str.replace(', ',' / ')
                        .str.replace('  ',' '))
        else:
            df_loc[v] = (df_loc[v].astype(str)
                        .str.replace('\', \'',' / ')
                        .str.replace(',','')
                        .str.replace('\'','')
                        .str.replace('  ',' '))
    df_loc.to_csv(path_out)

# %% netcdf functions
FLOAT_ENCODING = {"dtype": "float32", "zlib": True,"complevel": 3, "shuffle": True}
INT_ENCODING = {"dtype": "int32", "_FillValue":-9999, "zlib": True,"complevel": 3, "shuffle": True}
INT64_ENCODING = {"dtype": "int64", "_FillValue":-9999, "zlib": True,"complevel": 3, "shuffle": True}
STR_ENCODING = {"zlib": True, "complevel": 3, "shuffle": True}
STR_LENGTH = {"name": "S200", "profile": "S200", "method": "S50", "reference": "S1700", "reference_short":"S100"}

def parse_bibtex(bib_file):
    with open(bib_file, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Extract fields using regex
    authors_match = re.search(r'author\s*=\s*{(.+?)}', content, re.DOTALL)
    title_match = re.search(r'title\s*=\s*{(.+?)}', content, re.DOTALL)
    year_match = re.search(r'year\s*=\s*{(.+?)}', content)
    doi_match = re.search(r'doi\s*=\s*{(.+?)}', content)
    publisher_match = re.search(r'publisher\s*=\s*{(.+?)}', content)

    # Parse fields
    authors = authors_match.group(1) if authors_match else ""
    title = title_match.group(1).strip() if title_match else ""
    year = year_match.group(1) if year_match else "n.d."
    doi = doi_match.group(1) if doi_match else ""
    publisher = publisher_match.group(1) if publisher_match else ""

    # Format authors
    authors_list = [author.strip() for author in authors.split(" and ")]
    formatted_authors = []
    for author in authors_list:
        parts = author.split(",")
        last_name = parts[0].strip()
        first_names = parts[1].split() if len(parts) > 1 else []
        initials = [f"{name[0]}." for name in first_names]
        formatted_authors.append(f"{last_name}, {' '.join(initials)}")
    
    # Join authors
    if len(formatted_authors) > 1:
        authors_citation = ", ".join(formatted_authors[:-1]) + ", & " + formatted_authors[-1]
    else:
        authors_citation = formatted_authors[0]

    # Construct citation
    citation = (
        f"{authors_citation}: {title} ({year} release), {publisher}, "
        f"https://www.doi.org/{doi}, {year}."
    )
    return citation

SUMUP_CITATION = parse_bibtex("./doc/ReadMe_2024_src/sumup_reference.bib")
SUMUP_DOI = "10.18739/A2M61BR5M"

def make_key_ds(df_new, var):
    ds_meta_name =  (df_new[[f'{var}_key',var]]
                    .drop_duplicates()
                    .set_index(f'{var}_key')
                    .sort_index()
                    .to_xarray())
    ds_meta_method = (df_new[['method_key','method']]
                      .drop_duplicates()
                      .set_index('method_key')
                      .sort_index()
                      .to_xarray())
    ds_reference =(df_new[['reference_key', 'reference', 'reference_short']]
                           .drop_duplicates()
                           .set_index('reference_key')
                           .sort_index()
                           .to_xarray())
    ds_meta = xr.merge((ds_meta_name, ds_meta_method, ds_reference))

    for v in [var, 'method', 'reference', 'reference_short']:
        encoded_data = np.array([x.encode('utf-8', errors='ignore') for x in ds_meta[v].values],
                                dtype=STR_LENGTH[v])
        ds_meta[v] = xr.DataArray(encoded_data, dims=ds_meta[v].dims)

    return ds_meta


def apply_attributes(ds_sumup, ds_meta, var):
    df_attr = pd.read_csv(f'doc/attributes_{var}.csv',
                          skipinitialspace=True,
                          comment='#').set_index('var')
    for v in df_attr.index:
        for c in df_attr.columns:
            if (v in ds_sumup.keys()) & (df_attr.loc[v,c] != np.nan):
                    ds_sumup[v].attrs[c] = df_attr.loc[v,c]
            if (v in ds_meta.keys()) & (df_attr.loc[v,c] != np.nan):
                    ds_meta[v].attrs[c] = df_attr.loc[v,c]
    return ds_sumup, ds_meta


def make_ds_global_attributes(var_name, region_name):
    ds_global_attributes = xr.Dataset()

    ds_global_attributes.attrs['title'] = f'SUMup {var_name} dataset for the {region_name} ice sheet (2024 release)'
    ds_global_attributes.attrs['contact'] = 'Baptiste Vandecrux'
    ds_global_attributes.attrs['email'] = 'bav@geus.dk'
    ds_global_attributes.attrs['production date'] = pd.Timestamp.now().strftime('%Y-%m-%d')

    ds_global_attributes.attrs['dataset citation'] = SUMUP_CITATION
    ds_global_attributes.attrs['DOI'] = SUMUP_DOI
    return ds_global_attributes

def append_metadata_to_netcdf(ds_meta, filename, var):
    ds_meta.to_netcdf(filename,
                       group='METADATA',mode='a',
                       encoding = {
                           var+"_key": INT_ENCODING,
                           "reference_key": INT_ENCODING,
                           "method_key": INT_ENCODING,
                            var+"": STR_ENCODING,
                            "method": STR_ENCODING,
                            "reference": STR_ENCODING,
                            "reference_short": STR_ENCODING,
                        }
                       )


def write_density_to_netcdf(df_sumup, filename):
    df_new = df_sumup.copy()
    df_new['timestamp'] = pd.to_datetime(df_new.timestamp).dt.tz_localize(None)

    df_new.index.name='measurement_id'
    assert (~df_new.index.duplicated()).all(), 'non-unique measurement-id "'

    ds_meta = make_key_ds(df_new, 'profile')

    ds_sumup = df_new[['profile_key', 'reference_key', 'method_key', 'timestamp',
           'latitude', 'longitude', 'elevation', 'start_depth',
           'stop_depth', 'midpoint', 'density', 'error']].to_xarray()

    ds_sumup['elevation'] = ds_sumup.elevation.astype(int)
    ds_sumup['error'] = ds_sumup['error'].astype(float)
    # ds_sumup['notes'] = ds_sumup['notes'].astype(str)

    ds_sumup, ds_meta = apply_attributes(ds_sumup, ds_meta, 'density')

    ds_sumup['timestamp'].encoding = {}
    del ds_sumup['timestamp'].attrs['units']

    if ds_sumup.latitude.isel(measurement_id=0)>0:
        make_ds_global_attributes('density','Greenland').to_netcdf(filename)
    else:
        make_ds_global_attributes('density','Antarctica').to_netcdf(filename)

    ds_sumup.to_netcdf(filename,
                       group='DATA',
                       mode='a',
                       encoding={
                          "measurement_id": INT_ENCODING,
                          "timestamp": INT64_ENCODING|{'units': 'days since 1900-01-01'},
                          "density": FLOAT_ENCODING|{'least_significant_digit':2},
                          "start_depth": FLOAT_ENCODING|{'least_significant_digit':4},
                          "stop_depth": FLOAT_ENCODING|{'least_significant_digit':4},
                          "midpoint": FLOAT_ENCODING|{'least_significant_digit':4},
                          "error": FLOAT_ENCODING|{'least_significant_digit':4},
                          "longitude": FLOAT_ENCODING|{'least_significant_digit':4},
                          "latitude": FLOAT_ENCODING|{'least_significant_digit':4},
                          "elevation": INT_ENCODING,
                          "profile_key": INT_ENCODING,
                          "reference_key": INT_ENCODING,
                          "method_key": INT_ENCODING,
                          })
    append_metadata_to_netcdf(ds_meta, filename, "profile")


def write_temperature_to_netcdf(df_sumup, filename):
    df_new = df_sumup.copy()
    df_new['timestamp'] = pd.to_datetime(df_new.timestamp).dt.tz_localize(None)
    df_new.index.name='measurement_id'
    assert (~df_new.index.duplicated()).all(), 'non-unique measurement-id "'

    ds_meta = make_key_ds(df_new, 'name')

    ds_sumup = df_new.drop(columns=['name', 'method','reference','reference_short']).to_xarray()

    ds_sumup['elevation'] = ds_sumup.elevation.astype(int)
    ds_sumup['error'] = ds_sumup['error'].astype(float)
    ds_sumup['notes'] = ds_sumup['notes'].astype(str)

    ds_sumup, ds_meta = apply_attributes(ds_sumup, ds_meta, 'temperature')

    if ds_sumup.latitude.isel(measurement_id=0)>0:
        make_ds_global_attributes('temperature','Greenland').to_netcdf(filename)
    else:
        make_ds_global_attributes('temperature','Antarctica').to_netcdf(filename)

    del ds_sumup.timestamp.attrs['units']

    ds_sumup[['name_key', 'reference_key', 'method_key', 'timestamp',
              'latitude', 'longitude', 'elevation',
              'temperature',  'depth', 'duration','open_time',
              'error']].to_netcdf(filename,
                                  group='DATA',
                                  mode='a',
                                  encoding={
                                     "temperature": FLOAT_ENCODING |{'least_significant_digit':1},
                                     "depth": FLOAT_ENCODING |{'least_significant_digit':1},
                                      "timestamp": INT64_ENCODING|{'units': 'days since 1900-01-01'},
                                     "duration": INT_ENCODING,
                                     "open_time": INT_ENCODING,
                                     "error": FLOAT_ENCODING|{'least_significant_digit':1},
                                     "longitude": FLOAT_ENCODING|{'least_significant_digit':6},
                                     "latitude": FLOAT_ENCODING|{'least_significant_digit':6},
                                     "elevation": INT_ENCODING,
                                     "name_key": INT_ENCODING,
                                     "reference_key": INT_ENCODING,
                                     "method_key": INT_ENCODING,
                                     })
    append_metadata_to_netcdf(ds_meta, filename, "name")


def write_smb_to_netcdf(df_sumup, filename):
    df_new = df_sumup.copy()
    df_new['start_date'] = pd.to_datetime(df_new.start_date).dt.tz_localize(None)
    df_new['end_date'] = pd.to_datetime(df_new.end_date).dt.tz_localize(None)

    df_new.index.name='measurement_id'
    assert (~df_new.index.duplicated()).all(), 'non-unique measurement-id "'

    ds_meta = make_key_ds(df_new, 'name')

    ds_sumup = df_new.drop(columns=['name', 'method','reference','reference_short']).to_xarray()

    ds_sumup['elevation'] = ds_sumup.elevation.astype(int)
    ds_sumup['error'] = ds_sumup['error'].astype(float)
    ds_sumup['notes'] = '' #ds_sumup['notes'].astype(str)

    # ds_sumup.start_date.encoding['units'] = 'days since 1900-01-01'
    # ds_sumup.end_date.encoding['units'] = 'days since 1900-01-01'
    ds_sumup, ds_meta = apply_attributes(ds_sumup, ds_meta, 'smb')

    del ds_sumup.start_date.attrs['units']
    del ds_sumup.end_date.attrs['units']

    if ds_sumup.latitude.isel(measurement_id=0)>0:
        make_ds_global_attributes('SMB','Greenland').to_netcdf(filename)
    else:
        make_ds_global_attributes('SMB','Antarctica').to_netcdf(filename)

    ds_sumup[['name_key', 'reference_key', 'method_key', 'start_date', 'end_date',
              'start_year', 'end_year','latitude', 'longitude', 'elevation',  'smb',
              'error']].to_netcdf(filename,
                                  group='DATA',
                                  mode='a',
                                  encoding={
                                     "smb": FLOAT_ENCODING |{'least_significant_digit':4},
                                       "start_date": INT64_ENCODING|{'units': 'days since 1900-01-01'},
                                       "end_date": INT64_ENCODING|{'units': 'days since 1900-01-01'},
                                     "start_year": INT_ENCODING,
                                     "end_year": INT_ENCODING,
                                     "error": FLOAT_ENCODING|{'least_significant_digit':4},
                                     "longitude": FLOAT_ENCODING|{'least_significant_digit':6},
                                     "latitude": FLOAT_ENCODING|{'least_significant_digit':6},
                                     "elevation": INT_ENCODING,
                                     "name_key": INT_ENCODING,
                                     "reference_key": INT_ENCODING,
                                     "method_key": INT_ENCODING,
                                     "measurement_id": INT64_ENCODING,
                                     })
    append_metadata_to_netcdf(ds_meta, filename, "name")

import re
from simplekml import Kml
import pandas as pd

def clean_text(text):
    """Replace problematic characters for KML formatting and remove non-ASCII characters."""
    if pd.isnull(text):  # Handle NaN values
        return ""
    # Remove non-printable and non-ASCII characters
    text = re.sub(r'[^\x20-\x7E]+', '', str(text))
    # Replace XML-special characters
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def create_kmz(df_sumup,  output_prefix="output", chunk_size=2000):
    name_var = 'profile' if 'profile' in df_sumup.columns else 'name'
    time_var_start = 'timestamp' if 'timestamp' in df_sumup.columns else 'start_year'
    time_var_end = 'timestamp' if 'timestamp' in df_sumup.columns else 'end_year'

    # Degrade precision for latitude and longitude to group points within ~111 m radius
    df_sumup_degraded = df_sumup.copy()
    df_sumup_degraded['latitude'] = df_sumup_degraded['latitude'].round(3)
    df_sumup_degraded['longitude'] = df_sumup_degraded['longitude'].round(3)

    # Group to get timespan and details
    groups_ref = df_sumup_degraded.groupby(
        ['latitude', 'longitude', 'elevation', name_var, 'reference']
    ).agg(
        timespan_start=(time_var_start, 'min'),
        timespan_end=(time_var_end, 'max')
    ).reset_index()

    # Build the timespan and details for each unique combination
    groups_ref['timespan'] = groups_ref['timespan_start'].astype(str) + " - " + groups_ref['timespan_end'].astype(str)
    groups_ref['details'] = (
        "[" + groups_ref[name_var].astype(str) + ", coverage: " + groups_ref['timespan'] +
        ", " + groups_ref['elevation'].astype(str) + " m a.s.l., " + groups_ref['reference'].astype(str) + "]"
    )

    # Second aggregation by latitude and longitude to combine names and details
    df_unique = (
        groups_ref.groupby(['latitude', 'longitude']).agg(
            names=(name_var, lambda x: ' / '.join(x.unique())),
            details=('details', lambda x: ',\n'.join(x))
        ).reset_index()
    )

    # Split `df_unique` into chunks
    num_chunks = len(df_unique) // chunk_size + (1 if len(df_unique) % chunk_size != 0 else 0)

    for i in range(num_chunks):
        kml = Kml()
        chunk = df_unique.iloc[i * chunk_size : (i + 1) * chunk_size]

        # Add points to each chunk's KML
        for _, row in chunk.iterrows():
            pnt = kml.newpoint(
                coords=[(row['longitude'], row['latitude'])],
                name=row['names']
            )
            pnt.extendeddata.newdata(name="description", value=clean_text(row['details']))

        # Save each chunk as a separate KMZ file
        output_file = f"{output_prefix}_{i + 1}.kmz"
        try:
            kml.savekmz(output_file)
            print(f"KMZ file created: {output_file} containing {len(chunk)} points")
        except Exception as e:
            print(f"Error saving KMZ file: {e}")
