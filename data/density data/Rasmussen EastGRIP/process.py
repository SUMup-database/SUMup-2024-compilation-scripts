import os
import pandas as pd
import numpy as np

# Define the columns needed
col_needed = ['profile', 'reference_short', 
              'reference', 'method_key', 'method', 'date', 'timestamp', 'latitude', 
              'longitude', 'elevation', 'start_depth', 'stop_depth',
              'midpoint', 'density', 'error']

# Hard-coded values for the dataset
REFERENCE = 'Rasmussen, Sune Olander; Vinther, Bo Møllesøe; Freitag, Johannes; Kipfstuhl, Sepp (2023): EastGRIP snow-pack and ice-core densities (measured and modelled) [dataset bundled publication]. PANGAEA, https://doi.org/10.1594/PANGAEA.962754'
REFERENCE_SHORT = 'Rasmussen et al. (2023)'
METHOD_KEY = 4
METHOD = 'ice or firn core section'
LATITUDE = 75.630000
LONGITUDE = -36.000000
ELEVATION = 2704.0

def read_data(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find the end of the header
    header_end = None
    for i, line in enumerate(lines):
        if "*/" in line:
            header_end = i
            break

    if header_end is None:
        raise ValueError("Header end marker '*/' not found in file")

    # Column names are in the line immediately after the header
    column_names_line = lines[header_end + 1].strip()
    column_names = column_names_line.split('\t')

    # Read data after column names
    data_lines = lines[header_end + 2:]
    
    data = []
    for line in data_lines:
        fields = line.strip().split('\t')
        if len(fields) >= len(column_names):  # Ensure we have enough columns
            data.append(fields)
    
    return column_names, data

def process_file(filepath, profile_name, timestamp, depth_is_middepth, section_length):
    column_names, data = read_data(filepath)
    
    
    df = pd.DataFrame(data, columns=column_names)
    df =df.rename(columns={'Density ice [kg/m**3]':'density', 'Depth ice/snow [m]':'depth'})
    df = df.replace('nan', np.nan).dropna(subset=['density'])
    df['density'] = df['density'].astype(float)
    if depth_is_middepth:
        df['start_depth'] = df['depth'].astype(float) - section_length / 2
        df['stop_depth'] = df['depth'].astype(float) + section_length / 2
    else:
        df['start_depth'] = df['depth'].astype(float)
        df['stop_depth'] = df['depth'].astype(float) + section_length
    
    df['midpoint'] = df['start_depth'] + (df['stop_depth'] - df['start_depth']) / 2
    df['profile'] = profile_name
    df['timestamp'] = pd.to_datetime(timestamp).strftime('%Y-%m-%d')
    df['date'] = int(pd.to_datetime(timestamp).strftime('%Y%m%d'))
    df['latitude'] = LATITUDE
    df['longitude'] = LONGITUDE
    df['elevation'] = ELEVATION
    df['method_key'] = METHOD_KEY
    df['method'] = METHOD
    df['reference_short'] = REFERENCE_SHORT
    df['reference'] = REFERENCE
    df['error'] = np.nan

    return df

def process():
    # Define paths to your files
    files_info = [
        ('EGRIP_main_core_density_measurements.tab', 'EastGRIP Main Core', '2016-06-01', True, 0.55),
        ('EGRIP_S6_core_density_measurements.tab', 'EastGRIP S6 Core', '2018-06-12', False, 1.0),
        ('EGRIP_trench_density_measurements.tab', 'EastGRIP Trench', '2016-06-01', False, 1.0)
    ]

    list_add = []

    for file_info in files_info:
        filepath, profile_name, timestamp, depth_is_middepth, section_length = file_info
        if os.path.exists(filepath):
            df_formatted = process_file(filepath, profile_name, timestamp, depth_is_middepth, section_length)
            list_add.append(df_formatted)
        else:
            print(f"File {filepath} does not exist.")
    
    df_add = pd.concat(list_add, ignore_index=True)
    
    # Check if all columns are present
    for v in col_needed:
        if v not in df_add.columns:
            print(f'{v} is missing')
    
    df_add[col_needed].to_csv('data_formatted.csv', index=None)

if __name__ == "__main__":
    process()
