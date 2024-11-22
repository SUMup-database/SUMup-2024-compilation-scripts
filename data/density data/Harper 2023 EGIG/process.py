import os
import pandas as pd
import numpy as np

col_needed = ['profile',  'reference_short', 
              'reference', 'method_key','method', 'date', 'timestamp', 'latitude', 
              'longitude', 'elevation', 'start_depth', 'stop_depth',
              'midpoint', 'density', 'error']

def read_core_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Extract metadata
    core_info = {}
    core_info['profile'] = lines[0].split('\t')[1].strip()  # Name
    core_info['timestamp'] = lines[2].split('\t')[1].strip()  # Drilled date
    lat, lon = lines[1].split('\t')[1:3]  # Latitude and Longitude from the second line
    core_info['latitude'] = float(lat)
    core_info['longitude'] = float(lon)

    # Parse density data (from line 7 onwards)
    data = []
    for line in lines[7:]:
        fields = line.strip().split('\t')
        if len(fields) == 4:  # Check if the line has the expected number of columns
            data.append(fields)
    
    df = pd.DataFrame(data, columns=['start_depth_cm', 'stop_depth_cm', 'weight_g', 'density_kg/m3'])
    df = df.replace('nan', np.nan).dropna(subset=['density_kg/m3'])
    df['start_depth'] = df['start_depth_cm'].astype(float) / 100
    df['stop_depth'] = df['stop_depth_cm'].astype(float) / 100
    df['density'] = df['density_kg/m3'].astype(float)
    
    return df, core_info

def process():
    # Load elevation data
    elevation_df = pd.read_csv('Core_hole_locations_2023.txt', skiprows=3, delim_whitespace=True)
    
    data_folder = '.'  # Update to the actual path
    files = [f for f in os.listdir(data_folder) if f.endswith('_den.txt')]

    list_add = []

    for file in files:
        filepath = os.path.join(data_folder, file)
        df, core_info = read_core_file(filepath)
        
        # Create formatted DataFrame
        df_formatted = pd.DataFrame()
        df_formatted['start_depth'] = df['start_depth']
        df_formatted['stop_depth'] = df['stop_depth']
        df_formatted['density'] = df['density']
        
        # Calculate midpoint
        df_formatted['midpoint'] = df_formatted['start_depth'] + (df_formatted['stop_depth'] - df_formatted['start_depth']) / 2
        
        # Add core metadata
        profile_name = core_info['profile']
        print(profile_name)
        df_formatted['profile'] = profile_name
        df_formatted['timestamp'] = pd.to_datetime(core_info['timestamp']).strftime('%Y-%m-%d')
        df_formatted['date'] = int(pd.to_datetime(core_info['timestamp']).strftime('%Y%m%d'))
        df_formatted['latitude'] = core_info['latitude']
        df_formatted['longitude'] = -abs(core_info['longitude'])
        
        # Look up elevation data from the elevation file based on the profile name
        elevation_info = elevation_df.loc[elevation_df['Name'] == profile_name]
        if not elevation_info.empty:
            df_formatted['elevation'] = elevation_info['Altitude_m'].values[0]
        else:
            df_formatted['elevation'] = np.nan  # Handle missing elevation if needed
        
        df_formatted['method_key'] = 4
        df_formatted['method'] = 'ice or firn core section'
        df_formatted['reference_short'] = 'Harper and Humphrey (2023)'
        df_formatted['reference'] = 'Harper, J., Humphrey, N. (2023). Firn core density and ice content at sites along the lower Expéditions Glaciologiques Internationales au Groenland (EGIG) line, Western Greenland, 2023. Arctic Data Center. doi:10.18739/A2DB7VR82.'
        df_formatted['error'] = np.nan  # Add error column

        list_add.append(df_formatted)

    df_add = pd.concat(list_add)
    
    # Ensure all necessary columns are present
    for v in col_needed:
        assert v in df_add.columns, f'{v} is missing'

    df_add[col_needed].to_csv('data_formatted.csv', index=None)

if __name__ == "__main__":
    process()
