import os
import pandas as pd
import numpy as np

col_needed = ['profile', 'reference_short', 'reference', 'method_key', 'method', 'date', 'timestamp', 
              'latitude', 'longitude', 'elevation', 'start_depth', 'stop_depth', 'midpoint', 'density', 'error']

coordinates = {
    'B': (-81.66, -148.8120,'1996-12-10'),
    'C': (-81.66, -148.7943, '1996-12-15'),
    'D': (-81.65, -148.7860, '1997-01-11'),
    'E': (-81.30, -148.3023, '1996-12-27'),
    'F': (-81.91, -149.3370, '1996-12-05'),
    'G': (-81.57, -148.5975, '1997-01-13'),
    'H': (-81.74, -148.9768, '1997-01-15'),
    'I': (-81.65, -148.7860, '1997-01-16'),
    'J': (-81.93, -149.3763, '1997-01-16'),
}

reference_text = 'Jones, T. R., White, J. W. C., and Popp, T.: Siple Dome shallow ice cores: a study in coastal dome microclimatology, Clim. Past, 10, 1253–1267, https://doi.org/10.5194/cp-10-1253-2014, 2014. Data from: Lamorey, G. W. (2003) "Siple Shallow Core Density Data" U.S. Antarctic Program (USAP) Data Center. doi: https://doi.org/10.7265/N52F7KCD.'

reference_short = 'Jones et al. (2014); Lamorey (2003)'

def process_txt_file(file_path, site):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find the line containing "Depth(m)"
    start_idx = next(i for i, line in enumerate(lines) if "Depth(m)" in line) + 1

    # Extract data
    data = []
    for line in lines[start_idx:]:
        if line.strip():
            parts = line.split()
            depth, density = float(parts[0]), float(parts[1])
            data.append((depth, density))

    # Create DataFrame
    df_formatted = pd.DataFrame(data, columns=['midpoint', 'density'])
    
    # Calculate start and stop depths
    df_formatted['density'] = df_formatted['density'] * 1000
    df_formatted['start_depth'] = df_formatted['midpoint'] - 0.5
    df_formatted['stop_depth'] = df_formatted['midpoint'] + 0.5

    # Add metadata
    df_formatted['profile'] = 'Siple Dome shallow core ' +os.path.basename(file_path).replace('_density.txt', '').capitalize()
    df_formatted['reference_short'] = reference_short
    df_formatted['reference'] = reference_text
    df_formatted['method_key'] = 4
    df_formatted['method'] = 'ice or firn core section'
    df_formatted['timestamp'] = coordinates[site.capitalize()][2]
    df_formatted['date'] = int(coordinates[site.capitalize()][2].replace('-',''))
    df_formatted['latitude'], df_formatted['longitude'] = coordinates[site.capitalize()][:2]
    df_formatted['elevation'] = np.nan  # Assuming no elevation info provided
    df_formatted['error'] = np.nan

    return df_formatted

def process_all():
    list_add = []
    files = [f for f in os.listdir() if f.endswith('_density.txt')]  # Assuming files are in the current directory
    for file_path in files:
        site = file_path.split('_')[0]  # Assuming the site code is in the filename
        df_formatted = process_txt_file(file_path, site)
        list_add.append(df_formatted)

    df_add = pd.concat(list_add)
    for v in col_needed:
        assert v in df_add.columns, f'{v} is missing'
    df_add[col_needed].to_csv('data_formatted.csv', index=None)

if __name__ == "__main__":
    process_all()
