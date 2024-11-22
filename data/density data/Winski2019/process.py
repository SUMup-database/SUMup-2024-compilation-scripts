import os
import pandas as pd
import numpy as np

col_needed = ['profile', 'reference_short', 'reference', 'method_key', 'method', 'date', 'timestamp', 
              'latitude', 'longitude', 'elevation', 'start_depth', 'stop_depth', 'midpoint', 'density', 'error']

def read_spicecore_file(filepath):
    df = pd.read_csv(filepath, skiprows=6)  # Skip metadata lines
    df.columns =['midpoint_m',  'density_g_cm3', 'core']
    
    # Convert density to kg/m3 and calculate start and stop depth
    df['density'] = df['density_g_cm3'].astype(float) * 1000
    df['midpoint'] = df['midpoint_m'].astype(float)
    df['start_depth'] = df['midpoint'] - 0.5
    df['stop_depth'] = df['midpoint'] + 0.5
    
    return df

def process():
    # Load SPICEcore data
    filepath = 'SP19_Density.csv'  # Update to actual path
    df = read_spicecore_file(filepath)
    
    # Create formatted DataFrame
    df_formatted = pd.DataFrame()
    df_formatted['start_depth'] = df['start_depth']
    df_formatted['stop_depth'] = df['stop_depth']
    df_formatted['density'] = df['density']
    df_formatted['midpoint'] = df['midpoint']
    
    # Add metadata
    df_formatted['profile'] = 'South Pole Ice Core'
    df_formatted['timestamp'] = pd.to_datetime('2015-11-01').strftime('%Y-%m-%d')
    df_formatted['date'] = 20151101
    df_formatted['latitude'] = -90
    df_formatted['longitude'] = -180
    df_formatted['elevation'] = 2835
    
    df_formatted['method_key'] = 4
    df_formatted['method'] = 'ice or firn core section'
    df_formatted['reference_short'] = 'Winski et al. (2019)'
    df_formatted['reference'] = ('Winski, D. A., Alley, R., et al. (2019). The South Pole Ice Core (SPICEcore) '
                                 'chronology and supporting data. U.S. Antarctic Program (USAP) Data Center. '
                                 'doi: https://doi.org/10.15784/601206')
    df_formatted['error'] = np.nan  # No error data available

    # Ensure all necessary columns are present
    for v in col_needed:
        assert v in df_formatted.columns, f'{v} is missing'

    df_formatted[col_needed].to_csv('data_formatted.csv', index=None)

if __name__ == "__main__":
    process()
