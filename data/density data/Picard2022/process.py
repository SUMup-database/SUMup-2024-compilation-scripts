import os
import pandas as pd
import numpy as np
import warnings

col_needed = ['profile', 'reference_short', 'reference', 'method_key', 'method', 'date', 'timestamp',
              'latitude', 'longitude', 'elevation', 'start_depth', 'stop_depth', 'midpoint', 'density', 'error']

def read_file(filepath):
    df = pd.read_csv(filepath)
    df.columns = ['end_depth', 'density']

    # Convert density to kg/m3
    df['density'] = df['density'].astype(float)

    # Convert depth to positive increasing downwards
    df['end_depth'] = df['end_depth'].abs()
    df['start_depth'] = df['end_depth'].shift(1, fill_value=0)
    df['midpoint'] = (df['start_depth'] + df['end_depth']) / 2

    return df

def process():
    files = ['ago5-eaiist.csv', 'charcot-asuma.csv', 'd47-asuma.csv', 'faus-asuma.csv', 'paleo-eaiist.csv',
             's2-vanish.csv', 's2b-vanish.csv', 's4-vanish.csv', 'sortie-asuma.csv', 'sp1_domec-bipol.csv',
             'sp2_domec-bipol.csv', 'stop0-asuma.csv', 'stop1-asuma.csv', 'stop2-asuma.csv', 'stop3-asuma.csv',
             'stop4a-asuma.csv', 'stop4b-asuma.csv', 'stop5-asuma.csv']

    metadata = pd.read_csv('coords.csv').set_index('site')
    reference = ('Larue, F., Picard, G., Aublanc, J., Arnaud, L., Robledano-Perez, A., Meur, E. L., Favier, V., '
                 'Jourdain, B., Savarino, J., & Thibaut, P. (2021). Radar altimeter waveform simulations in Antarctica '
                 'with the Snow Microwave Radiative Transfer Model (SMRT). Remote Sensing of Environment, 263, 112534. '
                 'doi:10.1016/j.rse.2021.112534. Data: Picard, G., Löwe, H., Arnaud, L., Larue, F., Favier, V., Le Meur, E., '
                 'Lefebvre, E., Savarino, J., Royer, A. Krol, Q. , Jourdain, B. (2022) Snow properties in Antarctica, Canada '
                 'and the Alps for microwave emission and backscatter modeling [Data set]')
    reference_short = 'Larue et al. (2021) Picard et al. (2022)'

    df_all = pd.DataFrame()

    for file in files:
        profile = file.split('.')[0]
        filepath = os.path.join('data/', file)
        df = read_file(filepath)

        df_formatted = pd.DataFrame()
        df_formatted['start_depth'] = df['start_depth']
        df_formatted['stop_depth'] = df['end_depth']
        df_formatted['density'] = df['density']
        df_formatted['midpoint'] = df['midpoint']

        # Add metadata
        df_formatted['profile'] = profile
        df_formatted['timestamp'] = pd.to_datetime(metadata.loc[[profile]].date.values[0]).strftime('%Y-%m-%d')  # Update as needed
        df_formatted['date'] = 20220101  # Update as needed

        # Check and add coordinates
        df_formatted['latitude'] = metadata.loc[[profile]].latitude.values[0]
        df_formatted['longitude'] = metadata.loc[[profile]].longitude.values[0]
        df_formatted['elevation'] = metadata.loc[[profile]].elevation.values[0]

        df_formatted['method_key'] = 4
        df_formatted['method'] = 'ice or firn core section'
        df_formatted['reference_short'] = reference_short
        df_formatted['reference'] = reference
        df_formatted['error'] = np.nan  # No error data available

        # Ensure all necessary columns are present
        for v in col_needed:
            if v not in df_formatted.columns:
                df_formatted[v] = np.nan

        df_all = pd.concat([df_all, df_formatted[col_needed]], ignore_index=True)

    df_all.to_csv('data_formatted.csv', index=None)

if __name__ == "__main__":
    process()
