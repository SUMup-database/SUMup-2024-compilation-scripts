    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
col_needed = ['method_key','start_date', 'end_date', 'start_year','end_year', 'latitude', 
              'longitude', 'elevation', 'notes', 'smb', 'error', 'name', 
              'reference', 'method', 'reference_short']
def process():
    
    df = pd.read_csv('Moulin_L41A_ice_ablation.tab',
                     skiprows=19, sep='\t')
    df['end_date'] = pd.to_datetime(df['Date/Time'] + 'T20:00:00').dt.tz_localize(-3).dt.tz_convert('UTC')
    df['start_date'] = df['end_date'] - pd.Timedelta('1D')
    df['start_year'] = df['end_date'].dt.year
    df['end_year'] = df['end_date'].dt.year
    df['smb'] = -df['Ablation [mm]']*1000
    df['error'] = df['Ablation [±]']*1000
    df['name'] = 'Leverett catchment'
    df['reference'] = "Chandler, D. M., Alcock, J. D., Wadham, J. L., Mackie, S. L., and Telling, J.: Seasonal changes of ice surface characteristics and productivity in the ablation zone of the Greenland Ice Sheet, The Cryosphere, 9, 487–504, https://doi.org/10.5194/tc-9-487-2015, 2015. Data: Chandler, David M; Wadham, Jemma; Nienow, Peter; Doyle, Samuel H; Tedstone, Andrew; Telling, Jon; Hawkings, Jonathan; Alcock, Jonathan; Linhoff, Benjamin; Hubbard, Alun L (2021): Ice ablation record from the Greenland Ice Sheet measured in spring/summer 2012 [dataset]. PANGAEA, https://doi.org/10.1594/PANGAEA.926842"
    df['reference_short'] = "Chandler et al. (2015, 2021)"
    df['latitude'] = 66.97
    df['longitude'] = -49.27
    df['elevation'] = 1030
    df['notes'] = 'manual, from evening to evening'
    df['method'] = 'stake measurements'
    df['method_key'] = 5
    
    for v in col_needed:
        assert v in df.columns, f'{v} is missing'
    df[col_needed].to_csv('data_formatted.csv', index=None)

if __name__ == "__main__":
    process()
    