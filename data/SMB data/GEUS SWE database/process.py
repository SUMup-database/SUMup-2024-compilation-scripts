import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nead
col_needed = ['method_key','start_date', 'end_date', 'start_year','end_year', 'latitude', 
              'longitude', 'elevation', 'notes', 'smb', 'error', 'name', 
              'reference', 'method', 'reference_short']
def process():

    df = pd.read_csv('GEUS_SWE_compilation.csv')
    
    df = df.rename(columns={
                            'swe_mm':'smb',
                            'swe_error':'error',
                            'type': 'method',
                            'source':'reference_short',
                            'publication': 'reference',
                            })
    
    df['start_date'] = pd.to_datetime(df.start_time, errors='coerce')
    start_date = pd.to_datetime(df[['start_year', 'start_month', 'start_day']].rename(
            columns={'start_year':'year', 'start_month':'month', 'start_day':'day'}),
        errors='coerce')
    df['start_date'] = df['start_date'].fillna(start_date)
    
    df['end_date'] = pd.to_datetime(df.end_time, errors='coerce')
    end_date = pd.to_datetime(df[['end_year', 'end_month', 'end_day']].rename(
            columns={'end_year':'year', 'end_month':'month', 'end_day':'day'}),
        errors='coerce')
    df['end_date'] = df['end_date'].fillna(end_date)
    
    df['longitude'] = -df.longitude.abs()
    df['smb'] = df['smb']/1000
    df['error'] = df['error']/1000
    
    # List of NaNs in the SWE compilation. Correspond to suspicious values
    # print(df.loc[df.smb.isnull(), ['name','end_year','suspision']])
    df = df.loc[df.smb.notnull()]
    
    df.loc[df.reference.astype(str).str.startswith('Hermann'), 'reference'] =  \
     'Hermann, M., Box, J. E., Fausto, R. S., Colgan, W. T., Langen, P. L., Mottram, R., et al. (2018). Application of PROMICE Q-transect in situ accumulation and ablation measurements (2000–2017) to constrain mass balance at the southern tip of the Greenland ice sheet. Journal of Geophysical Research: Earth Surface, 123, 1235–1256. https://doi.org/10.1029/2017JF004408'
    df.loc[df.reference.astype(str).str.startswith('Hermann'), 'reference_short'] =  \
        'Hermann et al. (2018)'
        
    df.loc[df.reference_short.astype(str).str.startswith('Kjær'), 'reference_short'] =  \
        'Kjær et al. (2021)'
    df.loc[df.reference_short.astype(str).str.startswith('Kjær'), 'reference'] =  \
    'Kjær, H. A., Zens, P., Edwards, R., Olesen, M., Mottram, R., Lewis, G., Terkelsen Holme, C., Black, S., Holst Lund, K., Schmidt, M., Dahl-Jensen, D., Vinther, B., Svensson, A., Karlsson, N., Box, J. E., Kipfstuhl, S., and Vallelonga, P.: Recent North Greenland temperature warming and accumulation, The Cryosphere Discuss. [preprint], https://doi.org/10.5194/tc-2020-337, 2021.'
    
    
    df.loc[df.name.astype(str).str.startswith('NANOK'), 'reference'] =  'GEUS unpublished'
    df.loc[df.name.astype(str).str.startswith('NANOK'), 'reference_short'] =  'GEUS unpublished'
           
    df.loc[df.reference_short == 'Box/Wei/Mosley-Thompson', 'reference_short'] =  \
        'Burgress et al. (2010)'
    df.loc[df.reference == 'Box/Wei/Mosley-Thompson', 'reference'] =  \
    'Burgess, E. W., Forster, R. R., Box, J. E., Mosley-Thompson, E., Bromwich, D. H., Bales, R. C., and Smith, L. C. (2010), A spatially calibrated model of annual accumulation rate on the Greenland Ice Sheet (1958–2007), J. Geophys. Res., 115, F02004, doi:10.1029/2009JF001293. '
    
    df.loc[df.reference_short == 'Schaller et al 2016', 'reference_short'] =  \
        'Schaller et al. (2016)'
    df.loc[df.reference_short == 'Schaller et al. (2016)', 'reference'] =  \
    'Schaller, C. F., Freitag, J., Kipfstuhl, S., Laepple, T., Steen-Larsen, H. C., and Eisen, O.: A representative density profile of the North Greenland snowpack, The Cryosphere, 10, 1991–2002, https://doi.org/10.5194/tc-10-1991-2016, 2016.'
    
    df.loc[df.reference_short.astype(str).str.startswith('Niwano et al. (2020)'), 'reference_short'] =  'Niwano et al. (2020)'
    
    df.loc[df.notes.isnull(), 'notes'] =  ''
    
    df.loc[df.reference_short.isnull(), 'reference_full'] =  'GEUS unpublished'
    df.loc[df.reference_short.isnull(), 'reference_short'] =  'GEUS unpublished'
    
    df.loc[df.reference.isnull(), 'reference_short'] =  'GEUS unpublished'
    df.loc[df.reference.isnull(), 'reference'] =  'GEUS unpublished'
        
    for ref in ['Box/GEUS Q-transect', 'Steffen/Box/Albert/Cullen/Huff/Weber/Starkweather/Molotch/Vandecrux',
                'Colgan/GEUS', 'Steffen/Cullen/Huff/Colgan/Box/Vandecrux', 'GEUS unpublished',
                'Box/Niwano', 'Braithwaite-GGU', 'ACT-PROMICE', 'Summit']:
        df.loc[df.reference_short == ref, 'notes'] =  df.loc[df.reference_short == ref, 'notes'].astype(str) + ' from ' + ref
        df.loc[df.reference_short == ref, 'reference'] = 'GEUS unpublished'
        df.loc[df.reference_short == ref, 'reference_short'] =  'GEUS unpublished'
       
    df['method_key'] = -9999
    df.loc[df.regime=='ablation',  'method_key'] = 3
    df.loc[df.regime=='ablation',  'method'] = 'stake measurements'
    df.loc[df.method=='core',  'method_key'] = 5
    df.loc[df.method=='core',  'method'] = 'firn or ice core'
    df.loc[df.method=='snowpit',  'method_key'] = 4
    df.loc[df.method=='snowpit',  'method'] = 'snow pits'
    df.loc[df.method=='pit',  'method_key'] = 4
    df.loc[df.method=='pit',  'method'] = 'snow pits'
    # check_duplicates (df, df_sumup, plot=False)
                
    for v in col_needed:
        assert v in df.columns, f'{v} is missing'
    df[col_needed].to_csv('data_formatted.csv',index=None)

if __name__ == "__main__":
    process()
    