import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
import pandas as pd
import xarray as xr
import os, re
from datetime import datetime, timedelta

col_needed = ['start_date', 'end_date', 'start_year','end_year', 'latitude', 
              'longitude', 'elevation', 'notes', 'smb', 'error', 'name', 
              'reference', 'method', 'reference_short']


# def process():
    # %% 
import pandas as pd

# Load the Excel file
results = []
for file_path in ["PARCA-1997-cores.xls","PARCA-1998-cores_bapt.xls"]:
    xls = pd.ExcelFile(file_path)
    
    # Initialize lists to store extracted data
    
    # Loop through all sheets in the file
    for sheet_name in xls.sheet_names:
        if sheet_name.startswith('Read'):
            continue
        if sheet_name=='NDYE3C':
            continue
        df = xls.parse(sheet_name, header=None)
        
        # Extract name, latitude, longitude
        core_name = df.iloc[0, 0]
        lat = float(df.iloc[1, 3].split()[0])
        lon = -np.abs(float(df.iloc[2, 3].split()[0].replace('W','')))
        elev = float(df.iloc[3, 3].split()[0].replace('~',''))
        print(sheet_name, core_name, lat, lon, elev)
        
        # Extract accumulation data
        method = pd.Series(["dust",
                               "dO18", 
                               "H2O2"],index=[6,7,8])
        for col in [6,7,8]:
            accumulation_data = df.iloc[5:, [5,col]].copy()
            accumulation_data.columns = ["start_year", "smb"]
            accumulation_data['start_year'] = accumulation_data.start_year.astype(str).str.replace('*','')
            accumulation_data['start_year'] = pd.to_numeric(accumulation_data['start_year'], errors='coerce')   
            invalid_index = accumulation_data.start_year.apply(lambda x: not isinstance(x, (int, float)) or pd.isna(x)).idxmax()
            accumulation_data = accumulation_data.loc[:invalid_index-1].reset_index(drop=True)
            if len(accumulation_data)==0:
                print('no',method.loc[col] ,'data')
                continue
            elif accumulation_data.smb.isnull().all():
                print('no',method.loc[col] ,'data')
                continue
            else:
                print('found',method.loc[col] ,'data')
            
            # sometime one of the method doesnt cover the entire core, so removing invalid data
            invalid_index = accumulation_data.smb.apply(lambda x: not isinstance(x, (int, float)) or pd.isna(x)).idxmax()
            accumulation_data = accumulation_data.loc[:invalid_index-1].reset_index(drop=True)
            
            accumulation_data['smb'] = accumulation_data.smb.astype(float)
            accumulation_data['end_year'] = accumulation_data.start_year
            accumulation_data[['start_date','end_date','notes']] = ''
            accumulation_data['error']=np.nan
            accumulation_data['latitude']=lat
            accumulation_data['longitude']=lon
            accumulation_data['elevation']=elev
            accumulation_data['name']=core_name.strip()+' - '+method.loc[col] 
            accumulation_data['method'] = f"firn or ice core, {method.loc[col]} dating"
            if '1998' in file_path:
                accumulation_data.loc[accumulation_data.start_year==1998,'end_date'] = '1998-07-01'
                accumulation_data.loc[accumulation_data.start_year==1998,'notes'] = 'not all year measured'

            if '1997' in file_path:
                accumulation_data.loc[accumulation_data.start_year==1997,'end_date'] = '1997-07-01'
                accumulation_data.loc[accumulation_data.start_year==1997,'notes'] = 'not all year measured'


            results.append(accumulation_data)
# %%
# Combine all dataframes
final_df = pd.concat(results, ignore_index=True)
final_df['reference'] ='Mosley-Thompson, E., McConnell, J. R., Bales, R. C., Li, Z., Lin, P.-N., Steffen, K., Thompson, L. G., Edwards, R., and Bathke, D. (2001), Local to regional-scale variability of annual net accumulation on the Greenland ice sheet from PARCA cores, J. Geophys. Res., 106(D24), 33839–33851, doi:10.1029/2001JD900067.'
final_df['reference_short'] ='Mosley-Thompson et al. (2001)'

for v in col_needed:
    assert v in final_df.columns, f'{v} is missing'
    
final_df.to_csv('data_formatted.csv',index=None)

    
# if __name__ == "__main__":
#     process()
    
    
    

