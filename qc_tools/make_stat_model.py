# -*- coding: utf-8 -*-
"""
Created on %(date)s
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import geopandas as gpd

path_to_sumup = 'C:/Users/bav/GitHub/SUMup/SUMup-2024/SUMup 2024 beta/'
df_sumup = xr.open_dataset(path_to_sumup+'/SUMup_2024_SMB_greenland.nc',
                           group='DATA').to_dataframe()
ds_meta = xr.open_dataset(path_to_sumup+'/SUMup_2024_SMB_greenland.nc',
                           group='METADATA')

df_sumup.method_key = df_sumup.method_key.replace(np.nan,-9999)
df_sumup['method'] = ds_meta.method.sel(method_key = df_sumup.method_key.values).astype(str)
df_sumup['name'] = ds_meta.name.sel(name_key = df_sumup.name_key.values).astype(str)
df_sumup['reference'] = (ds_meta.reference
                         # .drop_duplicates(dim='reference_key')
                         .sel(reference_key=df_sumup.reference_key.values)
                         # .astype(str)
                         )
df_sumup['reference_short'] = (ds_meta.reference_short
                         # .drop_duplicates(dim='reference_key')
                         .sel(reference_key=df_sumup.reference_key.values)
                         # .astype(str)
                         )
df_ref = ds_meta.reference.to_dataframe()

# selecting Greenland metadata measurements
df_meta = df_sumup.loc[df_sumup.latitude>0,
                  ['latitude', 'longitude', 'name_key', 'name', 'method_key',
                   'reference_short','reference', 'reference_key']
                  ].drop_duplicates()

df_sumup.loc[df_sumup.start_year==df_sumup.end_year,'end_year'] = \
    df_sumup.loc[df_sumup.start_year==df_sumup.end_year,'end_year'] +1


# %%
from sklearn.ensemble import RandomForestRegressor

# Step 1: Feature engineering - use the midpoint of year_start and year_end
df_sumup['year_mid'] = (df_sumup['start_year'] + df_sumup['end_year']) / 2

df_sumup = df_sumup.loc[df_sumup.smb.notnull(),:]
# Step 2: Prepare features and target
X = df_sumup[['latitude', 'longitude', 'year_mid']].values
y = df_sumup['smb'].values

# Step 3: Train Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# Step 4: Query the model for new lat, lon, start_year, end_year
def predict_smb_rf(lat, lon, year_start, year_end):
    year_mid = (year_start + year_end) / 2
    X_new = np.array([[lat, lon, year_mid]])
    return rf.predict(X_new)[0]

# Example of querying the model
lat, lon, year_start, year_end = 70.0, -50.0, 1995, 2000
smb_pred = predict_smb_rf(lat, lon, year_start, year_end)
print(f"Predicted SMB: {smb_pred}")
