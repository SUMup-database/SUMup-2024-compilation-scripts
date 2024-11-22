# -*- coding: utf-8 -*-
"""
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
decode_utf8 = np.vectorize(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
for v in ['name','reference','reference_short','method']:
    ds_meta[v] = xr.DataArray(decode_utf8(ds_meta[v].values), dims=ds_meta[v].dims)


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

ds_mar = xr.open_dataset('ancil/SMB_mean_1990-2020_greenland.nc')

df_sumup['start_date'] = pd.to_datetime(df_sumup['start_date'])
df_sumup['end_date'] = pd.to_datetime(df_sumup['end_date'])

mask = (df_sumup['end_year'] == df_sumup['start_year']) & df_sumup['start_date'].isnull()
df_sumup.loc[mask,'end_year'] = df_sumup.loc[mask, 'start_year']+1

df_sumup['acc'] = np.where(
    df_sumup['start_date'].notnull() & df_sumup['end_date'].notnull(),
    df_sumup['smb'] / ((df_sumup['end_date'] - df_sumup['start_date']).dt.total_seconds() / (365.25 * 24 * 3600)),
    df_sumup['smb'] / (df_sumup['end_year'] - df_sumup['start_year'])
)

import rioxarray
import xarray as xr
import pandas as pd

# Ensure MAR dataset has a CRS defined
ds_mar = ds_mar.rio.write_crs("EPSG:3413")  # Define the original CRS

# Reproject MAR data to EPSG:4326
ds_mar_4326 = ds_mar.rio.reproject("EPSG:4326")

# Sample SMB_mean at points in df_sumup
df_sumup["SMB_mean"] = ds_mar_4326.SMB_mean.sel(
    x=xr.DataArray(df_sumup["longitude"], dims="points"),
    y=xr.DataArray(df_sumup["latitude"], dims="points"),
    method="nearest"
).values/1000

# Compare SMB_mean with acc
df_sumup["difference"] = df_sumup["SMB_mean"] - df_sumup["acc"]

# Check results
print(df_sumup.head())


# %%
import matplotlib.pyplot as plt
import numpy as np

# Calculate mean difference for each reference
mean_diff = df_sumup.groupby("reference_short")["difference"].mean()

# Rank references by mean difference
ranked_references = mean_diff.sort_values(ascending=False)

# Get unique references in ranked order
unique_references = ranked_references.index
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_references)))  # Use a colormap

for ref, color in zip(unique_references, colors):
    plt.figure(figsize=(10, 8))
    mask = df_sumup["reference_short"] == ref
    plt.scatter(
        df_sumup.loc[mask, "SMB_mean"],
        df_sumup.loc[mask, "acc"],
        alpha=0.7,
        s=10,
        label=f"{ref} (Mean Diff: {ranked_references[ref]:.2f})",
        color=color,
    )

    # Add one-to-one line
    x_min, x_max = plt.xlim()  # Get current x-axis limits
    plt.plot([x_min, x_max], [x_min, x_max], color="black", linestyle="--", linewidth=1, label="1:1 Line")

    plt.xlabel("SMB_mean (MAR)")
    plt.ylabel("acc (observed)")
    plt.title(f"Comparison of SMB_mean and acc for {ref}")
    plt.grid(True)

    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()



# Display ranked references and their mean differences
print(ranked_references.to_markdown())
