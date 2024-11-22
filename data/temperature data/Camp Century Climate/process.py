import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import os
import xarray as xr

def plot_string_dataframe(df_stack, filename):


    f=plt.figure(figsize=(9,9))
    sc=plt.scatter(df_stack.timestamp,
                -df_stack.depth,
                12, df_stack.temperature)
    plt.ylim(-df_stack.depth.max(), 0)
    plt.title(df_stack.site.unique().item())
    plt.colorbar(sc)
    f.savefig(filename+'.png', dpi=400)
    plt.show()


plot = False
needed_cols = ["timestamp", "name", "latitude", "longitude", "elevation", 'duration','method',
               "depth", "temperature", "reference", "reference_short", "note", 'error']

df = pd.read_csv("data_long.txt", sep=",", header=None)
df = df.rename(columns={0: "timestamp"})
df["timestamp"] = pd.to_datetime(df.timestamp)
df[df == -999] = np.nan
df = df.set_index("timestamp").resample("D").mean()
df = df.iloc[:, :-2]

df_promice = pd.read_csv("CEN_day.csv")   
df_promice["timestamp"] = pd.to_datetime(df_promice.time)
df_promice = df_promice.set_index("timestamp")["z_surf_combined"]
df_promice = df_promice.resample('D').mean().interpolate()

temp_label = ["T_" + str(i + 1) for i in range(len(df.columns))]
depth_label = ["depth_" + str(i + 1) for i in range(len(df.columns))]

df["surface_height"] = df_promice.loc[df.index[0].strftime('%Y-%m-%d'):df.index[-1].strftime('%Y-%m-%d')].values

depth = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 53, 58, 63, 68, 73]

for i in range(len(temp_label)):
    df = df.rename(columns={i + 1: temp_label[i]})
    df[depth_label[i]] = (
        depth[i] + df["surface_height"].values - df["surface_height"].iloc[0]
    )
df_stack = ( df[temp_label]
            .stack(future_stack=True)
            .to_frame(name='temperature')
            .reset_index()
            .rename(columns={'time':'time'}))
df_stack['depth'] = df[depth_label].stack(future_stack=True).values
df_stack=df_stack.loc[df_stack.temperature.notnull()]
df_stack=df_stack.loc[df_stack.depth.notnull()]

df_stack.loc[df_stack.temperature>-15, "temperature"] = np.nan
df_stack = df_stack.set_index("timestamp").reset_index()
site="CEN_THM_long"
df_stack["name"] = site
df_stack["latitude"] = 77.1333
df_stack["longitude"] = -61.0333
df_stack["elevation"] = 1880
df_stack["note"] = ""
df_stack[
    "reference"
] = "Vandecrux, B., Colgan, W., Solgaard, A.M., Steffensen, J.P., and Karlsson, N.B.(2021). Firn evolution at Camp Century, Greenland: 1966-2100, Frontiers in Earth Science, https://doi.org/10.3389/feart.2021.578978, 2021 dataset: https://doi.org/10.22008/FK2/SR3O4F"
df_stack["reference_short"] = "Vandecrux et al. (2021); Colgan and Vandecrux (2021)"
df_stack["method"] = "thermistors"
df_stack["durationOpen"] = 0
df_stack["durationMeasured"] =1
df_stack["error"] = 0.2
df_stack["duration"] = 1

if plot:
    plot_string_dataframe(df_stack, site)
    
df_all = df_stack[needed_cols]

df = pd.read_csv("data_short.txt", sep=",", header=None)
df = df.rename(columns={0: "timestamp"})
df["timestamp"] = pd.to_datetime(df.timestamp)
df[df == -999] = np.nan
df = df.set_index("timestamp").resample("D").mean()

temp_label = ["T_" + str(i + 1) for i in range(len(df.columns))]
depth_label = ["depth_" + str(i + 1) for i in range(len(df.columns))]
df["surface_height"] = df_promice.loc[df.index[0].strftime('%Y-%m-%d'):df.index[-1].strftime('%Y-%m-%d')].values

depth = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 17, 19, 22, 25, 28, 31, 34, 38, 42, 46, 50, 54]

for i in range(len(temp_label)):
    df = df.rename(columns={i + 1: temp_label[i]})
    df[depth_label[i]] = (
        depth[i] + df["surface_height"].values - df["surface_height"].iloc[0]
    )

df_stack = ( df[temp_label]
            .stack(future_stack=True)
            .to_frame(name='temperature')
            .reset_index()
            .rename(columns={'time':'time'}))
df_stack['depth'] = df[depth_label].stack(future_stack=True).values
df_stack=df_stack.loc[df_stack.temperature.notnull()]
df_stack=df_stack.loc[df_stack.depth.notnull()]

df_stack.loc[df_stack.temperature > -15, "temperature"] = np.nan
site="CEN_THM_short"
df_stack["name"] = site
df_stack["latitude"] = 77.1333
df_stack["longitude"] = -61.0333
df_stack["elevation"] = 1880
df_stack["note"] = ""
df_stack[
    "reference"
] = "Vandecrux, B., Colgan, W., Solgaard, A.M., Steffensen, J.P., and Karlsson, N.B.(2021). Firn evolution at Camp Century, Greenland: 1966-2100, Frontiers in Earth Science, https://doi.org/10.3389/feart.2021.578978, 2021 dataset: https://doi.org/10.22008/FK2/SR3O4F"

df_stack["reference_short"] = "Vandecrux et al. (2021); Colgan and Vandecrux (2021)"
df_stack["method"] = "thermistors"
df_stack["durationOpen"] = 0
df_stack["durationMeasured"] = 1
df_stack["error"] = 0.2
df_stack["duration"] = 1

if plot:
    plot_string_dataframe(df_stack, site)
    
df_all = pd.concat((df_all,  df_stack[needed_cols] ), ignore_index=True )

df_all.to_csv('data_formatted.csv')