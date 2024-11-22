
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
    plt.title(df_stack.name.unique().item())
    plt.colorbar(sc)
    f.savefig(filename+'.png', dpi=400)
    plt.show()


plot = False
needed_cols = ["timestamp", "name", "latitude", "longitude", "elevation", 'duration','method',
               "depth", "temperature", "reference", "reference_short", "note", 'error']
df_all = pd.DataFrame()
# %% loading Hills
# del df_stack, df_name, df_hs, df_humphrey, depth, depth_label, df,  i, name, temp_label
print("Loading Hills")
df_meta = pd.read_csv("./metadata.txt", sep=" ")
df_meta.date_start = pd.to_datetime(df_meta.date_start, format="%m/%d/%y")
df_meta.date_end = pd.to_datetime(df_meta.date_end, format="%m/%d/%y")

df_meteo = pd.read_csv("./Hills_33km_meteorological.txt", sep="\t")

df_meteo["timestamp"] = pd.to_datetime("2014-07-18") + pd.to_timedelta((df_meteo.Time.values-197)* 24 * 60 * 60, "seconds").round('60s')
df_meteo = df_meteo.set_index("timestamp").resample("D").mean()
df_meteo["surface_height"] = (
    df_meteo.DistanceToTarget.iloc[0] - df_meteo.DistanceToTarget
)

for name in df_meta.site[:-1]:
    print(name)
    df = pd.read_csv("./Hills_" + name + "_IceTemp.txt", sep="\t")
    df["timestamp"] = (
        df_meta.loc[df_meta.site == name, "date_start"].values[0]
        + pd.to_timedelta((df.Time.values - df.Time.values[0]) * 24 * 60 * 60, "seconds")
       )

    df = df.set_index("timestamp").resample('D').mean()
    df["surface_height"] = np.nan
    ind = df_meteo.index.intersection(df.index)
    df.loc[ind, "surface_height"] = df_meteo.loc[ind, "surface_height"]
    df["surface_height"] = df.surface_height.interpolate(
        method="linear", limit_direction="both"
    )
    if all(np.isnan(df.surface_height)):
        df["surface_height"] = 0

    depth = df.columns[1:-1].str.replace("Depth_", "").values.astype(float)
    temp_label = ["temp_" + str(len(depth) - i) for i in range(len(depth))]
    depth_label = ["depth_" + str(len(depth) - i) for i in range(len(depth))]

    for i in range(len(temp_label)):
        df = df.rename(columns={df.columns[i + 1]: temp_label[i]})
        df.iloc[:14, i + 1] = np.nan
        if name in ["T-14", "T-11b"]:
            df.iloc[:30, i + 1] = np.nan

        df[depth_label[i]] = (
            depth[i] + df["surface_height"].values - df["surface_height"].iloc[0]
        )

    df_stack = ( df[temp_label]
                .stack(future_stack=True)
                .to_frame(name='temperature')
                .reset_index()
                .rename(columns={'time':'timestamp'}))
    df_stack['depth'] = df[depth_label].stack(future_stack=True).values
    df_stack=df_stack.loc[df_stack.temperature.notnull()]
    df_stack=df_stack.loc[df_stack.depth.notnull()]


    df_stack["latitude"] = df_meta.latitude[df_meta.site == name].iloc[0]
    df_stack["longitude"] = df_meta.longitude[df_meta.site == name].iloc[0]
    df_stack["elevation"] = df_meta.elevation[df_meta.site == name].iloc[0]
    df_stack["name"] = name
    df_stack["duration"] = 1

    df_stack["note"] = ''
    df_stack[
        "reference"
    ] = "Hills, B. H., Harper, J. T., Meierbachtol, T. W., Johnson, J. V., Humphrey, N. F., and Wright, P. J.: Processes influencing heat transfer in the near-surface ice of Greenlands ablation zone, The Cryosphere, 12, 3215–3227, https://doi.org/10.5194/tc-12-3215-2018, 2018. data: https://doi.org/10.18739/A2QV3C418"
    
    df_stack["reference_short"] = "Hills et al. (2018)"
    df_stack[
        "method"
    ] = "digital temperature sensor model DS18B20 from Maxim Integrated Products, Inc."
    df_stack["durationOpen"] = 0
    df_stack["durationMeasured"] = 1
    df_stack["error"] = 0.0625
    df_stack=df_stack.loc[df_stack.depth>-0.1,:]
    df_stack=df_stack.loc[df_stack.depth.notnull(),:]
    df_stack=df_stack.loc[df_stack.temperature.notnull(),:]
    
    if plot: plot_string_dataframe(df_stack, name)
    
    df_all = pd.concat((df_all, df_stack[needed_cols]), ignore_index=True)

df_all=df_all.loc[
    (df_all.temperature.notnull() &
      df_all.latitude.notnull() &
      df_all.depth.notnull()), :]
df_all.to_csv('data_formatted.csv', index=False)
