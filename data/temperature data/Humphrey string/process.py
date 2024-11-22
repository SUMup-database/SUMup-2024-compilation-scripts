import os
import re
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def plot_string_dataframe(df_stack, filename):
    df_stack = df_stack.rename(columns={'timestamp':'time',
                                        'name':'site',
                                        'depth':'depth'})

    f=plt.figure(figsize=(9,9))
    sc=plt.scatter(df_stack.time,
                -df_stack.depth,
                12, df_stack.temperature)
    plt.ylim(-df_stack.depth.max(), 0)
    plt.title(df_stack.site.unique().item())
    plt.colorbar(sc)
    f.savefig('./'+filename+'.png', dpi=400)
    plt.show()
    

plot = False
df_all = pd.DataFrame()

needed_cols = [ 'timestamp', 'latitude', 'longitude', 'elevation', 'depth',
                'temperature', 'error', 'name', 'method', 'reference',
                'reference_short', 'duration']

# %% Load Humphrey data
# del df, filepath, note, num_therm, k
print("loading Humphrey")
df = pd.read_csv("./location.txt", sep=r'\s+')
df_humphrey = pd.DataFrame(
    columns=["site", "latitude", "longitude", "elevation", "timestamp", "T10m"]
)
for site in df.site:
    try:
        df_site = pd.read_csv(
            "data/" + site + ".txt",
            header=None,
            sep=r'\s+',
            names=["doy"] + ["IceTemperature" + str(i) + "(C)" for i in range(1, 33)],
        )
    except: 
        continue

    print(site)
    temp_label = df_site.columns[1:]
    # the first column is a time stamp and is the decimal days after the first second of January 1, 2007.
    df_site["time"] = [datetime(2007, 1, 1) + timedelta(days=d) for d in df_site.iloc[:, 0]]
    if site == "T1old":
        df_site["time"] = [
            datetime(2006, 1, 1) + timedelta(days=d) for d in df_site.iloc[:, 0]
        ]
    df_site = df_site.loc[df_site["time"] <= df_site["time"].values[-1], :]
    df_site = df_site.set_index("time")
    df_site = df_site.resample('h').mean()

    depth = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00, 5.25, 5.00, 5.50, 6.00, 6.50, 7.00, 7.50, 8.00, 8.50, 9.00, 9.50, 10.0]

    if site != "H5": df_site = df_site.iloc[24 * 30 :, :]
    if site == "T4": df_site = df_site.loc[:"2007-12-05"]
    if site == "H2": depth = np.array(depth) - 1
    if site == "H4": depth = np.array(depth) - 0.75
    if site in ["H3", "G165", "T1new"]: depth = np.array(depth) - 0.50

    df_hs = pd.read_csv("surface_heights/" + site + "_surface_height.csv")
    df_hs.time = pd.to_datetime(df_hs.time)
    df_hs = df_hs.set_index("time")
    df_hs = df_hs.resample('h').mean()
    df_site["surface_height"] = np.nan

    df_site["surface_height"] = df_hs.iloc[
        df_hs.index.get_indexer(df_site.index, method="nearest")
    ].values
    depth_label = ["depth_" + str(i) for i in range(1, len(temp_label) + 1)]
    plt.figure()
    for i in range(len(temp_label)):
        df_site[depth_label[i]] = (
            depth[i]
            + df_site["surface_height"].values
            - df_site["surface_height"].iloc[0]
        )
        df_site[temp_label[i]].plot(ax=plt.gca(), alpha=0.5, label=temp_label[i])

        df_site.loc[
            df_site[temp_label[i]] <  (df_site[temp_label[i]].rolling(24*7,
                                            min_periods=24*2,
                                            center=True).max()-1),
            temp_label[i]
            ] = np.nan
        df_site[temp_label[i]].plot(ax=plt.gca(), label=temp_label[i]+' filtered')
        plt.ylabel('Subsurface temperature (deg C)')
    plt.ylim(-30, 5)
    plt.title(site)
    # plt.legend()
    df_site = df_site.resample("D").mean()

    df_stack = ( df_site[temp_label]
                .stack(future_stack=True)
                .to_frame(name='temperature')
                .reset_index()
                .rename(columns={'time':'timestamp'}))
    df_stack['depth'] = df_site[depth_label].stack(future_stack=True).values
    df_stack=df_stack.loc[df_stack.temperature.notnull()]
    df_stack=df_stack.loc[df_stack.depth.notnull()]

    df_stack["name"] = site
    df_stack["latitude"] = df.loc[df.site == site, "latitude"].values[0]
    df_stack["longitude"] = df.loc[df.site == site, "longitude"].values[0]
    df_stack["elevation"] = df.loc[df.site == site, "elevation"].values[0]

    df_stack[
        "reference"
    ] = "Humphrey, N. F., Harper, J. T., and Pfeffer, W. T. (2012), Thermal tracking of meltwater retention in Greenlands accumulation area, J. Geophys. Res., 117, F01010, doi:10.1029/2011JF002083. Data available at: https://instaar.colorado.edu/research/publications/occasional-papers/firn-stratigraphy-and-temperature-to-10-m-depth-in-the-percolation-zone-of/"
    df_stack["reference_short"] = "Humphrey et al. (2012)"
    df_stack[
        "note"
    ] = "no surface height measurements, using interpolating surface height using CP1 and SwissCamp stations"
    
    df_stack["method"] = "sealed 50K ohm thermistors"
    df_stack["duration"] = 1
    df_stack["error"] = 0.5
    
    if plot:
        plot_string_dataframe(df_stack, site)
    
    df_all = pd.concat((df_all,  df_stack[needed_cols]), ignore_index=True)
    
df_all=df_all.loc[
    (df_all.temperature.notnull() &
     df_all.latitude.notnull() &
     df_all.depth.notnull()), :]
df_all.to_csv('data_formatted.csv', index=False)
