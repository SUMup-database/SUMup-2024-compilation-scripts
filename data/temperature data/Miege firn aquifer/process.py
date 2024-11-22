



import os
import re
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_string_dataframe(df_stack, filename):
    df_stack = df_stack.rename(columns={'date':'time',
                                        'timestamp':'time',
                                        'name':'site',
                                        'temperature':'temperature',
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


needed_cols = [ 'timestamp', 'latitude', 'longitude', 'elevation', 'depth',
                'temperature', 'error', 'name', 'method', 'reference',
                'reference_short', 'duration']

plot = False
df_all = pd.DataFrame()
print("Loading firn aquifer data")
metadata = np.array(
    [
        ["FA-13", 66.181, 39.0435, 1563],
        ["FA-15-1", 66.3622, 39.3119, 1664],
        ["FA-15-2", 66.3548, 39.1788, 1543],
    ]
)

# mean_accumulation = 1 # m w.e. from Miege et al. 2014
# thickness_accum = 2.7 # thickness of the top 1 m w.e. in the FA13 core
thickness_accum = 1.4  # Burial of the sensor between their installation in Aug 2015 and revisit in Aug 2016

for k, site in enumerate(["FA_13", "FA_15_1", "FA_15_2"]):
    depth = pd.read_csv(
        "./" + site + "_Firn_Temperatures_Depths.csv").transpose()
    print('    ', site)
    if k == 0:
        depth = depth.iloc[5:].transpose()
    else:
        depth = depth.iloc[5:, 0]
    temp = pd.read_csv("./" + site + "_Firn_Temperatures.csv")
    dates = pd.to_datetime(
        (
            temp.Year * 1000000
            + temp.Month * 10000
            + temp.Day * 100
            + temp["Hours (UTC)"]
        ).apply(str),
        format="%Y%m%d%H",
    )
    temp = temp.iloc[:, 4:]

    ellapsed_hours = (dates - dates[0]).dt.total_seconds()/60/60
    accum_depth = ellapsed_hours.values * thickness_accum / 365 / 24
    depth_cor = pd.DataFrame()
    depth_cor = depth.values.reshape((1, -1)).repeat(
        len(dates), axis=0
    ) + accum_depth.reshape((-1, 1)).repeat(len(depth.values), axis=1)

    temp.columns = temp.columns.str.replace('n','').astype(int)
    depth_ds = temp.copy()
    for i, col in enumerate(depth_ds.columns):
        depth_ds[col] = depth_cor[:,i]
    temp['timestamp'] = dates 
    depth_ds['timestamp'] = dates
    
    temp = temp.set_index('timestamp').resample('D').mean()
    depth_ds = depth_ds.set_index('timestamp').resample('D').mean()

    df_miege = temp.stack(future_stack=True).to_frame()
    df_miege.columns = ['temperature']
    df_miege['depth'] = depth_ds.stack(future_stack=True)
    df_miege = df_miege.reset_index()
    # df_miege['timestamp'] = pd.to_datetime(dates.loc[df_miege.level_0].values)
    df_miege = df_miege.drop(columns=['level_1'])
    
    df_miege = df_miege.loc[df_miege.depth.notnull(),:]
    df_miege = df_miege.loc[df_miege.temperature.notnull(),:]
    
    df_miege["name"]= site
    df_miege["latitude"] = float(metadata[k, 1])
    df_miege["longitude"] = -float(metadata[k, 2])
    df_miege["elevation"] = float(metadata[k, 3])
    df_miege["reference"] = "Miller, O., Solomon, D.K., Miège, C., Koenig, L., Forster, R., Schmerr, N., Ligtenberg, S.R., Legchenko, A., Voss, C.I., Montgomery, L. and McConnell, J.R., 2020. Hydrology of a perennial firn aquifer in Southeast Greenland: an overview driven by field data. Water Resources Research, 56(8), p.e2019WR026348. Dataset doi:10.18739/A2R785P5W"
    df_miege["reference_short"] = "Miller et al. (2020)"
    df_miege["note"] = ""

    df_miege["method"] = "digital thermarray system from RST©"
    df_miege["duration"] = 1
    df_miege["error"] = 0.07
    
    if plot: plot_string_dataframe(df_miege, site)

    df_all = pd.concat((df_all,  df_miege[needed_cols]), ignore_index=True)
df_all=df_all.loc[
    (df_all.temperature.notnull() &
     df_all.latitude.notnull() &
     df_all.depth.notnull()), :]
df_all.to_csv('data_formatted.csv', index=False)
