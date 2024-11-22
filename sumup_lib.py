# -*- coding: utf-8 -*-
"""
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import os
import xarray as xr
from datetime import datetime, timedelta

def resolve_reference_keys(df, df_sumup):
    '''
    Compares a dataframes 'reference_full' field  to the references that are already in
    SUMup. If a reference is already in SUMup (df_sumup), then it's reference_key
    is reused. If it is a new reference, then a reference_key is created.

    '''
    df_ref = df_sumup[['reference_key','reference']].drop_duplicates().set_index('reference')
    df_ref_to_add = (df.reference.drop_duplicates()
                        .reset_index(drop=True)
                        .reset_index(drop=False)
                        .set_index('reference')
                        )
    count=1
    for r in df_ref_to_add.index:
        tmp = df_ref.index.str.contains(r, regex=False)
        if tmp.any():
            # reference already in df_sumup, reusing key
            if len(df_ref.loc[tmp].reference_key.values)>1:
                print('\n', r, 'matching with')
                print(df_ref.loc[tmp].reference_key)
                print('using first one')
            df_ref_to_add.loc[r, 'index'] = df_ref.loc[tmp].reference_key.values[0]
        else:
            # new reference, creating new key
            df_ref_to_add.loc[r, 'index'] = df_sumup.reference_key.max() + count
            count = count+1
    return df_ref_to_add.loc[df.reference].values

def remove_and_print_duplicates(df, df_sumup):
    '''
    Check for duplicates based on 'name' and 'reference_short' between two DataFrames.
    Print min and max 'start_year' for matching rows and remove these rows from df.

    Input:
        df:        DataFrame with new data
        df_sumup:  DataFrame with existing data

    Output:
        DataFrame with duplicates removed from df
    '''
    # Create a dictionary to index df by 'name' and 'reference_short'
    df_index = df.set_index(['name', 'reference_short']).index
    df_dict = {}
    for idx in df_index:
        if idx not in df_dict:
            df_dict[idx] = []
        df_dict[idx].append(idx)

    # Initialize lists to store indices to remove and results
    to_remove = []

    for name, reference_short in df_sumup[['name', 'reference_short']].drop_duplicates().values:
        idx = (name, reference_short)
        if idx in df_dict:
            # Extract matching rows from df_sumup
            subset_new = df[(df['name'] == name) & (df['reference_short'] == reference_short)]
            subset_existing = df_sumup[(df_sumup['name'] == name) & (df_sumup['reference_short'] == reference_short)]

            # Calculate min and max of 'start_year'
            min_start_year_new = subset_new['start_year'].min()
            max_start_year_new = subset_new['start_year'].max()
            min_start_year_existing = subset_existing['start_year'].min()
            max_start_year_existing = subset_existing['start_year'].max()

            # Print the results
            print(f"Name: {name}, Reference Short: {reference_short}")
            print(f"New Data - Start Year Min: {min_start_year_new}, Max: {max_start_year_new}")
            print(f"Existing Data - Start Year Min: {min_start_year_existing}, Max: {max_start_year_existing}")
            print('')

            # Collect indices to remove
            to_remove.extend(df_dict[idx])

    # Remove duplicates from df using the collected indices
    df_cleaned = df[~df.index.isin(to_remove)]

    return df_cleaned

def check_duplicates(df_stack, df_sumup, verbose=True, plot=True, tol=0.3):
    '''
    Finding potential duplicates in new SMB data compared to what's in SUMup
    Input:
        df_stack:   dataframe with potentially multiple new SMB measurements to be
                    added to SUMup
        df_sumup:   dataframe with SUMup observations
        verbose:    if True print the candidates-duplicates
        plot:       if True print the candidates-duplicates
        tol:        tolerance, in degrees, accepted for coordinates of potential
                    duplicates compared to the location of new data
    Output:
        A dataframe containing the potential duplicates' name, reference_short,
        latitude, longitude and elevation
    '''
    df_all_candidates = pd.DataFrame()
    for p in df_stack.name.unique():
        df = df_stack.loc[df_stack.name == p, :]
        if len(df)==0:
            print('skipping',p)
            continue

        # looking for obs within defined tolerance
        msk1 = abs(df_sumup.latitude - df.latitude.iloc[0]) < tol
        msk2 = abs(df_sumup.longitude - df.longitude.iloc[0]) < tol
        # looking for cores or snow pits and not radar profiles
        msk3 = ~df_sumup.method.isin([2, 6, 7])
        msk = msk1 & msk2 & msk3

        if msk.any():
            # looping through the different observation groups (as defined by name)
            df_all_candidates_group = pd.DataFrame()  # Temporary DataFrame for current group
            for p_dup in df_sumup.loc[msk, 'name'].unique():
                # extracting potential duplicate from SUMup
                df_sumup_candidate_dupl = df_sumup.loc[df_sumup.name == p_dup, :]
                # making a last check that the new data (in df) and the candidate
                # duplicate ends more or less at the same year
                if abs(df_sumup_candidate_dupl.end_year.max() - df.end_year.max()) > 3:
                    continue

                # append that candidate to the list of candidates
                df_all_candidates_group = pd.concat((
                    df_all_candidates_group,
                    df_sumup_candidate_dupl[
                        ['name', 'reference_short', 'latitude', 'longitude', 'elevation']
                        ].drop_duplicates()))

            if len(df_all_candidates_group) >= 1:
                df_all_candidates = pd.concat((df_all_candidates, df_all_candidates_group))

                if verbose:
                    print('')
                    print(df_all_candidates_group.values)
                    print('might be the same as')
                    print(df[['name','reference_short', 'latitude', 'longitude']].drop_duplicates().values)
                    if df.name.iloc[0] == 'sdo2':
                        print('sdo2 has same coordinates as S.Domea and b but',
                              'different time range')

                if plot:
                    plt.figure()
                    plt.plot(df.end_year.values, df.smb.values,
                             marker='o', ls='None',
                             label=df.name.iloc[0] + ' '+df.reference_short.iloc[0])
                    for p_dup in df_all_candidates_group.name.unique():
                        plt.plot(df_sumup.loc[df_sumup.name == p_dup, 'end_year'].values,
                                 df_sumup.loc[df_sumup.name == p_dup, 'smb'].values,
                                 marker='^', ls='None', alpha=0.7,
                                 label=(df_sumup.loc[df_sumup.name == p_dup,
                                                    'name'].iloc[0]
                                        + ' ' +
                                        df_sumup.loc[df_sumup.name == p_dup,
                                                       'reference_short'].iloc[0]))
                    plt.title(df.name.iloc[0])
                    plt.legend()
                    plt.show()  # Ensure plot is displayed

    return df_all_candidates


def resolve_name_keys(df, df_sumup, v='name', df_existing=None):
    '''
    Creates name_keys for all the unique 'name' in df.
    '''
    df_names_to_add = (df[v]
                        .drop_duplicates()
                        .reset_index(drop=True)
                        .reset_index(drop=False)
                        .set_index(v)
                        )
    df_names_to_add.columns = [v+'_key']

    if df_existing is not None:
        df_existing = df_existing.reset_index().set_index(v)
        df_existing = df_existing[~df_existing.index.duplicated(keep='first')]
    else:
        df_existing = (df_sumup[[v,v+'_key']]
                            .drop_duplicates()
                            .set_index(v)
                            )
        df_existing.columns = ['key']
    for p in df_names_to_add.index:
        if p in df_existing.index:
            df_names_to_add.loc[p, v+'_key'] = df_existing.loc[p, 'key']
            print('reusing old key', df_existing.loc[p, 'key'], 'for',p)
        else:
            df_names_to_add.loc[p, v+'_key'] = df_names_to_add.loc[p, v+'_key'] + 1 + df_sumup[v+'_key'].max()

    return df_names_to_add.loc[df[v]].values


# import cartopy
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature


def parse_short_reference(df_in, verbose=False):
    df_sumup = df_in.copy()
    abc = 'bcdefghijklmnopqrstuvwxyz'
    df_sumup['reference'] = df_sumup.reference_full.astype(str)
    all_refs = np.array([['reference_short', 'reference']])
    for ref in df_sumup.reference.unique():
        if (',' not in ref) and (' ' not in ref):
            print(ref, 'left as is')
            ref_short = ref
        else:
            year = re.findall(r'\d{4}', ref)
            if len(year) > 0:
                year = year[0]
            else:
                year = ''

            # first guess
            name = ref.lstrip().split(',')[0].split(' ')[0]

            # some exceptions
            if name == 'U.S.': name = 'US_Army'
            if name == 'SIMBA': name = 'SIMBA: Lewis'
            if name == 'Paul': name = 'Smeets'
            if name == 'K.': name = 'Wegener'
            if name in ['van', 'Van']: name = ref.lstrip().split(',')[0].replace(' ',' ')
            ref_short = name + ' et al. ('+ year+')'
            if name == 'US': ref_short = 'US ITASE: Mayewski and Dixon ('+ year+')'
            if name == 'Satellite-Era': ref_short = 'SEAT11: Brucker and Koenig ('+ year+')'

        count = 0
        while ref_short in all_refs[:,0]:
            if count == 0:
                ref_short = ref_short[:-1] + 'b)'
                # tmp = all_refs[-1][0]
                # all_refs[-1] = tmp[:-1] + 'a)'
                count = count + 1
            else:
                ref_short = ref_short[:-2] + abc[count] +')'
                count = count + 1
        if verbose: print(ref_short)
        all_refs = np.vstack([all_refs, [ref_short, ref]])
    df_ref = pd.DataFrame(all_refs[1:,:], columns=all_refs[0,:])
    return df_ref


def check_conflicts(df_sumup, df_new, var=['name', 'depth','temperature'],
                    verbose=1):
    coords_sumup = df_sumup[['latitude','longitude']].drop_duplicates()
    coords_new = df_new[['latitude','longitude']].drop_duplicates()
    diff_lat = np.abs(coords_sumup.latitude.values - coords_new.latitude.values[:, np.newaxis])
    diff_lon = np.abs(coords_sumup.longitude.values - coords_new.longitude.values[:, np.newaxis])

    potential_duplicates = np.where( (diff_lat < 0.01) & (diff_lon < 0.01))
    for k in range(len(potential_duplicates[0])):
        i = potential_duplicates[0][k]
        j = potential_duplicates[1][k]
        tmp = pd.DataFrame()
        tmp['in SUMup']= df_sumup.loc[coords_sumup.iloc[j,:].name,:].T
        tmp['in new dataset'] = df_new.loc[coords_new.iloc[i,:].name,:].T
        if np.round(tmp.loc['date','in SUMup']/10000) == np.round(tmp.loc['date','in new dataset']/10000):
            if verbose:
                print('\nPotential duplicate found:\n')

                print(tmp.loc[var])
                print('reference in SUMup:')
                print(tmp.loc['reference_full', 'in SUMup'])
                print('reference in new dataset:')
                print(tmp.loc['reference_full', 'in new dataset'])
            return [tmp.loc[var[0], 'in SUMup']]
    return []



# %% New temperature addition

def stack_multidepth_df(df_in, temp_var, depth_var):
    print('    -> converting to multi-depth xarray')
    df_in = df_in.rename(columns={'date':'time'})

    # checking the time variable is no less than hourly
    # dif_time = df_in.time.diff()
    # if len(dif_time[dif_time<pd.Timedelta(minutes=50)])>0:
    #     print('found time steps under 1 h')
    #     print(df_in.time[dif_time<pd.Timedelta(minutes=50)])
    # df_in.time = df_in.time.dt.round('h')

    df_in = df_in.set_index(['site', 'time', 'reference_short','latitude',
                             'longitude', 'elevation', 'note', 'reference'])

    # some filtering
    df_in = df_in.dropna(subset=temp_var, how='all')
    for v in temp_var:
        df_in.loc[df_in[v] > 1, v] = np.nan
        df_in.loc[df_in[v] < -70, v] = np.nan
    df_in = df_in.loc[~df_in[temp_var].isnull().all(axis=1),:]

    df_stack = df_in[temp_var].rename(columns=dict(zip(temp_var,
         range(1,len(temp_var)+1)))).stack(future_stack=True).to_frame(name='temperature').reset_index()
    df_stack['depth'] = df_in[depth_var].rename(columns=dict(zip(depth_var,
             range(1,len(depth_var)+1)))).stack(future_stack=True).to_frame(name='depth').values

    return df_stack

def plot_string_dataframe(df_stack, filename):
    df_stack = df_stack.rename(columns={'date':'time',
                                        'temperatureObserved':'temperature',
                                        'depthOfTemperatureObservation':'depth'})

    f=plt.figure(figsize=(9,9))
    sc=plt.scatter(df_stack.time,
                -df_stack.depth,
                12, df_stack.temperature)
    plt.ylim(-df_stack.depth.max(), 0)
    plt.title(df_stack.site.unique().item())
    plt.colorbar(sc)
    f.savefig('figures/strings_raw/'+filename+'.png', dpi=400)
    plt.show()

# %% Vandecrux et al. full res
import tables as tb


def smooth(x, window_len=14, window="hanning"):
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        raise ValueError(
            "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        )

    s = np.r_[x[window_len - 1 : 0 : -1], x, x[-2 : -window_len - 1 : -1]]
    # print(len(s))
    if window == "flat":  # moving average
        w = np.ones(window_len, "d")
    else:
        w = eval("np." + window + "(window_len)")

    y = np.convolve(w / w.sum(), s, mode="valid")

    return y[int(window_len / 2 - 1) : -int(window_len / 2)]


def load_firncover_metadata(filepath, sites):
    CVNfile = tb.open_file(filepath, mode="r", driver="H5FD_CORE")
    datatable = CVNfile.root.FirnCover

    statmeta_df = pd.DataFrame.from_records(
        datatable.Station_Metadata[:].tolist(),
        columns=datatable.Station_Metadata.colnames,
    )
    statmeta_df.sitename = statmeta_df.sitename.str.decode("utf-8")
    statmeta_df.iridium_URL = statmeta_df.iridium_URL.str.decode("utf-8")
    statmeta_df["install_date"] = pd.to_datetime(
        statmeta_df.installation_daynumer_YYYYMMDD.values, format="%Y%m%d"
    )
    statmeta_df["rtd_date"] = pd.to_datetime(
        statmeta_df.RTD_installation_daynumber_YYYYMMDD.values, format="%Y%m%d"
    )
    firn_temp_cols = ['rtd'+str(ii) for ii in range(len(statmeta_df.RTD_depths_at_installation_m[0]))]
    firn_temp_cols = np.flip(firn_temp_cols)

    statmeta_df[firn_temp_cols] = pd.DataFrame(
        statmeta_df.RTD_depths_at_installation_m.values.tolist(),
        index=statmeta_df.index,
    )
    statmeta_df.set_index("sitename", inplace=True)
    statmeta_df.loc["Crawford", "rtd_date"] = statmeta_df.loc[
        "Crawford", "install_date"
    ]
    statmeta_df.loc["NASA-SE", "rtd_date"] = statmeta_df.loc[
        "NASA-SE", "install_date"
    ] - pd.Timedelta(days=1)

    # Meteorological_Daily to pandas
    metdata_df = pd.DataFrame.from_records(datatable.Meteorological_Daily[:])
    metdata_df.sitename = metdata_df.sitename.str.decode("utf-8")
    metdata_df["date"] = pd.to_datetime(
        metdata_df.daynumber_YYYYMMDD.values, format="%Y%m%d"
    )

    for site in sites:
        msk = (metdata_df["sitename"] == site) & (
            metdata_df["date"] < statmeta_df.loc[site, "rtd_date"]
        )
        metdata_df.drop(metdata_df[msk].index, inplace=True)
        if site == "NASA-SE":
            m3 = (
                (metdata_df["sitename"] == site)
                & (metdata_df["date"] > "2017-02-12")
                & (metdata_df["date"] < "2017-04-12")
            )
            metdata_df.loc[m3, "sonic_range_dist_corrected_m"] = np.nan

        if site == "EKT":

            m2 = (metdata_df["sitename"] == site) & (metdata_df["date"] > "2018-05-15")
            metdata_df.loc[m2, "sonic_range_dist_corrected_m"] = (
                metdata_df.loc[m2, "sonic_range_dist_corrected_m"] - 0.5
            )

        if site == "DYE-2":
            m2 = (metdata_df["sitename"] == site) & (metdata_df["date"] > "2016-04-29")
            metdata_df.loc[m2, "sonic_range_dist_corrected_m"] = (
                metdata_df.loc[m2, "sonic_range_dist_corrected_m"] - 0.3
            )


    metdata_df.reset_index(drop=True)

    sonic_df = metdata_df[
        ["sitename", "date", "sonic_range_dist_corrected_m"]
    ].set_index(["sitename", "date"])
    sonic_df.columns = ["sonic_m"]
    sonic_df.sonic_m[sonic_df.sonic_m < -100] = np.nan
    sonic_df.loc["Saddle", "2015-05-16"] = sonic_df.loc["Saddle", "2015-05-17"]

    # filtering
    gradthresh = 0.1

    for site in sites:
        if site in ["Summit", "NASA-SE"]:
            tmp = 0
        else:
            # applying gradient filter on KAN-U, Crawford, EwastGRIP, EKT, Saddle and Dye-2
            vals = sonic_df.loc[site, "sonic_m"].values
            vals[np.isnan(vals)] = -9999
            msk = np.where(np.abs(np.gradient(vals)) >= gradthresh)[0]
            vals[msk] = np.nan
            vals[msk - 1] = np.nan
            vals[msk + 1] = np.nan
            vals[vals == -9999] = np.nan
            sonic_df.loc[site, "sonic_m"] = vals
        sonic_df.loc[site, "sonic_m"] = (
            sonic_df.loc[site].interpolate(method="linear").values
        )
        sonic_df.loc[site, "sonic_m"] = smooth(sonic_df.loc[site, "sonic_m"].values)

    for site in sonic_df.index.unique(level="sitename"):
        dd = statmeta_df.loc[site]["rtd_date"]
        if site == "Saddle":
            dd = dd + pd.Timedelta("1D")
        sonic_df.loc[site, "delta"] = (
            sonic_df.loc[[site]].sonic_m - sonic_df.loc[(site, dd)].sonic_m
        )

    rtd_depth_df = statmeta_df[firn_temp_cols].copy()
    depth_cols = ["depth_" + str(i) for i in range(len(firn_temp_cols))]
    depth_cols = np.flip(depth_cols)
    rtd_depth_df.columns = depth_cols

    xx = statmeta_df.RTD_top_usable_RTD_num
    for site in sites:
        vv = rtd_depth_df.loc[site].values
        ri = np.arange(xx.loc[site], 24)
        vv[ri] = np.nan
        rtd_depth_df.loc[site] = vv
    rtd_d = sonic_df.join(rtd_depth_df, how="inner")
    rtd_dc = rtd_d.copy()
    rtd_dep = rtd_dc[depth_cols].add(-rtd_dc["delta"], axis="rows")

    rtd_df = pd.DataFrame.from_records(
        datatable.Firn_Temp_Daily[:].tolist(),
        columns=datatable.Firn_Temp_Daily.colnames,
    )
    rtd_df.sitename = rtd_df.sitename.str.decode("utf-8")
    rtd_df["date"] = pd.to_datetime(rtd_df.daynumber_YYYYMMDD.values, format="%Y%m%d")
    rtd_df = rtd_df.set_index(["sitename", "date"])

    rtd_df[firn_temp_cols] = pd.DataFrame(
        rtd_df.RTD_temp_avg_corrected_C.values.tolist(), index=rtd_df.index
    )

    # filtering
    for col in firn_temp_cols:
        rtd_df.loc[rtd_df[col]==-100.0, col] = np.nan

    for i in range(0, 4):
        vals = rtd_df.loc["Crawford", firn_temp_cols[i]].values
        vals[vals > -1] = np.nan
        rtd_df.loc["Crawford", firn_temp_cols[i]] = vals
    rtd_df = rtd_df.join(rtd_dep, how="inner").sort_index(axis=0)
    for site in sites:
        rtd_df.loc[site, firn_temp_cols][:14] = np.nan
    return statmeta_df, sonic_df, rtd_df


def add_vandecrux_full_res(plot=True):
    needed_cols = ["date", "site", "latitude", "longitude", "elevation", "depthOfTemperatureObservation", "temperatureObserved", "reference", "reference_short", "note", "error", "durationOpen", "durationMeasured", "method"]
    df_all = pd.DataFrame(columns=needed_cols)

    # %% Polashenski
    print("Loading Polashenski")
    df_Pol = pd.read_csv("data/temperature data/Polashenski/2013_10m_Temperatures.csv")
    df_Pol.columns = df_Pol.columns.str.replace(" ", "")
    df_Pol.date = pd.to_datetime(df_Pol.date, format="%m/%d/%y")
    df_Pol["reference"] = "Polashenski, C., Z. Courville, C. Benson, A. Wagner, J. Chen, G. Wong, R. Hawley, and D. Hall (2014), Observations of pronounced Greenland ice sheet firn warming and implications for runoff production, Geophys. Res. Lett., 41, 4238–4246, doi:10.1002/2014GL059806."
    df_Pol["reference_short"] = "Polashenski et al. (2014)"
    df_Pol["note"] = ""
    df_Pol["longitude"] = -df_Pol["longitude"]
    df_Pol["depthOfTemperatureObservation"] = (
        df_Pol["depthOfTemperatureObservation"].str.replace("m", "").astype(float)
    )
    df_Pol[
        "durationOpen"
    ] = "string lowered in borehole and left 30min for equilibrating with surrounding firn prior measurement start"
    df_Pol["durationMeasured"] = "overnight ~10 hours"
    df_Pol["error"] = 0.1
    df_Pol["method"] = "thermistor string"

    df_all = pd.concat((df_all, df_Pol[needed_cols] ), ignore_index=True)

    # %% McGrath (only adding non 10 m temp)
    print("Loading McGrath")
    df_mcgrath = pd.read_excel(
        "data/temperature data/McGrath/McGrath et al. 2013 GL055369_Supp_Material.xlsx"
    )
    df_mcgrath = df_mcgrath.loc[df_mcgrath["Data Type"] != "Met Station"]
    df_mcgrath["depthOfTemperatureObservation"] = np.array(
        df_mcgrath["Data Type"].str.split("m").to_list()
    )[:, 0].astype(int)
    df_mcgrath = df_mcgrath.loc[df_mcgrath.depthOfTemperatureObservation != 10,:]

    df_mcgrath = df_mcgrath.rename(
        columns={
            "Observed\nTemperature (°C)": "temperatureObserved",
            "Latitude\n(°N)": "latitude",
            "Longitude (°E)": "longitude",
            "Elevation\n(m)": "elevation",
            "Reference": "reference",
            "Location": "site",
        }
    )
    df_mcgrath["note"] = "as reported in McGrath et al. (2013)"

    df_mcgrath["date"] = pd.to_datetime(
        (df_mcgrath.Year * 10000 + 101).apply(str), format="%Y%m%d"
    )
    df_mcgrath["site"] = df_mcgrath["site"].str.replace("B4", "4")
    df_mcgrath["site"] = df_mcgrath["site"].str.replace("B5", "5")
    df_mcgrath["site"] = df_mcgrath["site"].str.replace("4-425", "5-0")

    df_mcgrath["method"] = "digital Thermarray system from RST©"
    df_mcgrath["durationOpen"] = 0
    df_mcgrath["durationMeasured"] = 30
    df_mcgrath["error"] = 0.07

    df_all = pd.concat((df_all,  df_mcgrath[needed_cols]), ignore_index=True)


    # %% Hawley GrIT
    del df_mcgrath
    print("Loading Hawley GrIT")
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')
    df_hawley = pd.read_excel("data/temperature data/Hawley GrIT/GrIT2011_9m-borehole_calc-temps.xlsx")
    df_hawley = df_hawley.rename(
        columns={
            "Pit name (tabs)": "site",
            "Date": "date",
            "Lat (dec.degr)": "latitude",
            "Long (dec.degr)": "longitude",
            "Elevation": "elevation",
            "9-m temp": "temperatureObserved",
        }
    )
    df_hawley["depthOfTemperatureObservation"] = 9
    df_hawley["note"] = ""
    df_hawley["reference"] = "Bob Hawley. 2014. Traverse physical, chemical, and weather observations. arcitcdata.io, doi:10.18739/A2W232. "
    df_hawley["reference_short"] = "Hawley (2014) GrIT"

    df_hawley = df_hawley.loc[[isinstance(x, float) for x in df_hawley.temperatureObserved]]
    df_hawley = df_hawley.loc[df_hawley.temperatureObserved.notnull()]

    df_hawley["method"] = "thermistor"
    df_hawley["durationOpen"] = 2
    df_hawley["durationMeasured"] = 0
    df_hawley["error"] = "not reported"

    df_all = pd.concat((df_all, df_hawley[needed_cols]), ignore_index=True)


    # %% Harper ice temperature
    print("Loading Harper ice temperature")
    df_harper = pd.read_csv(
        "data/temperature data/Harper ice temperature/harper_iceTemperature_2015-2016.csv"
    )
    df_harper["temperatureObserved"] = np.nan
    df_harper["note"] = ""
    df_harper[[pd.to_datetime("2015-01-01"),pd.to_datetime("2016-01-01")]] = df_harper[['temperature_2015_celsius','temperature_2016_celsius']]
    df_stack = (
        df_harper[[pd.to_datetime("2015-01-01"),pd.to_datetime("2016-01-01")]]
        .stack(future_stack=True).to_frame(name='temperatureObserved')
        .reset_index().rename(columns={'level_1':'date'})
                                       )

    df_stack['site'] = df_harper.loc[df_stack.level_0,'borehole'].values
    df_stack['latitude'] = df_harper.loc[df_stack.level_0,'latitude_WGS84'].values
    df_stack['longitude'] = df_harper.loc[df_stack.level_0,'longitude_WGS84'].values
    df_stack['elevation'] = df_harper.loc[df_stack.level_0,'Elevation_m'].values
    df_stack['depthOfTemperatureObservation'] = df_harper.loc[df_stack.level_0,'depth_m'].values-df_harper.loc[df_stack.level_0,'height_m'].values
    df_stack = df_stack.loc[df_stack.temperatureObserved.notnull()]

    plt.figure()
    plt.gca().invert_yaxis()
    for borehole in df_stack["site"].unique():
        df_stack.loc[df_stack.site==borehole].plot(ax=plt.gca(),
                                                   x='temperatureObserved',
                                                   y='depthOfTemperatureObservation',
                                                   label=borehole)
    plt.legend()

    df_stack[
        "reference"
    ] = "Hills, B. H., Harper, J. T., Humphrey, N. F., & Meierbachtol, T. W. (2017). Measured horizontal temperature gradients constrain heat transfer mechanisms in Greenland ice. Geophysical Research Letters, 44. https://doi.org/10.1002/2017GL074917; Data: https://doi.org/10.18739/A24746S04"

    df_stack["reference_short"] = "Hills et al. (2017)"

    df_stack["method"] = "TMP102 digital temperature sensor"
    df_stack["durationOpen"] = 0
    df_stack["durationMeasured"] = 30 * 24
    df_stack["error"] = 0.1
    df_stack["note"] = ""

    df_all = pd.concat((df_all,  df_stack[needed_cols]), ignore_index=True)

    # %%  FirnCover
    # del df_stack, df_harper, borehole
    print("Loading FirnCover")

    filepath = os.path.join("data/temperature data/FirnCover/FirnCoverData_2.0_2021_07_30.h5")
    sites = ["Summit", "KAN-U", "NASA-SE", "Crawford", "EKT", "Saddle", "EastGrip", "DYE-2"]
    statmeta_df, sonic_df,  rtd_df = load_firncover_metadata(filepath, sites)
    statmeta_df["elevation"] = [1840, 2119, 2361, 2370, 2456, 1942, 3208, 2666]

    rtd_df = rtd_df.reset_index()
    rtd_df = rtd_df.set_index(["sitename", "date"])
    df_firncover = pd.DataFrame()
    for site in sites:
        print('   ',site)
        df_d = rtd_df.xs(site, level="sitename").reset_index()
        df_stack = (
            df_d[[v for v in df_d.columns if v.startswith('rtd')]]
            .stack(future_stack=True).to_frame(name='temperatureObserved').reset_index()
            )
        df_stack['depthOfTemperatureObservation'] = df_d[
            [v.replace('rtd','depth_') for v in df_d.columns if v.startswith('rtd')]
            ].stack(future_stack=True).values

        df_stack['date'] = df_d.loc[df_stack.level_0, 'date'].values
        df_stack["site"] = site
        if site == "Crawford":
            df_stack["site"] = "CP1"
        df_stack["latitude"] = statmeta_df.loc[site, "latitude"]
        df_stack["longitude"] = statmeta_df.loc[site, "longitude"]
        df_stack["elevation"] = statmeta_df.loc[site, "elevation"]

        df_stack[
            "reference"
        ] = "MacFerrin, M. J., Stevens, C. M., Vandecrux, B., Waddington, E. D., and Abdalati, W. (2022) The Greenland Firn Compaction Verification and Reconnaissance (FirnCover) dataset, 2013–2019, Earth Syst. Sci. Data, 14, 955–971, https://doi.org/10.5194/essd-14-955-2022,"
        df_stack["reference_short"] = "MacFerrin et al. (2021, 2022)"
        df_stack["note"] = ""

        # Correction of FirnCover bias
        p = np.poly1d([1.03093649, -0.49950273])
        df_stack["temperatureObserved"] = p(df_stack["temperatureObserved"].values)

        df_stack["method"] = "Resistance Temperature Detectors + correction"
        df_stack["durationOpen"] = 0
        df_stack["durationMeasured"] = 1
        df_stack["error"] = 0.5

        if plot:
            plot_string_dataframe(df_stack, site)

        df_all = pd.concat((df_all, df_stack[needed_cols]), ignore_index=True)

    # %% SPLAZ KAN_U
    # del df_stack,  df_d,  df_firncover, rtd_df, site, sites, statmeta_df

    print("Loading SPLAZ at KAN-U")
    num_therm = [32, 12, 12]

    for k, note in enumerate(["SPLAZ_main", "SPLAZ_2", "SPLAZ_3"]):
        print('   ',note)
        ds = xr.open_dataset("data/temperature data/SPLAZ/T_firn_KANU_" + note + ".nc")
        ds=ds.where(ds['Firn temperature'] != -999)
        ds=ds.where(ds['depth'] != -999)

        if k ==0: lim = 10
        if k ==1: lim = 3
        if k ==2: lim = 3

        ds.loc[dict(level=slice(0,lim),time=slice('2012-09-01','2013-09-01'))] = np.nan

        ds=ds.where(ds['depth'] >0.1)
        ds = ds.resample(time='D').mean()
        df = ds.to_dataframe()
        df.reset_index(inplace=True)
        df = df.rename(columns={'time':'date',
                                'Firn temperature':'temperatureObserved',
                                'depth':'depthOfTemperatureObservation'})

        df["note"] = ''
        df["latitude"] = 67.000252
        df["longitude"] = -47.022999
        df["elevation"] = 1840
        df["site"] = "KAN_U "+note
        df[
            "reference"
        ] = "Charalampidis, C., Van As, D., Colgan, W.T., Fausto, R.S., Macferrin, M. and Machguth, H., 2016. Thermal tracing of retained meltwater in the lower accumulation area of the Southwestern Greenland ice sheet. Annals of Glaciology, 57(72), pp.1-10."
        df["reference_short"] = "Charalampidis et al. (2016); Charalampidis et al. (2022) "
        df["method"] = "RS 100 kΩ negative-temperature coefficient thermistors"
        df["durationOpen"] = 0
        df["durationMeasured"] = 1
        df["error"] = 0.2

        if plot:
            plot_string_dataframe(df, 'KAN_U_'+note)

        df_all = pd.concat((df_all,  df[needed_cols]), ignore_index=True)


    # %% Heilig Dye-2
    # del df_stack, df_m1eta, df_meteo, df, i, site, temp_label, depth_label, depth
    print("Loading Heilig Dye-2")

    # loading temperature data
    df = pd.read_csv("data/temperature data/Heilig/CR1000_PT100.txt", header=None)
    df.columns = [ "time_matlab", "temp_1", "temp_2", "temp_3", "temp_4", "temp_5", "temp_6", "temp_7", "temp_8"]
    df["time"] = pd.to_datetime(
        [
            datetime.fromordinal(int(matlab_datenum))
            + timedelta(days=matlab_datenum % 1)
            - timedelta(days=366)
            for matlab_datenum in df.time_matlab
        ]
    ).round('h')

    df = df.set_index("time").resample('D').mean()
    # loading surface height data
    df_surf = pd.read_csv("data/temperature data/Heilig/CR1000_SR50.txt", header=None)
    df_surf.columns = ["time_matlab", "sonic_m", "height_above_upgpr"]
    df_surf["time"] = pd.to_datetime(
        [
            datetime.fromordinal(int(matlab_datenum))
            + timedelta(days=matlab_datenum % 1)
            - timedelta(days=366)
            for matlab_datenum in df_surf.time_matlab
        ]
    ).round('h')
    df_surf = df_surf.set_index("time")[['sonic_m']].resample('D').mean()


    # loading surface height data from firncover
    filepath = os.path.join("data/temperature data/FirnCover/FirnCoverData_2.0_2021_07_30.h5")
    sites = ["Summit", "KAN-U", "NASA-SE", "Crawford", "EKT", "Saddle", "EastGrip", "DYE-2"]
    _, sonic_df, _ = load_firncover_metadata(filepath, sites)

    sonic_df = sonic_df.xs("DYE-2", level="sitename").reset_index().rename(columns={'date':'time'})
    sonic_df = sonic_df.set_index("time").drop(columns="delta").resample('h').interpolate()

    # aligning and merging the surface heights from Achim and Macferrin
    sonic_df = pd.concat(
        (sonic_df.reset_index(),
         (df_surf.loc[sonic_df.index[-1]:] - 1.83).reset_index()),
        ignore_index=True).set_index('time')


    df3 = sonic_df.loc[df.index[0]] - sonic_df.loc[df.index[0] : df.index[-1]]
    df3=df3[~df3.index.duplicated(keep='first')].loc[df.index.drop_duplicates()]
    df["surface_height"] = df3.values

    depth = 3.4 - np.array([3, 2, 1, 0, -1, -2, -4, -6])
    temp_label = ["temp_" + str(i + 1) for i in range(len(depth))]
    depth_label = ["depth_" + str(i + 1) for i in range(len(depth))]

    for i in range(len(depth)):
        df[depth_label[i]] = (
            depth[i] + df["surface_height"].values - df["surface_height"].iloc[0]
        )

    df.loc["2018-05-18":, "depth_1"] = df.loc["2018-05-18":, "depth_1"].values - 1.5
    df.loc["2018-05-18":, "depth_2"] = df.loc["2018-05-18":, "depth_2"].values - 1.84

    df_stack = ( df[temp_label]
                .stack(future_stack=True)
                .to_frame(name='temperatureObserved')
                .reset_index()
                .rename(columns={'time':'date'}))
    df_stack['depthOfTemperatureObservation'] = df[depth_label].stack(future_stack=True).values
    df_stack=df_stack.loc[df_stack.temperatureObserved.notnull()]
    df_stack=df_stack.loc[df_stack.depthOfTemperatureObservation.notnull()]
    df_stack["site"] = "DYE-2"
    df_stack["latitude"] = 66.4800
    df_stack["longitude"] = -46.2789
    df_stack["elevation"] = 2165.0
    df_stack["note"] = "using surface height from FirnCover station"
    df_stack[
        "reference"
    ] = "Heilig, A., Eisen, O., MacFerrin, M., Tedesco, M., and Fettweis, X.: Seasonal monitoring of melt and accumulation within the deep percolation zone of the Greenland Ice Sheet and comparison with simulations of regional climate modeling, The Cryosphere, 12, 1851–1866, https://doi.org/10.5194/tc-12-1851-2018, 2018. "
    df_stack["reference_short"] = "Heilig et al. (2018)"
    df_stack["method"] = "thermistors"
    df_stack["durationOpen"] = 0
    df_stack["durationMeasured"] = 1
    df_stack["error"] = 0.25

    if plot:
        plot_string_dataframe(df_stack, 'KAN_U_Heilig')
    df_all = pd.concat((df_all,  df_stack[needed_cols]), ignore_index=True)


    # %%  Echelmeyer Jakobshavn isbræ
    df_echel = pd.read_excel("data/temperature data/Echelmeyer jakobshavn/Fig5_points.xlsx")
    df_echel.date = pd.to_datetime(df_echel.date)
    df_echel = df_echel.rename(
        columns={"12 m temperature": "temperatureObserved", "Name": "site"}
    )
    df_echel["depthOfTemperatureObservation"] = 12

    df_profiles = pd.read_excel("data/temperature data/Echelmeyer jakobshavn/Fig3_profiles.xlsx")
    df_profiles = df_profiles[pd.DatetimeIndex(df_profiles.date).month > 3]
    df_profiles["date"] = pd.to_datetime(df_profiles.date)
    df_profiles = df_profiles.set_index(["site", "date"], drop=False).rename(
        columns={"temperature": "temperatureObserved",
                 "depth": "depthOfTemperatureObservation"}
    )
    df_profiles["note"] = "digitized"

    df_echel = pd.concat((df_echel, df_profiles), ignore_index=True)

    df_echel["reference"] = "Echelmeyer K, Harrison WD, Clarke TS and Benson C (1992) Surficial Glaciology of Jakobshavns Isbræ, West Greenland: Part II. Ablation, accumulation and temperature. Journal of Glaciology 38(128), 169–181, doi:10.3189/S0022143000009709"
    df_echel["reference_short"] = "Echelmeyer et al. (1992)"

    df_echel["method"] = "thermistors and thermocouples"
    df_echel["durationOpen"] = 0
    df_echel["durationMeasured"] = 0
    df_echel["error"] = 0.3

    df_all = pd.concat((df_all,  df_echel[needed_cols]), ignore_index=True)

    # %% Fischer de Quervain EGIG
    del df_echel, df_profiles
    df_fischer = pd.read_excel("data/temperature data/Fischer EGIG/fischer_90_91.xlsx")

    df_fischer[
        "reference"
    ] = "Fischer, H., Wagenbach, D., Laternser, M. & Haeberli, W., 1995. Glacio-meteorological and isotopic studies along the EGIG line, central Greenland. Journal of Glaciology, 41(139), pp. 515-527."
    df_fischer["reference_short"] = "Fischer et al. (1995)"

    df_all = pd.concat((df_all,  df_fischer[needed_cols]), ignore_index=True)

    df_dequervain = pd.read_csv("data/temperature data/Fischer EGIG/DeQuervain.txt", index_col=False)
    df_dequervain.date = pd.to_datetime(
        df_dequervain.date.str[:-2] + "19" + df_dequervain.date.str[-2:]
    )
    df_dequervain = df_dequervain.rename(
        columns={"depth": "depthOfTemperatureObservation", "temp": "temperatureObserved"}
    )
    df_dequervain["note"] = "as reported in Fischer et al. (1995)"
    df_dequervain["reference"] = "de Quervain, M, 1969. Schneekundliche Arbeiten der Internationalen Glaziologischen Grönlandexpedition (Nivologie). Medd. Grønl. 177(4)"
    df_dequervain["reference_short"] = "de Quervain (1969)"

    df_dequervain["method"] = "bimetallic, mercury, Wheastone bridge, platinium resistance thermometers"
    df_dequervain["durationOpen"] = 0
    df_dequervain["durationMeasured"] = 0
    df_dequervain["error"] = 0.2

    df_all = pd.concat((df_all, df_dequervain[needed_cols]), ignore_index=True)

    # %% Larternser EGIG
    del df_fischer, df_dequervain
    df_laternser = pd.read_excel("data/temperature data/Laternser 1992/Laternser94.xlsx")

    df_laternser["reference"] = "Laternser, M., 1994 Firn temperature measurements and snow pit studies on the EGIG traverse of central Greenland, 1992. Eidgenössische Technische Hochschule.  Versuchsanstalt für Wasserbau  Hydrologie und Glaziologic. (Arbeitsheft 15)."
    df_laternser["reference_short"] = "Laternser (1994)"

    df_laternser["method"] = "Fenwal 197-303 KAG-401 thermistors"
    df_laternser["durationOpen"] = 0
    df_laternser["error"] = 0.02

    df_all = pd.concat((df_all,  df_laternser[needed_cols]), ignore_index=True)

    # %% Wegener 1929-1930
    del df_laternser
    df1 = pd.read_csv("data/temperature data/Wegener 1930/200mRandabst_firtemperature_wegener.csv", sep=";")
    df3 = pd.read_csv("data/temperature data/Wegener 1930/ReadMe.txt", sep=";")

    df1["depthOfTemperatureObservation"] = df1.depth / 100
    df1["temperatureObserved"] = df1.Firntemp
    df1["date"] = df3.date.iloc[0]
    df1["latitude"] = df3.latitude.iloc[0]
    df1["longitude"] = df3.longitude.iloc[0]
    df1["elevation"] = df3.elevation.iloc[0]
    df1["reference"] = df3.reference.iloc[0]
    df1["site"] = df3.name.iloc[0]

    df2 = pd.read_csv(
        "data/temperature data/Wegener 1930/Eismitte_digitize_firntemperatures_wegener.csv", sep=";"
    )
    df2['date'] = "1930-" + df2.month.astype(str).apply(lambda x: x.zfill(2)) + "-15"
    df2 = df2.set_index('date').drop(columns=['month'])
    df_stack = ( df2
                .stack(future_stack=True)
                .to_frame(name='temperatureObserved')
                .reset_index()
                .rename(columns={'level_1':'depthOfTemperatureObservation'})
                )
    df_stack['depthOfTemperatureObservation'] = df_stack['depthOfTemperatureObservation'].str.replace('m','').astype(float)

    df_stack["latitude"] = df3.latitude.iloc[1]
    df_stack["longitude"] = df3.longitude.iloc[1]
    df_stack["elevation"] = df3.elevation.iloc[1]
    df_stack["site"] = df3.name.iloc[1]
    df_stack["reference"] = df3.reference.iloc[1]

    df_wegener = pd.concat((df1, df_stack), ignore_index=True,)
    df_wegener["reference_short"] = "Wegener (1940), Sorge (1940)"
    df_wegener["note"] = ""

    df_wegener["method"] = "electric resistance thermometer"
    df_wegener["durationOpen"] = "NA"
    df_wegener["durationMeasured"] = "NA"
    df_wegener["error"] = 0.2

    df_all = pd.concat((df_all,  df_wegener[needed_cols]), ignore_index=True)

    # %% Japanese stations
    del df1, df2, df3, df_stack, df_wegener
    df = pd.read_excel("data/temperature data/Japan/Sigma.xlsx")
    df["note"] = ""
    df_all = pd.concat((df_all,  df[needed_cols]), ignore_index=True)

    # %% Ambach
    del df
    meta = pd.read_csv(
        "data/temperature data/Ambach1979b/metadata.txt", sep="\t", header=None,
        names=["site", "file", "date", "latitude", "longitude", "elevation"],
    ).set_index('file')
    meta.date = pd.to_datetime(meta.date)
    for file in meta.index:
        df = pd.read_csv(
            "data/temperature data/Ambach1979b/" + file + ".txt", header=None,
            names=["temperatureObserved", "depthOfTemperatureObservation"]
        )

        plt.figure()
        df.set_index('depthOfTemperatureObservation').plot(ax=plt.gca(), marker = 'o')
        plt.title(file)
        for v in meta.columns: df[v] = meta.loc[file, v]

        df["reference"] = "Ambach, W., Zum Wärmehaushalt des Grönländischen Inlandeises: Vergleichende Studie im Akkumulations- und Ablationsgebiet,  Polarforschung 49 (1): 44-54, 1979"
        df["reference_short"] = "Ambach (1979)"
        df["note"] = "digitized"
        df["method"] = "NA"
        df["durationOpen"] = "NA"
        df["durationMeasured"] = "NA"
        df["error"] = "NA"

        df_all = pd.concat((df_all,  df[needed_cols]), ignore_index=True)

    # %% Kjær 2020 TCD data
    del df, file, meta, v
    df = pd.read_excel("data/temperature data/Kjær/tc-2020-337.xlsx")
    df["reference"] = "Kjær, H. A., Zens, P., Edwards, R., Olesen, M., Mottram, R., Lewis, G., Terkelsen Holme, C., Black, S., Holst Lund, K., Schmidt, M., Dahl-Jensen, D., Vinther, B., Svensson, A., Karlsson, N., Box, J. E., Kipfstuhl, S., and Vallelonga, P.: Recent North Greenland temperature warming and accumulation, The Cryosphere Discuss. [preprint], https://doi.org/10.5194/tc-2020-337 , 2021."
    df["reference_short"] = "Kjær et al. (2015)"
    df["note"] = ""
    df["method"] = "thermistor"
    df["durationOpen"] = 0
    df["durationMeasured"] = 0
    df["error"] = 0.1
    df_all = pd.concat((df_all,  df[needed_cols]), ignore_index=True)

    # %% Covi
    del df
    sites = ["DYE-2", "EKT", "SiteJ"]
    filenames = [
        "AWS_Dye2_20170513_20190904_CORR_daily",
        "AWS_EKT_20170507_20190904_CORR_daily",
        "AWS_SiteJ_20170429_20190904_CORR_daily",
    ]
    for site, filename in zip(sites, filenames):
        df = pd.read_csv("data/temperature data/Covi/" + filename + ".dat", skiprows=1)
        depth = np.flip(
            16 - np.concatenate((np.arange(16, 4.5, -0.5), np.arange(4, -1, -1)))
        )
        depth_ini = 1
        depth = depth + depth_ini
        df["date"] = df.TIMESTAMP.astype(str)
        temp_label = ["Tfirn" + str(i) + "(C)" for i in range(1, 29)]

        df.date = pd.to_datetime(df.date)
        # print(site, df.date.diff().unique())
        df = df.drop(columns=['TIMESTAMP'])

        if site in ["DYE-2", "EKT"]:
            # loading surface height from FirnCover station
            filepath = os.path.join("data/temperature data/FirnCover/FirnCoverData_2.0_2021_07_30.h5")
            sites = ["Summit",  "KAN-U", "NASA-SE", "Crawford", "EKT", "Saddle", "EastGrip", "DYE-2"]
            statmeta_df, sonic_df, rtd_df = load_firncover_metadata(filepath, sites)
            statmeta_df["elevation"] = [1840, 2119, 2361, 2370, 2456, 1942, 3208, 2666]

            df["surface_height"] = (
                -sonic_df.loc[site]
                .resample("D")
                .mean()
                .loc[df.date]
                .sonic_m.values
            )
        else:
            df["surface_height"] = df["SR50_corr(m)"]

        df["surface_height"] = df["surface_height"] - df["surface_height"].iloc[0]
        df["surface_height"] = df["surface_height"].interpolate().values

        depth_label = ["depth_" + str(i) for i in range(1, len(temp_label) + 1)]
        for i in range(len(temp_label)):
            df[depth_label[i]] = depth[i] + df["surface_height"].values
        df = df.set_index("date").resample('D').mean()

        df_stack = ( df[temp_label]
                    .stack(future_stack=True)
                    .to_frame(name='temperatureObserved')
                    .reset_index()
                    .rename(columns={'time':'date'}))
        df_stack['depthOfTemperatureObservation'] = df[depth_label].stack(future_stack=True).values
        df_stack=df_stack.loc[df_stack.temperatureObserved.notnull()]
        df_stack=df_stack.loc[df_stack.depthOfTemperatureObservation.notnull()]

        df_stack["site"] = site
        if site == "SiteJ":
            df_stack["latitude"] = 66.864952
            df_stack["longitude"] = -46.265141
            df_stack["elevation"] = 2060
        else:
            df_stack["latitude"] = statmeta_df.loc[site, "latitude"]
            df_stack["longitude"] = statmeta_df.loc[site, "longitude"]
            df_stack["elevation"] = statmeta_df.loc[site, "elevation"]
        df_stack["note"] = ""

        df_stack["reference"] = "Covi, F., Hock, R., and Reijmer, C.: Challenges in modeling the energy balance and melt in the percolation zone of the Greenland ice sheet. Journal of Glaciology, 69(273), 164-178. doi:10.1017/jog.2022.54, 2023. and Covi, F., Hock, R., Rennermalm, A., Leidman S., Miege, C., Kingslake, J., Xiao, J., MacFerrin, M., Tedesco, M.: Meteorological and firn temperature data from three weather stations in the percolation zone of southwest Greenland, 2017 - 2019. Arctic Data Center. doi:10.18739/A2BN9X444, 2022."
        df_stack["reference_short"] = "Covi et al. (2022, 2023)"
        df_stack["note"] = ""
        df_stack["method"] = "thermistor"
        df_stack["durationOpen"] = 0
        df_stack["durationMeasured"] = 1
        df_stack["error"] = 0.1

        if plot:
            plot_string_dataframe(df_stack, site)

        df_all = pd.concat((df_all,  df_stack[needed_cols] ), ignore_index=True )

    # %% Stauffer and Oeschger 1979
    del depth, depth_ini, depth_label, df, df_stack, filenames, filename, filepath
    del i,  rtd_df, site, sites, sonic_df, statmeta_df, temp_label
    df_s_o = pd.read_excel("data/temperature data/Stauffer and Oeschger 1979/Stauffer&Oeschger1979.xlsx")

    df_s_o["reference"] = "Clausen HB and Stauffer B (1988) Analyses of Two Ice Cores Drilled at the Ice-Sheet Margin in West Greenland. Annals of Glaciology 10, 23–27 (doi:10.3189/S0260305500004109)"
    df_s_o["reference_short"] = "Stauffer and Oeschger (1979)"
    df_s_o["note"] = "site location estimated by M. Luethi"
    df_s_o["method"] = "Fenwal Thermistor UUB 31-J1"
    df_s_o["durationOpen"] = 0
    df_s_o["durationMeasured"] = 0
    df_s_o["error"] = 0.1
    df_all = pd.concat((df_all, df_s_o[needed_cols]), ignore_index=True)

    # %% Schwager EGIG
    del df_s_o
    df_schwager = pd.read_excel("data/temperature data/Schwager/schwager.xlsx")

    df_schwager[
        "reference"
    ] = "Schwager, M. (2000): Eisbohrkernuntersuchungen zur räumlichen und zeitlichen Variabilität von Temperatur und Niederschlagsrate im Spätholozän in Nordgrönland - Ice core analysis on the spatial and temporal variability of temperature and precipitation during the late Holocene in North Greenland , Berichte zur Polarforschung (Reports on Polar Research), Bremerhaven, Alfred Wegener Institute for Polar and Marine Research, 362 , 136 p. . doi: 10.2312/BzP_0362_2000"
    df_schwager["reference_short"] = "Schwager (2000)"
    df_schwager["note"] = ""
    df_schwager.date = pd.to_datetime([str(y) + "-07-01" for y in df_schwager.date])
    df_schwager["method"] = "custom thermistors"
    df_schwager["durationOpen"] = 0
    df_schwager["durationMeasured"] = 0.5
    df_schwager["error"] = 0.5
    df_all = pd.concat((df_all,  df_schwager[needed_cols]), ignore_index=True)

    # %% Giese & Hawley
    del df_schwager
    df = pd.read_excel("data/temperature data/Giese and Hawley/giese_hawley.xlsx")

    df1 = df.iloc[:, :2]
    df1 = df1.loc[~np.isnan(df1.time1.values), :]
    df1["time"] = [
        datetime(int(d_y), 1, 1)
        + timedelta(seconds=(d_y - int(d_y)) * timedelta(days=365).total_seconds())
        for d_y in df1.time1.values
    ]
    df1 = df1.set_index("time").resample("D").mean().interpolate(method="cubic")

    df2 = df.iloc[:, 2:4]
    df2 = df2.loc[~np.isnan(df2.time2.values), :]
    df2["time"] = [
        datetime(int(d_y), 1, 1)
        + timedelta(seconds=(d_y - int(d_y)) * timedelta(days=365).total_seconds())
        for d_y in df2.time2.values
    ]
    df2 = df2.set_index("time").resample("D").mean().interpolate(method="cubic")

    df_giese = df.iloc[:, 4:]
    df_giese = df_giese.loc[~np.isnan(df_giese.time3.values), :]
    df_giese["time"] = [
        datetime(int(d_y), 1, 1)
        + timedelta(seconds=(d_y - int(d_y)) * timedelta(days=365).total_seconds())
        for d_y in df_giese.time3.values
    ]

    df_giese = df_giese.set_index("time").resample("D").mean().interpolate(method="cubic")

    df_giese["temp_8"] = df2.temp_8.values
    df_giese["depth_7"] = 6.5 + df1.depth1.values - df1.depth1.min()
    df_giese["depth_8"] = 9.5 + df1.depth1.values - df1.depth1.min()

    df_stack = ( df_giese[['temp_7','temp_8']]
                .stack(future_stack=True)
                .to_frame(name='temperatureObserved')
                .reset_index()
                .rename(columns={'time':'date'}))
    df_stack['depthOfTemperatureObservation'] = df_giese[['depth_7','depth_8']].stack(future_stack=True).values
    df_stack=df_stack.loc[df_stack.temperatureObserved.notnull()]
    df_stack=df_stack.loc[df_stack.depthOfTemperatureObservation.notnull()]
    site="Summit"
    df_stack["site"] = site
    df_stack["latitude"] = 72 + 35 / 60
    df_stack["longitude"] = -38 - 30 / 60
    df_stack["elevation"] = 3208

    df_stack["reference"] = "Giese AL and Hawley RL (2015) Reconstructing thermal properties of firn at Summit, Greenland, from a temperature profile time series. Journal of Glaciology 61(227), 503–510 (doi:10.3189/2015JoG14J204)"
    df_stack["reference_short"] = "Giese and Hawley (2015)"
    df_stack["note"] = "digitized and interpolated at 10m"

    df_stack["method"] = "thermistors"
    df_stack["durationOpen"] = 0
    df_stack["durationMeasured"] = 1
    df_stack["error"] = 0.5

    if plot:
        plot_string_dataframe(df_stack, site)

    df_all = pd.concat((df_all,  df_stack[needed_cols] ), ignore_index=True )

    # %% IMAU
    df_meta = pd.read_csv("data/temperature data/IMAU/meta.csv")
    depth_label = ["depth_" + str(i) for i in range(1, 6)]
    temp_label = ["temp_" + str(i) for i in range(1, 6)]
    df_imau = pd.DataFrame()
    for i, site in enumerate(["s5", "s6", "s9"]):
        df = pd.read_csv("data/temperature data/IMAU/" + site + "_tsub.txt")
        df["date"] = pd.to_datetime(df["year"], format="%Y") + pd.to_timedelta(
            df["doy"] - 1, unit="d"
        )
        df = df.set_index("date").drop(columns=["year", "doy"])

        for dep, temp in zip(depth_label, temp_label):
            df.loc[df[dep] < 0.2, temp] = np.nan
        if site == "s5":
            surface_height = df.depth_5 - df.depth_5.values[0]
            surface_height.loc["2011-09-01":] = surface_height.loc["2011-08-31":] - 9.38
            surface_height = surface_height.values
            df.loc["2011-08-29":"2011-09-05", temp_label] = np.nan
        else:
            surface_height = []
        if site == "s6":
            for dep, temp in zip(depth_label, temp_label):
                df.loc[df[dep] < 1.5, temp] = np.nan
            min_diff_to_depth = 3
        else:
            min_diff_to_depth = 3
        df = df.resample('D').mean()
        df_stack = (df[temp_label]
                    .stack(future_stack=True)
                    .to_frame(name='temperatureObserved')
                    .reset_index()
                    .rename(columns={'time':'date'}))
        df_stack['depthOfTemperatureObservation'] = df[depth_label].stack(future_stack=True).values
        df_stack=df_stack.loc[df_stack.temperatureObserved.notnull()]
        df_stack=df_stack.loc[df_stack.depthOfTemperatureObservation.notnull()]

        df_stack["site"] = site
        df_stack["note"] = ""
        df_stack["latitude"] = df_meta.loc[df_meta.site == site, "latitude"].values[0]
        df_stack["longitude"] = df_meta.loc[df_meta.site == site, "longitude"].values[0]
        df_stack["elevation"] = df_meta.loc[df_meta.site == site, "elevation"].values[0]

        df_stack["reference"] = " Paul C. J. P. Smeets, Peter Kuipers Munneke, Dirk van As, Michiel R. van den Broeke, Wim Boot, Hans Oerlemans, Henk Snellen, Carleen H. Reijmer & Roderik S. W. van de Wal (2018) The K-transect in west Greenland: Automatic weather station data (1993–2016), Arctic, Antarctic, and Alpine Research, 50:1, DOI: 10.1080/15230430.2017.1420954"
        df_stack["reference_short"] = "Smeets et al. (2018)"
        df_stack["method"] = "thermistor"
        df_stack["durationOpen"] = 0
        df_stack["durationMeasured"] = 1
        df_stack["error"] = 0.2

        if plot:
            plot_string_dataframe(df_stack, site)

        df_all = pd.concat((df_all,  df_stack[needed_cols] ), ignore_index=True )

    # %% Braithwaite
    del df_stack, depth_label, temp_label, df_imau, df_meta, i, min_diff_to_depth, dep
    del site, surface_height, temp, df
    df = pd.read_excel("data/temperature data/Braithwaite/data.xlsx")
    df = (
          df.set_index(['site', 'date', 'latitude', 'longitude', 'elevation'])
          .stack(future_stack=True).to_frame(name='temperatureObserved').reset_index()
          .rename(columns={'level_5': 'depthOfTemperatureObservation'})
          )
    df.site = df.site.astype(str)
    df["reference"] = "Braithwaite, R. (1993). Firn temperature and meltwater refreezing in the lower accumulation area of the Greenland ice sheet, Pâkitsoq, West Greenland. Rapport Grønlands Geologiske Undersøgelse, 159, 109–114. https://doi.org/10.34194/rapggu.v159.8218"
    df["reference_short"] = "Braithwaite (1993)"
    df["note"] = "from table"
    df["method"] = "thermistor"
    df["durationOpen"] = 0
    df["durationMeasured"] = 0
    df["error"] = 0.5

    df_all = pd.concat((df_all,  df[needed_cols]), ignore_index=True)

    # %% Clement
    df = pd.read_excel("data/temperature data/Clement/data.xlsx")
    df["depthOfTemperatureObservation"] = df.depth
    df["temperatureObserved"] = df.temperature

    df["reference"] = "Clement, P. “Glaciological Activities in the Johan Dahl Land Area, South Greenland, As a Basis for Mapping Hydropower Potential”. Rapport Grønlands Geologiske Undersøgelse, vol. 120, Dec. 1984, pp. 113-21, doi:10.34194/rapggu.v120.7870."
    df["reference_short"] = "Clement (1984)"
    df["note"] = "digitized"
    df["method"] = "thermistor"
    df["durationOpen"] = 0
    df["durationMeasured"] = 0
    df["error"] = 0.5

    df_all = pd.concat((df_all,  df[needed_cols]), ignore_index=True)

    # %% Nobles
    df = pd.read_excel("data/temperature data/Nobles Nuna Ramp/data.xlsx")
    df["depthOfTemperatureObservation"] = 8
    df["temperatureObserved"] = df["annual temp at 8m"]
    df["date"] = pd.to_datetime("1954-07-01")
    df["reference"] = "Nobles, L. H., Glaciological investigations, Nunatarssuaq ice ramp, Northwestern Greenland, Tech. Rep. 66, U.S. Army Snow, Ice and Permafrost Research Establishment, Corps of Engineers, 1960."
    df["reference_short"] = "Nobles (1960)"
    df["note"] = "digitized"
    df["method"] = "iron-constantan thermocouples"
    df["durationOpen"] = 0
    df["durationMeasured"] = 365
    df["error"] = 0.5

    df_all = pd.concat((df_all,  df[needed_cols]), ignore_index=True)
    # %% Schytt
    df = pd.read_excel("data/temperature data/Schytt Tuto/data.xlsx").rename(columns={'depth':'depthOfTemperatureObservation'})

    df["date"] = pd.to_datetime(df.date)
    df["reference"] = "Schytt, V. (1955) Glaciological investigations in the Thule Ramp area, U. S. Army Snow Ice and Permafrost Research Establishment, Corps of Engineers, Report 28, 88 pp. https://hdl.handle.net/11681/5989"
    df["reference_short"] = "Schytt (1955)"
    df["note"] = "from table"

    df["method"] = "copper-constantan thermocouples"
    df["durationOpen"] = 0
    df["durationMeasured"] = 0
    df["error"] = 0.5

    df_all = pd.concat((df_all,  df[needed_cols]), ignore_index=True)


    # %% Griffiths & Schytt
    df = pd.read_excel("data/temperature data/Griffiths Tuto/data.xlsx").rename(columns={'depth':'depthOfTemperatureObservation'})
    df["date"] = pd.to_datetime(df.date)
    df["reference"] = "Griffiths, T. M. (1960). Glaciological investigations in the TUTO area of Greenland., U. S. Army Snow Ice and Permafrost Research Establishment, Corps of Engineers, Report 47, 62 pp. https://hdl.handle.net/11681/5981"
    df["reference_short"] = "Griffiths (1960)"
    df["note"] = "from table"
    df["method"] = "copper-constantan thermocouples"
    df["durationOpen"] = 0
    df["durationMeasured"] = 0
    df["error"] = 0.5

    df_all = pd.concat((df_all,  df[needed_cols]), ignore_index=True)

    # %% Griffiths & Meier
    df = pd.read_excel("data/temperature data/Griffiths Tuto/data_crevasse3.xlsx").rename(columns={'depth':'depthOfTemperatureObservation'})
    df.note = "measurement made close to an open crevasse"
    df.temperatureObserved = pd.to_numeric(df.temperatureObserved, errors='coerce')
    df["date"] = pd.to_datetime(df.date)
    df["reference"] = "Griffiths, T. M. (1960). Glaciological investigations in the TUTO area of Greenland., U. S. Army Snow Ice and Permafrost Research Establishment, Corps of Engineers, Report 47, 62 pp. https://hdl.handle.net/11681/5981"
    df["reference_short"] = "Griffiths (1960)"

    df.loc[df.date <= '1955-12-31','reference'] = "Meier, M. F., Conel, J. E., Hoerni, J. A., Melbourne, W. G., & Pings, C. J. (1957). Preliminary Study of Crevasse Formation. Blue Ice Valley, Greenland, 1955. OCCIDENTAL COLL LOS ANGELES CALIF. https://hdl.handle.net/11681/6029"
    df.loc[df.date <= '1955-12-31',"reference_short"] = "Meier et al. (1957)"

    df["latitude"] = 76.43164
    df["longitude"] = -67.54949
    df["elevation"] = 800
    df["method"] = "copper-constantan thermocouples"
    df["durationOpen"] = 0
    df["durationMeasured"] = 0
    df["error"] = 0.5
    # only keeping measurements more than 1 m into the crevasse wall
    df = df.loc[df["distance from crevasse"] >= 1, :]

    df_all = pd.concat((df_all,  df[needed_cols]), ignore_index=True)

    # %% Vanderveen
    df = pd.read_excel("data/temperature data/Vanderveen et al. 2001/summary.xlsx")
    df = df.loc[df.date1.notnull(), :]
    df = df.loc[df.Temperature_celsius.notnull(), :]
    df["site"] = [str(s) for s in df.site]
    df["date"] = (
        pd.to_datetime(df.date1, errors='coerce') + (pd.to_datetime(df.date2, errors='coerce') - pd.to_datetime(df.date1, errors='coerce')) / 2
    )
    df["note"] = ""
    df.loc[np.isnan(df["date"]), "note"] = "only year available"
    df.loc[np.isnan(df["date"]), "date"] = pd.to_datetime(
        [str(y) + "-07-01" for y in df.loc[np.isnan(df["date"]), "date1"].values]
    )

    df["temperatureObserved"] = df["Temperature_celsius"]
    df["depthOfTemperatureObservation"] = df["Depth_centimetre"] / 100

    tmp, ind = np.unique(
        [str(x) + ' ' + str(y) for (x, y) in zip(df.site, df.date)], return_inverse=True
    )

    df["reference"] = "van der Veen, C. J., Mosley-Thompson, E., Jezek, K. C., Whillans, I. M., and Bolzan, J. F.: Accumulation rates in South and Central Greenland, Polar Geography, 25, 79–162, https://doi.org/10.1080/10889370109377709, 2001."
    df["reference_short"] = "van der Veen et al. (2001)"
    df["method"] = "thermistor"
    df["durationOpen"] = 8 * 24
    df["durationMeasured"] = 0
    df["error"] = 0.1

    df_all = pd.concat((df_all, df[needed_cols]), ignore_index=True)

    # %% Thomsen shallow thermistor
    df = pd.read_excel("data/temperature data/Thomsen/data-formatted.xlsx").rename(
        columns={'depth':"depthOfTemperatureObservation",
                 "temperature": "temperatureObserved"})

    df["note"] = "from unpublished pdf"
    df["date"] = pd.to_datetime(df.date)
    df["reference"] = "Thomsen, H. ., Olesen, O. ., Braithwaite, R. . and Bøggild, C. .: Ice drilling and mass balance at Pâkitsoq, Jakobshavn, central West Greenland, Rapp. Grønlands Geol. Undersøgelse, 152, 80–84, doi:10.34194/rapggu.v152.8160, 1991."
    df["reference_short"] = "Thomsen et al. (1991)"
    df["method"] = "thermistor"
    df["durationOpen"] = "NA"
    df["durationMeasured"] = "NA"
    df["error"] = 0.2

    df_all = pd.concat((df_all, df[needed_cols]), ignore_index=True)

    # %% Checking values
    # del df
    df_all['temperatureObserved'] = df_all.temperatureObserved.astype(float)
    df_all['depthOfTemperatureObservation'] = df_all.depthOfTemperatureObservation.astype(float)
    df_all = df_all.loc[~df_all.temperatureObserved.isnull(), :]
    df_all = df_all.loc[~df_all.depthOfTemperatureObservation.isnull(), :]
    df_all = df_all.loc[df_all.depthOfTemperatureObservation>0, :]
    df_all = df_all.loc[df_all.temperatureObserved<1, :]
    df_all.temperatureObserved = df_all.temperatureObserved.astype(float)
    df_all.depthOfTemperatureObservation = df_all.depthOfTemperatureObservation.astype(float)
    df_all = df_all.loc[df_all.temperatureObserved.notnull(),:]
    df_all = df_all.loc[df_all.depthOfTemperatureObservation.notnull(),:]

    # some renaming
    df_all.loc[df_all.method=='Thermistor', 'method'] = 'thermistors'
    df_all.loc[df_all.method=='Thermistors', 'method'] = 'thermistors'
    df_all.loc[df_all.method=='thermistor', 'method'] = 'thermistors'
    df_all.loc[df_all.method=='thermistor string', 'method'] = 'thermistors'
    df_all.loc[df_all.method=='custom thermistors', 'method'] = 'thermistors'
    df_all.loc[df_all.method=='NA', 'method'] = 'not available'
    df_all.loc[df_all.method.isnull(), 'method'] = 'not available'
    df_all.loc[df_all.method=='not_reported', 'method'] = 'not available'
    df_all.loc[df_all.method=='digital Thermarray system from RST©', 'method'] = 'RST ThermArray'
    df_all.loc[df_all.method=='digital thermarray system from RST©', 'method'] = 'RST ThermArray'

    return df_all
