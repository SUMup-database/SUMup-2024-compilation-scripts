import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
col_needed = ['profile',  'reference_short', 
                       'reference', 'method_key','method', 'date', 'timestamp', 'latitude', 
                       'longitude', 'elevation', 'start_depth', 'stop_depth',
                       'midpoint', 'density', 'error']

def readSnowEx(file_name, sheet_name):
    df = pd.read_excel(file_name, sheet_name=sheet_name)

    """
    indexing convention df.iloc[row, column]:
    rows: index = row number - 2
    columns: 0 = A, 1 = B, ...
    example cell A3 = df.iloc[1, 0]
    """
    
    # indexing metadata:
    site_id = df.iloc[1, 0]
    site_lat = df.iloc[6, 7]
    site_lon = df.iloc[6, 11]
    site_elev = df.iloc[6, 16]
    
    time = df.iloc[6, 0]
    observer = df.iloc[1, 7]
    gps_uncertainty = df.iloc[6, 18]
    comments = df.iloc[1, 20]

    pss = df.iloc[6, 4]
    profile_temp_start = df.iloc[4, 18]
    profile_temp_end = df.iloc[4, 19]
    
    # reformat date into year, month, and day to fit into SWE summary table:
    try: 
        date = df.iloc[4, 0]
        date = str(date)
        date = date.split(" ")[0]
        day = int(date.split("-")[2])
        month = int(date.split("-")[1])
        year = int(date.split("-")[0])
    except: # attempted to handle data without date information
        print("Incomplete date information in source data")
        day = np.nan
        month = np.nan
        year = np.nan
    
    # create metadata dataframes:
    observation_meta = pd.DataFrame(data=(site_id, site_lat, site_lon, site_elev,
                                          date, day, month, year, time, observer,
                                          gps_uncertainty, comments),
                                    index=["site_id", "site_lat", "site_lon", "site_elev",
                                           "date", "day", "month", "year", "time",
                                           "observer", "gps_uncertainty", 
                                           "comments"]).T
    profile_meta = pd.DataFrame(data=(pss, profile_temp_start, profile_temp_end),
                                index=["pss", "profile_temp_start", "profile_temp_end"]).T

    # indexing density and temperature data:
    depth_top = df.iloc[10:100, 0] # length of column extended to 100 to (hopefully) include the longest dataseries 
    depth_bottom = df.iloc[10:100, 2]
    density_a = df.iloc[10:100, 3]
    density_b = df.iloc[10:100, 4]
    density_note = df.iloc[10:100, 5]
    cutter = df.iloc[10:100, 6]
    temp_depth = df.iloc[10:100, 7]
    temp = df.iloc[10:100, 8]
    
    # create density dataframe:
    density = pd.DataFrame(
        data=(depth_top, depth_bottom, density_a, 
              density_b, density_note, cutter, temp_depth, temp),
        index=["depth_top", "depth_bottom", "density_a", "density_b",
               "density_note", "cutter", "temp_depth", "temp"]).T
    density.reset_index(inplace=True, drop=True)     
    
    df_snowex = pd.concat([observation_meta, profile_meta, density], axis=1)
    df_snowex.reset_index(inplace=True, drop=True) # reindex df_snowex
        
    return observation_meta, density

def process():
    list_of_paths = os.listdir("snowex/")
    list_of_paths = [ "snowex/"+f for f in list_of_paths]
    print('Loading GEUS snow pit and firn core dataset')
    # option to include a density vs. depth plot:
    do_plot = 0 # change to 1 to produce plot
    
    list_add = []
    for fn in list_of_paths:
        xl = pd.ExcelFile(fn)    
        sheets = xl.sheet_names
        print(fn.split('/')[-1])
        sentence_list=[]
        if do_plot:
            fig, ax = plt.subplots(figsize=(7, 6))
        for i, sheet in enumerate(sheets):
            print('   ',sheet)
            observation_meta, df = readSnowEx(file_name=fn, sheet_name=sheet)
            if observation_meta.day.isnull().all():
                continue
            if observation_meta.site_lat.isnull().sum():
                print('missing coordinates, skipping')
                continue

            df = df.rename(columns={'depth_top':'start_depth',
                                    'depth_bottom':'stop_depth'})
            df = df.loc[~df[['start_depth','stop_depth']].isnull().all(axis=1),:]
            df['start_depth'] = df.start_depth/100
            df['stop_depth'] = df.stop_depth/100
            df['density'] = df[['density_a', 'density_b']].mean(axis=1)
            df['error'] = df[['density_a', 'density_b']].std(axis=1)
            df['midpoint'] = df.start_depth + (df.stop_depth-df.start_depth)/2
            
            df['profile'] = fn.split('_')[1].split('.xlsx')[0]
            if len(sheets)>1:
                df['profile'] = fn.split('_')[1].split('.xlsx')[0]+'_'+sheet
                
            df['reference_short'] = 'GEUS snow and firn data (2023)'
            df['reference'] = 'Vandecrux, B.; Box, J.; Ahlstrøm, A.; Fausto, R.; Karlsson, N.; Rutishauser, A.; Citterio, M.; Larsen, S.; Heuer, J.; Solgaard, A.; Colgan, W.: GEUS snow and firn data in Greenland, https://doi.org/10.22008/FK2/9QEOWZ , GEUS Dataverse, 2023'            
    
            df['method_key'] = 6
            df['method'] = "Density cutter – size unknown"
    
            if observation_meta.time.isnull().all():
                df['timestamp'] = observation_meta.date.iloc[0]
            else:
                df['timestamp'] = observation_meta.date.iloc[0]+'T'+observation_meta.time.astype(str).iloc[0].replace(' (UTC',':00').replace(')','')
            df['date'] = int(observation_meta.date.iloc[0].replace('-',''))
            df['latitude'] = observation_meta.site_lat.iloc[0]
            df['longitude'] = -abs(observation_meta.site_lon.iloc[0])
            df['elevation'] = observation_meta.site_elev.iloc[0]

            list_add.append(df)
            
            if do_plot:
                plt.plot(df["midpoint"], df["density"],marker='o', label=sheet)

        if do_plot:
            plt.title(fn.split('/')[-1])
            plt.legend()
            plt.show()
            plt.xlabel('Depth (cm)')
            plt.ylabel('Density (kg m-3)')
        
    df_add = pd.concat(list_add)
    # df_add=df_add.loc[df_add.notnull()]
            
    for v in col_needed:
        assert v in df_add.columns, f'{v} is missing'
    df_add[col_needed].to_csv('data_formatted.csv', index=None)

if __name__ == "__main__":
    process()
    
    
    
