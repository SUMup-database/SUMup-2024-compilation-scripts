import os
import re
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Path to the sites folder
site_folder = r'C:\Users\bav\GitHub\PROMICE data\aws-l3-dev\sites'

# Variables to extract
temp_vars = [ 't_i_1', 't_i_2', 't_i_3', 't_i_4', 
                        't_i_5', 't_i_6', 't_i_7', 't_i_8', 't_i_9', 't_i_10', 't_i_11',
                        'd_t_i_1', 'd_t_i_2', 'd_t_i_3', 'd_t_i_4', 'd_t_i_5', 'd_t_i_6', 
                        'd_t_i_7', 'd_t_i_8', 'd_t_i_9', 'd_t_i_10', 'd_t_i_11', 't_i_10m']
variables_to_extract = ['time', 'lat', 'lon', 'alt', ] + temp_vars

# Collect all datasets
datasets = []

for root, dirs, files in os.walk(site_folder):
    for file in files:
        if file[:3] in ['ZAC', 'XXX', 'Roo','MIT','LYN','UWN','FRE']:
            continue
        if file.startswith('NUK_N') | file.endswith('_B_day.nc'):
            continue
        if file.endswith('day.nc'):
            print(file)
            # Get site name from the file path
            site_name = os.path.basename(root)
            
            # Open the NetCDF file
            filepath = os.path.join(root, file)
            ds = xr.open_dataset(filepath)
            
            # Select and rename variables
            ds_sel = ds[[v for v in variables_to_extract if v in ds.data_vars]]
            if all(ds_sel[temp_var].isnull().all(dim='time') for temp_var in temp_vars):
                print('No temperature data')
                continue
            ds_sel = ds_sel.assign_coords(site=site_name)
            
            # Add the dataset to the list
            datasets.append(ds_sel)

# Concatenate datasets along the 'site' dimension
merged_ds = xr.concat(datasets, dim='site')

# Save the merged dataset to a NetCDF file with compression
merged_ds.to_netcdf('./PROMICE_GC-Net_merged.nc', encoding={var: {"zlib": True} for var in merged_ds.data_vars})

#%% 
import xarray as xr
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
needed_cols = [ 'timestamp', 'latitude', 'longitude', 'elevation', 'depth',
                'temperature',  'name', ]
def stack_multidepth_df_from_ds(site_ds, temp_var, depth_var):
    print('    -> Converting to multi-depth DataFrame')

    # Convert the xarray Dataset to a DataFrame, while keeping time and site as separate columns
    df_in = site_ds.to_dataframe().reset_index()
    df_in = df_in.dropna(subset=['time', 'lat', 'lon'])
    df_in = df_in.dropna(subset=temp_var, how='all')

    # Filter temperature values (you can adjust the filter as needed)
    for v in temp_var:
        df_in.loc[df_in[v] > 1, v] = np.nan  # Assume valid temperature range is below 1°C
        df_in.loc[df_in[v] < -70, v] = np.nan  # Filter extreme low temperatures

    # Drop rows where all temp_var are still NaN after filtering
    df_in = df_in.loc[~df_in[temp_var].isnull().all(axis=1), :]

    # Separate out the variable that doesn't have a corresponding depth
    # if 't_i_10m' in temp_var:
    #     df_ti_10m = df_in[['site', 'time', 'lat', 'lon', 'alt']].dropna(subset=['t_i_10m'])
    #     df_ti_10m['depth'] = 10  # Assign a fixed depth of 10 meters for t_i_10m
    #     df_ti_10m = df_ti_10m.rename(columns={'t_i_10m': 'temperature'})
    #     temp_var.remove('t_i_10m')
    # else:
    #     df_ti_10m = pd.DataFrame()

    # Stack temperature and depth data (excluding t_i_10m)
    df_stack = df_in[temp_var].rename(columns=dict(zip(temp_var, 
        range(1, len(temp_var) + 1)))).stack(future_stack=True).to_frame(name='temperature').reset_index()

    # Add depth values from depth_var, ensuring alignment with temp_var
    df_stack['depth'] = df_in[depth_var].rename(columns=dict(zip(depth_var, 
        range(1, len(depth_var) + 1)))).stack(future_stack=True).to_frame(name='depth').values

    # Merge the stacked temperature and depth with time, lat, lon
    for var in ['time', 'lat', 'lon','alt','site']:
        df_stack[var] = df_in.loc[df_stack.level_0, var].values
    df_stack = (df_stack.drop(columns=['level_0','level_1'])
                    .rename(columns={'lat':'latitude',
                                    'lon':'longitude',
                                    'site':'name',
                                    'alt':'elevation',
                                    'time':'timestamp'}))
    df_stack = df_stack.loc[df_stack.latitude.notnull()]
    df_stack = df_stack.loc[df_stack.depth.notnull()]
    df_stack = df_stack.loc[df_stack.temperature.notnull()]
    # Concatenate df_ti_10m back into the stacked DataFrame (if it exists)
    # if not df_ti_10m.empty:
    #     df_stack = pd.concat([df_stack, df_ti_10m], ignore_index=True)

    return df_stack

def plot_temperature_depth(df_stack, site_name):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df_stack['time'], -df_stack['depth'], c=df_stack['temperature'], cmap='viridis', s=10)
    plt.ylim(-df_stack['depth'].max(), 0)
    plt.colorbar(scatter, label='Temperature (°C)')
    plt.xlabel('Time')
    plt.ylabel('Depth (m)')
    plt.title(f'Temperature Profile Over Time at {site_name}')
    plt.grid(True)
    
    # Save the plot
    plt.savefig(f'figures/{site_name}_temp_depth_plot.png', dpi=300)
    plt.show()

def process_merged_ds_and_plot(ds, temp_var_prefix='t_i_', depth_var_prefix='d_t_i_', plot=True):
    temp_var = [f'{temp_var_prefix}{i}' for i in range(1, 12)] #+ ['t_i_10m']  # Include t_i_10m
    depth_var = [f'{depth_var_prefix}{i}' for i in range(1, 12)]  # Only depth_var for t_i_1 to t_i_11

    all_sites_df = pd.DataFrame()

    # Iterate over each site in the Dataset
    for site in ds['site'].values:
        print(f"Processing site: {site}")
        site_ds = ds.sel(site=site)
        
        # Stack and process the data for this site
        df_stack = stack_multidepth_df_from_ds(site_ds, temp_var, depth_var)

        if plot:
            plot_temperature_depth(df_stack, site)

        # Concatenate the data for all sites
        all_sites_df = pd.concat([all_sites_df, df_stack], ignore_index=True)
    
    return all_sites_df

# Main execution block
# Assume merged_ds is already loaded as an xarray Dataset
ds = xr.open_dataset('./PROMICE_GC-Net_merged.nc')

# Process the dataset and generate the plot
all_sites_df = process_merged_ds_and_plot(ds, plot=False)

# all_sites_df['reference_short'] = "PROMICE/GC-Net: Fausto et al. (2021); How et al. (2023)"

# all_sites_df['reference'] = ("Fausto, R. S., van As, D., Mankoff, K. D., Vandecrux, B., Citterio, M., Ahlstrøm, A. P., Andersen, S. B., "
#               "Colgan, W., Karlsson, N. B., Kjeldsen, K. K., Korsgaard, N. J., Larsen, S. H., Nielsen, S., Pedersen, A Ø., "
#               "Shields, C. L., Solgaard, A. M., and Box, J. E.: Programme for Monitoring of the Greenland Ice Sheet "
#               "(PROMICE) automatic weather station data, Earth Syst. Sci. Data, 13, 3819–3845, https://doi.org/10.5194/essd-13-3819-2021, 2021."
#               " and How, P., Ahlstrøm, A.P., Andersen, S.B., Box, J.E., Citterio, M., Colgan, W.T., Fausto, R., Karlsson, N.B., "
#               "Jakobsen, J., Larsen, S.H., Mankoff, K.D., Pedersen, A.Ø., Rutishauser, A., Shields, C.L., Solgaard, A.M., "
#               "van As, D., Vandecrux, B., Wright, P.J., PROMICE and GC-Net automated weather station data in Greenland, "
#               "https://doi.org/10.22008/FK2/IW73UU, GEUS Dataverse, 2022.")
# Save the concatenated data to CSV
output_csv = './data_formatted.csv'
all_sites_df[needed_cols].to_csv(output_csv, index=False)

print(f"Data formatted and saved to {output_csv}")
