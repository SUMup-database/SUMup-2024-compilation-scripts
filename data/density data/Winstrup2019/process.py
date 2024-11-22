import pandas as pd
import numpy as np

# Define the necessary columns for the final formatted DataFrame
col_needed = ['profile', 'reference_short', 'reference', 'method_key', 'method', 'date', 'timestamp',
              'latitude', 'longitude', 'elevation', 'start_depth', 'stop_depth', 'midpoint', 'density', 'error']

# Metadata
reference_text = "Winstrup, M., Vallelonga, P., Kjær, H. A., Fudge, T. J., Lee, J. E., Riis, M. H., Edwards, R., Bertler, N. A. N., Blunier, T., Brook, E. J., Buizert, C., Ciobanu, G., Conway, H., Dahl-Jensen, D., Ellis, A., Emanuelsson, B. D., Hindmarsh, R. C. A., Keller, E. D., Kurbatov, A. V., Mayewski, P. A., Neff, P. D., Pyne, R. L., Simonsen, M. F., Svensson, A., Tuohy, A., Waddington, E. D., and Wheatley, S.: A 2700-year annual timescale and accumulation history for an ice core from Roosevelt Island, West Antarctica, Clim. Past, 15, 751–779, https://doi.org/10.5194/cp-15-751-2019, 2019. Data: Winstrup, Mai (2019): Roosevelt Island Climate Evolution (RICE) ice core: The RICE17 chronology and accumulation record for the past 2700 years. PANGAEA, https://doi.org/10.1594/PANGAEA.899147"
reference_short = "Winstrup et al. (2019); Winstrup (2019)"
latitude, longitude, elevation = -79.364000, -161.706000, 550.0

# Function to process the tab file
def process_tab_file(file_path):
    # Read the tab-separated data
    df = pd.read_csv(file_path, delimiter="\t", skiprows=35)
    # Rename columns for easier access and standard format
    df.columns = ['depth', 'age_AD_CE', 'age_ka_BP', 'age_max_ka', 'age_min_ka',
                  'annual_layer_thickness', 'density', 'thinning_factor_most_likely',
                  'thinning_factor_min', 'thinning_factor_max', 'accumulation_snow_most_likely',
                  'accumulation_snow_min', 'accumulation_snow_max']

    # Calculate midpoint, start and stop depth (assuming midpoint = depth)
    df['density'] = df['density']*1000
    df['stop_depth'] = df['depth']
    df['start_depth'] = df['stop_depth'].shift(1, fill_value=0)
    df['midpoint'] = (df['start_depth'] + df['stop_depth']) / 2

    # Fill additional metadata
    df['profile'] = 'RICE ice core'
    df['reference_short'] = reference_short
    df['reference'] = reference_text
    df['method_key'] = 5
    df['method'] = 'ice core'
    df['timestamp'] = '2013-01-01'
    df['date'] = 20130101
    df['latitude'] = latitude
    df['longitude'] = longitude
    df['elevation'] = elevation
    df['error'] = np.nan  # Assuming no error data available

    return df

# Main function to process the file and save the output
def process_all():
    file_path = 'Winstrup-etal_2019_V2.tab'  # Assuming the file is in the current directory
    df_formatted = process_tab_file(file_path)

    # Reorder and select the necessary columns
    df_formatted = df_formatted[col_needed]

    # Save the processed data to CSV
    df_formatted.to_csv('data_formatted.csv', index=False)

if __name__ == "__main__":
    process_all()
