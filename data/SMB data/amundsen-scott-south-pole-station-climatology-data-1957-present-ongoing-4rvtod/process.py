import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re, os
import warnings

# Columns needed in the final output
col_needed = ['method_key', 'start_date', 'end_date', 'start_year', 'end_year',
              'latitude', 'longitude', 'elevation', 'notes', 'smb', 'error',
              'name', 'reference', 'method', 'reference_short']

# Helper function to extract text following a specified string
def find_what_comes_after_str(text, str_to_find):
    match = re.search(f"{str_to_find}(.*)", text)
    return match.group(1).strip() if match else "m"

# Convert inches to meters
def inch_to_meter(inches):
    try:
        return float(inches) * 0.0254
    except ValueError:
        print(inches)
        return np.nan

# Create an empty list to store each month's data
all_lines = []

for year in range(1959, 2024):
    for month in range(1, 13):
        try:
            filename = f"{year}/{str(month).zfill(2)}{str(year)[-2:]}.lcd"

            if os.path.exists(filename):
                with open(filename, 'r') as file:
                    text = file.read()
            else:
                filename = f"{year}/new{str(month).zfill(2)}{str(year)[-2:]}.lcd"
                with open(filename, 'r') as file:
                    text = file.read()

            # Extract delta snow height
            delta_snow_height = find_what_comes_after_str(text, "NET CHANGE IN SNOWSTAKE FIELD:")
            delta_snow_height = (delta_snow_height
                                 .lower()
                                 .replace("inches", "")
                                 .replace("+/-", "").strip())
            delta_snow_height = np.nan if delta_snow_height == 'm' else inch_to_meter(delta_snow_height)

            # Initialize the data line with the extracted information
            line = {
                'start_year': year,
                'end_year': year,
                'start_date': pd.to_datetime(f"{year}-{month:02}-01"),
                'end_date': pd.to_datetime(f"{year}-{month:02}-{pd.Period(f'{year}-{month:02}').days_in_month}"),
                'method': 'stake measurements',
                'method_key': 4,
                'notes': '',
                # 'smb': 285.5 * delta_snow_height,
                'smb': -4.8E-4 / 3 * delta_snow_height**3 + 0.0196/2 * delta_snow_height**2 + 0.35*delta_snow_height,
                'error': 0.5 * delta_snow_height,
                'latitude': -90.0,
                'longitude': 0.0,
                'elevation': 2835,
                'reference': "South Pole Meteorology Office: Amundsen-Scott South Pole Station climatology data, 1957-present (ongoing). AMRDC Data Repository, accessed DD-MM-YYYY, https://doi.org/10.48567/szgp-6h49.",
                'reference_short': "South Pole Meteorology Office (2024)",
                'name': 'Amundsen-Scott South Pole Station'
            }

            all_lines.append(pd.DataFrame([line]))

        except FileNotFoundError:
            warnings.warn(f"File for {year}-{month:02} not found. Skipping.")
            continue

# Concatenate all data and save to CSV
df = pd.concat(all_lines, ignore_index=True)
df=df.loc[df.smb.notnull()]
df = df.loc[df.smb<40/1000,:]
df = df.loc[df.smb>=0, :]
df[col_needed].to_csv('data_formatted.csv', index=None)

# %%
import pandas as pd

# Load the Excel sheet with month columns (1-12) and year rows (1983-2021)
file_path = "Snow accumulation at SPS.xlsx"
sheet_name = "mm w.e."

# Read the data, specifying the correct header and columns
df_Zhai = pd.read_excel(file_path, sheet_name=sheet_name, header=1, usecols="A:M")
df_Zhai.columns = ["year"] + list(range(1, 13))  # Rename columns to make months explicit

# Reshape the DataFrame to have one row per month-year pair
df_long = df_Zhai.melt(id_vars="year", var_name="month", value_name="smb")

# Create a timestamp column as the start of each month
df_long["timestamp"] = pd.to_datetime(df_long["year"].astype(str) + "-" + df_long["month"].astype(str) + "-01")

# Select the final format with timestamp and smb columns
df_final = df_long[["timestamp", "smb"]].sort_values(by="timestamp").reset_index(drop=True)

# Display or save the final DataFrame
plt.figure()
(df.set_index('start_date').smb*1000).plot(label='SUMUp')
df_final.set_index('timestamp').smb.plot(label='Zhai et al.')
