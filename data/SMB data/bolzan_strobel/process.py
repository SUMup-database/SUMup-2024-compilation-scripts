import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
import pandas as pd
import xarray as xr
import os, re
from datetime import datetime, timedelta

col_needed = ['start_date', 'end_date', 'start_year','end_year', 'latitude', 
              'longitude', 'elevation', 'notes', 'smb', 'error', 'name', 
              'reference', 'method', 'reference_short']


# Function to convert year and fractional year to datetime
def convert_to_datetime(year_fraction):
    year = int(year_fraction)
    remainder = year_fraction - year
    # Calculate the number of days in the year (accounting for leap years)
    days_in_year = (datetime(year + 1, 1, 1) - datetime(year, 1, 1)).days
    # Calculate the exact date by adding the remainder in days
    exact_date = datetime(year, 1, 1) + timedelta(days=remainder * days_in_year)
    return exact_date


def extract_information(file_path):
    with open(file_path, 'r') as file:
        text = file.read()

    # Extract citations
    citation_pattern = re.compile(r'Citation:\s*(.*?)\n', re.DOTALL)
    related_to_pattern = re.compile(r'Related to:\s*(.*?)\n', re.DOTALL)
    citations = citation_pattern.findall(text)
    related_to = related_to_pattern.findall(text)

    # Extract event details
    event_pattern = re.compile(r'Event\(s\):\s*(.*?)\s*\*\s*LATITUDE:\s*([\d\.\-]+)\s*\*\s*LONGITUDE:\s*([\d\.\-]+)\s*\*\s*DATE/TIME:.*?\s*ELEVATION:\s*([\d\.\-]+)\s*', re.DOTALL)
    event_details = event_pattern.findall(text)

    return citations, related_to, event_details

def process():
    # %% 
    folder = '.'
    
    file_names = []
    
    for file_name in os.listdir(folder):
        if file_name.endswith('.tab'): 
            file_names.append(file_name)
    
    file_names.sort()
    plot = False
    
    if plot:
        path_to_sumup = '../../SUMup 2023/SMB/'
        df = pd.read_csv(path_to_sumup+'SUMup_2023_SMB_greenland.csv')
    
        fig, axes = plt.subplots(nrows = 3, ncols = 3, figsize=(18,9))
    
    list_add = []
    
    for i, file_name in enumerate(file_names):
        key = file_name[:2]
    
        file_path = os.path.join(folder, file_name)
        bolz = pd.read_csv(file_path, delimiter='\t', skiprows=19)
    
        bolz['mwe1'] = bolz['Depth ref [m]'] * 0.917  # ice equiv to water equiv
        bolz['mwe2'] = cumulative_trapezoid(bolz['Density [kg/m**3]'] / 1000, 
                                            bolz['Depth ice/snow [m]'], initial=0)
    
        # t = np.arange(np.min(bolz['Age [a AD/CE]']),np.max(bolz['Age [a AD/CE]']))
        bolz.sort_values(by='Age [a AD/CE]',inplace=True)
        t = np.arange(np.ceil(bolz['Age [a AD/CE]'].min()), bolz['Age [a AD/CE]'].max())
        if t[-1] != bolz['Age [a AD/CE]'].max():
            t = np.append(t, bolz['Age [a AD/CE]'].max())
        smb1 = -np.diff(np.interp(t,bolz['Age [a AD/CE]'], bolz['mwe1']))
        smb2 = -np.diff(np.interp(t,bolz['Age [a AD/CE]'], bolz['mwe2']))
    
    
        new_bolz = pd.DataFrame(smb2, columns=['smb'])
        new_bolz['start_date'] =[convert_to_datetime(dy) for dy in t[:-1]]
        new_bolz['end_date'] =[convert_to_datetime(dy) for dy in t[1:]]
            
        citations, related_to, event_details = extract_information(file_path)
        new_bolz['reference'] = related_to[0].strip() +'. Data: '+ citations[0].strip()
        new_bolz['reference_short'] = 'Bolzan and Strobel (1994, 1999a,b,c,d,e,f,g, 2001a,b)'
        new_bolz['name'] = event_details[0][0].strip().replace('GISP2','GISP2 ').replace('Site','Site ')
        new_bolz['latitude'] = float(event_details[0][1])
        new_bolz['longitude'] = float(event_details[0][2])
        new_bolz['elevation'] = float(event_details[0][3])
        new_bolz['method'] = "snow pit, dO18 dating, subannual resolution"
        new_bolz['notes'] = ""
        new_bolz['start_year'] = new_bolz['start_date'].dt.year
        new_bolz['end_year'] = new_bolz['end_date'].dt.year

        new_bolz['start_date'] = new_bolz['start_date'].dt.strftime('%Y-%m-%d')
        new_bolz['end_date'] = new_bolz['end_date'].dt.strftime('%Y-%m-%d')
        
        new_bolz['error'] = np.nan
        new_bolz['reference_key'] = int(key)

        list_add.append(new_bolz)
        print(file_path,event_details[0][0].strip())
    
        if plot:
            df_key = df[df['reference_key'] == int(key)] 
            axes=axes.flatten()
            ax = axes[i]
            ax.set_title(f'{file_name}')
            ax.plot(df_key.start_year, df_key.smb, label = 'old sumup', color='black', alpha=0.6)
            ax.plot(t[:-1],smb1,label='Bolz. published')
            ax.plot(t[:-1],smb2,label='Bolz. correct',linewidth=0.7)
            ax.plot(new_bolz.start_date.dt.year, new_bolz.smb,label='new sumup')
            ax.set_ylabel('smb') 
            ax.legend(loc='upper right')
            ax.grid()
            plt.tight_layout()
    
    df_add = pd.concat(list_add)
    # %% 
    for v in col_needed:
        assert v in df_add.columns, f'{v} is missing'
        
    df_add.to_csv('data_formatted.csv',index=None)

    
if __name__ == "__main__":
    process()
    
    
    

