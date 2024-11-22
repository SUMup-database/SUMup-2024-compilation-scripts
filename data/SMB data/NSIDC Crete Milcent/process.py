import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
col_needed = ['method_key','start_date', 'end_date', 'start_year','end_year', 'latitude', 
              'longitude', 'elevation', 'notes', 'smb', 'error', 'name', 
              'reference', 'method', 'reference_short']
def process():
    plot = False
    df_metadata = pd.read_excel('metadata.xlsx')
    list_add = []
    for f in df_metadata.file:
        print('   ', f)
        df_meta = pd.read_csv(f, sep='£', engine='python', header=None,
                              skip_blank_lines=False)
        df_meta.loc[df_meta[0].isnull(), 0] = ''
        ind_start = df_meta[0].str.startswith('DATA:')[::-1].idxmax()+1
        if ind_start == len(df_meta[0]):
            ind_start = df_meta[0].str.startswith('#')[::-1].idxmax()+1
        df = pd.read_csv(f, comment='#',
                          skiprows=ind_start+1, sep='\t', header=None)
        try:
            df.columns = ['end_year','smb']
        except:
            df.columns = ['end_year','smb','1','2','3','4','5','6','7','8']
        df=df.loc[df.end_year>1000,:]
        
    
        if plot:
            fig=plt.figure(figsize=(15,10))
            df.set_index('end_year').smb.plot(ax=plt.gca(), marker='o', ls='None')
            plt.title(f+' from NSIDC')
            plt.ylabel('Accumulation (cm ice yr-1)')
            plt.xlabel('year')
            # plt.close(fig)
        df['name'] = df_metadata.loc[df_metadata.file == f, 'name'].item()
        df['latitude'] = df_metadata.loc[df_metadata.file == f, 'latitude'].item()
        df['longitude'] = df_metadata.loc[df_metadata.file == f, 'longitude'].item()
        df['elevation'] = df_metadata.loc[df_metadata.file == f, 'elevation'].item()
        df['reference'] = 'Clausen, H.B., N.S. Gundestrup, S.J. Johnsen, R. Bindschadler, and J. Zwally. 1988. Glaciological investigations in the Crete area, Central Greenland:  A search for a new deep-drilling site. Annals of Glaciology, 10:10-15. Data: Clausen, H.B.; Gundestrup, N.; Johnsen, S.J.; Bindschadler, R.; Zwally, J. (1998): NOAA/WDS Paleoclimatology - Crete, Milcent - Oxygen Isotope and Accumulation Data. NOAA National Centers for Environmental Information. https://doi.org/10.25921/hhnz-ea64.'
        df['reference_short'] = 'Clausen et al. (1988)'
        df['error'] = np.nan
        df['smb'] = df.smb/100
        df['start_year'] = df.end_year
        df[['start_date','end_date']] = np.nan
        df['notes'] = 'discrete values, likely winter to winter'
        if df_metadata.loc[df_metadata.file == f, 'name'].str.contains('pit').all():
            df['method'] = 'firn or ice core, dO18 dating'
            df['method_key'] = 8
        else:
            df['method'] = 'firn or ice core, dO18 dating'
            df['method_key'] = 12
            
        duplicate_indices = df.index[df.index.duplicated()].unique()
        if len(duplicate_indices) > 0:
            print(df[df.index.duplicated()])

        list_add.append(df)
    
    df_add = pd.concat(list_add)
            
    for v in col_needed:
        assert v in df_add.columns, f'{v} is missing'
    df_add[col_needed].to_csv('data_formatted.csv',index=None)

if __name__ == "__main__":
    process()
    
    