import matplotlib.pyplot as plt

def plot_dataset_composition(df_in, path_to_figure='figures/dataset_composition.png'):
    df_sumup = df_in.copy()
    df_sumup['coeff'] = 1
    df_summary = (df_sumup.groupby('reference_short')
                  .apply(lambda x: x.coeff.sum())
                  .reset_index(name='num_measurements'))
    df_summary=df_summary.sort_values('num_measurements')
    explode = 0.2 + 0.3*(df_summary.num_measurements.max() - df_summary.num_measurements)/df_summary.num_measurements.max()

    cmap = plt.get_cmap("tab20c")

    fig, ax=plt.subplots(1,1, figsize=(12,8))
    # print(df_summary.shape[0],  (1-np.exp(-df_summary.shape[0]/500)))
    plt.subplots_adjust(bottom= 0.05)
    patches, texts = plt.pie( df_summary.num_measurements,
                             startangle=90,
                             explode=explode,
                             colors=plt.cm.tab20.colors)
    labels = df_summary.reference_short.str.replace(';','\n') + ' ' + (df_summary.num_measurements/df_summary.num_measurements.sum()*100).round(3).astype(str) + ' %'
    sort_legend = True
    if sort_legend:
        patches, labels, dummy =  zip(*sorted(zip(patches, labels,
                                                  df_summary.num_measurements),
                                              key=lambda x: x[2],
                                              reverse=True))

    plt.legend(patches, labels, loc='upper center', bbox_to_anchor=(0.5, -0.2),
               fontsize=15, ncol=2, title='Data origin (listed clock-wise)',
               columnspacing=0.5,title_fontsize='xx-large')
    plt.ylabel('')

    plt.savefig(path_to_figure,dpi=300, bbox_inches='tight')


import geopandas as gpd

def plot_map(df, filename, area='greenland'):
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")

    if area=='greenland':
        land = gpd.read_file('doc/GIS/greenland_land_3413.shp')
        ice = gpd.read_file('doc/GIS/greenland_ice_3413.shp')
        crs = 3413
    else:
        ice = gpd.read_file('doc/GIS/Medium_resolution_vector_polygons_of_the_Antarctic_coastline.shp')
        crs = 3031
        ice = ice.to_crs(crs)
        land = ice.loc[(ice.surface=='land') & (ice.area<5000)]
        ice = ice.loc[(ice.surface!='land') | (ice.area>5000)]

    gdf = gdf.to_crs(crs)

    if 'method' in df.columns:
        fig, ax_list = plt.subplots(1,2, figsize=(8,5))
        for i, ax in enumerate(ax_list):
            land.plot(ax=ax,color='k')
            ice.plot(ax=ax, color='lightblue')
            if i == 0:
                gdf.loc[~df.method.str.contains('radar'),:].plot(ax=ax, marker='d', markersize=15,
                                              alpha=0.5, edgecolor='tab:blue',
                                              facecolor='None')
                ax.set_title('Snowpits and ice cores')
            else:
                gdf.loc[df.method.str.contains('radar'),:].plot(ax=ax, marker='.', markersize=1,
                                              alpha=0.5, edgecolor='tab:blue',
                                              facecolor='None')
                ax.set_title('Radar profiles')
            ax.axis('off')
    else:
        fig, ax = plt.subplots(1,1, figsize=(8,5))
        land.plot(ax=ax,color='k')
        ice.plot(ax=ax, color='lightblue')

        gdf.plot(ax=ax, marker='d', markersize=20,
                                      alpha=0.8, edgecolor='tab:blue',
                                      facecolor='None')
        ax.axis('off')
    plt.tight_layout()
    fig.savefig(filename,dpi=200)
