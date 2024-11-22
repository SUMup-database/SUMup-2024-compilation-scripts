<p align="center">
<a href="https://www.scar.org/scar-news/antclimnow-news/antclimnow-grants-2021/"><img src="doc/misc/SCAR_logo_2018_white_background.png" alt="drawing" width="50"/></a>
<a href="https://www.promice.dk/"><img src="doc/misc/Promice_GC-Net_colour.jpg" alt="drawing" width=250"/></a>
<a href="https://ntrs.nasa.gov/citations/20180007574"><img src="doc/misc/nasa-logo-web-rgb.png" alt="drawing" width="50"/></a>
</p>


<h1 style="font-size:20px">
<p align="center">
<strong>
The SUMup dataset
<br />
Surface mass balance, subsurface temperature and density measurements from the Greenland and Antarctic ice sheets</p>
</strong>
</h1> 

## Quick links:
- [The SUMup 2024 release](https://www.doi.org/10.18739/A2M61BR5M)
- [The SUMup 2024 ReadMe file](SUMup%202024%20beta/SUMup_2024_ReadMe.pdf)
- [Example Python scripts / Jupyter notebooks to manipulate the 2023 files](https://github.com/SUMup-database/SUMup-example-scripts)
- [The new netcdf format](https://github.com/SUMup-database/SUMup-2023/blob/main/README.md#the-new-netcdf-structure)
- [The CSV files](https://github.com/SUMup-database/SUMup-2023/blob/main/README.md#the-csv-files)

## The 2023 release
### Reference:

Vandecrux, B., Amory, C., Ahlstrøm, A.P., Akers, P.D., Albert, M., Alley, R.B., Arnaud, L., Bales, R., Benson, C., Box, J.E., Buizert, C., Charalampidis, C., Clerx, N., Covi, F., Denis, G., Dibb, J.E., Ding, M., Eisen, O., Fausto, R., Fernandoy, F., Freitag, J., Gerland, S., Harper, J., Hawley, R.L., Hock, R., How, P., Hubbard, B., Humphrey, N., Iizuka, Y., Isaksson, E., Kameda, T., Karlsson, N.B., Kawakami, K., Kjær, H.A., Kuipers Munneke, P., Lewis, G., MacFerrin, M., Machguth, H., Mankoff, K.D., McConnell, J.R., Medley, B., Morris, E., Mosley-Thompson, E., Mulvaney, R., Niwano, M., Osterberg, E., Otosaka, I., Picard, G., Polashenski, C., Rennermalm, A., Rutishauser, A., Simonsen, S.B., Smith, A., Solgaard, A., Spencer, M., Steen-Larsen, H.C., Stevens, C.M., Sugiyama, S., Tedesco, M., Thompson-Munson, M., Tsutaki, S., van As, D., Van den Broeke, M.R., Wilhelms, F., Xiao, J., Xiao, C.: The SUMup collaborative database: Surface mass balance, subsurface temperature and density measurements from the Greenland and Antarctic ice sheets (1912 - 2023), Arctic Data Center, https://www.doi.org/10.18739/A2M61BR5M, 2023.

### Important notes on the 2023 release:
- Snow depth on sea ice is dropped. This was discussed with Nathan Kurtz who initiated this part of SUMup. The sea ice community did not reuse or feeded into SUMup's snow depth on sea ice data. In absence of volunteer to mobilize that community, offshore data is being removed from the next release of the data.
- The next release will contain SMB data, instead of just snow accumulation.

## In this working repository you will find:
- the scripts used to compile the SUMup dataset
- Q&A and suggestions for further additions in the [issues section](https://github.com/GEUS-Glaciology-and-Climate/sumup/issues)

## Additional ressources
The original dataset was described in:

Montgomery, L., Koenig, L., and Alexander, P.: **The SUMup dataset: compiled measurements of surface mass balance components over ice sheets and sea ice with analysis over Greenland**, Earth Syst. Sci. Data, 10, 1959–1985, [https://doi.org/10.5194/essd-10-1959-2018](https://doi.org/10.5194/essd-10-1959-2018), 2018.

The 2022 release of the SUMup data is available at:

Megan Thompson-Munson, Lynn Montgomery, Jan Lenaerts, and Lora Koenig. 2022. **Surface Mass Balance and Snow Depth on Sea Ice Working Group (SUMup) snow density, accumulation on land ice, and snow depth on sea ice datasets 1952-2019**. Arctic Data Center. [https://doi.org/10.18739/A24Q7QR58](https://doi.org/10.18739/A24Q7QR58).

with some illustration available here: https://github.com/MeganTM/SUMMEDup2022/blob/main/SUMMEDup_2022.ipynb

Read about the genesis of SUMup as first supported by NASA:

Koenig, L., Box, J., and Kurtz, N. (2013), **Improving Surface Mass Balance Over Ice Sheets and Snow Depth on Sea Ice**, Eos Trans. AGU, 94( 10), 100. [https://doi.org/10.1002/2013EO100006](https://doi.org/10.1002/2013EO100006)


## The new netcdf structure:

The data files have two groups "DATA": values given for each measurements and "METDATA": that gives more information about a variable present in DATA.
### density

|    | var             |  long_name                    |  unit                 |  description                                                   |
|---:|:----------------|:------------------------------|:----------------------|:---------------------------------------------------------------|
|  0 | # DATA          | nan                           | nan                   | nan                                                            |
|  1 | measurement_id  | measurement_index             | -                     | index of density measurement (unique for each observation)     |
|  2 | timestamp       | timestamp                     | days since 1900-01-01 | date at which measurement <measurement_id> was measured        |
|  3 | start_depth     | start_depth_of_measurement    | m                     | Top depth of density measurement <measurement_id>              |
|  4 | stop_depth      | stop_depth_of_measurement     | m                     | Bottom depth of density measurement <measurement_id>           |
|  5 | midpoint        | midpoint_depth_of_measurement | m                     | Midpoint depth of density measurement <measurement_id>         |
|  6 | density         | density                       | kg m^-3               | Measured density for measurement <measurement_id>              |
|  7 | error           | error                         | kg m^-3               | Error associated with the density measurement <measurement_id> |
|  8 | latitude        | latitude                      | degree North          | Latitude of measurement <measurement_id>                       |
|  9 | longitude       | longitude                     | degree East           | Longitude of measurement <measurement_id>                      |
| 10 | elevation       | elevation                     | m                     | Elevation of measurement <measurement_id> (datum may vary)     |
| 11 | profile_key     | profile_key                   | -                     | Profile key associated with measurement <measurement_id>       |
| 12 | method_key      | method_key                    | -                     | Method key of measurement <measurement_id>                     |
| 13 | reference_key   | reference_key                 | -                     | Reference key of measurement <measurement_id>                  |
| 14 | # METADATA      | nan                           | nan                   | nan                                                            |
| 15 | profile         | profile                       | -                     | name of the profile <profile_key>                              |
| 16 | reference       | reference                     | -                     | full reference associated with <reference_key>  (please cite)  |
| 17 | reference_short | reference_short               | -                     | short reference associated with <reference_key>  (please cite) |
| 18 | method          | method                        | -                     | method associated with <method_key>                            |

### Temperature

|    | var                  |  long_name                           |  unit                 |  description                                                                 |
|---:|:---------------------|:-------------------------------------|:----------------------|:-----------------------------------------------------------------------------|
|  0 | # DATA VARIABLES     | nan                                  | nan                   | nan                                                                          |
|  1 | measurement_id       | measurement_index                    | -                     | index of temperature measurement (unique for each observation)               |
|  2 | timestamp            | timestamp_of_temperature_measurement | days since 1900-01-01 | start date of measurement <measurement_id> (can be an estimation)            |
|  3 | temperature          | subsurface_temperature               | deg C                 | Measured temperature for measurement <measurement_id>                        |
|  4 | depth                | depth_of_subsurface_temperature      | m                     | Depth of temperature measurement <measurement_id>                            |
|  5 | error                | error                                | m w.e.                | Error associated with the temperature measurement <measurement_id>           |
|  6 | latitude             | latitude                             | degree North          | latitude of measurement <measurement_id>                                     |
|  7 | longitude            | longitude                            | degree East           | longitude of measurement <measurement_id>                                    |
|  8 | elevation            | elevation                            | m                     | elevation of measurement <measurement_id>                                    |
|  9 | name_key             | name_key                             | -                     | Name key associated to  measurement <measurement_id> (core name or location) |
| 10 | method_key           | method_key                           | -                     | method key of measurement <measurement_id>                                   |
| 11 | reference_key        | reference_key                        | -                     | reference key of measurement <measurement_id>                                |
| 12 | # METADATA VARIABLES | nan                                  | nan                   | nan                                                                          |
| 13 | name                 | name                                 | -                     | Name of the name key <name_key>                                              |
| 14 | reference            | reference                            | -                     | full reference corresponding to <reference_key> (please cite)                |
| 15 | reference_short      | reference_short                      | -                     | short reference corresponding to <reference_key> (please cite)               |
| 16 | method               | method                               | -                     | method corresponding to <method_key>                                         |

### SMB

|    | var                  |  long_name                    |  unit                 |  description                                                                 |
|---:|:---------------------|:------------------------------|:----------------------|:-----------------------------------------------------------------------------|
|  0 | # DATA VARIABLES     |                               |                       |                                                                              |
|  1 | measurement_id       | measurement_index             | -                     | index of smb measurement (unique for each observation)                       |
|  2 | start_date           | start_date_of_smb_measurement | days since 1900-01-01 | start date of measurement <measurement_id> (can be an estimation)            |
|  3 | end_date             | end_date_of_smb_measurement   | days since 1900-01-01 | end date of measurement <measurement_id> (can be an estimation)              |
|  4 | smb                  | surface_mass_balance          | m w.e.                | Measured surface mass balance for measurement <measurement_id>               |
|  5 | error                | error                         | m w.e.                | Error associated with the SMB measurement <measurement_id>                   |
|  6 | latitude             | latitude                      | degree North          | latitude of measurement <measurement_id>                                     |
|  7 | longitude            | longitude                     | degree East           | longitude of measurement <measurement_id>                                    |
|  8 | elevation            | elevation                     | m                     | elevation of measurement <measurement_id>                                    |
|  9 | name_key             | name_key                      | -                     | Name key associated to  measurement <measurement_id> (core name or location) |
| 10 | method_key           | method_key                    | -                     | method key of measurement <measurement_id>                                   |
| 11 | reference_key        | reference_key                 | -                     | reference key of measurement <measurement_id>                                |
| 12 | # METADATA VARIABLES |                               |                       |                                                                              |
| 13 | name                 | name                          | -                     | Name of the name key <name_key>                                              |
| 14 | reference            | reference                     | -                     | full reference corresponding to <reference_key> (please cite)                |
| 15 | reference_short      | reference_short               | -                     | short reference corresponding to <reference_key> (please cite)               |
| 16 | method               | method                        | -                     | method corresponding to <method_key>                                         |

## The CSV files:

```
> SUMup 2023 beta/density/csv
    SUMup_2023_density_antarctica.csv
	  SUMup_2023_density_greenland.csv
	  SUMup_2023_density_methods.tsv
	  SUMup_2023_density_profile_names.tsv
	  SUMup_2023_density_references.tsv
```


```python
import pandas as pd
df_density = pd.read_csv(path_to_SUMup_folder + 'density/csv/SUMup_2023_density_greenland.csv')
print(df_density.head(5).to_markdown())
```

|    |   profile_key |   reference_key |   method_key | timestamp   |   latitude |   longitude |   elevation |   start_depth |   stop_depth |   midpoint |   density |   error |
|---:|--------------:|----------------:|-------------:|:------------|-----------:|------------:|------------:|--------------:|-------------:|-----------:|----------:|--------:|
|  0 |            57 |               4 |            4 | 2013-04-08  |    66.1812 |    -39.0435 |        1563 |           nan |          nan |     0.06   |    365.45 |     nan |
|  1 |            57 |               4 |            4 | 2013-04-08  |    66.1812 |    -39.0435 |        1563 |           nan |          nan |     0.22   |    368.42 |     nan |
|  2 |            57 |               4 |            4 | 2013-04-08  |    66.1812 |    -39.0435 |        1563 |           nan |          nan |     0.425  |    394.59 |     nan |
|  3 |            57 |               4 |            4 | 2013-04-08  |    66.1812 |    -39.0435 |        1563 |           nan |          nan |     0.63   |    385    |     nan |
|  4 |            57 |               4 |            4 | 2013-04-08  |    66.1812 |    -39.0435 |        1563 |           nan |          nan |     0.8025 |    378.05 |     nan |

```python
df_methods = pd.read_csv(path_to_SUMup_folder + 'density/csv/SUMup_2023_density_methods.tsv', 
                         sep='\t').set_index('key')
print(df_methods.head(5).to_markdown())
```
|   key | method                                               |
|------:|:-----------------------------------------------------|
| -9999 | nan                                                  |
|     1 | 1000 cc density cutter                               |
|     2 | 250 cc density cutter                                |
|     3 | 100 cc density cutter                                |
|     4 | ice or firn core section                             |

```python
df_names = pd.read_csv(path_to_SUMup_folder + 'density/csv/SUMup_2023_density_profile_names.tsv', 
                         sep='\t').set_index('key')
print(df_names.head(5).to_markdown())
```

|   key | profile                 |
|------:|:------------------------|
|     1 | US_ITASE-02-6 (SPRESSO) |
|     2 | US_ITASE-07-4           |
|     3 | US_ITASE-02-5           |
|     4 | US_ITASE-03-1           |
|     5 | US_ITASE-02-4           |

```python
df_references = pd.read_csv(path_to_SUMup_folder + 'density/csv/SUMup_2023_density_references.tsv', 
                         sep='\t').set_index('key')
print(df_references.head(3).to_markdown())
```
|   key | reference                                                                                                                                                                                                                                                                                                                                        | reference_short                     |
|------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------|
|     1 | US International Trans-Antarctic Scientific Expedition (US ITASE) Glaciochemical Data, Version 2- Mayewski, P. A. and D. A. Dixon. 2013. US International Trans-Antarctic Scientific Expedition (US ITASE) Glaciochemical Data. Version 2. [US_ITASE_Core Info-SWE-Density_2013.xlsx]. Boulder, Colorado USA: National Snow and Ice Data Center. | US ITASE: Mayewski and Dixon (2013) |
|     2 | SIMBA 2007 data - Lewis, M. J., Tison, J. L., Weissling, B., Delille, B., Ackley, S. F., Brabant, F., Xie, H., 2011. Sea ice and snow cover characteristics during the winter- spring transition in the Bellingshausen Sea: an overview of SIMBA 2007, Deep Sea Research II , doi:10.1016/j.dsr2.2010.10.027.                                    | SIMBA: Lewis et al. (2007)          |
|     3 | Satellite-Era Accumulation Traverse 2011 (SEAT11) snowpit density data – Brucker, L. and Koenig, L., SEAT11 Traverse snowpit density data.                                                                                                                                                                                                       | SEAT11: Brucker and Koenig (2011)   |


