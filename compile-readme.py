import pandas as pd
import codecs
# %% updating author list
sheet_id = "1EdShh5Wg8luE2ifn2oz0GfneixOwGGe8acV_VZWsfuo"
sheet_name = "confirmed"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

df = pd.read_csv(url)
df = df.loc[:df['first name'].last_valid_index(),:]
df = df.iloc[:-1,:]

df_authors = df[['first name', 'last name', 'Affiliation N.']]
df_authors['Affiliation N.']=df_authors['Affiliation N.'].copy().astype(int).astype(str)
df_authors.loc[df_authors.index[df_authors.iloc[:,0].isnull()]-1, 'Affiliation N.'] = \
    (df_authors.loc[df_authors.index[df_authors.iloc[:,0].isnull()]-1, 'Affiliation N.'].values +', '+df_authors.loc[df_authors.index[df_authors.iloc[:,0].isnull()], 'Affiliation N.'].values)

df_authors.loc[df_authors.index[df_authors.iloc[:,0].isnull()]-1, 'Affiliation N.']
df_authors.drop(index=df_authors.index[df_authors.iloc[:,0].isnull()])
file_to_delete = open("doc/ReadMe_2024_src/author_list.tex",'w')
file_to_delete.close()

f = codecs.open("doc/ReadMe_2024_src/author_list.tex", "a", "utf-8")
for ind, row in df_authors.iterrows():
    # row  = [urllib.parse.quote(s) for s in row]
    # if ind == 1: break
    if not isinstance(row[0],str):
        continue
    f.write("\\Author[%s]{%s}{%s}\n"%(row[2],row[0],row[1]))

f.write("\n")
df_affil = df.iloc[df.iloc[:,14].first_valid_index():(df.iloc[:,14].last_valid_index()+1),14].reset_index(drop=True)
for ind, row in enumerate(df_affil.values):
    f.write("\\affil[%i]{%s}\n"%(ind+1,row))

f.close()

df_authors = df_authors.dropna()
# Generate author string
def generate_author_string(row):
    return f"{row['last name']}, {row['first name']}"

authors_list = df_authors.apply(generate_author_string, axis=1)
authors_string = " and ".join(authors_list)
title = "The SUMup collaborative database: Surface mass balance, subsurface temperature and density measurements from the Greenland and Antarctic ice sheets"
URL = "10.18739/A2M61BR5M"
year = 2024

# Define the BibTeX entry
bib_entry = (
    f"@misc{{sumup2024,\n"
    f"  author       = {{{authors_string}}},\n"
    f"  title        = {{{title}}},\n"
    f"  doi          = {{{URL}}},\n"
    f"  publisher    = {{Artic Data Center}},\n"
    f"  howpublished = {{\\url{{http://dx.doi.org/{URL}}}}},\n"
    f"  year         = {{{year}}}\n"
    f"}}\n"
)

# Save to sumup_reference.bib
with open("doc/ReadMe_2024_src/sumup_reference.bib", "w", encoding="utf-8") as bib_file:
    bib_file.write(bib_entry)

# %% copying the reference files to the table folder
df = pd.concat([pd.read_csv('SUMup 2024 beta/SUMup_2024_'+var+'_csv/SUMup_2024_'+var+'_references.tsv', sep='\t') for var in ['density','SMB','temperature']])
url_pattern = r'(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+(?<![.,]))'

tmp = (df['reference'].str.replace('\t',' ')
 .str.replace('&','and')
 .str.replace('_','\_')
 .str.replace('\doi','doi:')
 .str.replace('DOI ','doi:')
 .str.replace('DOI:','doi:')
 .str.replace('DOI: ','doi:')
 .str.replace('doi: ','doi:')
 .str.replace('doi:','https://doi.org/')
 .str.replace('‐','-')
 .str.replace('','-')
 .str.replace('and Steffen','\n\nSteffen')
 .str.replace('}',')')
 .str.replace(url_pattern, r'\\url{\1}', regex=True)
 .sort_values())
tmp = (pd.concat( (tmp.str.split(' as in ').str[0],
                  tmp.str.split(' as in ').str[1]), ignore_index=True)
       .drop_duplicates()
       .dropna()
       .sort_values())
f = open('doc/ReadMe_2024_src/tables/SUMup_2024_all_references.tsv', 'w', encoding="utf-8")
f.write('\n\n'.join(tmp.tolist()))
f.close()
# %%
import os
for f in os.listdir('doc/ReadMe_2024_src/tables/'):
    if f.startswith('composition_'):
        df = pd.read_csv('doc/ReadMe_2024_src/tables/'+f)
        for c in df.columns:
            try:
                df[c] = df[c].str.replace(',',' ')
                df[c] = df[c].str.replace('personal communication from','pers. comm.')
                # df[c] = df[c].str.replace('as in Spencer et al.','Spencer et al.')
                # df[c] = df[c].str.replace('Bolzan and Strobel (1999a b c d e f g h i j k l m n o)','Bolzan and Strobel (1999a-o)')
                # df[c] = df[c].str.replace('Oerter et al. (2008a b c d e f g h i j k l m n o p)','Oerter et al. (2008a-p)')
                # df[c] = df[c].str.replace('Graf and Oerter (2006a b c d e f g h i j k l m n o p q r s t u v w x y z)','Graf and Oerter (2006a-z)')
                # df[c] = df[c].str.replace('Graf et al. (1988a b c d e f g h i j k l m n o p q)','Graf and Oerter (1988a-q)')
                # df[c] = df[c].str.replace('Graf et al. (1999a b c d e f g h i j k l m n o p)','Graf and Oerter (1999a-p)')
                # df[c] = df[c].str.replace('Graf et al. (2002a b c d e f g h i j k l m n o p)','Graf and Oerter (2002a-o)')
            except:
                pass
        if 'reference_key' in df.columns:
            df = df.drop(columns='reference_key')
        df.to_csv('doc/ReadMe_2024_src/tables/'+f, index=None)


# %% compiling latex file
import os
import shutil
os.chdir('doc/ReadMe_2024_src/')
os.system("pdflatex SUMup_2024_ReadMe.tex")
os.system("pdflatex SUMup_2024_ReadMe.tex") # needs to run twice for the toc
os.system("pdflatex SUMup_2024_ReadMe.tex") # needs to run twice for the toc
shutil.move('SUMup_2024_ReadMe.pdf', '../../SUMup 2024 beta/SUMup_2024_ReadMe.pdf')

# cleanup
os.remove('SUMup_2024_ReadMe.toc')
os.remove('SUMup_2024_ReadMe.aux')
os.remove('SUMup_2024_ReadMe.log')
