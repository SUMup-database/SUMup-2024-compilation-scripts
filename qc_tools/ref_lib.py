import requests
import bibtexparser

def get_bibtex_from_doi(doi):
    url = f"https://doi.org/{doi}"
    headers = {"Accept": "application/x-bibtex"}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.text
    else:
        return None

def parse_short_reference(bibtex_entry):
    bib_database = bibtexparser.loads(bibtex_entry)
    entry = bib_database.entries[0]

    authors = entry.get('author', '').split(' and ')
    first_author_last_name = authors[0].split(',')[0]  # Get the first author's last name
    year = entry.get('year', 'n.d.')  # Get the year

    if len(authors) > 1:
        short_ref = f"{first_author_last_name} et al. ({year})"
    else:
        short_ref = f"{first_author_last_name} ({year})"

    return short_ref

def format_author(author):
    """Convert an author from 'Lastname, Firstname' to 'Lastname, F. I.' format."""
    parts = author.split(", ")
    last_name = parts[0]
    initials = ''.join([name[0] + '.' for name in parts[1].split()])
    return f"{last_name}, {initials}"

def parse_long_reference(bibtex_entry):
    bib_database = bibtexparser.loads(bibtex_entry)
    entry = bib_database.entries[0]

    # Extract necessary fields
    authors = entry.get('author', '').split(' and ')
    title = entry.get('title', '')
    journal = entry.get('journal', '')
    volume = entry.get('volume', '')
    pages = entry.get('pages', '')
    doi = entry.get('doi', '')
    year = entry.get('year', 'n.d.')

    # Format authors in "Lastname, F. I." style
    formatted_authors = ', '.join([format_author(a) for a in authors])

    # Construct long reference in Copernicus (ESSD) style
    long_ref = (f"{formatted_authors}: {title}, {journal}, {volume}, {pages}, "
                f"https://doi.org/{doi}, {year}.")

    return long_ref

# # Example usage
# doi = "10.5194/essd-13-3819-2021"  # Replace with your DOI
# bibtex = get_bibtex_from_doi(doi)

# if bibtex:
#     short_reference = parse_short_reference(bibtex)
#     long_reference = parse_long_reference(bibtex)

#     print("Short reference:", short_reference)
#     print("Long reference:", long_reference)
# else:
#     print(f"Error: Unable to retrieve BibTeX for DOI {doi}")
# -*- coding: utf-8 -*-
# %%
"""
Created on %(date)s
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""


df_good = df_sumup.loc[df_sumup.reference != df_sumup.reference_short,
                            ['reference_short','reference']].drop_duplicates()
df_uncertain = df_sumup.loc[df_sumup.reference == df_sumup.reference_short,
                            ['reference_short','reference']].drop_duplicates()

#%%
from difflib import get_close_matches

for index, row in df_uncertain.iterrows():
    query = row['reference_short']+' \"antarctica\"'
    possible_matches = get_close_matches(
        row['reference_short'], df_good['reference_short'], n=1, cutoff=0.9)
    if possible_matches:
        matched_reference = df_good.loc[df_good['reference_short'] == possible_matches[0],
                                        'reference'].values[0]
        print(f"\nUncertain ref: {row['reference_short']} -> Matching good ref: {matched_reference}")


# %%
from scholarly import scholarly
import requests
import bibtexparser

def format_bibtex(entry):
    # Extract BibTeX fields
    title = entry.get('title', '')
    author = entry.get('author', '')
    year = entry.get('pub_year', 'N/A')
    journal = entry.get('journal', '')
    volume = entry.get('volume', '')
    pages = entry.get('pages', '')

    # Format BibTeX entry
    bibtex_entry = f"@article{{{author.replace(' ', '')}{year},\n" \
                   f"  title = {{{title}}},\n" \
                   f"  author = {{{author}}},\n" \
                   f"  journal = {{{journal}}},\n" \
                   f"  year = {{{year}}},\n" \
                   f"  volume = {{{volume}}},\n" \
                   f"  pages = {{{pages}}}\n" \
                   "}"
    return bibtex_entry

def get_bibtex_from_doi(doi):
    url = f"https://doi.org/{doi}"
    headers = {"Accept": "application/x-bibtex"}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.text
    else:
        return None

def parse_short_reference(bibtex_entry):
    bib_database = bibtexparser.loads(bibtex_entry)
    entry = bib_database.entries[0]

    authors = entry.get('author', '').split(' and ')
    first_author_last_name = authors[0].split(',')[0]  # Get the first author's last name
    year = entry.get('year', 'n.d.')  # Get the year

    if len(authors) > 1:
        short_ref = f"{first_author_last_name} et al. ({year})"
    else:
        short_ref = f"{first_author_last_name} ({year})"

    return short_ref



def parse_long_reference(bibtex_entry):
    bib_database = bibtexparser.loads(bibtex_entry)
    entry = bib_database.entries[0]

    # Extract necessary fields
    authors = entry.get('author', '').split(' and ')
    title = entry.get('title', '')
    journal = entry.get('journal', '')
    volume = entry.get('volume', '')
    pages = entry.get('pages', '')
    doi = entry.get('doi', '')
    year = entry.get('year', 'n.d.')

    # Format authors in "Lastname, F. I." style
    formatted_authors = ', '.join([format_author(a) for a in authors])

    # Construct long reference in Copernicus (ESSD) style
    long_ref = (f"{formatted_authors}: {title}, {journal}, {volume}, {pages}, "
                f"https://doi.org/{doi}, {year}.")

    return long_ref

def format_author(author):
    """Convert an author from 'Lastname, Firstname' to 'Lastname, F. I.' format."""
    parts = author.split(" ")
    last_name = parts[1]
    initials = ''.join([name[0] + '.' for name in parts[1].split()])
    return f"{last_name}, {initials}"

def parse_long_reference_from_scholarly(first_result):


    # Extract necessary fields
    authors = first_result['bib'].get('author', '')
    title = first_result['bib'].get('title', '')
    journal = first_result['bib'].get('venue', '')
    volume = first_result['bib'].get('volume', '')
    pages = first_result['bib'].get('pages', '')
    year = first_result['bib'].get('pub_year', '')

    url = first_result.get('pub_url', '')
    if url == '':
        url = first_result.get('eprint_url', '')


    if "…" in journal:
        if "tc.copernicus" in url:
            journal = "The Cryosphere"
    # Format authors in "Lastname, F. I." style
    formatted_authors = ', '.join([format_author(a) for a in authors])

    # Construct long reference in Copernicus (ESSD) style
    long_ref = (f"{formatted_authors}: {title}, {journal}, "
                f"{url}, {year}.")

    return long_ref
import re

for index, row in df_uncertain.iloc[159:,:].iterrows():
    print(row['reference_short'],
          "\t",
          '')
    #%%
    if row['reference_short'] == "GEUS unpublished":
        continue

    # print('===')
    # print(row['reference_short'] )
    year_match = re.search(r'\b\d{4}\b', row['reference_short'])
    if year_match:
        year = int(year_match.group())
    else:
        print("No year found.")

    query = row['reference_short'] + ' "Antarctica"'
    search_query = scholarly.search_pubs(query)

    try:
        first_result = next(search_query)
        if first_result['bib']['pub_year'] ==str(year):
            if first_result['bib']['author'][0].split(' ')[1] == row['reference_short'].split(' ')[0] :

                print(row['reference_short'],
                      "\t",
                      parse_long_reference_from_scholarly(first_result))
            else:
                print(row['reference_short'],
                      "\t",
                      '')
        else:
            print(row['reference_short'],
                  "\t",
                  '')
    except StopIteration:
        print(row['reference_short'],
              "\t",
              '')
