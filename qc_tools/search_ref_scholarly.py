# -*- coding: utf-8 -*-
"""
Created on %(date)s
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""

df_ref.loc[df_ref.reference==df_ref.reference_short].reference.to_markdown().to_csv

with open('output.md', 'w') as f:
    f.write(df_ref.loc[df_ref['reference'] == df_ref['reference_short'], 'reference'].to_markdown())

# %%
from scholarly import scholarly
import pandas as pd

# List of short references
short_references = df_ref.loc[df_ref['reference'] == df_ref['reference_short'], 'reference'].values
# Create a DataFrame to store the short and long references
df = pd.DataFrame(short_references, columns=['short_reference'])
df['long_reference'] = None

def format_author_name(name):
    parts = name.split()
    last_name = parts[-1]
    initials = [f"{p[0]}." for p in parts[:-1] if p]  # Take the first letter of each part except the last name
    return f"{last_name}, {' '.join(initials)}"

# Function to search for the first result on Google Scholar
def find_long_reference(short_ref):
    query = f"{short_ref} antarctica"
    try:
        search_query = scholarly.search_pubs(query)
        result = next(search_query, None)
        if result:
            # Extract relevant information
            authors_raw = result['bib'].get('author', [])
            authors = ', '.join(format_author_name(author) for author in authors_raw)
            title = result['bib'].get('title', 'No title')
            journal = result['bib'].get('venue', 'No journal')
            year = result['bib'].get('pub_year', 'No year')
            doi = result['bib'].get('doi', result['pub_url'] or 'No DOI')
            formatted_reference = f"{authors}: {title}, {journal}, {year}, URL: {doi}"
            return formatted_reference
        else:
            return "No match found"
    except Exception as e:
        return f"Error: {e}"
results=[]
for short_ref in short_references:
    long_ref = find_long_reference(short_ref)
    print(short_ref, long_ref)
    results.append({'short_ref': short_ref, 'long_ref': long_ref})

# Create a DataFrame from the results
df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
df.to_csv('references.csv', index=False)

print("Results saved to references.csv")
