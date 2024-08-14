import pandas as pd
import os
import re
import sys
import numpy as np

'''
08.10.2024 by Haocheng
to do: read in data in dataframe All data.xlsx
'''
PsData = pd.read_csv(r'C:\Users\23163\Desktop\PS prediction\RnaPSP\all data\All data.csv')
PsData = PsData.dropna(axis=1, how='all')


'''
08.10.2024 by Haocheng
to do: only keep the data that is "-" in column "protein_sequence" and column "protein_name"
'''
PsSelf1Data = PsData.loc[(PsData['protein_sequence'] == '-') & (PsData['protein_name'] == '-')]
PsSelf1Data = PsSelf1Data.reset_index()
PsSelf1Data = PsSelf1Data.drop('level_0', axis=1)
PsSelf1Data = PsSelf1Data.drop('index', axis=1)
PsSelf1Data.to_csv(r'C:\Users\23163\Desktop\PS prediction\RnaPSP\all data\PsSelf1Data.csv', index=False, encoding='utf-8-sig')


'''
08.10.2024 by Haocheng
to do: read the column "rna_length" and find the number before "nt", 
if it is less than 500, keep the column in PsSelf1Less500. 
More than 500, keep the column in PsSelf1More500.
For those whose data type is not int, keep the column in PsSelf1NotInt
'''

# Extract the number before "nt" from the 'rna_length' column, handling different types
def extract_rna_length(row):
    rna_length = str(row['rna_length'])
    if 'nt' in rna_length:
        pattern = r'(\d{1,5})nt'
        match_obj = re.search(pattern, rna_length)
        if match_obj:
            num_str = match_obj.group(1)
            return int(num_str)
    elif rna_length.isdecimal():
        return int(rna_length)
    elif rna_length == '-' and row['rna_classification'] == 'poly RNA':
        return 0
    else:
        return ''

PsSelf1Data['rna_length_num'] = PsSelf1Data.apply(extract_rna_length, axis=1)

PsSelf1Less500 = PsSelf1Data[
    PsSelf1Data['rna_length_num'].apply(lambda x: isinstance(x, int) and x <= 500)
]
PsSelf1Less500 = PsSelf1Less500.reset_index()
PsSelf1Less500.to_csv(r'C:\Users\23163\Desktop\PS prediction\RnaPSP\all data\PsSelf1Less500.csv', index=True, encoding='utf-8-sig')

PsSelf1More500 = PsSelf1Data[
    PsSelf1Data['rna_length_num'].apply(lambda x: isinstance(x, int) and x > 500)
]
PsSelf1More500 = PsSelf1More500.reset_index()
PsSelf1More500.to_csv(r'C:\Users\23163\Desktop\PS prediction\RnaPSP\all data\PsSelf1More500.csv', index=True, encoding='utf-8-sig')

PsSelf1NotInt = PsSelf1Data[
    PsSelf1Data['rna_length_num'].apply(lambda x: not isinstance(x, int))
]
PsSelf1NotInt = PsSelf1NotInt.reset_index()
PsSelf1NotInt.to_csv(r'C:\Users\23163\Desktop\PS prediction\RnaPSP\all data\PsSelf1NotInt.csv', index=True, encoding='utf-8-sig')


'''
08.10.2024 by Haocheng
to do: devide poly RNA into a single document PsSelf2Data.csv
'''

PsSelf2Data = PsSelf1Data[PsSelf1Data['rna_classification'] == 'poly RNA']
PsSelf2Data = PsSelf2Data.reset_index()
PsSelf2Data = PsSelf2Data.drop('index', axis=1)
PsSelf2Data.to_csv(r'C:\Users\23163\Desktop\PS prediction\RnaPSP\all data\PsSelf2Data.csv', index=True, encoding='utf-8-sig')


'''
08.10.2024 by Haocheng
to do: devide repeat RNA into a single document PsSelf3Data.csv
'''

PsSelf3Data = PsSelf1Data[PsSelf1Data['rna_classification'] == 'repeat RNA']
PsSelf3Data = PsSelf3Data.reset_index()
PsSelf3Data = PsSelf3Data.drop('index', axis=1)
PsSelf3Data.to_csv(r'C:\Users\23163\Desktop\PS prediction\RnaPSP\all data\PsSelf3Data.csv', index=True, encoding='utf-8-sig')


'''
08.10.2024 by Haocheng
to do: RNA+proteins into a single document PsRnaPro.csv
'''
PsRnaPro = PsData.loc[PsData['components_type'] == 'RNAs + protein']
PsRnaPro = PsRnaPro.reset_index()
PsRnaPro = PsRnaPro.drop('level_0', axis=1)
PsRnaPro = PsRnaPro.drop('index', axis=1)
PsRnaPro.to_csv(r'C:\Users\23163\Desktop\PS prediction\RnaPSP\all data\PsRnaPro.csv', index=True, encoding='utf-8-sig')

