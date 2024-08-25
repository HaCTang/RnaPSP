import pandas as pd
import os
import re
import sys
import numpy as np

##############################################################################
'''
08.10.2024 by Haocheng
to do: read in data in dataframe All data.xlsx
'''
PsData = pd.read_csv('/home/thc/RnaPSP/RnaPSP/all data/All data.csv')
PsData = PsData.dropna(axis=1, how='all')
##############################################################################

##############################################################################
'''
08.10.2024 by Haocheng, 08.20.2024 modified by Haocheng
to do: only keep the data with one RNA in PsRna.csv
'''
PsSelf1Data = PsData.loc[PsData['components_type'] == 'RNA']
PsSelf1Data = PsSelf1Data.reset_index()
PsSelf1Data = PsSelf1Data.drop('level_0', axis=1)
PsSelf1Data = PsSelf1Data.drop('index', axis=1)
PsSelf1Data.to_csv('/home/thc/RnaPSP/RnaPSP/all data/PsRna/PsRna.csv', 
                   index=False, encoding='utf-8-sig')


'''
08.10.2024 by Haocheng, 08.20.2024 modified by Haocheng
to do: RNAs+proteins \  RNA + protein into a single document PsRnaPro.csv
'''
PsRnaPro = PsData.loc[(PsData['components_type'] == 'RNAs + protein') 
                      | (PsData['components_type'] == 'RNA + protein')]
PsRnaPro = PsRnaPro.reset_index()
PsRnaPro = PsRnaPro.drop('level_0', axis=1)
PsRnaPro = PsRnaPro.drop('index', axis=1)
PsRnaPro.to_csv('/home/thc/RnaPSP/RnaPSP/all data/PsRnaPro/PsRnaPro.csv', 
                index=True, encoding='utf-8-sig')
##############################################################################

##############################################################################
'''
08.21.2024 by Haocheng
to do: delete the repeated sequences acording to the column "rpsid"
'''
def delete_repeated_sequences(data):
    data = data.drop_duplicates(subset='rpsid')
    data = data.reset_index()
    data = data.drop('index', axis=1)
    return data

'''
08.10.2024 by Haocheng
to do: read the column "rna_length" and find the number before "nt", 
if it is less than 500, keep the column in Ps*Less500. 
More than 500, keep the column in Ps*More500.
For those whose data type is not int, keep the column in Ps*NotInt
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
PsRnaPro['rna_length_num'] = PsRnaPro.apply(extract_rna_length, axis=1)
PsData['rna_length_num'] = PsData.apply(extract_rna_length, axis=1)


PsSelf1Less500 = PsSelf1Data[
    PsSelf1Data['rna_length_num'].apply(lambda x: isinstance(x, int) and x <= 500)
]
PsSelf1Less500 = PsSelf1Less500.reset_index()
PsSelf1Less500.to_csv('/home/thc/RnaPSP/RnaPSP/all data/PsRna/PsRnaLess500.csv',
                    index=True, encoding='utf-8-sig')
PsRnaProLess500 = PsRnaPro[
    PsRnaPro['rna_length_num'].apply(lambda x: isinstance(x, int) and x <= 500)
]
PsRnaProLess500 = PsRnaProLess500.reset_index()
PsRnaProLess500.to_csv('/home/thc/RnaPSP/RnaPSP/all data/PsRnaPro/PsRnaProLess500.csv',
                    index=True, encoding='utf-8-sig')
PsLess500 = PsData[
    PsData['rna_length_num'].apply(lambda x: isinstance(x, int) and x <= 500)
]
# delete the rows if the column "rna_sequence" is "-"
PsLess500 = PsLess500[PsLess500['rna_sequence'] != '-']
# delete the repeated sequences
PsLess500 = delete_repeated_sequences(PsLess500)
PsLess500 = PsLess500.reset_index()
PsLess500.to_csv('/home/thc/RnaPSP/RnaPSP/all data/PsLess500.csv',
                    index=True, encoding='utf-8-sig')

PsSelf1More500 = PsSelf1Data[
    PsSelf1Data['rna_length_num'].apply(lambda x: isinstance(x, int) and x > 500)
]
PsSelf1More500 = PsSelf1More500.reset_index()
PsSelf1More500.to_csv('/home/thc/RnaPSP/RnaPSP/all data/PsRna/PsRnaMore500.csv',
                      index=True, encoding='utf-8-sig')
PsRnaProMore500 = PsRnaPro[
    PsRnaPro['rna_length_num'].apply(lambda x: isinstance(x, int) and x > 500)
]
PsRnaProMore500 = PsRnaProMore500.reset_index()
PsRnaProMore500.to_csv('/home/thc/RnaPSP/RnaPSP/all data/PsRnaPro/PsRnaProMore500.csv',
                        index=True, encoding='utf-8-sig')


PsSelf1NotInt = PsSelf1Data[
    PsSelf1Data['rna_length_num'].apply(lambda x: not isinstance(x, int))
]
PsSelf1NotInt = PsSelf1NotInt.reset_index()
PsSelf1NotInt.to_csv('/home/thc/RnaPSP/RnaPSP/all data/PsRna/PsSelf1NotInt.csv',
                     index=True, encoding='utf-8-sig')
PsRnaProNotInt = PsRnaPro[
    PsRnaPro['rna_length_num'].apply(lambda x: not isinstance(x, int))
]
PsRnaProNotInt = PsRnaProNotInt.reset_index()
PsRnaProNotInt.to_csv('/home/thc/RnaPSP/RnaPSP/all data/PsRnaPro/PsRnaProNotInt.csv',
                       index=True, encoding='utf-8-sig')
##############################################################################

##############################################################################
'''
08.10.2024 by Haocheng, 08.20.2024 modified by Haocheng
to do: devide poly RNA into a single document Ps*Poly.csv
'''

PsSelf2Data = PsSelf1Data[PsSelf1Data['rna_classification'] == 'poly RNA']
PsSelf2Data = PsSelf2Data.reset_index()
PsSelf2Data = PsSelf2Data.drop('index', axis=1)
PsSelf2Data.to_csv('/home/thc/RnaPSP/RnaPSP/all data/PsRna/PsRnaPoly.csv',
                   index=True, encoding='utf-8-sig')
PsRnaProPoly = PsRnaPro[PsRnaPro['rna_classification'] == 'poly RNA']
PsRnaProPoly = PsRnaProPoly.reset_index()
PsRnaProPoly = PsRnaProPoly.drop('index', axis=1)
PsRnaProPoly.to_csv('/home/thc/RnaPSP/RnaPSP/all data/PsRnaPro/PsRnaProPoly.csv',
                     index=True, encoding='utf-8-sig')
PsPoly = PsData[PsData['rna_classification'] == 'poly RNA']
PsPoly = PsPoly.reset_index()
PsPoly = PsPoly.drop('index', axis=1)
PsPoly.to_csv('/home/thc/RnaPSP/RnaPSP/all data/PsPoly.csv',
                index=True, encoding='utf-8-sig')
##############################################################################

##############################################################################
'''
08.10.2024 by Haocheng, 08.20.2024 modified by Haocheng
to do: devide repeat RNA into a single document Ps*Repeat.csv
'''

PsSelf3Data = PsSelf1Data[PsSelf1Data['rna_classification'] == 'repeat RNA']
PsSelf3Data = PsSelf3Data.reset_index()
PsSelf3Data = PsSelf3Data.drop('index', axis=1)
PsSelf3Data.to_csv('/home/thc/RnaPSP/RnaPSP/all data/PsRna/PsRnaRepeat.csv',
                   index=True, encoding='utf-8-sig')
PsRnaProRepeat = PsRnaPro[PsRnaPro['rna_classification'] == 'repeat RNA']
PsRnaProRepeat = PsRnaProRepeat.reset_index()
PsRnaProRepeat = PsRnaProRepeat.drop('index', axis=1)
PsRnaProRepeat.to_csv('/home/thc/RnaPSP/RnaPSP/all data/PsRnaPro/PsRnaProRepeat.csv',
                     index=True, encoding='utf-8-sig')
PsRepeat = PsData[PsData['rna_classification'] == 'repeat RNA']
PsRepeat = delete_repeated_sequences(PsRepeat)
PsRepeat = PsRepeat.reset_index()
PsRepeat = PsRepeat.drop('index', axis=1)
PsRepeat.to_csv('/home/thc/RnaPSP/RnaPSP/all data/PsRepeat.csv',
                index=True, encoding='utf-8-sig')
##############################################################################
