import pandas as pd
import os
import re
import sys
import numpy as np

'''
08.11.2024 by Haocheng
to do: compute base ratio for all data
'''

##############################################################################
# not use data with multiple RNAs, only use data with one RNA
def compute_base_ratio(row):
    
    rna_sequence = str(row['rna_sequence'])
    A_num = rna_sequence.count('A')
    U_num = rna_sequence.count('U')
    C_num = rna_sequence.count('C')
    G_num = rna_sequence.count('G')
    total_bases = A_num + U_num + G_num + C_num

    if total_bases == 0:
        if 'poly' in str(row['rnas']).lower():
            match = re.finditer(r'poly(\w)', str(row['rnas']).lower())
            poly_base = ''
            for i in match:
                # print (i.group(1))
                poly_base = poly_base + i.group(1).upper()
            A_num1 = poly_base.count('A')
            U_num1 = poly_base.count('U')
            C_num1 = poly_base.count('C')
            G_num1 = poly_base.count('G')
            total_bases1 = A_num1 + U_num1 + G_num1 + C_num1
            if total_bases1 == 0:
                return None, None, None, None
            else:
                return A_num1 / total_bases1, U_num1 / total_bases1, G_num1 / total_bases1, C_num1 / total_bases1
        else:
            return None, None, None, None
    else:
        return A_num / total_bases, U_num / total_bases, G_num / total_bases, C_num / total_bases
    

'''
08.11.2024 by Haocheng
to do: compute ratio for G/C, A/U  
'''

def compute_GC_ratio(row):
    G_ratio = row['G_ratio']
    C_ratio = row['C_ratio']
    if G_ratio == 0 or C_ratio == 0 or pd.isnull(G_ratio) or pd.isnull(C_ratio):
        return None
    else:
        return G_ratio / C_ratio
    
def compute_AU_ratio(row):
    A_ratio = row['A_ratio']
    U_ratio = row['U_ratio']
    if A_ratio == 0 or U_ratio == 0 or pd.isnull(A_ratio) or pd.isnull(U_ratio):
        return None
    else:
        return A_ratio / U_ratio
##############################################################################

##############################################################################
'''
One class classification
'''
PsData = pd.read_csv('/home/thc/RnaPSP/RnaPSP/all data/PsLess500.csv')
PsData = PsData.dropna(axis=1, how='all')

# Apply the function and expand the tuple into separate columns
PsData[['A_ratio', 'U_ratio', 'G_ratio', 'C_ratio']] = PsData.apply(compute_base_ratio, axis=1, result_type='expand')

PsData['GC_ratio'] = PsData.apply(compute_GC_ratio, axis=1)
PsData['AU_ratio'] = PsData.apply(compute_AU_ratio, axis=1)

if 'level_0' in PsData.columns:
    PsData = PsData.drop('level_0', axis=1)
PsData = PsData.reset_index()
if 'Unnamed: 0' in PsData.columns:
    PsData = PsData.drop('Unnamed: 0', axis=1)
# PsData = PsData.drop('level_0', axis=1)
if 'index' in PsData.columns:
    PsData = PsData.drop('index', axis=1)
PsData.to_csv('/home/thc/RnaPSP/RnaPSP/all data/PsLess500.csv',
              index=True, encoding='utf-8-sig')
##############################################################################

##############################################################################
'''
Two class classification
'''
PsData = pd.read_csv('/home/thc/RnaPSP/RnaPSP/all data/2 classification/TrainData.csv')
PsData = PsData.dropna(axis=1, how='all')

# Apply the function and expand the tuple into separate columns
PsData[['A_ratio', 'U_ratio', 'G_ratio', 'C_ratio']] = PsData.apply(compute_base_ratio, axis=1, result_type='expand')

PsData['GC_ratio'] = PsData.apply(compute_GC_ratio, axis=1)
PsData['AU_ratio'] = PsData.apply(compute_AU_ratio, axis=1)

if 'level_0' in PsData.columns:
    PsData = PsData.drop('level_0', axis=1)
PsData = PsData.reset_index()
if 'Unnamed: 0' in PsData.columns:
    PsData = PsData.drop('Unnamed: 0', axis=1)
# PsData = PsData.drop('level_0', axis=1)
if 'index' in PsData.columns:
    PsData = PsData.drop('index', axis=1)
PsData.to_csv('/home/thc/RnaPSP/RnaPSP/all data/2 classification/TrainData.csv',
              index=True, encoding='utf-8-sig')
##############################################################################