import pandas as pd
import numpy as np
import os

current_file_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.abspath(os.path.join(current_file_dir, '..', '..', 
                                         'all data', '2 classification'))
if not os.path.exists(input_dir):
    raise KeyError(f"Path does not exist: {input_dir}")

GenRepRna = pd.read_csv(os.path.join(input_dir, 'generated_repeat_rna.csv'))
PsRnaLess500 = pd.read_csv(os.path.join(input_dir, 'PsRnaLess500.csv'))
PsRnaProLess500 = pd.read_csv(os.path.join(input_dir, 'PsRnaProLess500.csv'))

'''
09.02.2024 by Haocheng
This script is used to merge the same RNAs who have 
totally same sequence in the column "rnas" or "rna_sequence" 
'''
def merge_same_seq(df)->pd.DataFrame:
    if 'rnas' not in df.columns:
        df = df.drop_duplicates(subset='rna_sequence', keep='first')
    else:
        df = df.drop_duplicates(subset='rnas', keep='first')
    return df

GenRepRna = merge_same_seq(GenRepRna)
PsRnaLess500 = merge_same_seq(PsRnaLess500)
PsRnaProLess500 = merge_same_seq(PsRnaProLess500)

'''
09.02.2024 by Haocheng
Delete lines in PsRnaProLess500 that duplicate PsRnaLess500
'''
PsRnaProLess500 = PsRnaProLess500[~PsRnaProLess500['rna_sequence'].isin(PsRnaLess500['rna_sequence'])]

# Concatenate GenRepRna and PsRnaLess500 
SynthPosData = pd.concat([GenRepRna, PsRnaLess500], axis=0)
# Add label column
SynthPosData['label'] = 1
PsRnaProLess500['label'] = 0

TrainData = pd.concat([SynthPosData, PsRnaProLess500], axis=0)

output_dir = os.path.abspath(os.path.join(current_file_dir, '..', '..', 
                                         'all data', '2 classification'))
TrainData.to_csv(os.path.join(output_dir, 'TrainData.csv'), index=False, encoding='utf-8-sig')