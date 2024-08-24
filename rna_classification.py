import pandas as pd

'''
08.22.2024 by Haocheng
to do: classify RNA sequences by the column "rna_classification"
Build two new columns: "poly_rna" and "repeat_rna"
if the "rna_classification" is "poly RNA", then "poly_rna" is 1 and "repeat_rna" is 0
elif the "rna_classification" is "repeat RNA", then "poly_rna" is 0 and "repeat_rna" is 1
else, "poly_rna" is 0 and "repeat_rna" is 0
'''

PsData = pd.read_csv('/home/thc/RnaPSP/RnaPSP/all data/PsLess500.csv')
# PsData = PsData.drop('level_0', axis=1)
# PsData = PsData.drop('Unnamed: 0', axis=1)

PsData['poly_rna'] = 0
PsData['repeat_rna'] = 0
PsData['else_rna'] = 0

PsData.loc[PsData['rna_classification'] == 'poly RNA', 'poly_rna'] = 1
PsData.loc[PsData['rna_classification'] == 'repeat RNA', 'repeat_rna'] = 1
PsData.loc[(PsData['rna_classification'] != 'poly RNA') & 
           (PsData['rna_classification'] != 'repeat RNA'), 'else_rna'] = 1


PsData.to_csv('/home/thc/RnaPSP/RnaPSP/all data/PsLess500.csv',
                index=False, encoding='utf-8-sig')

