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

PsData1 = pd.read_csv('/home/thc/RnaPSP/RnaPSP/all data/2 classification/TrainData.csv')
PsData1['poly_rna'] = 0
PsData1['repeat_rna'] = 0
PsData1['else_rna'] = 0
#如果'rna_classification'存在，且不为空
if 'rna_classification' in PsData1.columns:
    PsData1.loc[PsData1['rna_classification'].notnull() &
          (PsData1['rna_classification'] == 'poly RNA'), 'poly_rna'] = 1
    PsData1.loc[PsData1['rna_classification'].notnull() &
          (PsData1['rna_classification'] == 'repeat RNA'), 'repeat_rna'] = 1
    PsData1.loc[PsData1['rna_classification'].notnull() &
          (PsData1['rna_classification'] != 'poly RNA') & 
          (PsData1['rna_classification'] != 'repeat RNA'), 'else_rna'] = 1
#如果'repeat_unit'存在，且不为空
elif 'repeat_unit' in PsData1.columns:
        PsData1.loc[PsData1['repeat_unit'].notnull() & 
                (PsData1['rna_classification'] == 'repeat RNA'), 'repeat_rna'] = 1
PsData1.to_csv('/home/thc/RnaPSP/RnaPSP/all data/2 classification/TrainData.csv',
              index=True, encoding='utf-8-sig')
