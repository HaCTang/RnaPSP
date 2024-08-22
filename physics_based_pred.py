import RNA
import numpy as np
import pandas as pd

# https://viennarna.readthedocs.io/en/latest/api_python.html#RNA.fold_compound.mfe
# https://www.tbi.univie.ac.at/RNA/RNAfold.1.html

"""
08.21.2024-08.22.2024 by Haocheng 
Only be used in Linux environment
computing the Physics-based features of RNA sequences
"""
PsData = pd.read_csv(r"RnaFold/PsLess500.csv")
def calculate_rna_structures(seq):
    results = {}

    # Calculate MFE structure
    fold_seq = RNA.fold(seq)

    # Create a fold_compound object
    fc = RNA.fold_compound(seq)

    # Calculate the fc structure and free energy
    fc_struct, fc_energy = fc.pf()

    # Calculate the MEA structure
    mea_struct, mea_score = fc.MEA()

    # Calculate the centroid structure and distance
    cen_struct, cen_distance = fc.centroid()
    cen_energy = RNA.energy_of_struct(seq, cen_struct)

    # Calculate the inverse folding
    mfe_freq = fc.pr_structure(fold_seq[0])
    x = fc.ensemble_defect(fold_seq[0])
    if x == 0:
        ensemble_diver = 0
    else:
        ensemble_diver = 1/x

    # Calculate the mean base pair distance
    mean_bp_distance = fc.mean_bp_distance()

    # Store the results in a dictionary
    results['mfe_energy'] = fold_seq[1]
    results['fc_energy'] = fc_energy
    results['mea_score'] = mea_score
    results['cen_energy'] = cen_energy
    results['mfe_freq'] = mfe_freq
    results['ensemble_diver'] = ensemble_diver
    results['mean_bp_distance'] = mean_bp_distance

    return results

# Apply the function to the RNA sequences
df_results = PsData['rna_sequence'].apply(calculate_rna_structures)

# Convert the results to a DataFrame
PsData = pd.concat([PsData, df_results.apply(pd.Series)], axis=1)

# print(PsData)

PsData = PsData.drop('level_0', axis=1)
PsData = PsData.reset_index()
PsData.to_csv(r'RnaFold/PsLess500.csv', index=False, encoding='utf-8-sig')