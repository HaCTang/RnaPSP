'''
08.24.2024 by Haocheng
to do: repeat RNA generator
'''

def generate_repeat_rna(repeat_unit: str, length: int) -> str:
    """
    Generates a repeat RNA sequence of the specified length.

    :param repeat_unit: RNA fragments that act as repeating units
    :param length: The total length of the generated repeat RNA
    :return: The resulting repeat RNA sequence
    """
    repeat_unit = repeat_unit.upper()

    repeat_count = length // len(repeat_unit)
    repeat_rna = repeat_unit * repeat_count

    # If the generated sequence length is less than the specified length, complete the rest
    remaining_length = length % len(repeat_unit)
    repeat_rna += repeat_unit[:remaining_length]

    return repeat_rna

# # Test the function
# repeat_unit = "AUG"
# length = 30
# result = generate_repeat_rna(repeat_unit, length)
# print(f"The generated repeat RNA sequence is: {result}")

'''
08/25/2024 by Haocheng
to do: read the RNA sequence from a csv.file
, and figure out the repeat unit and the length of the repeat RNA
'''

import pandas as pd
import re

def read_rna_sequences_from_csv(file_path: str) -> list:
    """
    Reads RNA sequences from a CSV file and extracts the repeat unit and the length of each repeat RNA.
    Handles sequences in formats like '(UCUCUAAAAA)5', 'polyAU', or 'r(GGGGCC)4'.

    :param file_path: The path to the CSV file containing the RNA sequences
    :return: A list of dictionaries, each containing the repeat unit and the length of the repeat RNA
    """
    rna_data = pd.read_csv(file_path)
    
    if "rnas" not in rna_data.columns:
        raise ValueError("The 'rnas' column does not exist in the CSV file.")

    results = []

    for rna_structure in rna_data["rnas"]:
        repeat_unit = ""
        length = None  # Set length to None by default

        # Handle the case where the structure is in the format 'r(GGGGCC)4'
        match = re.match(r'r\((\w+)\)(\d+)', rna_structure)
        if match:
            repeat_unit = match.group(1)
            repeat_count = int(match.group(2))
            length = len(repeat_unit) * repeat_count
        # Handle the case where the structure is in the format '(UCUCUAAAAA)5'
        elif re.match(r'\((\w+)\)(\d+)', rna_structure):
            match = re.match(r'\((\w+)\)(\d+)', rna_structure)
            repeat_unit = match.group(1)
            repeat_count = int(match.group(2))
            length = len(repeat_unit) * repeat_count
        # Handle the case where the structure is like 'polyAU'
        elif rna_structure.startswith('poly'):
            repeat_unit = rna_structure[4:]
            # Length remains None since it's not fixed
        else:
            raise ValueError(f"Unrecognized RNA structure format: {rna_structure}")

        results.append({"repeat_unit": repeat_unit, "length": length})

    return results

# Example usage:
results = read_rna_sequences_from_csv("/home/thc/RnaPSP/RnaPSP/all data/PsRepeat.csv")
for result in results:
    print(result)
