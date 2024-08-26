'''
08.24.2024 by Haocheng
to do: repeat RNA generator
'''

def generate_repeat_rna_sequences(repeat_unit: str, rna_length: int) -> str:
    """
    Generates a repeat RNA sequence of the specified length.

    :param repeat_unit: RNA fragments that act as repeating units
    :param length: The total length of the generated repeat RNA
    :return: The resulting repeat RNA sequence
    """
    repeat_unit = repeat_unit.upper()

    repeat_count = rna_length // len(repeat_unit)
    repeat_rna = repeat_unit * repeat_count

    # If the generated sequence length is less than the specified length, complete the rest
    remaining_length = rna_length % len(repeat_unit)
    repeat_rna += repeat_unit[:remaining_length]

    return repeat_rna

# # Test the function
# repeat_unit = "AUG"
# length = 30
# result = generate_repeat_rna(repeat_unit, length)
# print(f"The generated repeat RNA sequence is: {result}")

'''
08.25.2024 by Haocheng
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
            repeat_rna = repeat_unit * repeat_count
        # Handle the case where the structure is in the format '(UCUCUAAAAA)5'
        elif re.match(r'\((\w+)\)(\d+)', rna_structure):
            match = re.match(r'\((\w+)\)(\d+)', rna_structure)
            repeat_unit = match.group(1)
            repeat_count = int(match.group(2))
            length = len(repeat_unit) * repeat_count
            repeat_rna = repeat_unit * repeat_count
        # Handle the case where the structure is like 'polyAU'
        elif rna_structure.startswith('poly'):
            repeat_unit = rna_structure[4:]
            # Length remains None since it's not fixed
        else:
            raise ValueError(f"Unrecognized RNA structure format: {rna_structure}")

        results.append({"repeat_unit": repeat_unit, "rna_sequence": repeat_rna ,"rna_length": length})

    return results

# # Example usage:
# results = read_rna_sequences_from_csv("/home/thc/RnaPSP/RnaPSP/all data/PsRepeat.csv")
# print(results)

'''
08.25.2024-08.26.2024 by Haocheng
to do: Generate repeat RNA sequences from the extracted data.
The length of the repeat RNA is no less than that from the
the extracted data and no more than 500.
The repeat times is generated randomly.
In addition, "the CAG units can be derived" GCA ", "AGC", "CAG", 
"CGA", "ACG", "GAC" three kinds of repetitive sequence.
Save the generated repeat RNA sequences to a new CSV file
'''

def generate_circular_variants(repeat_unit: str) -> list:
    """
    Generates all circular variants of a given repeat unit.

    :param repeat_unit: The repeat unit string
    :return: A list of all circular variants of the repeat unit
    """
    return [repeat_unit[i:] + repeat_unit[:i] for i in range(len(repeat_unit))]

def enlarge_circular_variants(data: list) -> list:
    generated_sequences = []
    for item in data:
        circular_variants = generate_circular_variants(item["repeat_unit"])
        for variant in circular_variants:
            if item["rna_length"] == None:
                repeat_rna = generate_repeat_rna_sequences(variant, 498)
                item["rna_length"] = 498
            else:
                repeat_rna = generate_repeat_rna_sequences(variant, item["rna_length"])
            generated_sequences.append({"repeat_unit": variant, "rna_sequence": repeat_rna, "rna_length": item["rna_length"]})
    data = data + generated_sequences
    return(data)

def filter_max_length_repeats(data: list) -> list:
    """
    Filters out repeat units with the maximum length if there are duplicates.

    :param data: A list of dictionaries containing the repeat unit and length of the repeat RNA
    :return: A new list with only the maximum length for each repeat unit
    """
    filtered_dict = {}
    
    for item in data:
        repeat_unit = item["repeat_unit"]
        length = item["rna_length"]
        
        # If the repeat_unit is not in the dictionary, or if the current length is greater than the stored length, update it
        if length is None:
            continue
        elif repeat_unit not in filtered_dict or length > filtered_dict[repeat_unit]["rna_length"]:
            filtered_dict[repeat_unit] = item
    
    # Convert the dictionary back to a list
    filtered_list = list(filtered_dict.values())
    
    return filtered_list

def generate_repeat_rna_dataset(data):
    """
    Generates repeat RNA sequences from the extracted data.

    :param data: A list of dictionaries containing the repeat unit and the length of the repeat RNA
    :return: A list of dictionaries containing the repeat unit, the generated repeat RNA sequence, and the RNA length
    """
    gen_list = []

    data1 = filter_max_length_repeats(data)

    for item in data1:
        repeat_unit = item["repeat_unit"]
        length = item["rna_length"]
        
        if length is None:
            length = 499 - 499 % len(repeat_unit)  # Set the length to 499 if it's not fixed
            repeat_rna = generate_repeat_rna_sequences(repeat_unit, length)
            gen_list.append({"repeat_unit": repeat_unit, "rna_sequence": repeat_rna, "rna_length": length})
        else:            
            for i in range(1, 6):
                length = length * (2 ** i)
                if length >= 500:
                    break
                else:
                    repeat_rna = generate_repeat_rna_sequences(repeat_unit, length)
                    gen_list.append({"repeat_unit": repeat_unit, "rna_sequence": repeat_rna, "rna_length": length})

    data = data + gen_list
    return data

def output_generated_rna(data: list, output_file):
    """
    Outputs the generated repeat RNA sequences to a CSV file.

    :param data: A list of dictionaries containing the repeat unit, the generated repeat RNA sequence, and the RNA length
    :param output_file: The path to the output CSV file
    """
    gen_data = generate_repeat_rna_dataset(data)
    generated_df = pd.DataFrame(gen_data)  
    generated_df = generated_df.astype({"rna_length": "int"})
    # Fill the columns “rna_length” with unit "nt"
    generated_df["rna_length"] = generated_df["rna_length"].apply(lambda x: "-" if x is None else str(x) + "nt")
    # Fill the “rna_classification” with "repeat RNA"
    generated_df["rna_classification"] = "repeat RNA"
    generated_df.to_csv(output_file, index=False)

# Example usage:
data = read_rna_sequences_from_csv("/home/thc/RnaPSP/RnaPSP/all data/PsRepeat.csv")
data = enlarge_circular_variants(data)
output_generated_rna(data, "/home/thc/RnaPSP/RnaPSP/all data/generated_repeat_rna.csv")
