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
