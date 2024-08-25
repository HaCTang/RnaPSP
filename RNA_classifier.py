'''
08.24.2024 by Haocheng
to do: RNA sequence classifier(polyRNA or repeatRNA)
'''

def is_repeat_or_poly_rna(sequence: str) -> str:
    # Convert the sequence to uppercase to avoid case sensitivity
    sequence = sequence.upper()

    # Check if the sequence is polyRNA (same nucleotide repeated)
    if sequence == sequence[0] * len(sequence) and len(sequence) > 40:
        return "PolyRNA"

    # Check if the sequence is a repeat RNA
    for i in range(1, len(sequence)):
        repeat_unit = sequence[:i]
        if (
            len(sequence) % len(repeat_unit) == 0 and  # Sequence length is divisible by repeat unit length
            (len(sequence) // len(repeat_unit)) >= 4 and  # Repeat unit appears at least 4 times
            repeat_unit * (len(sequence) // len(repeat_unit)) == sequence and  # Repeat unit matches the entire sequence
            len(sequence) > 20  # Sequence is longer than 20 nucleotides
        ):
            return "Repeat RNA"

    return "Neither"

# rna_sequence1 = "AUGAUGAUGAUGAUGAUGAUGAUG"
# rna_sequence2 = "AAAAAAAAAA"
# rna_sequence3 = "AUGAUG"
# result1 = is_repeat_or_poly_rna(rna_sequence1)
# result2 = is_repeat_or_poly_rna(rna_sequence2)
# result3 = is_repeat_or_poly_rna(rna_sequence3)
# print(f"The RNA sequence is: {result1}")
# print(f"The RNA sequence is: {result2}")
# print(f"The RNA sequence is: {result3}")
