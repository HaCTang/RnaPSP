import zlib
import math
import pandas as pd

'''
08.11.2024 by Haocheng
to do: compute Kolmogorov entropy
Using zlib compression
'''

def compute_kolmogorov_complexity(data: str)->float:
    compressed_data = zlib.compress(data.encode('utf-8'))
    return len(compressed_data)


'''
08.11.2024 by Haocheng
to do: compute shannon entropy, based on the code of
https://wenku.csdn.net/answer/1d38534721c24c4b9e22d7ad3135cb18?ydreferer=aHR0cHM6Ly93d3cuYmluZy5jb20v
'''

def compute_shannon_entropy(data: str)->float: 
    counts = {}
    for char in data: 
        if char in counts: 
            counts[char] += 1 
        else:
            counts[char] = 1
    entropy =0
    total = len(data)
    for count in counts.values(): 
        probability=count/total
    entropy -= probability * math.log2(probability) 
    
    return entropy

# test: showing the difference between Kolmogorov and Shannon
# data = 'CGCGCG'
# data1 = 'CCCGGG'
# data2 = 'CAGCAG'
# print(compute_kolmogorov_complexity(data))
# print(compute_kolmogorov_complexity(data1))
# print(compute_kolmogorov_complexity(data2))
# print(compute_shannon_entropy(data))
# print(compute_shannon_entropy(data1))
# print(compute_shannon_entropy(data2))


'''
08.11.2024 by Haocheng
to do: compute slide entropy, based on the code of
https://wenku.csdn.net/answer/1d38534721c24c4b9e22d7ad3135cb18?ydreferer=aHR0cHM6Ly93d3cuYmluZy5jb20v
08.18.2024 modified by Haocheng, change the range and average returning
'''

def slide_window_entropy(func: callable, data: str, window_size: int, step_size: int)->float:
    """
    * Using slide window to compute partial evenness
    * @param func: compute_kolmogorov_complexity / using compute_shannon_entropy
    * @param data: sequence
    * @param window_size: window size >= 1
    * @param step_size: step size >= 1
    * @return: entropy
    """
    entropy = 0
    for i in range(0, (len(data) - window_size + step_size - 1), step_size):
        subsequence = data[i:i+window_size]
        subsequence_entropy = func(subsequence)
        entropy += subsequence_entropy
    N_windows = (len(data) - window_size + step_size) // step_size
    if N_windows <= 1:
        if func == compute_kolmogorov_complexity:
            poly_base = 'A' * window_size
            return func(poly_base)
        elif func == compute_shannon_entropy:
            return 0 
    else:
        return entropy / N_windows

# test: showing the partial difference between Kolmogorov and Shannon
# seq1 = 'CAGCAGCAGCAGCAGCAGCAGCAGCAGCAG'
# seq2 = 'CCCCCAAAAACCCCCAAAAACCCCCAAAAA'
# print(slide_window_entropy(compute_shannon_entropy, seq1, 8, 2))
# print(slide_window_entropy(compute_shannon_entropy, seq2, 8, 2))
# print(slide_window_entropy(compute_kolmogorov_complexity, seq1, 8, 2))
# print(slide_window_entropy(compute_kolmogorov_complexity, seq2, 8, 2))

#todo: apply the above functions to the data
PsData = pd.read_csv(r'C:\Users\23163\Desktop\PS prediction\RnaPSP\all data\PsSelf1Less500.csv')
PsData = PsData.dropna(axis=1, how='all')

# Apply the sliding window entropy calculation to each RNA sequence
PsData['kolmogorov_complexity'] = PsData['rna_sequence'].apply(
    lambda seq: slide_window_entropy(compute_kolmogorov_complexity, seq, window_size=6, step_size=1)
)
PsData['shannon_entropy'] = PsData['rna_sequence'].apply(
    lambda seq: slide_window_entropy(compute_shannon_entropy, seq, window_size=6, step_size=1)
)

PsData = PsData.reset_index()
PsData = PsData.drop('level_0', axis=1)
PsData.to_csv(r'C:\Users\23163\Desktop\PS prediction\RnaPSP\all data\PsSelf1Less500.csv', index=False, encoding='utf-8-sig')

