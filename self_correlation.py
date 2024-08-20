import pandas as pd
import os
import re
import numpy as np
from scipy.stats import norm
from typing import Union

# to do: sequence cross-correlation and auto-correlation
# https://doi.org/10.1016/S0378-1119(02)01037-5
# https://blog.csdn.net/u013655530/article/details/47257525#:~:text=%E8%87%AA%E7%9B%B8%E5%85%B3%E6%98%AF%E4%BA%92%E7%9B%B8%E5%85%B3%E7%9A%84,%E5%8F%96%E5%80%BC%E7%9A%84%E7%9B%B8%E4%BC%BC%E7%A8%8B%E5%BA%A6%E3%80%82


'''
08.11.2024 by Haocheng
to do: define rules and corresponding assignment 
08.13.2024 added by Haocheng
(other vectors or rules? 2 dimen for 1 base)
'''

def filter_nucleotides(seq: str)->str:
    allowed_nucleotides = set("AUGC")
    filtered_seq = ''.join([nucleotide for nucleotide in seq if nucleotide in allowed_nucleotides])
    return filtered_seq

def rule_SW(seq: str):
    """
    * rule_SW: C or G = 1, A or U = 0
    """
    seq = filter_nucleotides(seq)
    return [1 if char == 'C' or char == 'G' else 0 for char in seq]

def rule_RY(seq: str):
    """
    * rule_RY: A or G = 1, C or U = 0
    """
    seq = filter_nucleotides(seq)
    return [1 if char == 'A' or char == 'G' else 0 for char in seq]

def rule_KM(seq: str):
    """
    * rule_KM: G or U = 1, C or A = 0
    """
    seq = filter_nucleotides(seq)
    return [1 if char == 'G' or char == 'U' else 0 for char in seq]

def rule_A(seq: str):
    """
    * rule_A: A = 1, other = 0
    """
    seq = filter_nucleotides(seq)
    return [1 if char == 'A' else 0 for char in seq]

def rule_U(seq: str):
    """
    * rule_U: U = 1, other = 0
    """
    seq = filter_nucleotides(seq)
    return [1 if char == 'U' else 0 for char in seq]

def rule_G(seq: str):
    """
    * rule_G: G = 1, other = 0
    """
    seq = filter_nucleotides(seq)
    return [1 if char == 'G' else 0 for char in seq]

def rule_C(seq: str):
    """
    * rule_C: C = 1, other = 0
    """
    seq = filter_nucleotides(seq)
    return [1 if char == 'C' else 0 for char in seq]

# print(type(rule_C("CAGCAGCAGCAGCAGCAGCAG")))
# print(rule_C("CAGCAGCAGCAGCAGCAGCAG"))


'''
08.11.2024 by Haocheng
to do: compute auto-correlation
'''

def seq_var(func: callable, seq: str) -> float:
    """
    @param func: rule_*
    """
    sigma_sq = np.var(func(seq))
    return sigma_sq

# 08.15.2024 added by Haocheng
def delta_auto_correl(l: int) -> float:
    if l >= 0:
        return 1 / (l.sqrt())
    else:
        raise ValueError("The length of RNA is wrong!")

# 08.15.2024 modified by Haocheng
def auto_correl(func: callable, seq: str, l: int) -> float:
    """
    @param func: rule_*
    @param l: distance between two nucleotides
    @return: Returns the calculated autocorrelation C and error terms delta_C
    """
    seq1 = filter_nucleotides(seq)
    N = len(seq1)
    A = 0
    B1 = 0
    B2 = 0
    
    for i in range(N - l):
        A = A + func(seq1[i])[0] * func(seq1[(i + l)])[0]
        B1 = B1 + func(seq1[i])[0]
        B2 = B2 + func(seq1[(i + l)])[0]

    sigma_sq = seq_var(func, seq)
    if sigma_sq == 0:
        return 0
    else: 
        C = (1 / seq_var(func, seq)) * (A / (N - l) - (B1 * B2) / ((N - l) ** 2))
        delta_C = delta_auto_correl(l)
        return C, delta_C

# print(auto_correl(rule_SW, "CAGCAGCAGCAGCAGCAGCAG", 3))
# print(auto_correl(rule_SW, "CAGCAGCAGCAGCAGCAGCAG", 2))
# print(auto_correl(rule_U, "CAGCAGCAGCAGCAGCAGCAG", 2))


# 08.15.2024 added by Haocheng
def par1_base(sub_seq: np.ndarray) -> float:
    """
    * return the percentage of ones in sub_seq
    """
    if sub_seq.ndim != 1:
        raise ValueError("The input must be 1d np.ndarray")
    count_ones = np.sum(sub_seq == 1)
    total_elements = len(sub_seq)
    return count_ones / total_elements if total_elements > 0 else 0.0

def auto_correl_cg(func: callable, seq: str, l: int) -> np.ndarray:
    """
    @param func: rule_*
    @param l: distance between two nucleotides
    @return: 
    """
    seq1 = filter_nucleotides(seq)
    N_b = len(seq1)
    seq1 = func(seq1)
    ext_b = N_b % l
    N_mat = N_b // l
    seq2 = []

    if ext_b % 2 == 0:
        del seq1[:(ext_b/2)]
        del seq1[(ext_b/2):] 
    else:
        del seq1[:((ext_b+1)/2)]
        del seq1[((ext_b-1)/2):] 

    # split the sequence into blocks
    split_mat = [seq1[i:i + l] for i in range(0, N_mat, l)]

    for i in range (N_mat):
        seq2.append(par1_base(split_mat[i]))

    return np.array(seq2)

'''
08.11.2024-08.13.2024 by Haocheng
# to do: self-atten matrix 
'''

def diagmat_gen(func: callable, seq: str) -> np.ndarray:
    """
    * diagonal matrix generator, based on the rule and sequence
    @param func: rule_*
    """
    vector = func(seq)
    return np.diag(vector)

# 08.13.2024 modified by Haocheng
def selfatten_mat_gen(N: str) -> np.ndarray:
    """
    * self-attention matrix generator, based on the length of sequence
    """
    selfatten_mat = np.zeros((N, N))

    # attention mechanism: using CDF(cumulative distribution function) of gaussion distribution
    mean = N/2    
    std_dev = N/4 
   
    for j in range(N):
        for i in range(N):
            l = abs(i - j)
            cdf = norm.cdf(N-l, mean, std_dev)
            selfatten_mat[j][i] = cdf

    return selfatten_mat


'''
08.13-15.2024 by Haocheng
# to do: Mercer's kernel function 
'''

def vector_gen(func: callable, seq: str) -> np.ndarray:
    vector = np.array(func(seq))
    return vector

def Mercers_kernel(vec1:list, vec2:list,
                    atten_mat:np.ndarray) -> np.ndarray:
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    K = np.dot(np.dot(vec1, atten_mat), vec2.T)
    return K

 # 08.20.2024 added by Haocheng
def trim_sequence(seq: np.ndarray, ext_b: int) -> np.ndarray:
    """
    * only used in slide_window_* functions
    * trim the edge of the sequence
    """
    if ext_b == 0:
        return seq
    elif ext_b % 2 == 0:
        half = ext_b // 2
        return seq[half:-half]
    elif ext_b % 2 == 1:
        front = (ext_b - 1) // 2
        back = (ext_b + 1) // 2
        return seq[front:-back]

def slide_window_kernel(func: callable, seq: str, window_size: int) -> np.ndarray:
    """
    @param func: rule_*
    """
    if seq == "":
        return np.array([])
    else:
        kernel = []
        seq1 = filter_nucleotides(seq)
        N_b = len(seq1)
        seq1 = func(seq1)
        # print(f"Filtered sequence: {seq1}")
        ext_b = N_b % window_size
        N_mat = N_b // window_size
        # print(f"ext_b: {ext_b}")
        
        seq1 = trim_sequence(seq1, ext_b)

        split_mat = [seq1[i:i + window_size] for i in range(0, N_mat * window_size, window_size)]

        for i in range(0, N_mat-1):
            if len(split_mat[i]) == window_size:
                kernel.append(Mercers_kernel(split_mat[i], split_mat[i+1], selfatten_mat_gen(window_size)))
            else:
                kernel.append(0)

        return np.array(kernel)


'''
08.14.2024 by Haocheng
# to do: Pooling of sequence 
'''

def seq_pooling(func: callable, seq: str, l: int, 
                gate_value: Union[int, float]) -> np.ndarray:
    """
    * Pooling of sequence
    @param seq: sequence
    @param l: length of pooling vector
    @param gate_value: nonlinear threshold to control the pooling
    """
    if seq == "":
        return np.array([])
    else:
        seq1 = filter_nucleotides(seq)
        N_b = len(seq1)
        seq1 = func(seq1)
        ext_b = N_b % l
        N_mat = N_b // l
        seq2 = []
        
        seq1 = trim_sequence(seq1, ext_b)
        
        # split the sequence into blocks
        split_mat = [seq1[i:i + l] for i in range(0, N_mat * l, l)]

        for i in range (N_mat):
            if sum(split_mat[i]) >= gate_value:
                seq2.append(1)
            else:
                seq2.append(0)

        return np.array(seq2)

#to do: apply the above functions to the data
PsData = pd.read_csv(r'C:\Users\23163\Desktop\PS prediction\RnaPSP\all data\PsSelf1Less500.csv')
PsData = PsData.dropna(axis=1, how='all')

# Apply the slide_window_kernel function to the PsData
PsData['SW_kernel_result'] = PsData['rna_sequence'].apply(
    lambda seq: slide_window_kernel(rule_SW, seq, window_size=3)
)

PsData['SW_pooling_result'] = PsData['rna_sequence'].apply(
    lambda seq: seq_pooling(rule_SW, seq, l=5, gate_value=2)
)

PsData = PsData.reset_index()
PsData = PsData.drop('level_0', axis=1)
PsData.to_csv(r'C:\Users\23163\Desktop\PS prediction\RnaPSP\all data\PsSelf1Less500.csv', index=False, encoding='utf-8-sig')