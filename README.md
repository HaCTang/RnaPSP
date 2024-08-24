# RNA

Only to be used on Linux.

# 0 Enlarging Dataset (not done)

Delete\Replace 20% of the sequence of RNA

Stick the head and tail of the RNA

# 1 Wash Data (done)


# 2 Base Ratio (done)

Computing the ratio of each types of bases

# 3 Evenness (done)

## 3.1 Kolmogorov complexity (done)

Using `zlib.compress()` to compress data and compute the length of compressed data. The value shows the symmetry of the sequence.

## 3.2 Shannon entropy (done)

Using Shannon entropy to compute the entropy of constitution:

$$
H(X) = - \sum_{i=1}^{n} p(x_i) \log_2 p(x_i)
$$

## 3.3 Sliding window statistics (done)

Employ function above to screening the evenness of a certain sequence.

# 4 Self-correlation

## 4. 1 Set up 7 different rules(label encoding vs one-hot-encoding?)

| Rule | Assignment |  | Rule | Assignment |  |
| --- | --- | --- | --- | --- | --- |
| A | A = 1 | else = 0 | SW | C or G = 1 | A or U = 0 |
| U | U = 1 | else = 0 | RY | A or G = 1 | C or U = 0 |
| C | C = 1 | else = 0 | KM | G or U = 1 | A or C = 0 |
| G | G = 1 | else = 0 |  |  |  |

**Study of statistical correlations in DNA sequences, Gene 300 (2002) 105–115*

## 4.2 Sliding kernel functions

### 4.2.1 Build up attention matrix for RNA blocks

Many different distribution are considered. Here, I just choose CDF(cumulative distribution function) of GD (Gaussian distribution) because the value is between 0-1 and the assignment is linear, so it doesn’t require normalization.  

$$
p_l(x) = \frac{1}{2} \left[1 + \text{erf}\left(\frac{x - \mu}{\sigma \sqrt{2}}\right)\right]
$$

$$
\text{erf}(z) = \frac{2}{\sqrt{\pi}} \int_{0}^{z} e^{-t^2} dt
$$

In Python, the function `scipy.stats.norm.cdf` are used to calculate this CAF of GD

Another way is to use UD (uniform distribution). Noting that UD equals GD when standard deviation is large, so I didn’t write it here.

We use this function to obtain the attention matrix, $Mat_{Atten}^{n \times n}$, to estimate the impact of nearby environment of certain nucleobase. **$*n$*** equals the length of the RNA sequence you choose. 

$$
Mat_{Atten}^{n \times n} =          \begin{bmatrix}
p_{0} & p_{1} & ... & p_{n-1} \\
p_{1} & p_{0} & ... & p_{n-2} \\
... & ... & ... & ... \\
p_{n-1} & p_{n-2} & ... & p_{0}
\end{bmatrix}   
$$

### 4.2.2 Build up Mercer’s kernel function

To compare the difference of two RNA sequences of the same length, we build up a vector, $V_{Seq*}$ , below, based on previous rules:

$$
V_{Seq1} = \begin{bmatrix}b_{1}^{'} & b_{2}^{'} & ... & b_{n}^{'} \end{bmatrix} \\                       V_{Seq2} = \begin{bmatrix}b_{1}^{''} & b_{2}^{''} & ... & b_{n}^{''} \end{bmatrix}   
$$

Then we build up a kernel function (**Mercer’s kernel function** here) to compute the similarity of 2 sequences.

$$
K = V_{Seq1} Mat_{Atten}^{n \times n} V_{Seq2}^{T}
$$

### 4.2.3 Build up sliding window

## 4.3 Autocorrelation (done)

### 4.3.1 autocorrelation of a single base in a sequence (done)

Variance: 

$$
\sigma^2 = \frac{1}{N} \sum_{i=1}^{N}x_i^2-(\frac{1}{N}\sum_{i=1}^{N}x_i)^2
$$

The autocorrelation of sequence at distance ‘ is defined by:

$$
c(𝒍)=\frac{1}{\sigma^2}\times[\frac{1}{N-𝒍}\sum_{i=1}^{N-𝒍}x_ix_{i+l}-\frac{1}{(N-𝒍)^2}\sum_{i=1}^{N-𝒍}x_i\sum_{i=1}^{N-𝒍}x_{i+𝒍}]
$$

Only considering linear correlations, and previous research indicated the correlation is essentially linear. 

The error in the determination c(𝒍), due to statistical fluctuations, can be easily estimated to be :

$$
\Delta c(𝒍) = \frac{1}{\sqrt{N}}
$$

### 4.3.2 autocorrelation of a subsequence in a RNA (Coarse Graining)

Use attention matrix to replace single base above to obtain updated formula

Variance Matrix:

$$
{\Sigma}{\Sigma}^T = \frac{1}{N} \sum_{i=1}^{N}X_iX_i^T-(\frac{1}{N}\sum_{i=1}^{N}X_i)^2
$$

Autocorrelation Matrix:

$$
C(𝒍)=({\Sigma}{\Sigma}^T)^{-1}\times[\frac{1}{N-𝒍}\sum_{i=1}^{N-𝒍}X_iX_{i+l}-\frac{1}{(N-𝒍)^2}\sum_{i=1}^{N-𝒍}X_i\sum_{i=1}^{N-𝒍}X_{i+𝒍}]
$$

However, formula above containing too much information. So, another way is to develop a renewable formula using compression.

$$
C_{comp}(𝒍)=\frac{(p_1-p_2)^2}{2(p_1+p_2)-(p_1+p_2)^2}[1-\frac{2𝒍}{𝒍_0}]\space for \space 1\leq𝒍\leq𝒍_0
$$

Whereas,  consider 2 subsequences, both of length   $𝒍$, and $p_1$ and $p_2$ respectively represent the proportions of 1’s in 2 subsequences.    $𝒍_0$ is the length of the whole sequence. Only consider adjacent $C_{comp}(𝒍)$, so we can get a vector $V_{comp}(l)$ to describe adjacent autocorrelation:

$$
V_{comp}(l) = [C_{comp12}(𝒍),\space C_{comp23}(𝒍), ... ]
$$

Computing average value of adjacent autocorrelation:

$$
<C_{comp}(𝒍)> 
$$

## 4.4 Pooling (done)

Pooling is something like a nonlinear dimensionality reduction technique here to deal with the sequence out of the idea that the symmetry is not that important, but the local component matters.

![image.png](images/pooling%20of%20RNA.png)

# 5 Machine Learning Models (not done)

## 5.1 One-class SVM

## 5.2 Linear Regression

## 5.3 Isolation Forest

random shuffling