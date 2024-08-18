import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

df = pd.read_csv(r'C:\Users\23163\Desktop\PS prediction\RnaPSP\all data\PsSelf1Less500.csv')

# Replace 0 in 'shannon_entropy' to the maximum value of the column
max_value = df['shannon_entropy'].replace(0, np.nan).max()
df['shannon_entropy'] = df['shannon_entropy'].replace(0, max_value)

weights = {
    'GC_ratio': 1.0,  
    'G_ratio': 2.0,
    'AU_ratio': 1.0,
    'kolmogorov_complexity': 0.05,
    'shannon_entropy': 1
}

# Choose the columns to be used for training
X = df[['GC_ratio', 'G_ratio', 'AU_ratio', 'kolmogorov_complexity', 'shannon_entropy']].copy()

# Fill NaN values with the mean of the column
X.fillna(X.mean(), inplace=True)

# Replace infinite values with a large boond
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X = X.clip(lower=-1e10, upper=1e10)

# Fill NaN values with the mean of the column
X.fillna(X.mean(), inplace=True)

for descriptor in weights:
    X[descriptor] = X[descriptor] * weights[descriptor]

# Create a OneClassSVM model
oc_svm = make_pipeline(StandardScaler(), OneClassSVM(kernel='rbf', gamma='auto', nu=0.1))
oc_svm.fit(X)

# Predict the class of each sample
predictions = oc_svm.predict(X)
df['Prediction'] = predictions

print(df)
