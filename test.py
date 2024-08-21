import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np

df = pd.read_csv(r'C:\Users\23163\Desktop\PS prediction\RnaPSP\all data\PsLess500.csv')

# Replace 0 in 'shannon_entropy' to the maximum value of the column
max_value = df['shannon_entropy'].replace(0, np.nan).max()
df['shannon_entropy'] = df['shannon_entropy'].replace(0, max_value)

# Choose the columns to be used for training
X = df[['GC_ratio', 
        'G_ratio', 
        'AU_ratio', 
        'kolmogorov_complexity', 
        'shannon_entropy',
        'SW_kernel_avg',
        'SW_kernel_var',
        'SW_pooling_avg',
        'SW_pooling_var']].copy()

# Fill NaN values with the mean of the column
X.fillna(X.mean(), inplace=True)

# Replace infinite values with a large boond
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X = X.clip(lower=-1e10, upper=1e10)

# Fill NaN values with the mean of the column
X.fillna(X.mean(), inplace=True)

# Biuld Isolation Forest model
iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
iso_forest.fit(X)

# Make predictions (1 indicates normal samples, -1 indicates anomaly samples)
df['anomaly'] = iso_forest.predict(X)

# Get anomaly scores (the smaller the value, the more likely it is an anomaly sample)
df['anomaly_score'] = iso_forest.decision_function(X)
# anomalies = df[df['anomaly'] == -1]
# print(anomalies)
df.to_csv(r'C:\Users\23163\Desktop\PS prediction\RnaPSP\all data\test_IsoFor.csv', 
          index=False, encoding='utf-8-sig')
