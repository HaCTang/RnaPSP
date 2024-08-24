import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv('/home/thc/RnaPSP/RnaPSP/all data/PsLess500.csv')

##############################################################################
# Replace 0 in 'shannon_entropy' to the maximum value of the column
max_value = df['shannon_entropy'].replace(0, np.nan).max()
df['shannon_entropy'] = df['shannon_entropy'].replace(0, max_value)

# Choose the columns to be used for training
X = df[['GC_ratio', 
        'G_ratio', 
        'AU_ratio', 
        # 'kolmogorov_complexity', 
        'shannon_entropy',
        'SW_kernel_avg',
        'SW_kernel_var',
        'SW_pooling_avg',
        'SW_pooling_var',
        'mfe_energy',
        'fc_energy',
        'mea_score',
        'cen_distance',
        'cen_energy',
        'mfe_freq',
        'ensemble_diver',
        'mean_bp_distance',
        'poly_rna',
        'repeat_rna',
        'else_rna']].copy()

# Fill NaN values with the mean of the column
X.fillna(X.mean(), inplace=True)
# Replace infinite values with a large bound
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X = X.clip(lower=-1e10, upper=1e10)
# Fill NaN values with the mean of the column
X.fillna(X.mean(), inplace=True)

#todo: Generate synthetic anomalies containing above features


# Extend the original dataset with synthetic anomalies
X_extended = pd.concat([X, X_anomalies], ignore_index=True)
y_extended = [1] * len(X) + [-1] * len(X_anomalies)  # 1表示正常，-1表示异常

# Initialize KFold cross-validation
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store the results of each fold
roc_auc_scores = []
ap_scores = []

# Start KFold cross-validation
for train_index, test_index in kf.split(X_extended):
    # Split training and test sets
    X_train, X_test = X_extended.iloc[train_index], X_extended.iloc[test_index]
    y_train, y_test = np.array(y_extended)[train_index], np.array(y_extended)[test_index]

    # Ensure the test set contains at least one positive and one negative class
    if len(np.unique(y_test)) < 2:
        continue  # Skip this fold if the test set contains only one class
    
    # Create Random Forest model
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Fit the model
    rf_classifier.fit(X_train, y_train)
    
    # Predict scores for the test set
    y_scores = rf_classifier.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    
    # Calculate ROC AUC score
    roc_auc = roc_auc_score(y_test, y_scores)
    roc_auc_scores.append(roc_auc)
    
    # Calculate Average Precision (AP) score
    ap_score = average_precision_score(y_test, y_scores)
    ap_scores.append(ap_score)

# Print average scores
print(roc_auc_scores)
print(ap_scores)
print(f"Average ROC AUC Score: {np.mean(roc_auc_scores)}")
print(f"Average Average Precision Score: {np.mean(ap_scores)}")

# Split data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_extended, y_extended, test_size=0.3, random_state=42)

# Build Random Forest model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions
df['predicted_label'] = rf_classifier.predict(X)
df['predicted_proba'] = rf_classifier.predict_proba(X)[:, 1]  # Probability estimates for the positive class

# Output results
print(classification_report(y_extended, rf_classifier.predict(X_extended)))
print("ROC AUC Score:", roc_auc_score(y_extended, rf_classifier.predict_proba(X_extended)[:, 1]))

# Save the results
df.to_csv('/home/thc/RnaPSP/RnaPSP/all data/test_RF.csv', 
          index=False, encoding='utf-8-sig')
