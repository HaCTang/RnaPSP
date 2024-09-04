import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, roc_curve, auc
from sklearn.model_selection import KFold, train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os

df = pd.read_csv('/home/thc/RnaPSP/RnaPSP/all data/2 classification/TrainData.csv')

# Output directory
output_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(output_dir, 'SVM')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

##############################################################################
# Replace 0 in 'shannon_entropy' to the maximum value of the column
max_value = df['shannon_entropy6_1'].replace(0, np.nan).max()
df['shannon_entropy6_1'] = df['shannon_entropy6_1'].replace(0, max_value)
max_value = df['shannon_entropy6_3'].replace(0, np.nan).max()
df['shannon_entropy6_3'] = df['shannon_entropy6_3'].replace(0, max_value)
max_value = df['shannon_entropy6_1'].replace(0, np.nan).max()
df['shannon_entropy10_5'] = df['shannon_entropy10_5'].replace(0, max_value)

# Choose the columns to be used for training
X = df[['GC_ratio', 
        'G_ratio', 
        'AU_ratio', 
        # 'kolmogorov_complexity', 
        'shannon_entropy6_1',
        'shannon_entropy6_3',
        'shannon_entropy10_5',
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

y = df['label']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Biuld SVM model
svm_classifier = SVC(kernel='linear', probability=True, random_state=42)
svm_classifier.fit(X_train, y_train)

# make predictions
y_scores = svm_classifier.predict_proba(X_test)[:, 1] 

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
plt.close()

# Output results
print(classification_report(y_test, svm_classifier.predict(X_test)))
print("ROC AUC Score:", roc_auc_score(y_test, y_scores))

# Save the results
df['predicted_label'] = svm_classifier.predict(X)
df['predicted_proba'] = svm_classifier.predict_proba(X)[:, 1] 

output_path = os.path.join(output_dir, 'SVM_results.csv')
df.to_csv(output_path, index=False)

##############################################################################
# Initialize cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Variables to store cross-validation results
tprs = []
mean_fpr = np.linspace(0, 1, 100)
roc_aucs = []
classification_reports = []
i = 0

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Build SVM model
    svm_classifier = SVC(kernel='linear', probability=True, random_state=42)
    svm_classifier.fit(X_train, y_train)
    
    # Make predictions
    y_scores = svm_classifier.predict_proba(X_test)[:, 1] 
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    roc_aucs.append(roc_auc)
    
    # Plot ROC curve for the fold
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC fold {i+1} (area = {roc_auc:.2f})')
    i += 1

# Plot mean ROC curve
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=0.8)
plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (area = {mean_auc:.2f})', lw=2, alpha=0.8)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='gray', alpha=0.2)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM_ROC)')
plt.legend(loc='lower right')
plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
plt.close()

# Output results
print("Average ROC AUC Score:", np.mean(roc_aucs))
print("Classification Reports:")
for i, report in enumerate(classification_reports):
    print(f"\nFold {i+1}:\n{report}")

# Save the results
df['predicted_label'] = svm_classifier.predict(X)
df['predicted_proba'] = svm_classifier.predict_proba(X)[:, 1] 

output_path = os.path.join(output_dir, 'SVM_results.csv')
df.to_csv(output_path, index=False)