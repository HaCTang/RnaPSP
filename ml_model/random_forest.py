import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, roc_curve, auc
from sklearn.model_selection import KFold, train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os


df = pd.read_csv('/home/thc/RnaPSP/RnaPSP/all data/2 classification/TrainData.csv')

output_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(output_dir, 'random_forest')
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

# Build Random Forest model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions
y_scores = rf_classifier.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class

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
print(classification_report(y_test, rf_classifier.predict(X_test)))
print("ROC AUC Score:", roc_auc_score(y_test, y_scores))

# Save the results
output_path = os.path.join(output_dir, 'ttest_RF.csv')
df.to_csv(output_path, index=False)

##############################################################################
# Initialize KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

plt.figure()
i = 0

for train_index, test_index in kf.split(X):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Make sure the test set contains both classes
    if len(np.unique(y_test)) < 2:
        continue  
    
    # Build Random Forest model
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    
    # Make predictions and calculate ROC curve and AUC
    y_scores = rf_classifier.predict_proba(X_test)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (area = %0.2f)' % (i+1, roc_auc))

    i += 1

# Plot ROC curve
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=0.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (area = %0.2f ± %0.2f)' % (mean_auc, std_auc), lw=2, alpha=0.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='gray', alpha=0.2, label=r'± 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.savefig(os.path.join(output_dir, 'roc_curve_cross_validation.png'))
plt.close()

# print and save the results
print(f"Average ROC AUC Score: {np.mean(aucs)} ± {np.std(aucs)}")
output_path = os.path.join(output_dir, 'kfold_RF_results.txt')
with open(output_path, 'w') as f:
    f.write(f"Average ROC AUC Score: {np.mean(aucs)} ± {np.std(aucs)}\n")