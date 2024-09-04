import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, roc_curve, auc
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

df = pd.read_csv('/home/thc/RnaPSP/RnaPSP/all data/2 classification/TrainData.csv')

output_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(output_dir, 'xgboost')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

max_value = df['shannon_entropy6_1'].replace(0, np.nan).max()
df['shannon_entropy6_1'] = df['shannon_entropy6_1'].replace(0, max_value)
max_value = df['shannon_entropy6_3'].replace(0, np.nan).max()
df['shannon_entropy6_3'] = df['shannon_entropy6_3'].replace(0, max_value)
max_value = df['shannon_entropy6_1'].replace(0, np.nan).max()
df['shannon_entropy10_5'] = df['shannon_entropy10_5'].replace(0, max_value)

X = df[['GC_ratio', 
        'G_ratio', 
        'AU_ratio', 
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

X.fillna(X.mean(), inplace=True)
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X = X.clip(lower=-1e10, upper=1e10)
X.fillna(X.mean(), inplace=True)

y = df['label']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize KFold for cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

plt.figure()
i = 0

for train_index, test_index in kf.split(X_scaled):
    
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Make sure there are both positive and negative samples in the test set
    if len(np.unique(y_test)) < 2:
        continue  
    
    # Build XGBoost model
    xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    
    xgb_classifier.fit(X_train, y_train)
    
    y_scores = xgb_classifier.predict_proba(X_test)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (area = %0.2f)' % (i+1, roc_auc))

    i += 1

# Draw diagonal line
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=0.8)

# Calculate and draw mean ROC curve
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (area = %0.2f ± %0.2f)' % (mean_auc, std_auc), lw=2, alpha=0.8)

# Draw standard deviation area
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='gray', alpha=0.2, label=r'± 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBoost_ROC')
plt.legend(loc='lower right')
plt.savefig(os.path.join(output_dir, 'XGBoost_KFold_ROC.png'))
plt.close()

# Print and save results 
print(f"Average ROC AUC Score: {np.mean(aucs)} ± {np.std(aucs)}")
output_path = os.path.join(output_dir, 'kfold_XGBoost_results.txt')
with open(output_path, 'w') as f:
    f.write(f"Average ROC AUC Score: {np.mean(aucs)} ± {np.std(aucs)}\n")

##############################################################################
'''
09.04.2024 by Haocheng
to do: PCA analysis Visualization
'''
pca = PCA(n_components=3)
X_reduced = pca.fit_transform(X_test)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

predictions = xgb_classifier.predict(X_test)
scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=predictions, cmap='coolwarm')

ax.set_title('XGBoost_PCA_3d')
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')

fig.colorbar(scatter)
plt.savefig(os.path.join(output_dir, 'XGBoost_PCA_3d.png'))
plt.close()

##############################################################################
'''
09.04.2024 by Haocheng
to do: t-SNE analysis Visualization
'''
tsne = TSNE(n_components=3, random_state=42)
X_tsne = tsne.fit_transform(X_test)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=predictions, cmap='coolwarm')

ax.set_title('XGBoost_tSNE_3d.png')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_zlabel('t-SNE 3')

fig.colorbar(scatter)
plt.savefig(os.path.join(output_dir, 'XGBoost_tSNE_3d.png'))
plt.close()
