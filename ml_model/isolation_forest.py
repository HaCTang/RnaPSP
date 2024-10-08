import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, roc_curve
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv('/home/thc/RnaPSP/RnaPSP/all data/PsLess500.csv')

output_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(output_dir, 'isolation_forest')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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

# Fill NaN values
X = X.apply(lambda col: col.fillna(col.mean()), axis=0)
# Check and handle infinity and out-of-range values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.dropna()

# Biuld Isolation Forest model
iso_forest = IsolationForest(n_estimators=100, max_samples=0.75, contamination=0.05, 
                             max_features=1.0, bootstrap=False, random_state=42, warm_start=False)
iso_forest.fit(X)

# Make predictions (1 indicates normal samples, -1 indicates anomaly samples)
df['label'] = iso_forest.predict(X)

# Get scores (the smaller the value, the more likely it is an anomaly sample)
df['score'] = iso_forest.decision_function(X)
# anomalies = df[df['anomaly'] == -1]
# print(anomalies)
sum_feature1 = df['label'].sum()
mean_feature2 = df['score'].mean()
print(sum_feature1, mean_feature2)
print(df[df.label==-1])
output_path = os.path.join(output_dir, 'test_IsoFor.csv')
df.to_csv(output_path, index=False)

# df['anomaly'] = df['label'].apply(lambda x: 'outlier' if x==-1  else 'inlier') 
# fig = px.histogram(df,x='scores',color='anomaly') 
# fig.show()
##############################################################################

##############################################################################
# Plot the histogram of scores
df['anomaly'] = df['label'].apply(lambda x: 'outlier' if x == -1 else 'inlier')

outliers = df[df['anomaly'] == 'outlier']['score']
inliers = df[df['anomaly'] == 'inlier']['score']

plt.hist([outliers, inliers], bins=10, color=['red', 'blue'], label=['outlier', 'inlier'])

plt.legend()

plt.title('Histogram of Scores')
plt.xlabel('Scores')
plt.ylabel('Frequency')

# save the plot
output_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(output_dir, 'isolation_forest')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_path = os.path.join(output_dir, 'iso_fore_histogram.png')
plt.savefig(output_path)
plt.close()

# ROC 曲线
# fpr, tpr, _ = roc_curve(df['label'], df['score'])
# roc_auc = auc(fpr, tpr)
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# output_path = os.path.join(output_dir, 'iso_fore_ROC.png') 
# plt.savefig(output_path)
# plt.close()

# # Precision-Recall 曲线
# precision, recall, _ = precision_recall_curve(df['label'], df['score'])
# pr_auc = auc(recall, precision)
# plt.figure(figsize=(8, 6))
# plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall (PR) Curve')
# plt.legend(loc='lower left')
# output_path = os.path.join(output_dir, 'iso_fore_Precision-Recall.png') 
# plt.savefig(output_path)
# plt.close()
##############################################################################

# visualize the anomalies detected by Isolation Forest
plt.figure(figsize=(10, 6))
sns.scatterplot(x='GC_ratio', y='shannon_entropy', hue='label', palette={1: 'blue', -1: 'red'}, data=df)
plt.title('Anomalies Detected by Isolation Forest')
output_path = os.path.join(output_dir, 'iso_fore_points.png') 
plt.savefig(output_path)
plt.close()
