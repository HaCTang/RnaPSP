import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import numpy as np

df = pd.read_csv('/home/thc/RnaPSP/RnaPSP/all data/PsLess500.csv')

##############################################################################
'''
08.18.2024 by Haocheng
to do: Biuld OneClassSVM model
'''
# Replace 0 in 'shannon_entropy' to the maximum value of the column
max_value = df['shannon_entropy'].replace(0, np.nan).max()
df['shannon_entropy'] = df['shannon_entropy'].replace(0, max_value)

# weights = {
#     'GC_ratio': 1.0,  
#     'G_ratio': 2.0,
#     'AU_ratio': 1.0,
#     'kolmogorov_complexity': 0.05,
#     'shannon_entropy': 1,
#     'SW_kernel_avg': 1,
#     'SW_kernel_var': 1,
#     'SW_pooling_avg': 1,
#     'SW_pooling_var': 1
# }

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
        'cen_energy',
        'mfe_freq',
        'ensemble_diver',
        'mean_bp_distance',
        'mean_bp_distance',
        'poly_rna',
        'repeat_rna',
        'else_rna']].copy()

# Fill NaN values with the mean of the column
X.fillna(X.mean(), inplace=True)

# Replace infinite values with a large boond
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X = X.clip(lower=-1e10, upper=1e10)

# Fill NaN values with the mean of the column
X.fillna(X.mean(), inplace=True)

# for descriptor in weights:
#     X[descriptor] = X[descriptor] * weights[descriptor]

# Create a OneClassSVM model
oc_svm = make_pipeline(StandardScaler(), OneClassSVM(kernel='rbf', gamma='auto', nu=0.1))
oc_svm.fit(X)

# Predict the class of each sample
predictions = oc_svm.predict(X)
df['label'] = predictions
sum_feature1 = df['label'].sum()
print(sum_feature1)
print(df[df.label==-1])
df.to_csv('/home/thc/RnaPSP/RnaPSP/all data/test_svm.csv', 
          index=False, encoding='utf-8-sig')


##############################################################################
'''
08.22.2024 by Haocheng
to do: KFold cross-validation
'''
# Initialize cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
classification_reports = []

for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    oc_svm = make_pipeline(StandardScaler(), OneClassSVM(kernel='rbf', gamma='auto', nu=0.1))
    oc_svm.fit(X_train)
    
    # Predict on test set
    predictions = oc_svm.predict(X_test)
    
    # Create dummy true labels for demonstration (since this is unsupervised)
    true_labels = np.ones(len(X_test))
    
    # Get classification report and convert it to a dictionary
    report = classification_report(true_labels, predictions, output_dict=True, zero_division=0)
    
    # Flatten the report dictionary and add fold number
    report_df = pd.DataFrame(report).transpose().reset_index()
    report_df['fold'] = fold
    
    # Append the report for this fold to the list
    classification_reports.append(report_df)

# Concatenate all reports into a single DataFrame
final_report_df = pd.concat(classification_reports, ignore_index=True)

# Save the DataFrame to a CSV file
final_report_df.to_csv('classification_report.csv', index=False)

print("Classification report for each fold has been saved to 'classification_report.csv'.")
##############################################################################

##############################################################################
'''
08.23.2024 by Haocheng
to do: PCA analysis Visualization
'''
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Dimensionality reduction to 3D using PCA
pca = PCA(n_components=3)
X_reduced = pca.fit_transform(X_test)

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Use the predictions as colors
scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=predictions, cmap='coolwarm')

# Add title and labels
ax.set_title('SVM Decision Boundary (PCA 3D reduced)')
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')

# Add color bar
fig.colorbar(scatter)

plt.show()
##############################################################################

##############################################################################
'''
08.23.2024 by Haocheng
to do: t-SNE analysis Visualization
'''
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

tsne = TSNE(n_components=3, random_state=42)
X_tsne = tsne.fit_transform(X_test)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=predictions, cmap='coolwarm')

ax.set_title('t-SNE Visualization of Clusters (3D)')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_zlabel('t-SNE 3')

fig.colorbar(scatter)

plt.show()
##############################################################################
