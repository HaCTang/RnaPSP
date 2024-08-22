import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import numpy as np

df = pd.read_csv(r'C:\Users\23163\Desktop\PS prediction\RnaPSP\all data\PsLess500.csv')

##############################################################################
# Replace 0 in 'shannon_entropy' to the maximum value of the column
max_value = df['shannon_entropy'].replace(0, np.nan).max()
df['shannon_entropy'] = df['shannon_entropy'].replace(0, max_value)

# Choose the columns to be used for training
X = df[['GC_ratio', 
        'G_ratio', 
        'AU_ratio', 
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

# Initialize cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    oc_svm = make_pipeline(StandardScaler(), OneClassSVM(kernel='rbf', gamma='auto', nu=0.1))
    oc_svm.fit(X_train)
    
    # Predict on test set
    predictions = oc_svm.predict(X_test)
    
    # Create dummy true labels for demonstration
    # Ideally, you would have true labels for evaluation if it were supervised
    true_labels = np.ones(len(X_test))  
    
    print(f"Fold {fold}:")
    print(classification_report(true_labels, predictions, zero_division=0))
    fold += 1
##############################################################################

##############################################################################
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 对数据进行PCA降维到三维
pca = PCA(n_components=3)
X_reduced = pca.fit_transform(X_test)

# 创建三维绘图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 使用预测结果进行颜色编码
scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=predictions, cmap='coolwarm')

# 添加标题和标签
ax.set_title('SVM Decision Boundary (PCA 3D reduced)')
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')

# 添加颜色条
fig.colorbar(scatter)

plt.show()


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 对数据进行t-SNE降维到三维
tsne = TSNE(n_components=3, random_state=42)
X_tsne = tsne.fit_transform(X_test)

# 创建三维绘图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 使用预测结果进行颜色编码
scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=predictions, cmap='coolwarm')

# 添加标题和标签
ax.set_title('t-SNE Visualization of Clusters (3D)')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_zlabel('t-SNE 3')

# 添加颜色条
fig.colorbar(scatter)

plt.show()
##############################################################################