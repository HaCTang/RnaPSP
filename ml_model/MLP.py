import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interp
import os

# 读取数据集
df = pd.read_csv('/home/thc/RnaPSP/RnaPSP/all data/2 classification/TrainData.csv')

output_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(output_dir, 'MLP')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Replace 0 in 'shannon_entropy' to the maximum value of the column
max_value = df['shannon_entropy6_1'].replace(0, np.nan).max()
df['shannon_entropy6_1'] = df['shannon_entropy6_1'].replace(0, max_value)
max_value = df['shannon_entropy6_3'].replace(0, np.nan).max()
df['shannon_entropy6_3'] = df['shannon_entropy6_3'].replace(0, max_value)
max_value = df['shannon_entropy6_1'].replace(0, np.nan).max()
df['shannon_entropy10_5'] = df['shannon_entropy10_5'].replace(0, max_value)

# 选择用于训练的特征列
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

# Fill NaN values with the mean of the column
X.fillna(X.mean(), inplace=True)
# Replace infinite values with a large bound
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X = X.clip(lower=-1e10, upper=1e10)
# Fill NaN values with the mean of the column
X.fillna(X.mean(), inplace=True)

# 标签列
y = df['label']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 转换为Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# 构建Dataset和DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(21, 64)  # 输入层到隐藏层1
        self.fc2 = nn.Linear(64, 32)  # 隐藏层1到隐藏层2
        self.fc3 = nn.Linear(32, 1)   # 隐藏层2到输出层
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# 实例化模型、定义损失函数和优化器
model = MLP()
criterion = nn.BCELoss()  # 二分类任务使用二元交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    # 每个epoch打印损失
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# 测试模型并计算ROC AUC
model.eval()
with torch.no_grad():
    y_scores = model(X_test)
    y_pred = y_scores.round()  # 将预测值四舍五入为0或1
    accuracy = (y_pred == y_test).float().mean()
    print(f'Accuracy: {accuracy.item() * 100:.2f}%')
    
    # 计算ROC曲线和AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC Score: {roc_auc:.2f}")
    
    # 打印分类报告
    print(classification_report(y_test, y_pred))

# 绘制ROC曲线
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

# 定义五折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

plt.figure()
i = 0

for train_index, val_index in kf.split(X_scaled):
    print(f'Fold {i+1}')
    
    # 分割训练集和验证集
    X_train_fold, X_val_fold = X_scaled[train_index], X_scaled[val_index]
    y_train_fold, y_val_fold = y[train_index], y[val_index]

    # 转换为Tensor
    X_train_fold = torch.tensor(X_train_fold, dtype=torch.float32)
    y_train_fold = torch.tensor(y_train_fold.values, dtype=torch.float32).unsqueeze(1)
    X_val_fold = torch.tensor(X_val_fold, dtype=torch.float32)
    y_val_fold = torch.tensor(y_val_fold.values, dtype=torch.float32).unsqueeze(1)

    # 构建Dataset和DataLoader
    train_dataset = TensorDataset(X_train_fold, y_train_fold)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 实例化模型
    model = MLP()

    # 定义损失函数和优化器
    criterion = nn.BCELoss()  # 二分类任务使用二元交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # 在验证集上测试模型并计算ROC AUC
    model.eval()
    with torch.no_grad():
        y_scores = model(X_val_fold)
        fpr, tpr, _ = roc_curve(y_val_fold, y_scores)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (area = %0.2f)' % (i+1, roc_auc))

    i += 1

# 画对角线
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=0.8)

# 计算并画出平均ROC曲线
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (area = %0.2f ± %0.2f)' % (mean_auc, std_auc), lw=2, alpha=0.8)

# 计算并画出标准差区域
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
