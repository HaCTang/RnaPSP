##############################################################################
'''
09.04.2024 by Haocheng
to do: PCA analysis Visualization
'''
pca = PCA(n_components=3)
X_reduced= pca.fit_transform(X_test)

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Use the predictions as colors
predictions = model.eval()
with torch.no_grad(): 
    predictions = model(X_test)
scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=predictions, cmap='coolwarm')

# Add title and labels
ax.set_title('MLP_PCA_3d')
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')

# Add color bar
fig.colorbar(scatter)
plt.savefig(os.path.join(output_dir, 'MLP_PCA_3d.png'))
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

ax.set_title('MLP_tSNE_3d')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_zlabel('t-SNE 3')

fig.colorbar(scatter)
plt.savefig(os.path.join(output_dir, 'MLP_tSNE_3d.png'))
plt.close()