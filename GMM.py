import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# 加载降维后的TF-IDF矩阵
reduced_matrix_path = r'D:\machine_learning_class_project\datasets\Reduced_TFIDF_Matrix.csv'
reduced_tfidf_matrix = pd.read_csv(reduced_matrix_path)

# 提取前100维的数据进行聚类
reduced_tfidf_matrix_100 = reduced_tfidf_matrix.iloc[:, :100]

# 查看提取的维度
print(f"提取的数据维度: {reduced_tfidf_matrix_100.shape}")

# 使用GMM进行聚类
n_components = 5  # 设置高斯混合模型的组件数量，相当于聚类的数量
gmm = GaussianMixture(n_components=n_components, random_state=42)
gmm.fit(reduced_tfidf_matrix_100)
cluster_labels = gmm.predict(reduced_tfidf_matrix_100)

# 将聚类结果添加到数据框中
reduced_tfidf_matrix_100['cluster'] = cluster_labels

# 加载文章基本信息
basic_info_path = r'D:\machine_learning_class_project\datasets\process_Sampled_Dataset.csv'
basic_info_df = pd.read_csv(basic_info_path)

# 合并聚类结果和文章基本信息
clustered_data_with_info = pd.concat([reduced_tfidf_matrix_100, basic_info_df[['publication', 'author', 'year', 'month']]], axis=1)

# 保存带有聚类结果和基本信息的数据
clustered_data_with_info_path = r'D:\machine_learning_class_project\datasets\Clustered_GMM_100.csv'
clustered_data_with_info.to_csv(clustered_data_with_info_path, index=False)
print(f"带有聚类结果和基本信息的数据已保存到 {clustered_data_with_info_path}")
print(f"合并后的数据的维度: {clustered_data_with_info.shape}")

# Part 1: 出版物及其文章数量和比例的统计
publication_counts = clustered_data_with_info['publication'].value_counts()
total_articles = publication_counts.sum()
publication_proportions = publication_counts / total_articles
print("出版物的数目:", len(publication_counts))
print("每个出版物的文章数:\n", publication_counts)
print("每个出版物所占文章比例:\n", publication_proportions)

# Part 2: 每个publication找到每一类（cluster）中的文章数目；
publication_cluster_counts = clustered_data_with_info.groupby(['publication', 'cluster']).size().unstack(fill_value=0)
# print("每个出版社找到每一类中的文章数目:\n", publication_cluster_counts)

# Calculate proportions per cluster for each publication
publication_cluster_proportions = publication_cluster_counts.div(publication_cluster_counts.sum(axis=1), axis=0)
# print("Proportions of articles per publication per cluster:\n", publication_cluster_proportions)

# Table visualization for Part 1 and Part 2
# Part 1: Table for publication statistics
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.axis('tight')
ax1.axis('off')
table_data = pd.DataFrame({
    'Publication': publication_counts.index,
    'Total Articles': publication_counts.values,
    'Proportion': publication_proportions.values
})
table = ax1.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc='center', loc='center')
for key, cell in table.get_celld().items():
    cell.set_edgecolor('white')
    cell.set_facecolor('#696969')
    cell.set_text_props(color='white')
    if key[0] == 0 or key[1] == -1:
        cell.set_text_props(weight='bold', color='white')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
ax1.set_title('Number of articles per publishing house and its proportion')


# Part 2: Table for publication cluster counts
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.axis('tight')
ax2.axis('off')
# Table customization
table_data = publication_cluster_counts
table = ax2.table(cellText=table_data.values, colLabels=table_data.columns, rowLabels=table_data.index, cellLoc='center', loc='center')

# Customize the table cells
for key, cell in table.get_celld().items():
    cell.set_edgecolor('white')
    cell.set_facecolor('#696969')
    cell.set_text_props(color='white')
    if key[0] == 0 or key[1] == -1:
        cell.set_text_props(weight='bold', color='white')

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
ax2.set_title('Article Counts per Publication per Cluster', color='black')


fig3, ax3 = plt.subplots(figsize=(12, 6))
ax3.axis('tight')
ax3.axis('off')
table_data = publication_cluster_proportions
table = ax3.table(cellText=table_data.values, colLabels=table_data.columns, rowLabels=table_data.index, cellLoc='center', loc='center')
for key, cell in table.get_celld().items():
    cell.set_edgecolor('white')
    cell.set_facecolor('#696969')
    cell.set_text_props(color='white')
    if key[0] == 0 or key[1] == -1:
        cell.set_text_props(weight='bold', color='white')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
ax3.set_title('The proportion of articles produced by publication')

# Part3 Plotting the Gantt chart
fig4, ax4 = plt.subplots(figsize=(10, 8))
# Create a color map for clusters using a more vibrant colormap
colors = plt.cm.Paired(np.linspace(0, 1, n_components))

for i, (publication, row) in enumerate(publication_cluster_proportions.iterrows()):
    left = 0
    for cluster in range(n_components):
        width = row[cluster]
        ax4.barh(i, width, left=left, color=colors[cluster], label=f'Cluster {cluster}' if i == 0 else "")
        left += width

ax4.set_yticks(range(len(publication_counts)))
ax4.set_yticklabels(publication_counts.index)
ax4.set_xlabel('Proportion of Articles')
ax4.set_ylabel('Publication')
ax4.set_title('Proportion of Articles per Publication by Cluster')

# Create a legend
handles, labels = ax4.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax4.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')


# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, random_state=42)
tsne_representation = tsne.fit_transform(reduced_tfidf_matrix_100)

# 绘制散点图
plt.figure(figsize=(10, 8))
for cluster in range(n_components):
    plt.scatter(tsne_representation[cluster_labels == cluster, 0],
                tsne_representation[cluster_labels == cluster, 1],
                label=f'Cluster {cluster}', alpha=0.7)
plt.title('t-SNE Visualization of GMM')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.show()

