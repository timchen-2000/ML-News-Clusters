import pandas as pd
import hdbscan
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
# 加载降维后的TF-IDF矩阵
reduced_tfidf_matrix_path = r'D:\machine_learning_class_project\datasets\Reduced_TFIDF_Matrix.csv'
reduced_tfidf_matrix = pd.read_csv(reduced_tfidf_matrix_path)

# 提取前100维的数据进行聚类
reduced_tfidf_matrix_100 = reduced_tfidf_matrix.iloc[:, :100]

# 查看提取的维度
print(f"提取的数据维度: {reduced_tfidf_matrix_100.shape}")

# 使用HDBSCAN进行聚类
clusterer = hdbscan.HDBSCAN(min_cluster_size=5,min_samples=1, gen_min_span_tree=True)
cluster_labels = clusterer.fit_predict(reduced_tfidf_matrix_100)

# 将聚类结果添加到数据框中
reduced_tfidf_matrix_100['cluster'] = cluster_labels

# 加载文章基本信息
basic_info_path = r'D:\machine_learning_class_project\datasets\process_Sampled_Dataset.csv'
basic_info_df = pd.read_csv(basic_info_path)

# 合并聚类结果和文章基本信息
clustered_data_with_info = pd.concat([reduced_tfidf_matrix_100, basic_info_df[['publication', 'author', 'year', 'month']]], axis=1)

# 保存带有聚类结果和基本信息的数据
clustered_data_with_info_path = r'D:\machine_learning_class_project\datasets\Clustered_hdbscan.csv'
clustered_data_with_info.to_csv(clustered_data_with_info_path, index=False)
print(f"带有聚类结果和基本信息的数据已保存到 {clustered_data_with_info_path}")
print(f"合并后的数据的维度: {clustered_data_with_info.shape}")

# 可视化聚类结果
plt.figure(figsize=(10, 6))
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(set(cluster_labels)))]
for cluster, color in zip(sorted(set(cluster_labels)), colors):
    cluster_points = reduced_tfidf_matrix_100[cluster_labels == cluster]
    plt.scatter(cluster_points.iloc[:, 0], cluster_points.iloc[:, 1], color=color, label=f'Cluster {cluster}', alpha=0.5)
plt.title('HDBSCAN Clustering Result')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.show()

