import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# 加载降维后的TF-IDF矩阵
reduced_matrix_path = r'D:\machine_learning_class_project\datasets\Reduced_TFIDF_Matrix.csv'
reduced_tfidf_matrix = pd.read_csv(reduced_matrix_path)

# 提取前100维的数据进行聚类
reduced_tfidf_matrix_100 = reduced_tfidf_matrix.iloc[:, :100]
X = reduced_tfidf_matrix.iloc[:, :100].values  # 转换为 numpy 数组

# 设置初始的 min_samples
min_samples = 5

# 计算每个点的 k-距离
neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(X)
distances, indices = neighbors_fit.kneighbors(X)

# 对距离排序
distances = np.sort(distances, axis=0)
distances = distances[:, 1]
plt.plot(distances)
plt.xlabel('Data Points sorted by distance')
plt.ylabel('Epsilon (eps) distance')
plt.title('k-distance Graph')
plt.show()
