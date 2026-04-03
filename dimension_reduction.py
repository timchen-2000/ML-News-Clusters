import pandas as pd
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import load_npz
from tqdm import tqdm

# 加载原始数据集以获取文章的基本信息
original_data_path = r'E:\machine_learning_class_project\datasets\process_Sampled_Dataset.csv'
original_df = pd.read_csv(original_data_path)

# 加载 TF-IDF 矩阵
tfidf_matrix_path = r'E:\machine_learning_class_project\datasets\TFIDF_Matrix.npz'
tfidf_matrix = load_npz(tfidf_matrix_path)

# 查看TF-IDF矩阵的维度
print(f"TF-IDF 矩阵的维度: {tfidf_matrix.shape}")

# 对每篇文章的单词进行降维
n_components = 100  # 选择降维后的维度
svd = TruncatedSVD(n_components=n_components)

# 使用进度条显示进度
print("正在对每篇文章的单词进行降维...")
tfidf_matrix_reduced = svd.fit_transform(tfidf_matrix)

# 将降维后的数据转换为DataFrame
reduced_df = pd.DataFrame(tfidf_matrix_reduced, columns=[f'component_{i+1}' for i in range(n_components)])

# 确保原始数据和降维后的数据行数匹配
assert len(original_df) == len(reduced_df), "原始数据和降维数据行数不匹配"

# 合并文章基本信息和降维后的数据
combined_df = pd.concat([reduced_df, original_df[['publication', 'author', 'year', 'month']]], axis=1)

# 保存合并后的数据
combined_data_path = r'E:\machine_learning_class_project\datasets\Reduced_TFIDF_Matrix.csv'
combined_df.to_csv(combined_data_path, index=False)
print(f"合并后的数据已保存到 {combined_data_path}")
print(f"合并后的数据的维度: {combined_df.shape}")
