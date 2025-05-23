import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

# 加载预处理后的文本数据
preprocessed_data_path = r'E:\machine_learning_class_project\datasets\process_Sampled_Dataset.csv'
preprocessed_data_df = pd.read_csv(preprocessed_data_path)

# 加载词汇表count数据
vocab_count_path = r'E:\machine_learning_class_project\datasets\Filtered_Vocab_Table.csv'
vocab_count_df = pd.read_csv(vocab_count_path)

# 计算词汇表count中第40分位数和第99.5分位数之间的词汇数量
percentile_40 = vocab_count_df['nums'].quantile(0.4)
percentile_995 = vocab_count_df['nums'].quantile(0.995)
selected_vocab_count = vocab_count_df[(vocab_count_df['nums'] >= percentile_40) & (vocab_count_df['nums'] <= percentile_995)]

# 选择第40分位数和第99.5分位数之间的词汇
selected_vocab = selected_vocab_count['word'].tolist()

# 筛选出选定的词汇对应的文本数据
filtered_data_df = preprocessed_data_df[preprocessed_data_df['content'].apply(lambda x: any(word in x for word in selected_vocab))]

# 创建 TfidfVectorizer
vectorizer = TfidfVectorizer(vocabulary=selected_vocab)
tfidf_matrix = vectorizer.fit_transform(filtered_data_df['content'])

# 保存 TF-IDF 矩阵为稀疏矩阵
tfidf_matrix_path = r'E:\machine_learning_class_project\datasets\TFIDF_Matrix.npz'
save_npz(tfidf_matrix_path, tfidf_matrix)

print(f"TF-IDF 矩阵已保存到 {tfidf_matrix_path}")


