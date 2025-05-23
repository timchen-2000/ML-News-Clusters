from scipy.sparse import load_npz
import pandas as pd
from scipy.sparse import load_npz
from sklearn.feature_extraction.text import TfidfVectorizer

# 加载 TF-IDF 矩阵
tfidf_matrix_path = r'E:\machine_learning_class_project\datasets\TFIDF_Matrix.npz'
tfidf_matrix = load_npz(tfidf_matrix_path)

# 查看矩阵的维度
print("TF-IDF 矩阵的维度:", tfidf_matrix.shape)





import pandas as pd

# 加载词汇表count数据
vocab_count_path = r'E:\machine_learning_class_project\datasets\Filtered_Vocab_Table.csv'
vocab_count_df = pd.read_csv(vocab_count_path)

# 计算第40分位数
percentile_40 = vocab_count_df['nums'].quantile(0.39)

# 根据第40分位数筛选词汇
selected_vocab_count = vocab_count_df[vocab_count_df['nums'] == percentile_40]

# 打印第40分位数的词汇列表及其nums值
print("第40分位数的词汇及其nums值：")
print(selected_vocab_count[['word', 'nums']])



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

# 创建 TfidfVectorizer
vectorizer = TfidfVectorizer(vocabulary=selected_vocab)

# 矢量化处理
tfidf_matrix = vectorizer.fit_transform(preprocessed_data_df['content'])

# 查看词汇表
feature_names = vectorizer.get_feature_names_out()

# 提取和展示第一个文档的TF-IDF值及其对应的词汇
doc_id = 0  # 可以更改为想查看的文档ID
first_document_tfidf = tfidf_matrix[doc_id].todense().tolist()[0]

# 将词汇及其对应的TF-IDF值存储在DataFrame中
df_tfidf = pd.DataFrame({'word': feature_names, 'tfidf': first_document_tfidf})

# 仅显示有非零TF-IDF值的词汇
df_tfidf = df_tfidf[df_tfidf['tfidf'] > 0]

print(f"第{doc_id + 1}个文档的TF-IDF值及其对应的词汇:")
print(df_tfidf)

