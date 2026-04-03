import pandas as pd
from collections import Counter
from nltk import word_tokenize
import nltk
import re
from nltk.stem import PorterStemmer
from tqdm import tqdm

# 确保下载了所需的nltk数据包
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# 读取数据集
file_path = r'E:\machine_learning_class_project\datasets\Sampled_Dataset.csv'
df = pd.read_csv(file_path)

# 定义预处理函数
def preprocess_text(text):
    # 将文本转换为小写
    text = text.lower()

    # 去除数字和标点符号
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)  # 保留字母、下划线和空格

    # 分词
    words = nltk.word_tokenize(text)

    # 词干提取
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words]

    # 词性标注
    pos_tagged_words = nltk.pos_tag(words)

    # 去除专有名词（词性为NNP或NNPS）
    filtered_words = [word for word, pos in pos_tagged_words if pos not in ['NNP', 'NNPS']]

    # 将单词列表连接成字符串
    return ' '.join(filtered_words)

# 对DataFrame中的content列应用预处理
tqdm.pandas()  # 添加进度条功能
df['content'] = df['content'].progress_apply(preprocess_text)

# 合并所有文本
all_text = ' '.join(df['content'])

# 分词
tokens = word_tokenize(all_text)

# 统计词频
word_freq = Counter(tokens)

# 创建词汇表（保留高频词）
vocab_size = len(word_freq)  # 使用唯一单词的数量作为词汇表大小
common_words = word_freq.most_common(vocab_size)
vocab = {word: freq for word, freq in common_words}

# 转换为 DataFrame
vocab_df = pd.DataFrame(list(vocab.items()), columns=['word', 'nums'])

# 输出 DataFrame
print(vocab_df)

# 保存为 CSV 文件
output_file_path = r'E:\machine_learning_class_project\datasets\Vocab1_Table.csv'
vocab_df.to_csv(output_file_path, index=False)
print(f"词汇表已保存到 {output_file_path}")
