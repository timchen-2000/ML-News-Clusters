import pandas as pd
import nltk
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from tqdm import tqdm

# 确保下载了所需的nltk数据包
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# 读取数据集
file_path = r'E:\machine_learning_class_project\datasets\Sampled_Dataset.csv'
df = pd.read_csv(file_path)


# 定义预处理函数
def preprocess_text(text):
    # 将文本转换为小写
    text = text.lower()

    # 去除半角符号"."和数字
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\.', '', text)

    # 分词
    words = nltk.word_tokenize(text)

    # 词干提取
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words]

    # 词性标注
    pos_tagged_words = nltk.pos_tag(words)

    # 去除专有名词（词性为NNP或NNPS）
    filtered_words = [word for word, pos in pos_tagged_words if pos not in ['NNP', 'NNPS']]

    return ' '.join(filtered_words)


# 对DataFrame中的content列应用预处理并显示进度条
tqdm.pandas(desc="Processing")
df['content'] = df['content'].progress_apply(preprocess_text)

# 保存预处理后的数据集到新的CSV文件
processed_file_path = r'E:\machine_learning_class_project\datasets\process_Sampled_Dataset.csv'
df.to_csv(processed_file_path, index=False)

print(f"预处理后的数据集已保存到 {processed_file_path}")
