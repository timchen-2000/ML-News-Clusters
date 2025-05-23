import pandas as pd
from nltk.corpus import words
from tqdm import tqdm

# 加载原始词汇表
original_vocab_path = r'E:\machine_learning_class_project\datasets\Vocab1_Table.csv'
original_vocab_df = pd.read_csv(original_vocab_path)

# 获取nltk完整英语语料库中的单词
english_vocab = set(words.words())

# 筛选词汇表，保留在英语语料库中的单词
filtered_vocab_list = []
filtered_vocab_nums = []
for index, row in tqdm(original_vocab_df.iterrows(), total=len(original_vocab_df)):
    word = row['word']
    nums = row['nums']
    if word in english_vocab:
        filtered_vocab_list.append(word)
        filtered_vocab_nums.append(nums)

# 创建简化后的词汇表 DataFrame
filtered_vocab_df = pd.DataFrame({'word': filtered_vocab_list, 'nums': filtered_vocab_nums})

# 输出简化后的词汇表
print(filtered_vocab_df)

# 保存简化后的词汇表为CSV文件
filtered_output_file_path = r'E:\machine_learning_class_project\datasets\Filtered_Vocab_Table.csv'
filtered_vocab_df.to_csv(filtered_output_file_path, index=False)
print(f"简化后的词汇表已保存到 {filtered_output_file_path}")
