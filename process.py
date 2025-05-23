import pandas as pd

# 读取数据集
file_path = r'E:\machine_learning_class_project\datasets\Dataset.csv'  # 替换为你的实际文件路径
df = pd.read_csv(file_path)

# 删除 content 为空值的行
df_cleaned = df.dropna(subset=['content'])

# 指定保存清理后数据集的新文件路径
cleaned_file_path = r'E:\machine_learning_class_project\datasets\Cleaned_Dataset.csv'  # 替换为你希望保存的路径

# 保存清理后的数据集到新的CSV文件
df_cleaned.to_csv(cleaned_file_path, index=False)

print(f"清理后的数据集已保存到 {cleaned_file_path}")

