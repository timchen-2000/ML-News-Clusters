import pandas as pd

# 读取数据集
file_path = '/kaggle/working/Cleaned_Dataset.csv'
df = pd.read_csv(file_path)

# 计算要取出的行数
num_rows = df.shape[0]
num_rows_10_percent = int(num_rows * 0.1)

# 取出前10%的行数
sampled_df = df.head(num_rows_10_percent)

# 指定保存抽样后数据集的新文件路径
sampled_file_path = '/kaggle/working/Sampled1_Dataset.csv'

# 保存抽样后的数据集到新的CSV文件
sampled_df.to_csv(sampled_file_path, index=False)

print(f"抽样后的数据集已保存到 {sampled_file_path}")


# 生成sample——dataset.csv