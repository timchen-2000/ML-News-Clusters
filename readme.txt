
dataset：

Clustered_Reduced_TFIDF_Matrix     是下降维度到100 的矩阵 对应论文2.1 这里我用了SVD(奇异值分解) ，没有用PCA      最后面有publication author year mouth  
Filtered_Vocab_Table            是简化后的词汇表                    对应论文1.2
process_Sampled_Dataset       是做完预处理后的数据          对应论文1.1
Clustered_kmeans_100_.csv       kmeans处理后的文件，最后面有聚类的类别(cluster)   publication author year mouth  

脚本说明：
dimension_reduction :词汇降维
Tfldf: 构建词频矩阵的
kmeans：我试着做了下原型聚类



要做聚类的话记得指定
# 提取前100维的数据进行聚类
reduced_tfidf_matrix_100 = reduced_tfidf_matrix.iloc[:, :100]

因为
Clustered_Reduced_TFIDF_Matrix 里还有文字

可视化统计
如何对上述代码中的聚类结果进行统计，要求：
①要进行统计有多少不同的publication，以及publication所拥有的总的文章数，以及所占比例；
②根据每个publication找到每一类（cluster）中的文章数目；
③对聚类结果进行采用甘特图进行可视化。

