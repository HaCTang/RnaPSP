from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 示例文本
text1 = "This is the first document."
text2 = "This document is the second document."
text3 = "And this is the third one."
text4 = "Is this the first document?"

# 创建TfidfVectorizer对象，用于将文本转换为TF-IDF向量
vectorizer = TfidfVectorizer()

# 将文本向量化
tfidf_matrix = vectorizer.fit_transform([text1, text2, text3, text4])
print(tfidf_matrix)

# 计算余弦相似性
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 打印相似性矩阵
print("相似性矩阵：")
print(cosine_sim)

# 打印文本之间的相似性
print("\n文本之间的相似性：")
print("文本1与文本2的相似性:", cosine_sim[0][1])
print("文本1与文本3的相似性:", cosine_sim[0][2])
print("文本1与文本4的相似性:", cosine_sim[0][3])
