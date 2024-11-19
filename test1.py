from gensim import corpora, models, similarities
from stop_words import get_stop_words

# 假设我们有一些文本数据
texts = [
    "人工智能是未来的发展方向".split(),
    "我们需要学习人工智能".split(),
    "未来的教育将会重要人工智能".split(),
]

documents = [['human', 'interface', 'computer'],
             ['survey', 'user', 'computer', 'system', 'response', 'time'],
             ['eps', 'user', 'interface', 'system'],
             ['system', 'human', 'system', 'eps'],
             ['user', 'response', 'time'],
             ['trees'],
             ['graph', 'trees'],
             ['user', 'graph', 'trees'],
             ['user', 'survey', 'system'],
             ['graph', 'minors', 'trees']]

# 获取英语的停用词列表
stoplist = get_stop_words('en')

# 创建一个新的Corpus并剔除停用词
new_corpus = []
for text in texts:
    # 过滤停用词
    no_stopwords = [word for word in text if word not in stoplist]
    new_corpus.append(no_stopwords)

# 使用新的无停用词Corpus训练一个TF-IDF模型
dictionary = corpora.Dictionary(new_corpus)
dict = corpora.Dictionary(documents)

corpus_tfidf = [dictionary.doc2bow(text) for text in new_corpus]
corpus = [dict.doc2bow(doc) for doc in documents]

index = similarities.MatrixSimilarity(corpus, num_best=1)

# 输出字典和TF-IDF Corpus
# print(dictionary.token2id)
# print(corpus_tfidf)

vector = dict.doc2bow(documents[0])  # 转换为bow形式
similar_documents = index[vector]  # 查询相似文档
print(similar_documents)