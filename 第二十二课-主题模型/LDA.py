# coding:utf-8

from gensim import corpora, models, similarities

if __name__ == "__main__":
	path = r"E:\python\PythonSpace\Git\ML\第二十二课-主题模型\data\LDA_test.txt"
	with open(path, "r") as fr:
		lines = fr.readlines()
	stop_list = set("for a of the and to in".split())
	texts = [[word for word in line.strip().lower().split() if word not in stop_list] for line in lines]
	print(lines)
	print(texts)

	# 构造字典，字典中每个单词是唯一的并且是乱序的，编号从1开始
	dictionary = corpora.Dictionary(texts)
	# 构造词袋，doc2bow将每一个文档中的每个词转化成（a,b）数组对，
	# 其中a表示当前文档中出现了字典中编号为a的词，并且这个词在当前
	# 文档中出现的次数为b
	# corpus是一个迭代器
	corpus = [dictionary.doc2bow(text) for text in texts]
	# 计算每个单词的TF-IDF值
	tfidf = models.TfidfModel(corpus)
	corpus_tfidf = tfidf[corpus]
	# corpus_tfidf = corpus
	# for c in corpus_tfidf:
	# 	print(c)

	print("LSI Model：")
	lsi = models.LsiModel(corpus, num_topics=2, id2word=dictionary)
	topic_result = [a for a in lsi[corpus]]
	print(topic_result)
	print("LSI Topics：")
	# 有两个主题，打印最相关的前5个词
	print(lsi.print_topics(num_topics=2, num_words=5))
	similarity = similarities.MatrixSimilarity(lsi[corpus])
	print("Similarity：")
	print(list(similarity))

	print("LDA Model：")
	# alpha表示主题密度，值越大则文档中包含的主题数量越多
	# passes指遍历语料库的次数，遍历的次数越多，则模型越精确
	lda = models.LdaModel(corpus, num_topics=2, id2word=dictionary, alpha="auto",
						  eta="auto", minimum_probability=0.001, passes=10)
	doc_topic = [doc_t for doc_t in lda[corpus]]
	print("Document Topic：")
	print(doc_topic)
	for doc_topic in lda.get_document_topics(corpus):
		print(doc_topic)
	for topic_id in range(2):
		print("Topic", topic_id)
		print(lda.show_topic(topic_id))
	similarity = similarities.MatrixSimilarity(lda[corpus])
	print("Similarity：")
	print(list(similarity))