# coding:utf-8

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from matplotlib import rcParams
from matplotlib import pyplot as plt

def test_clf(clf):
	print("分类器：", clf)
	alpha_can = np.logspace(-3, 2, 10)
	model = GridSearchCV(clf, param_grid={"alpha": alpha_can}, cv=5)
	m = alpha_can.size
	# 判断当前模型中是否有“alpha”这个属性
	if hasattr(clf, "alpha"):
		model.set_params(param_grid={"alpha": alpha_can})
		m = alpha_can.size
	if hasattr(clf, "n_neighbors"):
		neigbors_can = np.arange(1, 15)
		model.set_params(param_grid={"n_neighbors": neigbors_can})
		m = neigbors_can.size
	if hasattr(clf, "C"):
		C_can = np.logspace(1, 3, 3)
		gamma_can = np.logspace(-3, 0, 3)
		model.set_params(param_grid={"C": C_can, "gamma": gamma_can})
		m = C_can.size * gamma_can.size
	if hasattr(clf, "max_depth"):
		max_depth_can = np.arange(4, 10)
		model.set_params(param_grid={"max_depth": max_depth_can})
		m = max_depth_can.size
	train_start = time()
	model.fit(x_train, y_train)
	train_end = time()
	train_time = (train_end - train_start) / (5 * m)
	print("5折交叉验证的训练时间为：%.3f秒/(5*%d)=%.3f秒" % ((train_end - train_start), m, train_time))
	print("最优超参数为：", model.best_params_)
	test_start = time()
	y_pred = model.predict(x_test)
	test_end = time()
	test_time = test_end - test_start
	print("测试时间：%.3f秒" % test_time)
	accuracy = accuracy_score(y_test, y_pred)
	print("测试集准确率为：%.2f%%" % (100 * accuracy))
	print("-----------------")
	name = str(clf).split("(")[0]
	index = name.find("Classifier")
	if index != -1:
		name = name[:index]
	if name == "SVC":
		name = "SVM"
	return train_time, test_time, 1-accuracy, name

if __name__ == "__main__":
	data_home = r"E:\python\PythonSpace\Git\ML\第二十二课-主题模型\data\news"
	print("开始下载/加载数据...")
	start = time()
	categories = ["alt.atheism", "talk.religion.misc", "comp.graphics", "sci.space"]
	# remove：是一个元组，用来去除一些停用词的，例如标题引用之类的。
	# ‘headers’, ‘footers’, ‘quotes’，分别表示页眉、页脚和引用
	remove = ("headers", "footers", "quotes")
	data_train = fetch_20newsgroups(data_home=data_home, subset="train", categories=categories,
									shuffle=True, random_state=0, remove=remove)
	data_test = fetch_20newsgroups(data_home=data_home, subset="test", categories=categories,
								   shuffle=True, random_state=0, remove=remove)
	end = time()
	print("总用时：%.3f秒" % (end - start))
	print("数据类型：", type(data_train))
	print("训练集包含的文本数目：%d" % len(data_train.data))
	print("测试集包含的文本数目：%d" % len(data_test.data))
	categories = data_train.target_names
	print(categories)
	# pprint(categories)

	y_train = data_train.target
	y_test = data_test.target
	# print("----前10个文本----")
	# for i in range(10):
	# 	print("文本%d（属于类别：%s）" % (i+1, categories[y_train[i]]))
	# 	print(data_train.data[i])
	# 	print("\n\n")
	# max_df：有些词在文档中出现的频率很高，但其实是没有啥用的，
	# 可以设置一个阈值，当频率高于这个阈值时就去除这些词
	vectorizer = TfidfVectorizer(input="content", stop_words="english",
								 max_df=0.5, sublinear_tf=True)
	x_train = vectorizer.fit_transform(data_train.data)
	x_test = vectorizer.transform(data_test.data)
	print("训练集样本数：%d，特征个数：%d" % x_train.shape)
	print(x_train)
	print("--------------")
	print(x_test)
	print(type(x_train), type(x_test))
	# print("停用词：")
	# pprint(vectorizer.get_stop_words())

	# get_feature_names()：获得所有词汇
	# feature_names = np.asarray(vectorizer.get_feature_names())

	print("------分类器比较------")
	clfs = [MultinomialNB(), BernoulliNB(), KNeighborsClassifier(),
			RidgeClassifier(), RandomForestClassifier(n_estimators=200), SVC()]
	result = []
	for clf in clfs:
		a = test_clf(clf)
		result.append(a)
	result = np.array(result)
	train_time, test_time, err, names = result.T
	train_time = train_time.astype(np.float)
	test_time = test_time.astype(np.float)
	err = err.astype(np.float)
	x = np.arange(len(train_time))
	rcParams["font.sans-serif"] = "SimHei"
	rcParams["axes.unicode_minus"] = False
	plt.figure(figsize=(8, 6))
	ax = plt.axes()
	b1 = ax.bar(x, err, width=0.25, color="#77E0A0", edgecolor="k")
	ax_t = ax.twinx()
	b2 = ax_t.bar(x+0.25, train_time, width=0.25, color="#FFA0A0", edgecolor="k")
	b3 = ax_t.bar(x+0.5, test_time, width=0.25, color="#FF8080", edgecolor="k")
	plt.xticks(x+0.5, names)
	plt.legend([b1[0], b2[0], b3[0]], ("错误率", "训练时间", "测试时间"))
	plt.title("新闻文本数据不同分类器间的比较", fontsize=15)
	plt.xlabel("分类器名称")
	plt.grid(True)
	plt.tight_layout(2)
	plt.show()

