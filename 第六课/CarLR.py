# coding:utf-8

# 使用逻辑回归预测对当前的车满不满意

import pandas as pd
import numpy as np
'''
	LogisticRegression、LogisticRegressionCV都有L1、L2正则化参数，
	但LogisticRegressionCV使用了交叉验证来选择正则化系数，而LogisticRegression
	需要自己每次指定一个正则化系数
'''
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import label_binarize
# rcParams用于指定图片的像素等信息
from matplotlib import rcParams
from matplotlib import pyplot as plt


if __name__ == "__main__":
	# set_option可以修改pandas的默认参数
	# display.width表示横向最多展示的字符数量
	# display.max_columns代表显示的最大列数，如果超额就显示省略号，
	# 当DataFrame存在多个列，并且一行内展示不完时，可以使用省略号
	pd.set_option("display.width", 300)
	pd.set_option("display.max_columns", 300)

	path = r'E:\python\PythonSpace\Git\ML\Data\6\car.data'
	# 列名
	columns = ["buy", "maintain", "doors", "persons", "boot", "safety", "accept"]
	data = pd.read_csv(path, header=None, names=columns)
	# print(data.head())

	# 使用one-hot编码，将6个特征映射成向量
	x = pd.DataFrame()
	for col in columns:
		# 将类别变量变化成指示变量
		t = pd.get_dummies(data[col])
		t = t.rename(columns=lambda x: col + "_" + str(x))
		x = pd.concat((x, t), axis=1)
	# print(x.shape)
	# print(x.head())

	# Categorical可以得到不同label的所有情况，
	# codes表示当前label对应的序列号
	y = np.array(pd.Categorical(data["accept"]).codes)

	# 训练和测试样本比例划分
	x, x_test, y, y_test = train_test_split(x, y, test_size=0.3)

	# logspace()，创建等比数列，开始点和结束点都是10的幂次，
	# 比如logspace(-1，2，4)，开始点就是10^(-1)，结束点就是10^2

	# Cs参数是用来表示你更相信数据还是更相信模型，它可以设置多个值，
	# 然后用交叉验证的方式选出最好的一个值
	clf = LogisticRegressionCV(Cs=np.logspace(-3, 4, 8), cv=5)
	clf.fit(x, y)
	# clf.coef_输出系数，它是一个矩阵，行数等于类别数量，列数等于特征数量
	# print(len(clf.coef_),len(clf.coef_[0]))
	# print(clf.coef_)

	y_hat = clf.predict(x)
	print("训练集精确度：", metrics.accuracy_score(y, y_hat))
	y_test_hat = clf.predict(x_test)
	print("测试集精确度：", metrics.accuracy_score(y_test, y_test_hat))

	# AUC、ROC
	n_class = len(np.unique(y))
	if n_class > 2:
		y_test_one_hot = label_binarize(y_test, classes=np.arange(n_class))
		# predict_proba表示样本在每个类别上的预测值
		y_test_one_hot_hat = clf.predict_proba(x_test)
		# ravel()将numpy中的多维数组将成一维的
		fpr, tpr, _ = metrics.roc_curve(y_test_one_hot.ravel(), y_test_one_hot_hat.ravel())
		# metrics.auc：计算AUC值
		# print("Micro AUC :", metrics.auc(fpr, tpr))
		auc = metrics.roc_auc_score(y_test_one_hot, y_test_one_hot_hat)
		print(auc)
	else:
		auc = metrics.roc_auc_score(y_test, y_test_hat)
		print(auc)

	# 这两个可以使得图片显示中文
	rcParams["font.sans-serif"] = "SimHei"
	rcParams["axes.unicode_minus"] = False

	plt.figure(figsize=(8, 7), dpi=80, facecolor='w')
	plt.plot(fpr, tpr, 'r-', lw=2, label="AUC=%.4f" % auc)
	plt.legend(loc='lower right')
	plt.xlim((-0.01, 1.02))
	plt.ylim((-0.01, 1.02))
	plt.xticks(np.arange(0, 1.1, 0.1))
	plt.yticks(np.arange(0, 1.1, 0.1))
	plt.xlabel('False Positive Rate', fontsize=14)
	plt.ylabel('True Positive Rate', fontsize=14)
	plt.grid(b=True, ls=':')
	plt.title('ROC曲线和AUC', fontsize=18)
	plt.show()