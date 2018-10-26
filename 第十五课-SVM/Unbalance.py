# coding:utf-8

import numpy as np
from sklearn import svm
from sklearn.metrics import recall_score
from matplotlib import colors, rcParams
from matplotlib import pyplot as plt
from sklearn.exceptions import UndefinedMetricWarning
import warnings

if __name__ == "__main__":
	warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)

	np.random.seed(0)  # 保证每次生成的随机数都相同
	c1 = 990
	c2 = 10
	n = c1 + c2
	# 产生990个二维的服从正态分布的数据
	x_c1 = 3 * np.random.randn(c1, 2)
	x_c2 = 0.5 * np.random.randn(c2, 2) + (4, 4)
	x = np.vstack((x_c1, x_c2))
	y = np.ones(n)
	# 前990个样本都是负样本
	y[:c1] = -1

	# 分类器，对不平衡的样本附上不同的权重
	# class_weight={-1: 1, 1: 2}表示负样本的权重是1，正样本的权重是2
	clfs = [svm.SVC(C=1, kernel="linear", class_weight={-1: 1, 1: 2}),
			svm.SVC(C=1, kernel="linear", class_weight={-1: 1, 1: 30}),
			svm.SVC(C=1, kernel="rbf", gamma=0.5, class_weight={-1: 1, 1: 2}),
			svm.SVC(C=1, kernel="rbf", gamma=0.5, class_weight={-1: 1, 1: 30})]
	titles = ("Linear Weight = %d" % 2), ("Linear Weight = %d" % 30),\
			 ("RBF Weight = %d" % 2), ("RBF Weight = %d" % 30)

	rcParams["font.sans-serif"] = "SimHei"
	rcParams["axes.unicode_minus"] = False
	x1_min, x2_min = np.min(x, axis=0)
	x1_max, x2_max = np.max(x, axis=0)
	t1 = np.linspace(x1_min+0.1, x1_max+0.1, 500)
	t2 = np.linspace(x2_min+0.1, x2_max+0.1, 500)
	x1, x2 = np.meshgrid(t1, t2)
	grid_test = np.stack((x1.flat, x2.flat), axis=1)

	cm_light = colors.ListedColormap(['#77E0A0', '#FF8080'])
	cm_dark = colors.ListedColormap(["g", "r"])

	plt.figure(figsize=(8, 6))
	for i, clf in enumerate(clfs):
		clf.fit(x, y)
		y_hat = clf.predict(x)
		# recall_score中的pos_label参数用于指定哪个类别是正类
		print("模型%d的召回率为：%.3f" % (i, recall_score(y, y_hat)))

		# 画图
		plt.subplot(2, 2, i+1)
		grid_hat = clf.predict(grid_test)
		grid_hat = grid_hat.reshape(x1.shape)
		plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
		point_size = np.ones(n) * 30
		point_size[:c1] = 10
		plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors="k", s=point_size, cmap=cm_dark)
		plt.xlim(x1_min+0.1, x1_max+0.1)
		plt.ylim(x2_min+0.1, x2_max+0.1)
		plt.title(titles[i])
		plt.grid(b=True, ls=":")
	plt.suptitle("不平衡样本的处理", fontsize=15)
	plt.tight_layout()
	plt.subplots_adjust(top=0.92)
	plt.show()
