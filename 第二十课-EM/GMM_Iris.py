# coding:utf-8

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
from matplotlib import rcParams, colors

if __name__ == "__main__":
	iris_feature = "花萼长度", "花萼宽度", "花瓣长度", "花瓣宽度"
	path = r"E:\python\PythonSpace\Git\ML\第九课-回归\data\iris.data"
	data = pd.read_csv(path, header=None)
	x_prime = data[np.arange(4)]
	y = pd.Categorical(data[4]).codes
	n_components = 3
	# 两两特征组合，为了画图方便
	feature_pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
	rcParams["font.sans-serif"] = "SimHei"
	rcParams["axes.unicode_minus"] = False
	plt.figure(figsize=(8, 6))
	for k, pair in enumerate(feature_pairs, start=1):
		x = x_prime[pair]
		# 求每种鸢尾花的两种特征值的平均值
		m = np.array([np.mean(x[y == i], axis=0) for i in range(3)])
		# print("实际均值：")
		# print(m)

		model = GaussianMixture(n_components=n_components, covariance_type="full",
								tol=1e-5, max_iter=300, random_state=0)
		model.fit(x)
		# print("预测均值：")
		# print(model.means_)
		# print("-------------")

		cm_light = colors.ListedColormap(["#FF8080", "#77E0A0", "#A0A0FF"])
		cm_dark = colors.ListedColormap(["r", "g", "#606060"])
		x1_min, x2_min = np.min(x, axis=0)
		x1_max, x2_max = np.max(x, axis=0)
		t1 = np.linspace(x1_min-0.5, x1_max+0.5, 200)
		t2 = np.linspace(x2_min-0.5, x2_max+0.5, 200)
		x1, x2 = np.meshgrid(t1, t2)
		grid_test = np.stack((x1.flat, x2.flat), axis=1)
		grid_test_hat = model.predict(grid_test)
		grid_test_hat = grid_test_hat.reshape(x1.shape)

		plt.subplot(2, 3, k)
		plt.pcolormesh(x1, x2, grid_test_hat, cmap=cm_light)
		plt.scatter(x[pair[0]], x[pair[1]], s=20, c=y, marker="o", cmap=cm_dark,
					edgecolors="k")
		acc_str = "准确率：%.2f%%" % (100*np.mean(model.predict(x) == y))
		xx = 0.98 * x1_min + 0.02 * x1_max
		yy = 0.05 * x2_min + 0.95 * x2_max
		plt.text(xx, yy, acc_str, fontsize=10)
		plt.xlim(x1_min-0.5, x1_max+0.5)
		plt.ylim(x2_min-0.5, x2_max+0.5)
		plt.xlabel(iris_feature[pair[0]], fontsize=10)
		plt.ylabel(iris_feature[pair[1]], fontsize=10)
		plt.grid(b=True, ls=":", color="#606060")
	plt.suptitle("EM算法与鸢尾花分类", fontsize=13)
	plt.tight_layout(1, rect=(0, 0, 1, 0.95))
	plt.show()
