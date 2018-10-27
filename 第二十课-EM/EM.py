# coding:utf-8

import numpy as np
from sklearn.mixture import GaussianMixture
from matplotlib import rcParams
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances_argmin
from scipy.stats import multivariate_normal
# 画三维图时，必须导入这个包
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
	style = "sklearn"

	np.random.seed(0)
	# 定义均值和协方差阵，用于构造多元高斯分布
	# 三维空间中的多元高斯分布
	mu1_fact = (0, 0, 0)
	# 对角阵
	cov1_fact = np.diag((1, 2, 3))
	data1 = np.random.multivariate_normal(mu1_fact, cov1_fact, 400)
	mu2_fact = (2, 2, 1)
	cov2_fact = np.array([[1, 1, 3], [1, 2, 1], [3, 1, 4]])
	data2 = np.random.multivariate_normal(mu2_fact, cov2_fact, 400)
	data = np.vstack((data1, data2))
	y = np.array([True] * 400 + [False] * 400)

	if style == "sklearn":
		# n_components表示混合高斯分布的个数，
		# covariance_type表示每个高斯分布的协方差类型，
		# tol表示收敛的阈值，
		# max_iter表示最大迭代次数
		g = GaussianMixture(n_components=2, covariance_type="full", tol=1e-6)
		g.fit(data)
		print("每个高斯分布的权重：%.3f" % g.weights_[0])
		print("每个高斯分布的均值：", g.means_)
		print("每个高斯分布的协方差矩阵：", g.covariances_)
		mu1, mu2 = g.means_
		sigma1, sigma2 = g.covariances_
	else:
		# 自定义的EM算法
		num_iter = 200
		n, d = data.shape
		# 随机初始化均值
		# mu1 = np.random.standard_normal(d)
		# mu2 = np.random.standard_normal(d)
		mu1 = data.min(axis=0)
		mu2 = data.max(axis=0)
		# 生成一个d*d维的单位阵
		sigma1 = np.identity(d)
		sigma2 = np.identity(d)
		pi = 0.5   # 表示数据来自第一个高斯分布的概率
		for i in range(num_iter):
			# E步
			norm1 = multivariate_normal(mu1, sigma1)
			norm2 = multivariate_normal(mu2, sigma2)
			# norm1相当于是一个高斯分布，pdf表示data中的每一个数是
			# 产生于当前这个高斯分布的概率
			tau1 = pi * norm1.pdf(data)
			tau2 = (1-pi) * norm2.pdf(data)
			# gamma相当于做了归一化，表示data中的每个数据来自第一个
			# 高斯分布的概率
			gamma = tau1 / (tau1 + tau2)

			# M步
			mu1 = np.dot(gamma, data)/np.sum(gamma)
			mu2 = np.dot((1-gamma), data)/np.sum((1-gamma))
			sigma1 = np.dot(gamma*(data-mu1).T, data-mu1)/np.sum(gamma)
			sigma2 = np.dot((1-gamma)*(data-mu2).T, data-mu2)/np.sum((1-gamma))
			pi = np.sum(gamma)/n
			print(i+1, "：", mu1, mu2)
		print("第一个高斯分布的权重：%.3f" % pi)
		print("每个高斯分布的均值：", mu1, mu2)
		print("第一个高斯分布的方差：")
		print(sigma1)
		print("第二个高斯分布的方差：")
		print(sigma2)

	# 预测分类
	norm1 = multivariate_normal(mu1, sigma1)
	norm2 = multivariate_normal(mu2, sigma2)
	tau1 = norm1.pdf(data)
	tau2 = norm2.pdf(data)

	rcParams["font.sans-serif"] = "SimHei"
	rcParams["axes.unicode_minus"] = False

	fig = plt.figure(figsize=(10, 5))
	ax = fig.add_subplot(121, projection="3d")
	ax.scatter(data[:, 0], data[:, 1], data[:, 2], c="b", s=30,
			   marker="o", edgecolors="k", depthshade=True)
	ax.set_xlabel("X")
	ax.set_ylabel("Y")
	ax.set_zlabel("Z")
	ax.set_title("原始数据", fontsize=15)

	ax = fig.add_subplot(122, projection="3d")
	# 计算mu1向量与mu1_fact和mu2_fact两个向量的欧式距离，
	# 返回距离最小的类标签，比如mu1_fact和mu2_fact分别
	# 表示0、1两种类别，如果与前者的距离最小，则返回0；同理计算mu2
	# 与mu1_fact和mu2_fact两个向量的欧式距离；
	# euclidean表示欧式距离
	order = pairwise_distances_argmin([mu1_fact, mu2_fact], [mu1, mu2],
									  metric="euclidean")
	if order[0] == 0:
		c1 = tau1 > tau2
	else:
		c1 = tau1 < tau2
	c2 = ~c1
	acc = np.mean(y == c1)
	print("准确率：%.2f%%" % (100*acc))
	ax.scatter(data[c1, 0], data[c1, 1], data[c1, 2], c="r", s=30,
			   marker="o", edgecolors="k", depthshade=True)
	ax.scatter(data[c2, 0], data[c2, 1], data[c2, 2], c="b", s=30,
			   marker="^", edgecolors="k", depthshade=True)
	ax.set_xlabel("X")
	ax.set_ylabel("Y")
	ax.set_zlabel("Z")
	ax.set_title("EM算法分类", fontsize=15)
	plt.suptitle("EM算法的实现", fontsize=18)
	plt.subplots_adjust(top=0.9)
	plt.tight_layout()
	plt.show()