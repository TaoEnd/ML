# coding:utf-8

import numpy as np
from matplotlib import colors, rcParams
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets as ds
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, silhouette_score

if __name__ == "__main__":
	n = 400
	# 生成4类数据
	centers = 4
	# cluster_std表示每个类簇数据的方差，默认为1，
	# n_features表示生成几维的数据，
	# centers表示数据类簇数量,
	# shuffle默认是True，表示打乱所有数据
	data1, y1 = ds.make_blobs(n, n_features=2, centers=4, random_state=2)
	data2, y2 = ds.make_blobs(n, n_features=2, centers=4,
							  cluster_std=(1, 2, 1.5, 0.5), random_state=2)
	# 将第一类数据的全部，第二类数据的前50个，第三类数据的前60个，
	# 第四类数据的前40个拼成最终训练的数据
	data3 = np.vstack((data1[y1==0][:], data1[y1==1][:50],
					   data2[y2==2][:60], data1[y1==2][:40]))
	y3 = np.array([0]*100 + [1]*50 + [2]*60 + [2]*40)
	# 让原始的数据乘上一个矩阵，即对原数据做一下拉伸或转向等操作
	m = np.array([[1, 1], [1, 3]])
	data_r = data1.dot(m)

	rcParams["font.sans-serif"] = "SimHei"
	rcParams["axes.unicode_minus"] = False
	cm = colors.ListedColormap(list("rgbm"))
	data_list = data1, data1, data_r, data_r, data2, data2, data3, data3
	y_list = y1, y1, y1, y1, y2, y2, y3, y3
	titles = "原始数据", "Kmeans++聚类", "旋转后的数据", "旋转后的Kmeans++",\
			 "方差不相等数据", "方差不相等的Kmeans++聚类", "数量不相等数据",\
			 "数量不相等数据的Kmeans++聚类"

	# n_init表示运行10轮，选择最好的一轮结果作为Kmeans++的最终结果，
	# max_iter表示每一轮运行次数
	model = KMeans(n_clusters=4, init="k-means++", n_init=10, max_iter=300)
	# start表示从1开始编号
	plt.figure(figsize=(7, 8))
	for i, (x, y, title) in enumerate(zip(data_list, y_list, titles), start=1):
		plt.subplot(4, 2, i)
		plt.title(title)
		if i % 2 == 1:
			y_pred = y
		else:
			y_pred = model.fit_predict(x)
		print(i)
		print("一致性（Homogeneity）：%.4f" % homogeneity_score(y, y_pred))
		print("完整性（Completeness）：%.4f" % completeness_score(y, y_pred))
		print("V Measure：%.4f" % v_measure_score(y, y_pred))
		print("AMI：%.4f" % adjusted_mutual_info_score(y, y_pred))
		print("ARI：%.4f" % adjusted_rand_score(y, y_pred))
		print("轮廓系数（Silhouette）：%.4f" % silhouette_score(x, y_pred))
		print("--------------------")

		plt.scatter(x[:, 0], x[:, 1], c=y_pred, s=20, cmap=cm)
		x1_min, x2_min = np.min(x, axis=0)
		x1_max, x2_max = np.max(x, axis=0)
		plt.xlim(x1_min-0.5, x1_max+0.5)
		plt.ylim(x2_min-0.5, x2_max+0.5)
		plt.grid(b=True, ls=":")
	# rect=(0, 0, 1, 0.97)将子图整体看成画布中的一个矩形，
	# (0, 0),(1, 0.97)分别表示矩形的左下角和右上角，0.97表示占比
	plt.tight_layout(2, rect=(0, 0, 1, 0.97))
	plt.suptitle("Kmeans聚类", fontsize=15)
	plt.show()


