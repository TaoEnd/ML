# coding:utf-8

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import euclidean_distances
from matplotlib import pyplot as plt
from matplotlib import rcParams

if __name__ == "__main__":
	x = np.arange(0, 2*np.pi, 0.1)
	data1 = np.vstack((np.sin(x), np.cos(x))).T
	data2 = np.vstack((2*np.sin(x), 2*np.cos(x))).T
	data3 = np.vstack((3*np.sin(x), 3*np.cos(x))).T
	data = np.vstack((data1, data2, data3))

	n_clusters = 3
	# 计算向量之间的距离
	m = euclidean_distances(data, squared=True)

	rcParams["font.sans-serif"] = "SimHei"
	rcParams["axes.unicode_minus"] = False

	plt.figure(figsize=(10, 7))
	plt.suptitle("谱聚类", fontsize=15)
	clrs = plt.cm.Spectral(np.linspace(0, 0.8, n_clusters))
	for i, s in enumerate(np.logspace(-2, 0, 6)):
		print(s)
		af = np.exp(-m**2/(s**2)) + 1e-6
		# affinity表示使用哪种方式计算相似矩阵
		model = SpectralClustering(n_clusters=n_clusters, affinity="precomputed", assign_labels="kmeans")
		y_hat = model.fit_predict(af)
		plt.subplot(2, 3, i+1)
		for k, clr in enumerate(clrs):
			cur = (y_hat == k)
			plt.scatter(data[cur, 0], data[cur, 1], s=40, c=clr, edgecolors="k")
		x1_min, x2_min = np.min(data, axis=0)
		x1_max, x2_max = np.max(data, axis=0)
		plt.xlim(x1_min-0.5, x1_max+0.5)
		plt.ylim(x2_min-0.5, x2_max+0.5)
		plt.grid(b=True, ls=":", color="#808080")
		plt.title(r"$\sigma$ = %.2f" % s, fontsize=13)
	plt.tight_layout()
	plt.subplots_adjust(top=0.9)
	plt.show()

