# coding:utf-8

import numpy as np
from sklearn import datasets as ds
from sklearn.preprocessing import StandardScaler
from matplotlib import rcParams
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN

if __name__ == "__main__":
	n = 1000
	centers = [[1, 2], [-1, -1], [1, -1], [-1, 1]]
	data, y = ds.make_blobs(n, n_features=2, centers=centers, cluster_std=[0.5, 0.25, 0.7, 0.4])
	# 规范化
	data = StandardScaler().fit_transform(data)
	# DBSCAN的delta大小和核心对象个数
	params = ((0.2, 5), (0.2, 10), (0.2, 15), (0.3, 5), (0.3, 10), (0.3, 15))

	rcParams["font.sans-serif"] = "SimHei"
	rcParams["axes.unicode_minus"] = False

	plt.figure(figsize=(9, 7))
	plt.suptitle("DBSCAN聚类", fontsize=15)
	for i in range(6):
		eps, min_samples = params[i]
		model = DBSCAN(eps=eps, min_samples=min_samples)
		model.fit(data)
		y_hat = model.labels_

		y_unique = np.unique(y_hat)
		# 如果y_hat中存在-1，则说明DBSCAN认为数据中有噪声
		num_clusters = y_unique.size - (1 if -1 in y_hat else 0)
		print("类簇数量：", num_clusters)

		core_indices = np.zeros_like(y_hat, dtype=bool)
		# core_sample_indices_表示核心样本在原本数据中的位置（索引）
		core_indices[model.core_sample_indices_] = True

		clrs = plt.cm.Spectral(np.linspace(0, 0.8, num_clusters))
		# print(clrs)
		plt.subplot(2, 3, i+1)
		for k, clr in zip(y_unique, clrs):
			cur = (y_hat == k)
			if k == -1:
				# 将噪声数据全部表示成黑色的
				plt.scatter(data[cur, 0], data[cur, 1], s=10, c="k")
				continue
			plt.scatter(data[cur, 0], data[cur, 1], s=15, c=clr, edgecolors="k")
			plt.scatter(data[cur & core_indices][:, 0], data[cur & core_indices][:, 1],
						s=30, c=clr, marker="o", edgecolors="k")
			x1_min, x2_min = np.min(data, axis=0)
			x1_max, x2_max = np.max(data, axis=0)
			plt.xlim(x1_min-0.5, x1_max+0.5)
			plt.ylim(x2_min-0.5, x2_max+0.5)
			plt.grid(b=True, ls=":", color="#606060")
			plt.title(r"$\epsilon$ = %.1f m = %d，聚类数目：%d" % (eps, min_samples, num_clusters),
					  fontsize=12)
	plt.tight_layout()
	plt.subplots_adjust(top=0.9)
	plt.show()


