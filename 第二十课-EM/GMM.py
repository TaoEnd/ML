# coding:utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from matplotlib import rcParams, colors
from matplotlib import pyplot as plt

# 使用高斯分布估计男生和女生的身高、体重数据满足的高斯分布

if __name__ == "__main__":
	path = r"E:\python\PythonSpace\Git\ML\第二十课-EM\data\HeightWeight.csv"
	data = pd.read_csv(path, header=0)
	x = data[["Height(cm)", "Weight(kg)"]]
	y = data[["Sex"]]
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
	model = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
	model.fit(x_train)
	# print("均值：", model.means_)
	# print("方差：")
	# print(model.covariances_)
	y_train_hat = model.predict(x_train)
	y_test_hat = model.predict(x_test)
	# z = y_test_hat == 0 等价于z = (y_test_hat == 0)，
	# 首先判断y_test_hat数组中哪些元素等于0，等于0的就表示成True，
	# 否则表示成False，然后将判断的boolean数组赋值给z

	acc_train = np.mean(y_train_hat == np.array(y_train).ravel())
	acc_test = np.mean(y_test_hat == np.array(y_test).ravel())
	acc_train_str = "训练集准确率：%.2f%%" % (acc_train*100)
	acc_test_str = "测试集准确率：%.2f%%" % (acc_test*100)
	print(acc_train_str)
	print(acc_test_str)

	cm_light = colors.ListedColormap(["#FF8080", "#77E0A0"])
	cm_dark = colors.ListedColormap(["r", "g"])
	x1_min, x2_min = np.min(x, axis=0)
	x1_max, x2_max = np.max(x, axis=0)
	t1 = np.linspace(x1_min-0.5, x1_max+0.5, 200)
	t2 = np.linspace(x2_min-0.5, x2_max+0.5, 200)
	x1, x2 = np.meshgrid(t1, t2)
	# x1.flat表示将x1变成1*n的向量，只有一行
	grid_test = np.stack((x1.flat, x2.flat), axis=1)
	grid_test_hat = model.predict(grid_test)
	grid_test_hat = grid_test_hat.reshape(x1.shape)

	rcParams["font.sans-serif"] = "SimHei"
	rcParams["axes.unicode_minus"] = False
	plt.figure(figsize=(7, 6))
	plt.pcolormesh(x1, x2, grid_test_hat, cmap=cm_light)
	plt.scatter(x_train.iloc[:, 0], x_train.iloc[:, 1], s=50, c=y_train, marker="o",
				cmap=cm_dark, edgecolors="k")
	plt.scatter(x_test.iloc[:, 0], x_test.iloc[:, 1], s=60, c=y_test, marker="^",
				cmap=cm_dark, edgecolors="k")

	# 绘制等高线
	p = model.predict_proba(grid_test)
	p = p[:, 0].reshape(x1.shape)
	cs = plt.contour(x1, x2, p, levels=(0.1, 0.5, 0.8), colors=list("rgb"),
					 linewidths=2)

	# 在画布上加一些文字说明
	# xx、yy分别表示相对于画布的位置
	ax1_min, ax1_max, ax2_min, ax2_max = plt.axis()
	xx = 0.95*ax1_min + 0.05*ax1_max
	yy = 0.05*ax2_min + 0.95*ax2_max
	plt.text(xx, yy, acc_train_str, fontsize=12)
	yy = 0.1*ax2_min + 0.9*ax2_max
	plt.text(xx, yy, acc_test_str, fontsize=12)

	plt.xlim(x1_min-0.5, x1_max+0.5)
	plt.ylim(x2_min-0.5, x2_max+0.5)
	plt.xlabel("身高（cm）", fontsize=13)
	plt.ylabel("体重（kg）", fontsize=13)
	plt.title("EM估计GMM的参数", fontsize=13)
	plt.grid(b=True, ls=":")
	plt.tight_layout(2)
	plt.show()