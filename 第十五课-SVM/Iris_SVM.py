# coding:utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from matplotlib import rcParams, colors
from matplotlib import pyplot as plt

if __name__ == "__main__":
	path = r"E:\python\PythonSpace\Git\ML\第十五课-SVM\data\iris.data"
	data = pd.read_csv(path, header=None)
	x = data[[0, 1]]
	y = pd.Categorical(data[4]).codes.ravel()
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

	params = {"C": range(1, 5), "gamma": np.linspace(0.01, 1, 10)}
	iris_svm = svm.SVC(kernel="rbf", decision_function_shape="ovr")
	model = GridSearchCV(estimator=iris_svm, param_grid=params, scoring="accuracy", cv=5)
	model.fit(x_train, y_train)

	best_C = model.best_params_["C"]
	best_Gamma = model.best_params_["gamma"]
	print("C：%.3f，Gamma：%.3f" % (best_C, best_Gamma))

	model = svm.SVC(C=best_C, kernel="rbf", gamma=best_Gamma, decision_function_shape="ovr")
	model.fit(x_train, y_train)
	accuracy = accuracy_score(y_test, model.predict(x_test))

	print("样本数量：%d，准确率：%.4f" % (len(x_test), accuracy))
	# decision_function：计算点到分类超平面的函数距离，因此存在负数
	# print(model.decision_function(x_train))

	rcParams["font.sans-serif"] = "SimHei"
	rcParams["axes.unicode_minus"] = False

	# 画图
	x1_min, x2_min = x.min()
	x1_max, x2_max = x.max()
	t1 = np.linspace(x1_min-0.1, x1_max+0.1, 50)
	t2 = np.linspace(x2_min-0.1, x2_max+0.1, 50)
	x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
	grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
	grid_hat = model.predict(grid_test)  # 预测分类值
	grid_hat = grid_hat.reshape(x1.shape)

	cm_light = colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
	cm_dark = colors.ListedColormap(["g", "r", "b"])
	plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
	plt.scatter(x[0], x[1], c=y, edgecolors="k", s=50, cmap=cm_dark)
	plt.scatter(x_test[0], x_test[1], s=120, facecolors="none")
	plt.ylabel("花萼长度", fontsize=13)
	plt.xlabel("花萼宽度", fontsize=13)
	plt.xlim(x1_min-0.1, x1_max+0.1)
	plt.ylim(x2_min-0.1, x2_max+0.1)
	plt.title("鸢尾花SVM分类", fontsize=15)
	plt.grid(b=True, ls=":")
	plt.tight_layout()
	plt.show()
