# coding:utf-8

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib import colors
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pydotplus

if __name__ == "__main__":
	path = r"E:\python\PythonSpace\Git\ML\第十一课-决策树\data\iris.data"
	data = pd.read_csv(path, header=None)
	x = data.iloc[:, :data.shape[1]-1]
	# 为了可视化，所以只使用x的前两列特征
	x = x.iloc[:, :2]
	y = LabelEncoder().fit_transform(data[4])
	# random_state是一个随机数种子，确保每一次随机抽样得到的样本都是相同的，
	# 如果不设置时，那么每次抽样的结果都不相同
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
	# min_samples_split：表示内部节点包含的最少样本数量
	# min_samples_left：叶节点包含的最少样本数量
	model = DecisionTreeClassifier()
	model.fit(x_train, y_train)
	y_pred = model.predict(x_test)
	accuracy = accuracy_score(y_test, y_pred)
	print(accuracy)

	# 保存
	# 保存成.dot文件
	dotPath = r"E:\python\PythonSpace\Git\ML\第十一课-决策树\data\iris.dot"
	with open(dotPath, 'w') as fw:
		# 保存的dot文件需要使用graphviz软件打开
		tree.export_graphviz(model, out_file=fw)

	iris_feature_E = "sepal length", "sepal width", "petal length", "petal width"
	iris_feature = "花萼长度", "花萼宽度", "花瓣长度", "花瓣宽度"
	iris_class = "Iris-setosa", "Iris-versicolor", "Iris-virginica"

	# 保存成.pdf和.png文件
	dot_data = tree.export_graphviz(model, out_file=None,
									feature_names=iris_feature_E[:2],
									class_names=iris_class, filled=True,
									rounded=True, special_characters=True)
	graph = pydotplus.graph_from_dot_data(dot_data)
	pdfPath = r"E:\python\PythonSpace\Git\ML\第十一课-决策树\data\iris.pdf"
	pngPath = r"E:\python\PythonSpace\Git\ML\第十一课-决策树\data\iris.png"
	graph.write_pdf(pdfPath)
	with open(pngPath, "wb") as fw:
		fw.write(graph.create_png())

	# 画图
	rcParams["font.sans-serif"] = "SimHei"
	rcParams["axes.unicode_minus"] = False
	N, M = 50, 50  # 用来控制横纵各采样多少个值
	x1_min, x2_min = x.min()
	x1_max, x2_max = x.max()
	t1 = np.linspace(x1_min, x1_max, N)
	t2 = np.linspace(x2_min, x2_max, M)
	x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
	# print(x1)
	# print(x1.flat)
	x_show = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
	print(x_show.shape)

	cm_light = colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
	cm_dark = colors.ListedColormap(["g", "r", "b"])
	y_show = model.predict(x_show)  # 预测值
	print(y_show.shape)
	y_show = y_show.reshape(x1.shape)  # 使之与输入的形状相同
	plt.pcolormesh(x1, x2, y_show, cmap=cm_light)  # 预测结果的显示
	plt.scatter(x_test[0], x_test[1], c=y_test.ravel(), edgecolors="k", s=100,
				zorder=10, cmap=cm_dark, marker="*")
	plt.scatter(x[0], x[1], c=y.ravel(), edgecolors="k", s=20, cmap=cm_dark) # 全部数据
	plt.xlabel(iris_feature[0], fontsize=13)
	plt.ylabel(iris_feature[1], fontsize=13)
	plt.xlim(x1_min, x1_max)
	plt.ylim(x2_min, x2_max)
	plt.grid(b=True, ls=":", color="#606060")
	plt.title("鸢尾花数据的决策树分类", fontsize=15)
	plt.show()