# coding:utf-8

import numpy as np
import pandas as pd
import warnings
from matplotlib import pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 波士顿房价预测

def not_empty(s):
	return s != ""

if __name__=="__main__":
	warnings.filterwarnings(action="ignore")
	np.set_printoptions(suppress=True)

	path = r'E:\python\PythonSpace\Git\ML\第九课\data\housing.data'
	source = pd.read_csv(path, header=None)
	data = np.empty((len(source), 14))
	# enumerate：将一个可遍历的对象组合成一个索引序列，同时列出数据和数据下标
	for i, d in enumerate(source.values):
		temp = list(map(float, list(filter(not_empty, d[0].split(' ')))))
		data[i] = temp
	# 将特征值和label分开
	x, y = np.split(data, (13, ), axis=1)
	# 将label转化成一维的
	y = y.ravel()

	# 数据集划分
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

	# 随机森林回归
	model = RandomForestRegressor(n_estimators=50, criterion="mse")
	print("开始训练...")
	model.fit(x_train, y_train)

	# 获得参数，每个特征的重要程度
	print(len(model.feature_importances_), model.feature_importances_)

	# 对房价排序，并返回排序后的下标
	# order = y_test.argsort(axis=0)
	# y_test = y_test[order]
	# x_test = x_test[order, :]
	y_pred = model.predict(x_test)
	r2 = model.score(x_test, y_test)
	mse = mean_squared_error(y_test, y_pred)
	print("R2：", r2)
	print("均方误差：", mse)

	t = np.arange(len(y_pred))
	rcParams["font.sans-serif"] = "SimHei"
	rcParams["axes.unicode_minus"] = False
	plt.plot(t, y_test, 'r-', lw=2, label="真实值")
	plt.plot(t, y_pred, 'g-', lw=2, label="预测值")
	plt.legend()
	plt.title("波士顿房价预测", fontsize=15)
	plt.xlabel("样本编号", fontsize=15)
	plt.ylabel("房屋价格", fontsize=15)
	plt.grid()
	plt.show()