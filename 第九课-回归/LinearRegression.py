# coding:utf-8

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

# 使用线性回归预测广告销量

if __name__ == "__main__":
	path = r'E:\python\PythonSpace\Git\ML\第九课\data\Advertising.csv'
	data = pd.read_csv(path, header=0, usecols=[1, 2, 3, 4])
	y = data["Sales"]

	rcParams["font.sans-serif"] = "SimHei"
	rcParams["axes.unicode_minus"] = False

	plt.plot(data["TV"], y, "ro", label="TV")
	plt.plot(data["Radio"], y, "g^", label="Radio")
	plt.plot(data["Newspaper"], y, "mv", label="Newspaper")
	plt.legend()
	plt.xlabel("广告花费", fontsize=15)
	plt.ylabel("销售额", fontsize=15)
	plt.title("销售额与广告花费", fontsize=15)
	plt.grid(b=True, ls=":")
	plt.show()

	plt.subplot(311)
	plt.plot(data["TV"], y, "ro")
	plt.title("TV")
	plt.grid(b=True, ls=":")
	plt.subplot(312)
	plt.plot(data["Radio"], y, "g^")
	plt.title("Radio")
	plt.grid(b=True, ls=":")
	plt.subplot(313)
	plt.plot(data["Newspaper"], y, "b*")
	plt.title("Newspaper")
	plt.grid(b=True, ls=":")
	plt.show()

	x = data[["TV", "Radio"]]
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
	# 普通线性回归，没有归一化
	model1 = LinearRegression()
	model1.fit(x_train, y_train)
	y_pred1 = model1.predict(x_test)
	mse = mean_squared_error(y_test, y_pred1)
	r2 = model1.score(x_test, y_test)
	print(model1.coef_, model1.intercept_, mse, r2)

	# 普通线性回归，归一化
	model2 = LinearRegression()
	model2.fit(x_train, y_train)
	y_pred2 = model2.predict(x_test)
	mse = mean_squared_error(y_test, y_pred2)
	r2 = model2.score(x_test, y_test)
	print(model2.coef_, model2.intercept_, mse, r2)

	# 岭回归
	model3 = Ridge(alpha=0.3)
	model3.fit(x_train, y_train)
	y_pred3 = model3.predict(x_test)
	mse = mean_squared_error(y_test, y_pred3)
	r2 = model3.score(x_test, y_test)
	print(model3.coef_, model3.intercept_, mse, r2)

	plt.subplot(311)
	xlabel = np.arange(len(y_test))
	plt.plot(xlabel, y_test, "r", lw=1, label="真实值")
	plt.plot(xlabel, y_pred1, "g", lw=1, label="预测值")
	plt.title("普通线性回归，未归一化")
	plt.subplot(312)
	plt.plot(xlabel, y_test, "r", lw=1, label="真实值")
	plt.plot(xlabel, y_pred2, "g", lw=1, label="预测值")
	plt.title("普通线性回归，归一化")
	plt.subplot(313)
	plt.plot(xlabel, y_test, "r", lw=1, label="真实值")
	plt.plot(xlabel, y_pred3, "g", lw=1, label="预测值")
	plt.title("岭回归")
	plt.tight_layout(2)
	plt.show()
