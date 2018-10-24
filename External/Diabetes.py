# coding:utf-8

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rcParams
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
from sklearn.preprocessing import PolynomialFeatures

# 使用糖尿病数据集预测糖尿病

if __name__ == "__main__":
	diabetes = load_diabetes()
	x_data = diabetes.data
	# 使用PolynomialFeatures做特征交叉，默认两两特征进行交叉
	# poly = PolynomialFeatures()
	# x_data = poly.fit_transform(x_data)
	# print(x_data.shape)
	y = diabetes.target
	x = pd.DataFrame()

	for i in range(len(x_data[0])):
		x_hat = x_data[:, i:i+1].ravel()
		# 计算皮尔逊相似度
		p = pearsonr(x_hat, y)[0]
		print("%.4f" % p, end=" ")
		if p > 0.4:
			x = pd.concat((x, pd.Series(x_hat)), axis=1)
	print(x.shape)
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

	# 普通线性回归
	model = LinearRegression()
	model.fit(x_train, y_train)
	y_pred_lr = model.predict(x_test)
	mse = mean_squared_error(y_test, y_pred_lr)
	r2 = model.score(x_test, y_test)
	print("普通线性回归的MSE：%.4f，R方：%.4f" % (mse, r2))

	# Lasso回归
	model = Ridge(alpha=0.5)
	model.fit(x_train, y_train)
	y_pred_la = model.predict(x_test)
	mse = mean_squared_error(y_test, y_pred_la)
	r2 = model.score(x_test, y_test)
	print("Ridge回归的MSE：%.4f，R方：%.4f" % (mse, r2))

	# 随机森林回归
	model = RandomForestRegressor(n_estimators=50, criterion="mse", max_depth=6)
	model.fit(x_train, y_train)
	y_pred_rf = model.predict(x_test)
	mse = mean_squared_error(y_test, y_pred_rf)
	r2 = model.score(x_test, y_test)
	print("随机森林回归的MSE：%.4f，R方：%.4f" % (mse, r2))

	rcParams["font.sans-serif"] = "SimHei"
	rcParams["axes.unicode_minus"] = False
	xlabel = list(range(x_test.shape[0]))

	plt.plot(xlabel, y_test, "r*", markersize=15)
	plt.plot(xlabel, y_pred_lr, "g-", lw=2)
	plt.plot(xlabel, y_pred_la, "b-", lw=2)
	plt.plot(xlabel, y_pred_rf, "k-", lw=2)
	plt.grid(b=True, ls=":")
	plt.tight_layout()
	plt.show()