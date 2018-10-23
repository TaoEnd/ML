# coding:utf-8

import numpy as np
from matplotlib import rcParams
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 使用决策树或者随机森林做回归

if __name__ == "__main__":
	n = 100   # 生成的样本数量
	x = np.random.rand(n) * 6 -3
	x.sort()
	y = np.sin(x) + np.random.randn(n) * 0.05
	# reshape(-1, 1)将1行n列的矩阵，变成了n行1列的矩阵
	x = x.reshape(-1, 1)

	# 使用普通决策树进行回归
	drmodel = DecisionTreeRegressor(criterion="mse", max_depth=6)
	drmodel.fit(x, y)
	# 生成测试数据
	x_test = np.linspace(-3, 3, 50)
	y_test = np.sin(x_test) + np.random.randn(50) * 0.05
	x_test = x_test.reshape(-1, 1)
	y_pred_dr = drmodel.predict(x_test)
	mse_dr = mean_squared_error(y_test, y_pred_dr)

	# 使用随机森林进行回归
	rfmodel = RandomForestRegressor(n_estimators=50, criterion="mse", max_depth=6)
	rfmodel.fit(x, y)
	y_pred_rf = rfmodel.predict(x_test)
	mse_rf = mean_squared_error(y_test, y_pred_rf)
	print("普通决策树的MSE：%.4f，随机森林的MSE：%.4f" % (mse_dr, mse_rf))

	rcParams["font.sans-serif"] = "SimHei"
	rcParams["axes.unicode_minus"] = False

	plt.plot(x, y, "r*", markersize=10, markeredgecolor="k", label="实际值")
	plt.plot(x_test, y_pred_dr, "g-", lw=2, label="普通决策树预测值")
	plt.plot(x_test, y_pred_rf, "b-", lw=2, label="随机森林预测值")
	plt.legend()
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.grid(b=True, ls=":")
	plt.title("决策树回归", fontsize=15)
	plt.tight_layout()
	plt.show()