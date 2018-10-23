# coding:utf-8

import numpy as np
from matplotlib import rcParams
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# 使用决策树或者随机森林做回归

if __name__ == "__main__":
	n = 100   # 生成的样本数量
	x = np.random.rand(n) * 6 -3
	x.sort()
	y = np.sin(x) + np.random.randn(n) * 0.05
	# reshape(-1, 1)将1行n列的矩阵，变成了n行1列的矩阵
	x = x.reshape(-1, 1)

	model = DecisionTreeRegressor(criterion="mse", max_depth=5)
	model.fit(x, y)
	# 生成测试数据
	x_test = np.linspace(-3, 3, 50).reshape(-1, 1)
	y_pred = model.predict(x_test)

	rcParams["font.sans-serif"] = "SimHei"
	rcParams["axes.unicode_minus"] = False

	plt.plot(x, y, "r*", markersize=10, markeredgecolor="k", label="实际值")
	plt.plot(x_test, y_pred, "g-", lw=2, label="预测值")
	plt.legend()
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.grid(b=True, ls=":")
	plt.title("决策树回归", fontsize=15)
	plt.tight_layout()
	plt.show()