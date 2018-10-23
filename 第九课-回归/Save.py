# coding:utf-8

import pandas as pd
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import os

# 使用线性回归预测广告销量

if __name__ == "__main__":
	path = r'E:\python\PythonSpace\Git\ML\第九课\data\Advertising.csv'
	data = pd.read_csv(path, header=0, usecols=[1, 2, 3, 4])
	y = data["Sales"]

	rcParams["font.sans-serif"] = "SimHei"
	rcParams["axes.unicode_minus"] = False

	x = data[["TV", "Radio"]]
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
	# 模型保存与加载
	modelPath = r'E:\python\PythonSpace\Git\ML\第九课\data\lr.model'
	if os.path.exists(modelPath):
		print("模型加载...")
		model = joblib.load(modelPath)
	else:
		print("模型保存...")
		model = LinearRegression()
		model.fit(x_train, y_train)
		joblib.dump(model, modelPath)
	y_pred = model.predict(x_test)
	mse = mean_squared_error(y_test, y_pred)
	r2 = model.score(x_test, y_test)
	print(model.coef_, model.intercept_, mse, r2)