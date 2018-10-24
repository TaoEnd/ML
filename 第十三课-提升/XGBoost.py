# coding:utf-8

import xgboost as xgb

if __name__ == "__main__":
	trainPath = r'E:\python\PythonSpace\Git\ML\agaricus_train.txt'
	testPath = r'E:\python\PythonSpace\Git\ML\agaricus_test.txt'
	data_train = xgb.DMatrix(trainPath)
	data_test = xgb.DMatrix(testPath)
	# print(type(data_test.get_label()))
	# print(data_test.get_label())

	# 设置参数
	# eta是学习率，
	# silent=1表示不打印运行过程的信息，默认为0，表示打印信息
	# object表示设置当前问题类型
	param = {"max_depth": 3, "eta": 0.2, "silent": 1, "object": "binary:logistic"}
	watchlist = [(data_train, "train"), (data_test, "eval")]
	n_round = 5  # 表示训练7轮
	# obj和feral可以分别用来设置损失函数和最终的模型评估函数
	bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist)

	y_hat = bst.predict(data_test)
	y_test = data_test.get_label()
	# 计算错误率
	error = sum(y_test != (y_hat>0.5))
	error_rate = float(error) / len(y_hat)
	print("样本总数：", len(y_hat))
	print("错误数目：", error)
	print("错误率：%.4f%%" % (100*error_rate))