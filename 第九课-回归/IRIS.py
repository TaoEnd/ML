# coding:utf-8

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 使用逻辑回归对鸢尾花分类

if __name__=="__main__":
	path = r'E:\python\PythonSpace\Git\ML\第九课\data\iris.data'
	source = pd.read_csv(path, header=None)
	x = pd.DataFrame()
	for i in range(source.shape[1]-1):
		max = np.max(source[i])
		temp = [num/max for num in source[i]]
		temp = pd.Series(temp)
		x = pd.concat((x, temp), axis=1)
	# 得到所有label
	labels = pd.Categorical(source[4]).unique()
	y = pd.Categorical(source[4]).codes
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

	# 训练模型
	# class_weight：比如当前有2类样本，0和1，它们是不平衡的样本，
	# 比如0的样本有100个，1的样本有20个，此时可以通过class_weight来
	# 设置0、1样本的权重，使得模型更关注数量更少的1，比如可以设置成
	# class_weight={0:1, 1:5},1、5分别表示权重。这是一种处理不平衡
	# 样本的方法。
	model = LogisticRegressionCV(Cs=np.logspace(-1, 4, 10), cv=5, n_jobs=3)
	model.fit(x_train, y_train)
	y_pred = model.predict(x_test)
	accuracy = accuracy_score(y_pred, y_test)
	print(model.predict_proba(x_test))
	print(y_test)
	print(y_pred)
	print(accuracy)