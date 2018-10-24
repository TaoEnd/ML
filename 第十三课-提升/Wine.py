# coding:utf-8

from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

if __name__ == "__main__":
	data = load_wine().data
	target = load_wine().target
	# 数据归一化
	norm_data = MinMaxScaler().fit_transform(data)

	x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=0)

	# XGBoost分类结果
	data_train = xgb.DMatrix(x_train, label=y_train)
	data_test = xgb.DMatrix(x_test, label=y_test)
	watchlist = [(data_train, "train"), (data_test, "eval")]
	param = {"max_depth": 6, "eta": 0.4, "silent": 1,
			 "object": "miulti:softmax", "num_class": 3}
	model = xgb.train(param, data_train, num_boost_round=10, evals=watchlist)
	y_pred = model.predict(data_test)
	accuracy_xgb = accuracy_score(y_test, y_pred)

	# 逻辑回归分类结果
	model = LogisticRegression(penalty="l1", C=5)
	model.fit(x_train, y_train)
	y_pred = model.predict(x_test)
	accuracy_lr = accuracy_score(y_test, y_pred)

	# 随机森林分类结果
	model = RandomForestClassifier(n_estimators=40, max_depth=6)
	model.fit(x_train, y_train)
	y_pred = model.predict(x_test)
	accuracy_rf = accuracy_score(y_test, y_pred)

	# AdaBoost预测结果
	# 首先设置基分类器
	base_model = DecisionTreeClassifier(criterion="gini", max_depth=3, min_samples_split=4)
	model = AdaBoostClassifier(base_estimator=base_model, n_estimators=50,
							   learning_rate=0.6, algorithm="SAMME.R")
	model.fit(x_train, y_train)
	y_pred = model.predict(x_test)
	accuracy_ab = accuracy_score(y_test, y_pred)

	print("测试样本数量：%d" % len(y_test))
	print("XGBoost分类准确率：%.4f" % accuracy_xgb)
	print("LR分类准确率：%.4f" % accuracy_lr)
	print("随机森林分类准确率：%.4f" % accuracy_rf)
	print("AdaBoost分类准确率：%.4f" % accuracy_ab)
