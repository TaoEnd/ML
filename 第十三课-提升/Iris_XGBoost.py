# coding:utf-8

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":
	x = load_iris().data
	y = load_iris().target
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

	# XGBoost的网格搜索
	classfier = XGBClassifier(silent=1, objective="multi:softprob", num_class=3,
						eval_metic="auc", num_boost_round=5)

	param = {"max_depth": range(2, 5), "learning_rate": [0.3, 0.4, 0.5, 0.6]}

	model = GridSearchCV(estimator=classfier, param_grid=param, cv=5)
	model.fit(x_train, y_train)
	max_depth = model.best_params_["max_depth"]
	eta = model.best_params_["learning_rate"]
	print(model.best_params_, model.best_score_)

	data_train = xgb.DMatrix(x_train, label=y_train)
	data_test = xgb.DMatrix(x_test, label=y_test)
	watchlist = [(data_train, "train"), (data_test, "eval")]
	param = {"max_depth": max_depth, "eta": eta, "silent":1,
			 "object": "multi:softmax", "num_class": 3}
	model = xgb.train(param, data_train, num_boost_round=5, evals=watchlist)

	y_pred = model.predict(data_test)
	print(y_test)
	print(y_pred)
	print(y_test != y_pred)
	accuracy = accuracy_score(y_test, y_pred)
	print("准确率：%.4f" % accuracy)