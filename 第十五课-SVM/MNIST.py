# coding:utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn import svm

if __name__ == "__main__":
	path = r"E:\python\PythonSpace\Git\ML\第十五课-SVM\data\MNIST.train.csv"
	data_train = pd.read_csv(path, header=0, dtype=int)
	x = data_train.iloc[:, 1:]
	y = np.array(data_train.iloc[:, :1])
	y = y.ravel()
	data_test = pd.read_csv(path, header=0, dtype=int)
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

	print("SVM开始...")
	model = svm.SVC(C=2, kernel="rbf", gamma=0.1)
	model.fit(x_train, y_train)
	y_pred = model.predict(x_test)
	accuracy_svm = accuracy_score(y_test, y_pred)

	print("XGBoost开始...")
	watchlist = [(x_train, "train"), (x_test, "eval")]
	param = {"max_depth": 6, "learning_rate": 0.3, "slient": 1,
			 "objective": "multi:softmax", "num_class": 10,
			 "min_samples_split": 4, "n_estimators": 150}
	d_train = xgb.DMatrix(x_train, label=y_train)
	d_test = xgb.DMatrix(y_test, label=y_test)
	model = xgb.train(param, d_train, num_boost_round=10, evals=watchlist)
	y_pred = model.predict(d_test)
	accuracy_xgb = accuracy_score(y_test, y_pred)

	print("RF开始...")
	model = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_split=4)
	model.fit(x_train, y_train)
	y_pred = model.predict(x_test)
	accuracy_rf = accuracy_score(y_test, y_pred)

	print("SVM的正确率：%.3f" % accuracy_svm)
	print("XGBoost的正确率：%.3f" % accuracy_xgb)
	print("RF的正确率：%.3f" % accuracy_rf)

