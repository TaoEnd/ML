# coding:utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_curve, auc
from matplotlib import rcParams
from matplotlib import pyplot as plt

if __name__ == "__main__":
	pd.set_option("display.width", 400)
	path = r'E:\python\PythonSpace\Git\ML\data\adult.data'
	column_names = 'age', 'workclass', 'fnlwgt', 'education', 'education-num', \
				   'marital-status', 'occupation', 'relationship', 'race', \
				   'sex', 'capital-gain', 'capital-loss', 'hours-per-week', \
				   'native-country', 'income'
	data = pd.read_csv(path, header=None, names=column_names)
	for name in column_names:
		data[name] = pd.Categorical(data[name]).codes
	x = data[data.columns[:-1]]
	y = data[data.columns[-1]]

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

	# AdaBoost
	base_model = DecisionTreeClassifier(criterion="gini", max_depth=3, min_samples_split=4)
	model_ab = AdaBoostClassifier(base_estimator=base_model, n_estimators=80, learning_rate=0.6)
	model_ab.fit(x_train, y_train)
	y_tain_pred = model_ab.predict(x_train)
	y_test_pred = model_ab.predict(x_test)
	print("训练集准确率：%.3f" % accuracy_score(y_train, y_tain_pred))
	print("训练集查准率：%.3f" % precision_score(y_train, y_tain_pred))
	print("训练集召回率：%.3f" % recall_score(y_train, y_tain_pred))
	print("训练集F1值：%.3f" % f1_score(y_train, y_tain_pred))
	print()
	print("测试集准确率：%.3f" % accuracy_score(y_test, y_test_pred))
	print("测试集查准率：%.3f" % precision_score(y_test, y_test_pred))
	print("测试集召回率：%.3f" % recall_score(y_test, y_test_pred))
	print("测试集F1值：%.3f" % f1_score(y_test, y_test_pred))
	print("------------------------------")

	# 随机森林
	model_rf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=3)
	model_rf.fit(x_train, y_train)
	y_train_pred = model_rf.predict(x_train)
	y_test_pred = model_rf.predict(x_test)
	print("训练集准确率：%.3f" % accuracy_score(y_train, y_tain_pred))
	print("训练集查准率：%.3f" % precision_score(y_train, y_tain_pred))
	print("训练集召回率：%.3f" % recall_score(y_train, y_tain_pred))
	print("训练集F1值：%.3f" % f1_score(y_train, y_tain_pred))
	print()
	print("测试集准确率：%.3f" % accuracy_score(y_test, y_test_pred))
	print("测试集查准率：%.3f" % precision_score(y_test, y_test_pred))
	print("测试集召回率：%.3f" % recall_score(y_test, y_test_pred))
	print("测试集F1值：%.3f" % f1_score(y_test, y_test_pred))

	# 得到每一种类别的预测概率，并且只取其中一种类的概率
	y_test_prob_ab = model_ab.predict_proba(x_test)
	y_test_prob_ab = y_test_prob_ab[:, 1]
	fpr_ab, tpr_ab, thresholds_ab = roc_curve(y_test, y_test_prob_ab)
	auc_ab = auc(fpr_ab, tpr_ab)

	y_test_prob_rf = model_rf.predict_proba(x_test)
	y_test_prob_rf = y_test_prob_rf[:, 1]
	fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_test_prob_rf)
	auc_rf = auc(fpr_rf, tpr_rf)

	rcParams["font.sans-serif"] = "SimHei"
	rcParams["axes.unicode_minus"] = False

	plt.plot((0, 1), (0, 1), "b--", lw=1.2)
	# 在plot中设置label相当于在图例中设置名字
	plt.plot(fpr_ab, tpr_ab, "r-", lw=1.2, label="AdaBoost AUC=%.3f" % auc_ab)
	plt.plot(fpr_rf, tpr_rf, "g-", lw=1.2, label="RF AUC=%.3f" % auc_rf)
	plt.xlim((-0.01, 1.02))
	plt.ylim((-0.01, 1.02))
	plt.xticks(np.arange(0, 1.1, 0.1))
	plt.yticks(np.arange(0, 1.1, 0.1))
	plt.xlabel("False Positive Rate", fontsize=12)
	plt.ylabel("True Positive Rate", fontsize=12)
	plt.grid(b=True)
	plt.legend()
	plt.title("ROC-AUC")
	plt.show()
