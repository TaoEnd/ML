# coding:utf-8

import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tools.sm_exceptions import HessianInversionWarning
from matplotlib import rcParams
from matplotlib import pyplot as plt

# 航班流量预测

def date_parser(date):
	return pd.datetime.strptime(date, '%Y-%m')

if __name__=="__main__":
	warnings.filterwarnings(action="ignore", category=HessianInversionWarning)
	pd.set_option("display.width", 100)
	np.set_printoptions(linewidth=100, suppress=True)

	path = r'E:\python\PythonSpace\Git\ML\第九课\data\AirPassengers.csv'

	# 解析日期，并设置索引列
	data = pd.read_csv(path, header=0, parse_dates=["Month"],
					   date_parser=date_parser, index_col=["Month"])
	# 修改列名
	data.rename(columns={"#Passengers": "Passengers"}, inplace=True)
	# print(data.head())

	rcParams["font.sans-serif"] = "SimHei"
	rcParams["axes.unicode_minus"] = False

	# 将字符串转化成小数
	x = data["Passengers"].astype(np.float)
	# 随着时间增加，波动越来越剧烈，可以对数字取对数，从而缓解这种情况
	x = np.log(x)
	# print(x.head())

	p = 8
	d = 1
	q = 8
	model = ARIMA(endog=x, order=(p, d, q))
	# disp=-1表示不输出过程
	arima = model.fit(disp=-1)
	prediction = arima.fittedvalues
	# print(prediction)
	y = prediction.cumsum() + x[0]
	mse = ((x - y)**2).mean()
	rmse = np.sqrt(mse)

	show = "prime"
	# 差分结果
	diff = x - x.shift(periods=d)
	ma = pd.rolling_mean(x, window=12)
	xma = x - ma

	if show == "diff":
		plt.plot(x, "r-", lw=2, label="原始数据")
		plt.plot(diff, "g-", lw=2, label="%d阶差分" % d)
		title = "乘客人数变化曲线 - 取对数"
	elif show == "ma":
		plt.plot(xma, "g-", lw=2, label="ln原始数据 - Ln滑动平均数据")
		plt.plot(prediction, 'r-', lw=2, label="预测数据")
		title = "滑动平均值与MA预测值"
	else:
		plt.plot(x, "r-", lw=2, label="原始数据")
		plt.plot(y, "g-", lw=2, label="预测数据")
		title = "对数乘客人数与预测值（AR=%d，d=%d，MA=%d）:RMSE=%.4f" % (p, d, q, rmse)
	plt.legend()
	plt.grid(b=True, ls=":")
	plt.title(title, fontsize=16)
	# 自动调整图片中的参数
	plt.tight_layout(2)
	plt.show()
