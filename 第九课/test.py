# coding:utf-8

import numpy as np
from pandas import Series

def get_norm(num, max):
	return num/max

x = Series([1, 2, 3, 4])
for num in x:
	print(num)
max = max(x)
y = [num/max for num in x]
print(y)