# coding:utf-8

import pandas as pd

s = pd.Series(["a", "b", "c", "a"], name="tag")
print(s)
t = pd.get_dummies(s)
print(t)