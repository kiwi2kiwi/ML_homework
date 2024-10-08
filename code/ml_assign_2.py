import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dp = pd.read_csv("../data/data_problem2.csv", header=None).T
dp2 = dp.T
indices1 = dp2.T.index[dp2.loc[1] == 1]
indices0 = dp2.T.index[dp2.loc[1] == 0]
df = dp2.T
dis0 = df.loc[indices0]
dis1 = df.loc[indices1]

plt.hist(dis0.iloc[:,0], bins=20, label="first")
plt.hist(dis1.iloc[:,0], bins=20, label="second")

plt.title('Histogram of distributions')

plt.ylabel('Frequency')

plt.show()

print("test")