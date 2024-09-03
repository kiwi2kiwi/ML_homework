import pandas as pd
temp = pd.read_csv("data/global-temperatures.csv", sep="\s+", skiprows = [0,1,2,3,4,5,6], header=None)

cov_matrix = temp.cov()
cov_singular = temp[1].cov(temp[0])
std_x = temp[0].std()
std_y = temp[1].std()
y_mean = temp[1].mean()

pcc = cov_singular/(std_x * std_y)

x = temp[1]-1880


slope = pcc * (std_y/std_x)
intercept = temp[1].mean() - (slope*temp[0].mean())

import matplotlib.pyplot as plt
plt.scatter(temp[0],temp[1])
plt.plot([0, 1880 + 137], [intercept, (1880 + 137) * slope + intercept], color='k', linestyle='-', linewidth=2)
plt.show()

import numpy as np
y_pred = [i*slope+intercept for i in temp[0]]
r_sq_top = sum([(temp[1][i] - y_pred[i])**2 for i in np.arange(138)])
r_sq_bot = sum([(temp[1][i] - y_mean)**2 for i in np.arange(138)])

r_sq = 1 - (r_sq_top/r_sq_bot)

print("end")