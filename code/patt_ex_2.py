import numpy as np
import matplotlib.pyplot as plt

def normal_dist(x, mean, sd):
    prob_density = (1/(sd*np.sqrt(2*np.pi)) * np.exp(-0.5*(((x-mean)**2)/sd)))
    return prob_density

def p(x):
    if x > 0 and 2 > x:
        return 0.5 #np.array(out)
    return 0

seeds = np.random.uniform(0, 2, 32)
datapoints = np.linspace(-1,3,100)
# datapoints = datapoints[datapoints.argsort()]
data = np.array([[],[]])
probabs = np.array([])
for samples in datapoints:
    probab = 0
    for seed in seeds:
        using = p(seed)
        if using != 0:
            probab += normal_dist(((seed - samples) / 0.1), 0, 1)

    probab = probab / seeds.shape[0]
    probab = probab / 0.1
    probabs = np.append(probabs, probab)

probabs = probabs / datapoints.shape[0]
#a[a[:, 1].argsort()]
plt.plot(datapoints, probabs)
plt.show()


print("test")