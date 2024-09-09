import numpy as np
import pandas as pd

sf = pd.read_csv("../data/spotifyFeatures.csv")

#1a
number_of_songs = sf.shape[0]
song_features = sf.shape[1]

#1b
pc = sf[sf["genre"].isin(["Pop","Classical"])]

#lbl = ((pc["genre"] == "Pop").astype(int))
pc["label"] = ((pc["genre"] == "Pop").astype(int))
#pc = pc.drop("label", axis=1)
#pc = pc.drop("label", axis=0)

pop_count = pc["label"].sum()
classical_count = pc.shape[0]-pop_count

reduced = pc[["liveness","loudness","label"]]
dataset = reduced

train = dataset.sample(frac=0.8)
test = dataset.drop(train.index)
x_train = np.array(train[["liveness","loudness"]])
x_test = np.array(test[["liveness","loudness"]])
y_train = np.array(train["label"])
y_test = np.array(test["label"])

import matplotlib.pyplot as plt

dataset_cla = dataset[dataset["label"] == 0]
dataset_pop = dataset[dataset["label"] == 1]

plt.plot(dataset_cla["liveness"], dataset_cla["loudness"], 'b.', alpha=0.5)
plt.plot(dataset_pop["liveness"], dataset_pop["loudness"], 'r.', alpha=0.5)

plt.ylabel("loudness")
plt.xlabel("liveness")

plt.grid()
plt.show()
plt.clf()

#2a


# this is my own code
weights = np.array([0.0,0.0])
bias = 0
losses = []
# predicting
#np.dot(X, weights) + bias

# error function
#np.mean((y_true - y_pred) ** 2)

# computing the gradient
# y_pred = np.dot(X, weights) + bias
# error = y_pred - y_batch
# gradient_weights = np.dot(X_batch.T, error) / X_batch.shape[0]
# gradient_bias = np.mean(error)

# do gradient descent for 100 samples
for i in np.arange(1000):
    # data selection

    data_sample = train.sample(n=1)
    # data_sample = train.iloc[0]
    x_sample = np.array(data_sample[["liveness","loudness"]])
    y_sample = np.array(data_sample["label"])
    # x_sample = np.array([[ 0.5, 1 ]])
    # y_sample = np.array([1.0])

    # compute the gradients
    # doing the sigmoid
    y_pred = 1 / (1 + np.exp(-(np.dot(x_sample, weights) + bias)))
    error = (y_pred - y_sample)
    gradient_weights = np.dot(x_sample.T, error)
    gradient_bias = np.mean(error)

    # update step
    weights -= gradient_weights * 0.1
    bias -= gradient_bias * 0.1

    # is it learning????????
    loss = (y_sample - y_pred)**2
    losses.append(loss)
    print(f"Epoch {i}: Loss {loss}")

errors = pd.DataFrame({"loss":losses})
plt.plot(np.arange(len(losses)), losses)
plt.xlabel("Epochs")
plt.ylabel("Squared Error")
plt.title("2a")
plt.show()

train_results = 1 / (1 + np.exp(-(np.dot(train[["liveness","loudness"]], weights) + bias)))
np.round(train_results,0)

# accuracy score on training set
from sklearn.metrics import accuracy_score
print("accuracy for training: ", accuracy_score(train[["label"]], np.round(train_results,0)))

test_results = 1 / (1 + np.exp(-(np.dot(test[["liveness","loudness"]], weights) + bias)))

print("accuracy for testing: ", accuracy_score(test[["label"]], np.round(test_results,0)))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cfm = confusion_matrix(test[["label"]], np.round(test_results,0))
cmd = ConfusionMatrixDisplay(confusion_matrix=cfm)
cmd.plot(cmap=plt.cm.Greens)
plt.title('Confusion Matrix')
plt.show()
print("test")