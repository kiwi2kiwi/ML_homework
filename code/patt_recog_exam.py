import scipy
from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# loading the data
test = scipy.io.loadmat("../data/TestingData_N1000_d2_M4-v7.mat")
train = scipy.io.loadmat("../data/TrainingData_N800_d2_M4-v7.mat")
# conversion to pandas DataFrame for easier handling
df_train = pd.DataFrame(
    data=np.array([train["Labels"].flatten(), train["DataVecs"][:, 0], train["DataVecs"][:, 1]]).T,
    columns=["Label", "X_Coordinate", "Y_Coordinate"]
)
df_test = pd.DataFrame(
    data=np.array([test["TestLabels"].flatten(), test["TestVecs"][:, 0], test["TestVecs"][:, 1]]).T,
    columns=["Label", "X_Coordinate", "Y_Coordinate"]
)

from scipy.stats import multivariate_normal
unique_labels_train = df_train['Label'].unique()
unique_labels_test = df_test['Label'].unique()
classes = {}
for i, label in enumerate(unique_labels_train):
    class_data = df_train[df_train['Label'] == label]
    mean_x = np.mean(class_data["X_Coordinate"])
    mean_y = np.mean(class_data["Y_Coordinate"])
    matrix = np.vstack((class_data["X_Coordinate"],class_data["Y_Coordinate"])).T
    column_means = np.mean(matrix, axis=0)
    covariance_matrix = np.cov(matrix, rowvar=False)
    print("Class: ", label, ": mean for x and y: ", column_means, "\\\\")
    print("Class: ", label, ": covariance matrix: ", covariance_matrix, "\\\\")
    classes[int(label)] = multivariate_normal(mean=column_means, cov=covariance_matrix)
    # variance_x = np.mean([(x - mean_x) ** 2 for x in class_data["X_Coordinate"]])
    # variance_y = np.mean([(y - mean_y) ** 2 for y in class_data["Y_Coordinate"]])
    # print("$Class: ", label, ": \[\sigma\] for x: ", "%.3f" % variance_x, "\\\\")
    # print("$Class: ", label, ": \[\sigma\] for y: ", "%.3f" % variance_y, "\\\\")

prediction_test = []
for index, i in df_test.iterrows():
    cl1 = classes[1].pdf(i.iloc[[1, 2]])
    cl2 = classes[2].pdf(i.iloc[[1, 2]])
    cl3 = classes[3].pdf(i.iloc[[1, 2]])
    cl4 = classes[4].pdf(i.iloc[[1, 2]])
    prediction_test.append(np.array([cl1,cl2,cl3,cl4]).argmax()+1)

prediction_train = []
for index, i in df_train.iterrows():
    cl1 = classes[1].pdf(i.iloc[[1, 2]])
    cl2 = classes[2].pdf(i.iloc[[1, 2]])
    cl3 = classes[3].pdf(i.iloc[[1, 2]])
    cl4 = classes[4].pdf(i.iloc[[1, 2]])
    prediction_train.append(np.array([cl1,cl2,cl3,cl4]).argmax()+1)
    # print(np.array([cl1,cl2,cl3,cl4]).argmax()+1)

from sklearn.metrics import accuracy_score
import scipy.linalg as la
print("accuracy for training set: ", accuracy_score(df_train["Label"], prediction_train))
print("accuracy for testing set: ", accuracy_score(df_test["Label"], prediction_test))

count = 0
for i in range(len(df_train["Label"])):
  if df_train["Label"][i] == prediction_train[i]:
    count += 1

countt = 0
for i in range(len(df_test["Label"])):
  if df_test["Label"][i] == prediction_test[i]:
    countt += 1

print("accuracy for training set: ", count/len(prediction_train))
print("accuracy for testing set: ", countt/len(prediction_test))

pred_df_train = df_train.copy()
pred_df_test = df_test.copy()
pred_df_train["Label"] = prediction_train
pred_df_test["Label"] = prediction_test

#colours
colors = ['red', 'blue', 'green', 'orange']
plt.figure(figsize=(8, 6))
# plotting the data with different colours and markers
for i, label in enumerate(unique_labels_train):
    class_data = pred_df_train[pred_df_train['Label'] == label]
    # plt.scatter(class_data['X_Coordinate'], class_data['Y_Coordinate'],
    #             color=colors[i], label=f'Class {str(label)[0]}', alpha=0.6, marker='o')

    theta = np.linspace(0, 2 * np.pi, 100)
    matrix = np.vstack((class_data["X_Coordinate"], class_data["Y_Coordinate"])).T
    column_means = np.mean(matrix, axis=0)
    covariance_matrix = np.cov(matrix, rowvar=False)

    # Eigen decomposition for the ellipse
    eigvals, eigvecs = np.linalg.eigh(covariance_matrix)
    order = eigvals.argsort()[::-1]  # Sort eigenvalues
    eigvals, eigvecs = la.eig(covariance_matrix)#eigvals[order], eigvecs[:, order]

    # Scale eigenvalues for 2-sigma (95%) confidence interval
    ellipse_radii = 2 * np.sqrt(eigvals)  # 2-sigma scaling

    # Parametrize the ellipse
    ellipse_points = np.array([
        ellipse_radii[0] * np.cos(theta) * eigvecs[0, 0] + ellipse_radii[1] * np.sin(theta) * eigvecs[1, 0],
        ellipse_radii[0] * np.cos(theta) * eigvecs[0, 1] + ellipse_radii[1] * np.sin(theta) * eigvecs[1, 1]
    ])
    plt.plot(column_means[0] + ellipse_points[0, :], column_means[1] + ellipse_points[1, :], color=colors[i], label=f'Class {int(label)}')



for i, label in enumerate(unique_labels_test):
    class_data = pred_df_test[pred_df_test['Label'] == label]
    plt.scatter(class_data['X_Coordinate'], class_data['Y_Coordinate'],
                color=colors[i], label=f'Class {str(label)[0]}', alpha=0.6, marker='^')


plt.title('2c Plotting the datasets with different markers and \nthe classes with different colours. \nTriangle markers are from test set, ellipses are from train set')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.savefig('../plots/2c.png', bbox_inches='tight')
plt.show()







# dataset shape
df_test["Label"].shape
df_train["Label"].shape
# counting the classes
df_train["Label"].value_counts()
df_test["Label"].value_counts()


#colours
colors = ['red', 'blue', 'green', 'orange']
plt.figure(figsize=(8, 6))
# plotting the data with different colours and markers
for i, label in enumerate(unique_labels_train):
    class_data = df_train[df_train['Label'] == label]
    plt.scatter(class_data['X_Coordinate'], class_data['Y_Coordinate'],
                color=colors[i], label=f'Class {str(label)[0]}', alpha=0.6, marker='o')

for i, label in enumerate(unique_labels_test):
    class_data = df_test[df_test['Label'] == label]
    plt.scatter(class_data['X_Coordinate'], class_data['Y_Coordinate'],
                color=colors[i], label=f'Class {str(label)[0]}', alpha=0.6, marker='^')


plt.title('1c Plotting the datasets with different markers and \nthe classes with different colours. \nCircles are from train set, triangles are from test set')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.savefig('../plots/1c.png', bbox_inches='tight')
plt.show()

plt.cla()
plt.scatter(x=df_train["X_Coordinate"],y=df_train["Y_Coordinate"],label=df_train["Label"], color="b")
plt.scatter(x=df_test["X_Coordinate"],y=df_test["Y_Coordinate"],label=df_test["Label"], color="b")
plt.title("single colour whole dataset")
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.savefig('../plots/1b.png', bbox_inches='tight')
plt.show()

plt.cla()
plt.figure()
df_train.plot()
plt.legend(loc='best')
plt.show()





################# testing:
# Because I always forget how to do this.
# https://gist.github.com/gwgundersen/087da1ac4e2bad5daf8192b4d8f6a3cf
# Credit: https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/

import numpy as np
import matplotlib.pyplot as plt
from   matplotlib import cm
from   mpl_toolkits.mplot3d import Axes3D
from   scipy.stats import multivariate_normal

# Our 2-dimensional distribution will be over variables X and Y
N = 60
X = np.linspace(-3, 3, N)
Y = np.linspace(-3, 4, N)
X, Y = np.meshgrid(X, Y)

# Mean vector and covariance matrix
mu = np.array([-2.42459259, 0.74933983])
Sigma = np.array([[ 7.07803297, 4.37427947],[ 4.37427947, 12.14063639]])

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

# The distribution on the variables X, Y packed into pos.
F = multivariate_normal(mu, Sigma)
Z = F.pdf(pos)

# Create a surface plot and projected filled contour plot under it.
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                cmap=cm.viridis)

cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)

# Adjust the limits, ticks and view angle
ax.set_zlim(-0.15,0.2)
ax.set_zticks(np.linspace(0,0.2,5))
ax.view_init(27, -21)

plt.show()


print("test")