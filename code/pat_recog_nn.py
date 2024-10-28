import numpy as np
import scipy
from scipy.io import loadmat
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


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


from scipy.stats import norm, multivariate_normal


class ParzenWindowClassifier:
    def __init__(self, h, X, y):
        self.h = h  # Bandwidth
        self.X = X
        self.y = y


    def pred(self, X):
        preds = []
        for i,x in X.iterrows():
            post_probs = self.post_prob(x)
            preds.append(np.argmax(post_probs))
        return np.array(preds)

    def post_prob(self, x):
        num_classes = 4
        post_probs = []
        for c in range(num_classes):
            class_indices = np.where(self.y == c+1)[0]
            class_samples = self.X.iloc[class_indices,:]
            prior = 200 / len(self.X)
            likelihood = self.parzen_window_estimation(x, class_samples)
            post = prior * likelihood
            post_probs.append(post)
        return np.array(post_probs)

    def parzen_window_estimation(self, x, samples):
        # Calculate the multivariate normal density (Gaussian KDE)
        cov = np.identity(len(x)) * (self.h ** 2)  # Covariance matrix for Gaussian KDE
        likelihood = 0
        for i, sample in samples.iterrows():
            likelihood += multivariate_normal.pdf(x, mean=sample, cov=cov)
        return likelihood / len(samples)



# Create and train the classifier
h = 0.2  # Bandwidth parameter

classifier = ParzenWindowClassifier(h,df_train[["X_Coordinate","Y_Coordinate"]],df_train["Label"])

# Make predictions on the training set itself
preds_test = classifier.pred(df_test[["X_Coordinate","Y_Coordinate"]])
preds_train = classifier.pred(df_train[["X_Coordinate","Y_Coordinate"]])
df_test_pred = df_test[["X_Coordinate","Y_Coordinate"]].copy()
df_test_pred["Label"] = preds_test
unique_labels_test = df_test['Label'].unique()
colors = ['red', 'blue', 'green', 'orange']
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))

for i, label in enumerate(unique_labels_test):
    class_data = df_test_pred[df_test_pred["Label"] == label-1]
    plt.scatter(class_data['X_Coordinate'], class_data['Y_Coordinate'],
                color=colors[i], label=f'Class {str(label)[0]}', alpha=0.6, marker='^')

plt.title('2g Plotting the prediction of the parzen window classifier on the test set')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.savefig('../plots/2g.png', bbox_inches='tight')
plt.show()

print("accuracy for training set: ", accuracy_score(df_train["Label"], preds_train+1))
print("accuracy for testing set: ", accuracy_score(df_test["Label"], preds_test+1))





# mlp = MLPClassifier(hidden_layer_sizes=(64, 32),
#                     max_iter=1000, random_state=42)
# mlp.fit(X_train, y_train)
# y_pred = mlp.predict(X_test)
# print (mlp.score(X_train,y_train))
# plt.plot(mlp.loss_curve_)
# plt.plot(mlp.validation_scores_)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.2f}")













def to_one_hot(labels, num_classes=4):
    num_labels = labels.shape[0]
    one_hot = np.zeros((num_labels, num_classes))
    one_hot[np.arange(num_labels), labels.astype(int)-1] = 1
    return one_hot

print(df_train["Label"][10:])
print(to_one_hot(df_train["Label"])[10:])

print("toast")
class nn:
    def __init__(self, input_size, output_size, learning_rate=0.01):
        # Initialize weights and biases
        self.weights = np.random.randn(input_size, output_size) * 0.1  # Small random weights
        self.biases = np.zeros((1, output_size))
        self.learning_rate = learning_rate

    # Sigmoid activation function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Derivative of the sigmoid function
    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    # Forward pass
    def forward(self, X):
        # Linear combination (z) of inputs and weights plus biases
        self.z = np.dot(X, self.weights) + self.biases
        # Activation output (y_hat)
        self.y_hat = self.sigmoid(self.z)
        return self.y_hat

    # Compute the squared error cost
    def error_function(self, y_true, y_pred):
        return (y_true - y_pred) ** 2

    # Backpropagation
    def backprop(self, X, y_true):
        # Calculate the error (y_pred - y_true) * derivative of cost w.r.t. y_hat
        error = self.y_hat - y_true
        # Derivative of cost with respect to z
        delta = error * self.sigmoid_derivative(self.z)

        # Gradients for weights and biases
        grad_weights = np.dot(X.T, delta) / X.shape[0]
        grad_biases = np.mean(delta, axis=0, keepdims=True)

        # Update weights and biases
        self.weights -= self.learning_rate * grad_weights
        self.biases -= self.learning_rate * grad_biases
        return error

    # Train the neural network
    def train(self, X, y, epochs=1000):
        losses = []
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            # Backpropagation and weight update
            losses.append(self.backprop(X, y))
            # Calculate and print the cost every 100 epochs
            if epoch % 100 == 0:
                cost = self.error_function(y, y_pred)
                print(f"Epoch {epoch}: Cost = {np.mean(cost):.4f}")
        return losses


nn = nn(2,4,0.1)

df_train["Label"] = to_one_hot(df_train["Label"])
df_train = df_train.sample(frac = 1)
nn.train(df_train[["X_Coordinate","Y_Coordinate"]],to_one_hot(df_train["Label"]))
test_pred = nn.forward(df_test[["X_Coordinate","Y_Coordinate"]])

not_one_hot_pred = np.argmax(test_pred, axis=1)
df_test_pred = df_test[["X_Coordinate","Y_Coordinate"]].copy()
df_test_pred["Label"] = not_one_hot_pred
print("accuracy: ", accuracy_score(df_test["Label"], not_one_hot_pred))

unique_labels_test = df_test['Label'].unique()
colors = ['red', 'blue', 'green', 'orange']
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))

for i, label in enumerate(unique_labels_test):
    class_data = df_test_pred[df_test_pred["Label"] == label-1]
    plt.scatter(class_data['X_Coordinate'], class_data['Y_Coordinate'],
                color=colors[i], label=f'Class {str(label)[0]}', alpha=0.6, marker='^')

plt.title('3c Plotting the prediction of the neural network on the test set')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.savefig('../plots/3c.png', bbox_inches='tight')
plt.show()







print("test")
