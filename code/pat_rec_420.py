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

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# squared error cost function and its derivative
def se(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))


def se_derivative(y_true, y_pred):
    return y_pred - y_true


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, hidden_layers=4, learning_rate=0.01):
        self.learning_rate = learning_rate

        # Initialize weights and biases for each layer
        self.weights = []
        self.biases = []

        # Input layer to first hidden layer
        self.weights.append(np.random.randn(input_size, hidden_size) * 0.1)
        self.biases.append(np.zeros((1, hidden_size)))

        # Hidden layers
        for _ in range(hidden_layers - 1):
            self.weights.append(np.random.randn(hidden_size, hidden_size) * 0.1)
            self.biases.append(np.zeros((1, hidden_size)))

        # Last hidden layer to output layer
        self.weights.append(np.random.randn(hidden_size, output_size) * 0.1)
        self.biases.append(np.zeros((1, output_size)))

    def forward(self, x):
        # Store activations for each layer
        activations = [x]

        # Forward pass through hidden layers
        for i in range(len(self.weights) - 1):
            x = sigmoid(np.dot(x, self.weights[i]) + self.biases[i])
            activations.append(x)

        # Forward pass through output layer
        x = sigmoid(np.dot(x, self.weights[-1]) + self.biases[-1])
        activations.append(x)

        return activations

    def backpropagate(self, activations, y_true):
        # Calculate output error
        y_pred = activations[-1]
        error = se_derivative(y_true, y_pred) * sigmoid_derivative(y_pred)

        # Gradient descent update for output layer
        deltas = [error]

        # Backpropagate through hidden layers
        for i in reversed(range(len(self.weights) - 1)):
            error = np.dot(error, self.weights[i + 1].T) * sigmoid_derivative(activations[i + 1])
            deltas.append(error)

        # Reverse deltas (to align with forward order)
        deltas.reverse()

        # Gradient descent weight and bias updates
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * np.dot(activations[i].T, deltas[i])
            self.biases[i] -= self.learning_rate * np.sum(deltas[i], axis=0, keepdims=True)

        return error

    def train(self, X, y, xtest, ytest, epochs=1000):
        losses = []
        ca_dict = {}
        ca_dict[1.0] = []
        ca_dict[2.0] = []
        ca_dict[3.0] = []
        ca_dict[4.0] = []
        for epoch in range(epochs):
            # Forward pass
            activations = self.forward(X)
            # Backpropagation
            losses.append(np.mean(se(y, activations[-1]).tolist()))
            out = self.forward(xtest)[-1]
            fw = np.argmax(out, axis=1)
            for i, label in enumerate([1.0,2.0,3.0,4.0]):
                preds = fw[np.argmax(ytest, axis=1) == label-1]
                class_target = np.argmax(ytest[np.argmax(ytest, axis=1) == label-1], axis=1)
                ca_dict[label].append(accuracy_score(class_target, preds))
            # Calculate and print loss every 100 epochs
            if epoch % 100 == 0:
                loss = se(y, activations[-1])
                print(f'Epoch {epoch}, Loss: {loss}')
        return losses, ca_dict

    def predict(self, X):
        return self.forward(X)[-1]


def to_one_hot(labels, num_classes=4):
    num_labels = labels.shape[0]
    one_hot = np.zeros((num_labels, num_classes))
    one_hot[np.arange(num_labels), labels.astype(int)-1] = 1
    return one_hot

X = np.array(df_train[["X_Coordinate","Y_Coordinate"]])
y = np.array(to_one_hot(df_train["Label"]))

X_test = np.array(df_test[["X_Coordinate","Y_Coordinate"]])
y_test = np.array(to_one_hot(df_test["Label"]))

# Initialize and train the network
nn = NeuralNetwork(input_size=2, hidden_size=8, output_size=4, hidden_layers=4, learning_rate=0.1)
losses, ca_dict = nn.train(X, y, X_test, y_test, epochs=1000)

# Test predictions
predictions = nn.predict(X)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))  # Adjust figure size as needed

for class_name, accuracies in ca_dict.items():
    epochs = range(1, len(accuracies) + 1)  # Assuming accuracies are per epoch
    plt.plot(epochs, accuracies, label=class_name)
plt.plot(epochs, losses, label = "loss")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Class Accuracies over Epochs")
plt.legend()  # Display the legend
plt.grid(True)  # Add a grid for better readability
plt.show()

print("\nPredictions:")
print(predictions)