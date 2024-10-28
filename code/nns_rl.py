import numpy as np
import scipy
from scipy.io import loadmat
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from collections import Counter

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

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for test_point in X_test:
            # Calculate distances to all training points
            distances = np.linalg.norm(self.X_train - test_point, axis=1)
            # Find the indices of k nearest neighbors
            k_nearest_indices = np.argsort(distances)[:self.k]
            # Find the labels of the k nearest neighbors
            k_nearest_labels = self.y_train[k_nearest_indices]
            # Assign the most common label among neighbors
            most_common_label = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common_label)
        return np.array(predictions)

# Load and prepare data
# Note: Assuming df_train and df_test are already prepared as in your previous code
X_train = np.array(df_train[["X_Coordinate", "Y_Coordinate"]])
y_train = np.array(df_train["Label"])
X_test = np.array(df_test[["X_Coordinate", "Y_Coordinate"]])
y_test = np.array(df_test["Label"])

# Instantiate and fit the classifier
k = 3  # Choose a value for k
knn = KNNClassifier(k=k)
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Set Accuracy: {accuracy * 100:.2f}%")

# Plotting the classified test set
plt.figure(figsize=(10, 6))
# Plot each class with a different color
for label in np.unique(y_test):
    plt.scatter(X_test[y_pred == label, 0], X_test[y_pred == label, 1], label=f"Class {label}")

plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title(f"k-NN Classification Results (k={k})")
plt.legend()
plt.grid(True)
plt.savefig('../plots/4e.png', bbox_inches='tight')
plt.show()


import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SmallNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(torch.tensor(x, dtype=torch.float32)))
        x = self.fc2(x)
        return x

# Example usage
input_size = 2
hidden_size = 20
output_size = 4

model = SmallNet(input_size, hidden_size, output_size)
criterion = nn.MSELoss() # Squared Error Cost Function
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


def to_one_hot(labels, num_classes=4):
    num_labels = labels.shape[0]
    one_hot = np.zeros((num_labels, num_classes))
    one_hot[np.arange(num_labels), labels.astype(int)-1] = 1
    return one_hot
X = np.array(df_train[["X_Coordinate","Y_Coordinate"]])
y = np.array(to_one_hot(df_train["Label"]))

X_test = np.array(df_test[["X_Coordinate","Y_Coordinate"]])
y_test = np.array(to_one_hot(df_test["Label"]))

train_losses = []
test_accuracies = []
ca_dict = {}
ca_dict[1.0] = []
ca_dict[2.0] = []
ca_dict[3.0] = []
ca_dict[4.0] = []


class KernelMultiClassNetwork:
    def __init__(self, sigma=1.0, learning_rate=0.01, num_classes=4, epochs=1000):
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.epochs = epochs
        self.alpha = None  # Alpha values will serve as weights

    def rbf_kernel(self, X1, X2):
        """Compute RBF (Gaussian) kernel."""
        dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        return np.exp(-dists / (2 * self.sigma**2))

    def fit(self, X, y):
        """Train using a one-vs-rest approach."""
        N = X.shape[0]
        self.alpha = np.zeros((self.num_classes, N))

        # Precompute the kernel matrix for training data
        K = self.rbf_kernel(X, X)

        # One-hot encoding of labels for multi-class
        # y_one_hot = np.eye(self.num_classes)[y.astype(int) - 1]

        losses = []
        for epoch in range(self.epochs):
            output = np.dot(self.alpha, K).T
            error = y - output
            for c in range(self.num_classes):
                self.alpha[c] += self.learning_rate * np.dot(error[:, c], K)
            loss = np.mean(np.square(error))
            losses.append(loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

        return losses

    def predict(self, X_train, X_test):
        """Predict classes for test data based on trained alphas."""
        K_test = self.rbf_kernel(X_test, X_train)  # Shape (N_test, N_train)
        output = np.dot(self.alpha, K_test.T)  # Align shapes (num_classes, N_train) x (N_train, N_test)
        predictions = np.argmax(output, axis=0) + 1  # +1 to adjust for class labels (1-4)
        return predictions

# Example data
# X_train = np.array([[1, 2], [1, -1], [-2, -2], [-3, 3], [3, -3]])  # Sample train data
# y_train = np.array([1, 2, 3, 4, 1])  # Sample labels (classes 1 to 4)
# X_test = np.array([[0, 0], [2, -1], [-1, 3]])  # Sample test data
# y_test = np.array([1, 2, 4])  # Test labels for accuracy check

# Initialize and train the kernel network
kernel_net = KernelMultiClassNetwork(sigma=1.0, learning_rate=0.01, num_classes=4, epochs=1000)
losses = kernel_net.fit(X, y)

# Predict and calculate accuracy
predictions = kernel_net.predict(X, X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Plotting the loss
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.show()














for epoch in range(200):
    # Train
    optimizer.zero_grad()
    outputs = model(torch.tensor(X, dtype=torch.float32))
    new_stuff = nn.functional.one_hot(torch.tensor(y).type(torch.int64), num_classes=4).float()
    loss = criterion(outputs, torch.tensor(y, dtype=torch.float32))
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    # Test
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs, 1)
        correct_per_class = [0] * 4

        for i, label in enumerate([1.0,2.0,3.0,4.0]):
            preds = test_outputs[np.argmax(y_test, axis=1) == label - 1]
            trues = y_test[np.argmax(y_test, axis=1) == label - 1]
            ca_dict[label].append(accuracy_score(np.argmax(trues, axis=1), np.argmax(preds, axis=1)))

        for i in range(len(y_test)):
            if predicted[i] == y_test[i]:
                correct_per_class[y_test[i]] += 1

        class_accuracies = [c / (np.sum(y_test == j) + 1e-8) for j,c in enumerate(correct_per_class)]
        test_accuracies.append(class_accuracies)
        print(f"Epoch {epoch+1}/{200}, Loss: {loss.item():.4f}")
        # Print class accuracies for each epoch (optional)
        print(f"Class accuracies: {class_accuracies}")



# Plot the loss curve
plt.figure()
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('../plots/4b_loss.png', bbox_inches='tight')
plt.show()


plt.figure(figsize=(10, 6))
for class_name, accuracies in ca_dict.items():
    epochs = range(0, len(accuracies))  # Assuming accuracies are per epoch
    plt.plot(epochs, accuracies, label=class_name)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy per Class')
plt.legend()
plt.grid(True)
plt.savefig('../plots/4b_ac.png', bbox_inches='tight')
plt.show()

unique_labels_test = df_test['Label'].unique()
colors = ['red', 'blue', 'green', 'orange']
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))

to = np.argmax(test_outputs, axis=1)
for i, label in enumerate(unique_labels_test):
    class_data = X_test[to == label-1]

    plt.scatter(class_data[:,0], class_data[:,1],
                color=colors[i], label=f'Class {str(label)[0]}', alpha=0.6, marker='^')

plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.savefig('../plots/4b_sc.png', bbox_inches='tight')
plt.show()


