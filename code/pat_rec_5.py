import numpy as np
import matplotlib.pyplot as plt


# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)


# Mean Squared Error
def mse(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))


# Define the two-layer perceptron
class TwoLayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        # Forward propagation
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.output = sigmoid(self.z2)
        return self.output

    def backward(self, X, y, learning_rate):
        # Backpropagation
        # Output layer error and delta
        error_output = y - self.output
        delta_output = error_output * sigmoid_derivative(self.output)

        # Hidden layer error and delta
        error_hidden = np.dot(delta_output, self.W2.T)
        delta_hidden = error_hidden * sigmoid_derivative(self.a1)

        # Update weights and biases
        self.W2 += np.dot(self.a1.T, delta_output) * learning_rate
        self.b2 += np.sum(delta_output, axis=0, keepdims=True) * learning_rate
        self.W1 += np.dot(X.T, delta_hidden) * learning_rate
        self.b1 += np.sum(delta_hidden, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, iterations, learning_rate):
        errors = []
        for i in range(iterations):
            # Forward pass
            output = self.forward(X)

            # Calculate error (Mean Squared Error)
            error = mse(y, output)
            errors.append(error)

            # Backward pass
            self.backward(X, y, learning_rate)

        return errors


# Prepare the data (from previous generation)
n_samples = 500
mu1 = np.array([0, 0])
mu2 = np.array([1, 1])
mu3 = np.array([0, 1])
mu4 = np.array([1, 0])
Sigma = np.array([[0.01, 0.0], [0.0, 0.01]])

sequence1 = np.random.multivariate_normal(mu1, Sigma, n_samples)
sequence2 = np.random.multivariate_normal(mu2, Sigma, n_samples)
sequence3 = np.random.multivariate_normal(mu3, Sigma, n_samples)
sequence4 = np.random.multivariate_normal(mu4, Sigma, n_samples)

# Stack all sequences together as input data (X)
X = np.vstack((sequence1, sequence2, sequence3, sequence4))

# Create some arbitrary labels for this problem (e.g., for binary classification or regression)
y1 = np.zeros((n_samples, 1))  # Label 0 for sequence1
y2 = np.ones((n_samples, 1))  # Label 1 for sequence2
y3 = np.zeros((n_samples, 1))  # Label 0 for sequence3
y4 = np.ones((n_samples, 1))  # Label 1 for sequence4

# Stack labels together (y)
y = np.vstack((y1, y2, y3, y4))

# Initialize the two-layer perceptron
input_size = 2  # 2D input
hidden_size = 2  # 2 neurons in the hidden layer
output_size = 1  # 1 output neuron
learning_rate = 0.1
iterations = 1000

perceptron = TwoLayerPerceptron(input_size, hidden_size, output_size)

# Train the perceptron and get the error curve
errors = perceptron.train(X, y, iterations, learning_rate)

# Plot the error curve
plt.plot(errors)
plt.title('Error Curve')
plt.xlabel('Iteration steps')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.show()

print("test")


# Prepare the data (from previous generation)
n_samples = 50
mu1 = np.array([0, 0])
mu2 = np.array([1, 1])
mu3 = np.array([0, 1])
mu4 = np.array([1, 0])
Sigma = np.array([[0.01, 0.0], [0.0, 0.01]])

sequence1 = np.random.multivariate_normal(mu1, Sigma, n_samples)
sequence2 = np.random.multivariate_normal(mu2, Sigma, n_samples)
sequence3 = np.random.multivariate_normal(mu3, Sigma, n_samples)
sequence4 = np.random.multivariate_normal(mu4, Sigma, n_samples)

# Stack all sequences together as input data (X)
X = np.vstack((sequence1, sequence2, sequence3, sequence4))

# Create some arbitrary labels for this problem (e.g., for binary classification or regression)
y1 = np.zeros((n_samples, 1))  # Label 0 for sequence1
y2 = np.ones((n_samples, 1))  # Label 1 for sequence2
y3 = np.zeros((n_samples, 1))  # Label 0 for sequence3
y4 = np.ones((n_samples, 1))  # Label 1 for sequence4

# Stack labels together (y)
y = np.vstack((y1, y2, y3, y4))


















import numpy as np
import matplotlib.pyplot as plt

# Mean vectors
mu1 = np.array([0, 0])
mu2 = np.array([1, 1])
mu3 = np.array([0, 1])
mu4 = np.array([1, 0])

# Covariance matrix
Sigma = np.array([[0.01, 0.0],
                  [0.0, 0.01]])

# Number of samples for each sequence
n_samples = 500

# Generate random sequences
sequence1 = np.random.multivariate_normal(mu1, Sigma, n_samples)
sequence2 = np.random.multivariate_normal(mu2, Sigma, n_samples)
sequence3 = np.random.multivariate_normal(mu3, Sigma, n_samples)
sequence4 = np.random.multivariate_normal(mu4, Sigma, n_samples)

# Plotting the sequences
plt.figure(figsize=(8, 8))
plt.scatter(sequence1[:, 0], sequence1[:, 1], label='Sequence 1 (μ1)', alpha=0.6)
plt.scatter(sequence2[:, 0], sequence2[:, 1], label='Sequence 2 (μ2)', alpha=0.6)
plt.scatter(sequence3[:, 0], sequence3[:, 1], label='Sequence 3 (μ3)', alpha=0.6)
plt.scatter(sequence4[:, 0], sequence4[:, 1], label='Sequence 4 (μ4)', alpha=0.6)
plt.legend()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Two-Dimensional Gaussian Random Sequences')
plt.grid(True)
plt.axis('equal')  # Equal scaling for both axes
plt.show()





# Logistic activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Mean Squared Error
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Initialize network parameters
np.random.seed(42)  # Set random seed for reproducibility
input_size = 2      # Number of inputs
hidden_size = 2     # Number of neurons in the hidden layer
output_size = 1     # Number of neurons in the output layer

# Randomly initialize weights and biases
W1 = np.random.randn(input_size, hidden_size)
b1 = np.random.randn(hidden_size)
W2 = np.random.randn(hidden_size, output_size)
b2 = np.random.randn(output_size)

# Training parameters
learning_rate = 0.1
epochs = 10000  # Number of iterations
error_history = []

# Training loop
for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(X, W1) + b1  # Linear combination in the hidden layer
    a1 = sigmoid(z1)  # Activation (sigmoid) for hidden layer
    z2 = np.dot(a1, W2) + b2  # Linear combination in the output layer
    y_pred = sigmoid(z2)  # Activation (sigmoid) for output layer

    # Compute the error (Mean Squared Error)
    error = mean_squared_error(y, y_pred)
    error_history.append(error)

    # Backpropagation
    # Output layer error
    dL_dy_pred = (y_pred - y)  # Derivative of loss w.r.t. prediction
    dL_dz2 = dL_dy_pred * sigmoid_derivative(z2)  # Chain rule for output layer

    # Hidden layer error
    dL_da1 = np.dot(dL_dz2, W2.T)  # Derivative of loss w.r.t. hidden activations
    dL_dz1 = dL_da1 * sigmoid_derivative(z1)  # Chain rule for hidden layer

    # Gradients for weights and biases
    dL_dW2 = np.dot(a1.T, dL_dz2)  # Gradient of loss w.r.t. W2
    dL_db2 = np.sum(dL_dz2, axis=0)  # Gradient of loss w.r.t. b2
    dL_dW1 = np.dot(X.T, dL_dz1)  # Gradient of loss w.r.t. W1
    dL_db1 = np.sum(dL_dz1, axis=0)  # Gradient of loss w.r.t. b1

    # Update weights and biases
    W2 -= learning_rate * dL_dW2
    b2 -= learning_rate * dL_db2
    W1 -= learning_rate * dL_dW1
    b1 -= learning_rate * dL_db1

    # Optionally print error every 1000 epochs
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Error: {error}')

# Plot the error curve
plt.figure(figsize=(8, 6))
plt.plot(error_history)
plt.title('Error Curve')
plt.xlabel('Iteration steps')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.show()




print("test")