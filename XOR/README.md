# Simple Neural Network for XOR Problem

This project implements a simple feedforward neural network to solve the XOR problem using NumPy. The network has one hidden layer and uses the sigmoid activation function. Below are the key steps of the project:

## Key Steps in the Project

### 1. Data Setup
- The input data `x` is the XOR truth table, with corresponding target outputs `T`.
- XOR problem is a classification problem, where inputs (0,0), (0,1), (1,0), and (1,1) are expected to output `0`, `1`, `1`, and `0`, respectively.

### 2. Sigmoid Activation and Derivatives
- The `sigmoid` function is used as the activation function for both hidden and output layers.
- The derivative of the sigmoid function is used during backpropagation to update the network weights and biases.

### 3. Neural Network Architecture
- The network consists of:
  - Input layer: 2 units (for the 2 features of XOR problem)
  - Hidden layer: 2 units
  - Output layer: 1 unit
- The weight matrices `W1` and `W2`, as well as the bias vectors `B1` and `B2`, are randomly initialized.

### 4. Forward and Backward Propagation
- **Forward propagation**:
  - Calculates the activations of the hidden and output layers using the dot product of inputs and weights, followed by the sigmoid function.
- **Backward propagation**:
  - Computes the errors at the output and hidden layers and updates the weights and biases using gradient descent.

### 5. Training Loop
- The network is trained for 10,000 epochs.
- During each epoch, forward and backward propagation are performed, and the loss is calculated using Mean Squared Error (MSE).
- Every 1,000 epochs, the loss is printed to monitor training progress.

### 6. Model Accuracy
- After training, the model's predictions for the XOR inputs are printed to evaluate accuracy.

