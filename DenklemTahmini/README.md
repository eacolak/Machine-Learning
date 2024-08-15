# Neural Network for Predicting Mathematical Formula Outputs

This project implements a neural network using NumPy and pandas to predict the results of a mathematical formula based on input variables. The network is designed with Leaky ReLU activation functions and dropout regularization to prevent overfitting. It also includes robust standardization and data augmentation for improved learning. Below are the key steps of the project:

## Key Steps in the Project

### 1. Data Generation and Augmentation
- Input data (`XveYinputlari`) and output data (`GercekCiktilar`) are initialized.
- Data augmentation is performed by generating 50 additional data points based on random inputs and a mathematical formula.

### 2. Data Preprocessing
- Robust standardization is applied to scale the input and output data using the interquartile range (IQR).
- Data is sorted based on the Euclidean distances from the origin.

### 3. Neural Network Architecture
- The neural network consists of two hidden layers and an output layer. The sizes are as follows:
  - Input layer: 2 units
  - Hidden layer 1: 32 units
  - Hidden layer 2: 128 units
  - Output layer: 1 unit
- The activation function used is Leaky ReLU, with dropout regularization applied after the second hidden layer.

### 4. Forward and Backward Propagation
- Forward propagation calculates the output predictions from the input data using matrix multiplication and Leaky ReLU activations.
- Backward propagation computes the errors and updates the weights and biases using gradient descent.

### 5. Training Loop
- The model is trained for 40,000 epochs using Mean Squared Error (MSE) as the loss function.
- The total loss is printed every 1000 epochs to monitor training progress.

### 6. Prediction and Evaluation
- After training, the network generates predictions (`Tahmin`) for the input data.
- The actual outputs (`Gerçek Çikti`) are compared with the predicted outputs, and the differences (`Fark`) are calculated.
- A pandas DataFrame is created to display the inputs, actual outputs, predicted outputs, and their differences.

