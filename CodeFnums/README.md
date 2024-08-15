# Image Classification with CNN and Data Augmentation

This project demonstrates how to build and train a Convolutional Neural Network (CNN) model using Python and Keras to classify images into different categories. The model is developed with various preprocessing steps and data augmentation techniques to improve accuracy and reduce overfitting. Finally, the trained model is saved for future use.

## Key Steps in the Project

### 1. Data Loading and Preprocessing
- Images are loaded from different class directories.
- The images are resized to 32x32 pixels and converted to grayscale.
- Preprocessing includes histogram equalization and normalization to enhance image contrast and model performance.

### 2. Data Splitting
- The dataset is split into training, validation, and test sets using `train_test_split()` from `sklearn`.

### 3. Data Augmentation
- To avoid overfitting, `ImageDataGenerator` is used for data augmentation.
- Augmentation includes width/height shifts, zoom, and rotations.

### 4. Model Building
- A CNN is built with `Conv2D`, `MaxPooling2D`, and `Dense` layers.
- Dropout layers are added to prevent overfitting.
- The model uses ReLU and Sigmoid activation functions and ends with a Softmax layer for classification.

### 5. Model Training
- The model is compiled using the Adam optimizer and `categorical_crossentropy` loss.
- It is trained for 15 epochs using the augmented training data and validated with a validation set.

### 6. Model Saving
- After training, the model is saved to a specified directory for later use in image classification tasks.

