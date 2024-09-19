import numpy as np

x = np.array([[0,0],
              [0,1],
              [1,1],
              [1,0]])
T = np.array([[0],
              [1],
              [0],
              [1]])

def sigmoid(x):
    return 1/(1+np.exp(-x))
#sigmoid derivative normalde sigmoid(x) * (1-sigmoid(x)) halinde yazılır
#fakat burada zaten sigmoid_derivative fonksiyonuna girdi verirken sigmoid
#fonksiyonundan geçmiş bir değişken kullandığımız için gerek yoktur.
def sigmoid_derivative(x):
    return x*(1-x)
input_size = 2
hidden_size = 2
output_size = 1
W1 = np.random.uniform(size=(input_size,hidden_size))
B1 = np.random.uniform(size=(1, hidden_size))
W2 = np.random.uniform(size=(hidden_size, output_size))
B2 = np.random.uniform(size=(1, output_size))
def forward(X):
    Z1 = np.dot(X, W1) + B1
    A1 = sigmoid(Z1)          
    Z2 = np.dot(A1, W2) + B2  
    A2 = sigmoid(Z2)          
    return Z1, A1, Z2, A2
Z1, A1, Z2, A2 = forward(x)
def backward(x,T,Z1,A1,Z2,A2,learning_rate = 0.1):
    global W1,W2,B1,B2  
    #burada türev(mevcut)/türev(önceki) fonksiyonuyla öncekinin mevcuta duyarlılığı hesaplanır
    output_error = A2-T
    output_delta = output_error * sigmoid_derivative(A2)        
    hidden_error = output_delta.dot(W2.T)
    hidden_delta = hidden_error * sigmoid_derivative(A1)   
    w2_update = A1.T.dot(output_delta)
    b2_update = np.sum(output_delta , axis = 0, keepdims = True)
    w1_update = x.T.dot(hidden_delta)
    b1_update = np.sum(hidden_delta , axis = 0, keepdims = True)
    W2 -= learning_rate * w2_update
    B2 -= learning_rate * b2_update
    W1 -= learning_rate * w1_update
    B1 -= learning_rate * b1_update
for epoch in range(10000):
    Z1,A1,Z2,A2 = forward(x)
    backward(x,T,Z1,A1,Z2,A2)
    if epoch %1000 == 0:
        loss = np.mean(np.square(T-A2))
        print(f"epoch: {epoch} , loss: {loss} ")
Z1,A1,Z2,A2 = forward(x)    
print("Model accuracy degerleri: ")
print(A2)






