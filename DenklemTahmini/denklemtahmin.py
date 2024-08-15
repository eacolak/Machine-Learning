import numpy as np
import random
import pandas as pd

def robusedstandardize(data):
    q1 = np.percentile(data, 25, axis=0)
    q3 = np.percentile(data, 75, axis=0)
    iqr = q3 - q1
    return (data - q1) / iqr

def dropout(x,dropout_rate):
    mask =np.random.binomial(1, 1-dropout_rate, size=x.shape) / (1 - dropout_rate)
    out = x * mask
    return out

XveYinputlari = np.array([[0,0],[0,1],[1,0],[2,1],[4,5],[5,3],[6,2],[7,3]])
GercekCiktilar = np.array([[3],[4],[3],[8],[108],[87],[79],[159]])

def veriArttirma(x,T):
    for i in range(50):
        a = random.randint(0, 200)
        b = random.randint(0, 200)
        formula = ((a**2)*b) + (b**2) + 3 
        
        x = np.append(x,[[a,b]],axis = 0)
        T = np.append(T,[[formula]], axis = 0)

    return x, T

XveYinputlari, GercekCiktilar = veriArttirma(XveYinputlari, GercekCiktilar)

# Verileri sıralama
# Öklid uzaklıklarını hesapla
uzakliklar = np.sqrt(XveYinputlari[:, 0]**2 + XveYinputlari[:, 1]**2)

# Uzaklıklara göre sıralama indekslerini al
siralama_indexleri = uzakliklar.argsort()


XveYinputlari = XveYinputlari[siralama_indexleri]
GercekCiktilar = GercekCiktilar[siralama_indexleri]


XveYinputlari = robusedstandardize(XveYinputlari)
GercekCiktilar = robusedstandardize(GercekCiktilar)

input_size = 2
hidden_size1 = 32
hidden_size2 = 128
output_size = 1

# Ağırlık ve bias başlatma
W1 = np.random.randn(input_size, hidden_size1)
B1 = np.zeros((1, hidden_size1))
W2 = np.random.randn(hidden_size1, hidden_size2)
B2 = np.zeros((1, hidden_size2))
W3 = np.random.randn(hidden_size2, output_size)
B3 = np.zeros((1, output_size))

# Leaky ReLU aktivasyonu
def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

# İleri yayılım
def forward(X, dropout_rate):
    Z1 = np.dot(X, W1) + B1
    A1 = leaky_relu(Z1)
    Z2 = np.dot(A1, W2) + B2
    A2 = leaky_relu(Z2)
    X = dropout(X , dropout_rate)
    Z3 = np.dot(A2, W3) + B3
    A3 = leaky_relu(Z3)
    return Z1, A1, Z2, A2, Z3, A3

# Geri yayılım
def backward(X, T, Z1, A1, Z2, A2, Z3, A3, learning_rate=0.001):
    global W1, W2, W3, B1, B2, B3
    
    # Çıkış katmanı hatası
    delta3 = (A3 - T) * leaky_relu_derivative(Z3)
    
    # İkinci gizli katman hatası
    delta2 = np.dot(delta3, W3.T) * leaky_relu_derivative(Z2)
    
    # İlk gizli katman hatası
    delta1 = np.dot(delta2, W2.T) * leaky_relu_derivative(Z1)
    
    # Ağırlık güncellemeleri
    dW3 = np.dot(A2.T, delta3)
    dB3 = np.sum(delta3, axis=0, keepdims=True)
    dW2 = np.dot(A1.T, delta2)
    dB2 = np.sum(delta2, axis=0, keepdims=True)
    dW1 = np.dot(X.T, delta1)
    dB1 = np.sum(delta1, axis=0, keepdims=True)
    
    W3 -= learning_rate * dW3
    B3 -= learning_rate * dB3
    W2 -= learning_rate * dW2
    B2 -= learning_rate * dB2
    W1 -= learning_rate * dW1
    B1 -= learning_rate * dB1

# Eğitim döngüsü
epochs = 40000
for epoch in range(epochs):
    Z1, A1, Z2, A2, Z3, A3 = forward(XveYinputlari, 0.1)
    backward(XveYinputlari, GercekCiktilar, Z1, A1, Z2, A2, Z3, A3)
    
    total_loss = 0
    total_loss += np.mean(np.square(A3 - GercekCiktilar))
    
    if epoch % 1000 == 0:
        loss = np.mean(np.square(A3 - GercekCiktilar))
        print(f"Epoch: {epoch}, Loss: {loss:.4f}")

# Tahmin
Z1, A1, Z2, A2, Z3, A3 = forward(XveYinputlari, 0.1)


# DataFrame oluşturma
df_tahmin = pd.DataFrame(A3, columns=['Tahmin'])
df_input = pd.DataFrame(XveYinputlari, columns=['X', 'Y'])
df_gercek_cikti = pd.DataFrame(GercekCiktilar, columns=['Gerçek Çikti'])
df_cikti_fark = pd.DataFrame(abs(GercekCiktilar-A3),columns=['Fark'])

# Tüm DataFrame'leri birleştirme
df_sonuc = pd.concat([df_input, df_gercek_cikti, df_tahmin,df_cikti_fark], axis=1)

print(df_sonuc)
print(f"Total Loss: {total_loss/epochs}")


# print("Model tahminleri:")
# print(np.around(A3, decimals=4))
  
# print("Modelin x,y değerleri: ")
# print(np.around(XveYinputlari, decimals=4))

# print("Modelin T değerleri: ")
# print(np.around(GercekCiktilar, decimals=4))
# print(f"Total Loss: {total_loss/epochs}")
