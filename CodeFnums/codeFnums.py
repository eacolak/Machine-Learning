import numpy as np
import cv2
import os 
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Flatten,BatchNormalization
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import pickle

path = "myData"

myList = os.listdir(path)
noOfClasses = len(myList)

images = []
classNo = []

for i in range(noOfClasses):
    myImageList = os.listdir( path + "\\"+str(i))
    for j in myImageList:
        img =cv2.imread(path + "\\"+str(i)+"\\" + j) 
        img = cv2.resize(img,(32,32))
        images.append(img)
        classNo.append(i)


images = np.array(images)
classNo = np.array(classNo)


#veri ayırma

x_train, x_test, y_train, y_test = train_test_split(images,classNo,test_size=0.5,random_state=42)
x_train, x_validation, y_train, y_validation = train_test_split(x_train,y_train,test_size=0.2,random_state=42)

print("images: ",images.shape)
print("x_train: ",x_train.shape)
print("x_Test: ",x_test.shape)
print("x_val: ",x_validation.shape)

#preprocess
def preProcess(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/225
    
    return img

x_train = np.array(list(map(preProcess,x_train)))
x_test = np.array(list(map(preProcess,x_test)))
x_validation = np.array(list(map(preProcess,x_validation)))

x_train = x_train.reshape(-1,32,32,1)
x_test = x_test.reshape(-1,32,32,1)
x_validation = x_validation.reshape(-1,32,32,1)

# Data Generator
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range = 0.1,
                             rotation_range = 10
                             )

dataGen.fit(x_train)

y_train = to_categorical(y_train,noOfClasses)
y_test = to_categorical(y_test,noOfClasses)
y_validation = to_categorical(y_validation,noOfClasses)

model = Sequential()
model.add(Conv2D(filters = 16,kernel_size =(3,3) ,activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units = 256,activation="sigmoid"))
model.add(Dropout(0.2))
model.add(Dense(units = noOfClasses,activation="softmax"))

model.compile(loss = "categorical_crossentropy", optimizer = ("adam"),metrics=["accuracy"])

batch_size = 250

hist = model.fit_generator(dataGen.flow(x_train,y_train,batch_size=batch_size ),
                                        validation_data = (x_validation,y_validation),
                                        epochs = 15,steps_per_epoch = x_train.shape[0]//batch_size,shuffle = 1)    

model.save('C:\\Users\\Asus\\SayiTanıma')




















