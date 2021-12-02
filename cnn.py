import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import cv2
#from tensorflow.keras import layers,datasets,models
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

image=cv2.imread(r'C:\Users\HP\Downloads\Transpo_G70_TA-518126.jpg')
print(image.shape)
input=cv2.resize(image,(32,32))
input.resize(1,32,32,3)
print(input.shape)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
labels=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
x_train=x_train/255
x_test=x_test/255


#cnn
cnn=keras.Sequential([
                  keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',input_shape=(32,32,3)),
                  keras.layers.MaxPooling2D((2,2)),

                  keras.layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu'),
                  keras.layers.MaxPooling2D((2,2)),  

                  keras.layers.Flatten(),
                  keras.layers.Dense(100,activation='relu'),
                  keras.layers.Dense(10,activation='softmax')
])
cnn.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
cnn.fit(x_train,y_train,epochs=10)
print(cnn.evaluate(x_test,y_test))
predicted=cnn.predict(x_test)
output=[labels[np.argmax(i)] for i in predicted]
#print(output)
print(labels[np.argmax(cnn.predict(input))])
