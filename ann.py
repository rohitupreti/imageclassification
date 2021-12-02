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

#ann

ann=keras.Sequential([
                  keras.layers.Flatten(input_shape=(32,32,3)),
                  keras.layers.Dense(3000,activation='relu'),
                  keras.layers.Dense(1000,activation='relu'),
                  keras.layers.Dense(10,activation='sigmoid')
])
ann.compile(optimizer='SGD',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
ann.fit(x_train,y_train,epochs=5)

print(ann.evaluate(x_test,y_test))
predicted=ann.predict(x_test)
output=[labels[np.argmax(i)] for i in predicted]
#print(output)
print(labels[np.argmax(ann.predict(input))])
