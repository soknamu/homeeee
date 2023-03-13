import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping

#1.데이터

datasets = load_iris()

x = datasets.data

y = datasets.target

#print(x.shape, y.shape) #(150, 4) (150,)
#1. 판다스
# import pandas as pd
# y = pd.get_dummies(y)
# print(y.shape) #(150, 3)

#2.사이킷런
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder()
# y = y.reshape(-1,1)
# y = ohe.fit_transform(y)
# print(y.shape) #(150, 3)

#3. 케라스
from keras.utils import to_categorical
y = to_categorical(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state= 2005)

print('y의 라벨값 :', np.unique(y)) #2개

#2. 모델구성

model = Sequential()
model.add(Dense(5,input_dim = 4))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일
es = EarlyStopping(monitor = 'val_acc', patience= 50, verbose= 1, mode = 'max', restore_best_weights= True)
model.complie(loss = 'binary_crossentropy', optimizer ='adam')
model.fit(x_train, y_train, epochs = 100, batch_size =32, verbose =1, callbacks =[es])

#4. 평가

loss = model.evaluate(x_test,y)
