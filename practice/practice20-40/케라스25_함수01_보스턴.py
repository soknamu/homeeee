from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.callbacks import EarlyStopping
import numpy as np
datasets = load_boston()
#1. 데이터
x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x,
    y, train_size=0.7, shuffle=True,random_state=999)

#1-1. 스케일러

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_test),np.max(x_test))


#2. 모델링 (함수)

input1 = Input(shape = (13,),name = 'h1')
dense1 = Dense(30,name = 'h2')(input1)
dense2 = Dense(40,name = 'h3')(dense1)
dense3 = Dense(50)(dense2)
dense4 = Dense(30)(dense3)
dense5 = Dense(20)(dense4)
output1 = Dense(1)(dense5)
model = Model(inputs = input1, outputs = output1)

#3 컴파일 훈련
import time
es = EarlyStopping(monitor = 'Val_loss', 
                   patience= 300, mode= 'min', 
                   verbose = 1, 
                   restore_best_weights=True)
start_time = time.time()
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 1000, 
          verbose = 1, batch_size = 30, 
          validation_split =0.2,
          callbacks = [es])

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)



