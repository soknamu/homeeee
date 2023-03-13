import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. data

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([10,9,8,7,6,5,4,3,2,1])

x_train = x[:7]
y_train = y[:7]
x_test = x[7:]
y_test = y[7:]

#2. modeling

model = Sequential()

model.add(Dense(5,input_dim = 1))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(1))

#3. compile fit

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 100, batch_size =3)

#4. evaulate, predict

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([4])
print('[4]의 예측값은 : ', result)
