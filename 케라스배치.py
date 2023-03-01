import numpy as np

x= np.array([1,2,3,4,5])
y= np.array([1,2,3,5,4,])\

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(5, input_dim = 1))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(1))


#compile

model.compile(loss = 'mse', optimizer ='adam')
model.fit(x, y, epochs = 100, batch_size =3)


loss = model.evaluate(x,y)
print('loss : ', loss)

result = model.predict([4])
print('[4]의 예측값은 :',  result)

