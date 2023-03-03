import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([[1,2,3], # 쉼표와 array를 쓰기
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8]]
       )

y= np.array([1,2,3,4,5,6])

model = Sequential()
model.add(Dense(5, input_dim = 3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(1))

model.compile(loss = 'mae', optimizer = 'adam')
model.fit(x,y, epochs = 30 ,  batch_size =2)

loss = model.evaluate(x,y)
print('loss : ', loss)

result = model.predict([[6,7,8]])
print('[6,7,8]의 예측값 : ', result)


