#input 4 output 2 range로 범위 설정

#1. data

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. modeling

x = np.array([range(10), range(21,31), range(83,93), range(102,112)]) # (10)이면 0 ~ 9 (21,31)이면 21~30

x = x.T

# y = np.array([[1,2,3,4,5,6,7,8,9,10],
#                [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9 ],
#                [9,8,7,6,5,4,3,2,1,0]])

y = np.array([range(10), range(21,31)])

y = y.T

model = Sequential()
model.add(Dense(5, input_dim = 4))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(2))

#3. compile

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x,y, epochs = 2000, batch_size =2)

#4. evaluate, predict

loss = model.evaluate(x,y)
print('loss : ', loss)

result = model.predict([[9,30,92,111]])
print('[9,30,92,111]의 예측치는 : ', result)

'''
1/1 [==============================] - 0s 68ms/step
[9,30,92,111]의 예측치는 :  [[ 9.001249 29.999554]]
'''

