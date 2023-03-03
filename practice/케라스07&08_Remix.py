import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. data

x = np.array([ Random.range(10), range(21,31), range(51,61)])
np.random.shuffle(x)
y = np.array([range(10), range(44,54), range(67,77)])

#y= np.array([[1,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,19,18,20],[21,22,23,24,25,26,27,28,29,30]])





print(x)
x = x.T
y = y.T
'''
#2. modeling

model = Sequential()

model.add(Dense(4, input_dim =3))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(9))
model.add(Dense(12))
model.add(Dense(15))
model.add(Dense(17))
model.add(Dense(12))
model.add(Dense(9))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(3))

#3.compile

model.compile(loss = 'mae', optimizer = 'adam')
model.fit(x,y, epochs =5000, batch_size =1)

#4. evaluate, predict

loss = model.evaluate(x,y)
print('loss : ', loss)

result = model.predict([[9,31,61]])
print('[9,31,61]의 값은 : ', result)


#값이 9와 53와 76에 근접 해야됨.
# 랜덤으로 돌리기전 의 값
#1/1 [==============================] - 0s 102ms/step
# [9,31,61]의 값은 :  [[ 9.118988 53.974087 77.43406 ]]

# 1/1 [==============================] - 0s 100ms/step
# [9,31,61]의 값은 :  [[ 8.772327 55.46763  79.67876 ]]
'''