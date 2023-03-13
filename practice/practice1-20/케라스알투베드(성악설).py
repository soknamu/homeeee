# 신뢰도 바닥 케라스

#최적의 값 : 레이어 8줄,에포크 300 , 배치 3, 제곱값


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.data
x= np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y= np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

from sklearn.model_selection import train_test_split

x_test, x_train, y_test, y_train = train_test_split(
    x,y, train_size= 0.7, shuffle= True, random_state= 428)

#2. 모델구성

model = Sequential()
model.add(Dense(5, input_dim = 1))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일 훈련

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs =300, batch_size = 3)

#4. 평가,

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict =model.predict(x_test)

from sklearn.metrics import r2_score
r2 =r2_score(y_test, y_predict)
print('r2스코어는 : ', r2)

'''
loss :  21.26185417175293
1/1 [==============================] -1/1 [==============================] - 0s 72ms/step
r2스코어는 :  0.26579926334402737  

loss :  21.861806869506836
1/1 [==============================] -1/1 [==============================] - 0s 78ms/step
r2스코어는 :  0.24508203623404123 

loss :  20.831661224365234
1/1 [==============================] -1/1 [==============================] - 0s 84ms/step
r2스코어는 :  0.28065440000684894

loss :  20.975963592529297
1/1 [==============================] -1/1 [==============================] - 0s 72ms/step
r2스코어는 :  0.2756714338232862  

'''
