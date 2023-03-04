from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np # -> 생략해버림
#1 data

x= np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
y= np.array([1,2,3,4,5,6,7,8,9,10,13,15,31,32,23,45,41,56,44,53,56,58,59,63,76,56,54,43,65,43])

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle= True, random_state= 34)

#2 model

model = Sequential()
model.add(Dense(5,input_dim= 1))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(1))

#3 compile

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 100, batch_size = 32)

#4 evaluate

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x)  #y_predict = predict(x_test) -> model을 붙히지 않음 

# 나머지는 기억이 안남

#5. 시각화
import matplotlib.pyplot as plt   # -> 그림그리는 것을 단축키로 설장
plt.scatter(x,y)     # -> 현재 자료에있는 점들 표시
#plt.scatter(x,y_predict)         # ->점으로 찍는다.
plt.plot(x,y_predict, color = 'green')  # -> 선, 색깔변경
plt.show()                        # -> 그림실행 명령어
