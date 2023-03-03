import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#모델링
x = np.array(
   [[1,1],
    [2,1],
    [3,1],
    [4,1],
    [5,2],
    [6, 1.3],
    [7, 1.4],
    [8, 1.5],
    [9, 1.6],
    [10, 1.4],
    [11, 2.8]]
   ) #10행 2열 -> 2개의 특성을 가진 10개의 데이터
y = np.array([11,12,13,14,15,16,17,18,19,20,21]) # -> 삼성전자의 주가
model = Sequential()
model.add(Dense(5, input_dim=2))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#컴파일
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x,y, epochs = 1300, batch_size = 3)

#예측, 평가

loss = model.evaluate(x,y)
print("loss : ", loss)

result = model.predict([[11, 2.8]])
print("[11, 2.8]의 값은 " , result)