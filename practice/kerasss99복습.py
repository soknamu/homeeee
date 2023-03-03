#데이터
import numpy as np
x = np.array([1,2,3])
y = np.array([1,2,3])


#모델구성
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#컴파일

model.compile(loss = 'mae', optimizer = 'adam')
model.fit(x,y, epochs= 100)

#평가예측

loss = model.evaluate(x,y)
print("loss : ", loss)

result = model.predict([4])
print("[4]의 예측값 : ", result)