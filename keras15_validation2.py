from tensorflow.python.keras.models import Sequential #파이썬을 붙이면 자동완성됨.
from tensorflow.python.keras.layers import Dense
import numpy as np

#1. 데이터
x_train = np.array(range(1,17))  #(10,1) -> 스칼라10개 벡터1개 input_dim =1
y_train = np.array(range(1,17))
# x_val = np.array([14,15,16])
# y_val = np.array([14,15,16])     #  여기까지가 train데이터다. 그중에 3개가 또 train데이터다.
# x_test = np.array([11,12,13])      #16개중 10개가 train = val 13개, 3개는 테스트
# y_test = np.array([11,12,13])
#실습 : : 잘라봐!
x_val = np.array([91,92,93,94])
y_val = np.array([91,92,93,94])
x_test = np.array([14,15,16,17])
y_test = np.array([14,15,16,17])


#2. 모델구성
model = Sequential()
model.add(Dense(12,activation = 'linear', input_dim =1))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train,y_train, epochs= 550, batch_size = 4, verbose = 1,
          validation_data=(x_val, y_val)) # validation_data= 데이터 수정. train 데이터 수정.

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss :' , loss)

result = model.predict([18])
print('18의 예측값은 : ', result)

# loss : 9.094947017729282e-13
# 18의 예측값은 :  [[17.999996]]

# loss : 3.183231456205249e-12
# 18의 예측값은 :  [[18.]]

import matplotlib.pyplot as plt
plt.scatter(x_train,y_train)
plt.plot(x_val, y_val, color = 'yellow')
plt.scatter(x_train, y_train, color = 'green')
plt.plot(x_test, y_test, color = 'pink')
plt.show()
