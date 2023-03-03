#x값과 y값에 범위를 넣어보기


import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.data
# x =np.array([1,2,3,4,5,6,7,8,9,10])
x =np.array(range(13))

y = np.array(range(13))


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.4, random_state= 410, shuffle= True
) 


print(x_test, y_test)



#2. modeling

model = Sequential()
model.add(Dense(3, input_dim = 1))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(1))

#3. compile
model.compile(loss = 'mse',  optimizer = 'adam')
model.fit(x_train, y_train, epochs = 1234, batch_size = 3)

#4. evaluate, predict

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([3])
print('[3]의 예측값은 : ', result)

# 1/1 [==============================] - 0s 85ms/step
# [3]의 예측값은 :  [[3.0000002]]
