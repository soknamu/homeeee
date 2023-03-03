import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size = 0.3, 
    random_state= 12, 
    shuffle= True)
print(x)

#train_test_split 괄호안에 x,y,빼먹지말기.


#modeling

model = Sequential()
model.add(Dense(5, input_dim = 1))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(1))

#3. compile

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs =300, batch_size =3)

#4. predict, evaluate

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([4])
print('[4]의 예측값은 : ', result)