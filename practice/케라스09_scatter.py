import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
#1.data
x= np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y= np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])




x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle= True, random_state= 123)

#2.모델구성

model = Sequential()

model.add(Dense(5, input_dim =3))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(1))

#3. compile

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs =33, batch_size =3)

#4. evaluate, predict

loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x)

import matplotlib.pyplot as plt

plt.scatter(x,y)
plt.plot(x, y_predict, color = 'green')

plt.show()