import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1 data

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,
test_size =0.3, random_state=15, shuffle= True)

#2. modeling

model = Sequential()
model.add(Dense(10, input_dim = 1 ))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(1))

#3. compile, train

model.compile(loss = 'mse' , optimizer = 'adam') #adam은 최적화
model.fit(x_train, y_train, epochs = 2700, batch_size =1) #one F5에 4번돔(train 1~7이기때문에 2 2 2 1)

#4. evaluate, predict

loss = model.evaluate(x_test ,y_test)
print('loss : ', loss)

result = model.predict([4])
print('[4]의 예측값은 : ', result)



