from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_california_housing

#1 데이터
datasets = fetch_california_housing()

x = datasets.data
y = datasets.target

#print(x.shape, y.shape) 

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
x,y, train_size = 0.3 , shuffle= True, random_state= 33)

#2. 모델구성

model = Sequential()
model.add(Dense(5, input_dim= 8))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 44, batch_size =800)

#4. 평가 예측

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print('r2의 스코어는 : ', r2)
