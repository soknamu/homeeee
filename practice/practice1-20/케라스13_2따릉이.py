import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

#1 데이터

path = './_data/ddarung/'

train_csv = pd.read_csv(path + 'train.csv', index_col= 0) #scv라고 적음

test_csv = pd.read_csv(path + 'test.csv', index_col= 0)  #이거는 왜 틀린지 모르겠음
print(train_csv)
print(train_csv.columns)
print(train_csv.info())
print(train_csv.describe())

#1-1 결측치 제거

print(train_csv.isnull().sum()) # .(점)대신 , (쉼표) 씀
train_csv = train_csv.dropna()

print(train_csv.isnull().sum())
print(train_csv.info())

#1-2 x값과 y값

x = train_csv.drop(['count'],axis = 1)

y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state= 88, shuffle= True, train_size= 0.7)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

#2. 모델구성

model = Sequential()
model(Dense(5, input_dim = 9)) #input_dim 인데 _ 안씀
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

#3.컴파일, 훈련

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 100, batch_size =500, verbose = 1)

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_predict, y_test)
print('r2 score :', r2)

#submission.csv 만들기

print(test_csv.isnull().sum()) #isnull인데 isfull이라 적음
y_submit = model.predict(test_csv)
print(y_submit)

#submission = pd.read_csv(path +'submisson.csv',index_col = 0)
submission = pd.read_csv(path + 'submission.csv', index_col = 0)  #+인데 =을 붙임, submission인데 submisson 이라고 적음
print(submission)

submission['count'] = y_submit
print(submission)

submission.to_csv(path + 'submit_ 이름. csv')

#함수

def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)

print("RMSE : ",rmse)
