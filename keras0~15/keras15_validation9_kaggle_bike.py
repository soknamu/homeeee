import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

#1. 데이터

path = './_data/kaggle_bike/'
path_save = './_save/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)

print(train_csv.shape)
print(test_csv.shape)

print(train_csv.isnull().sum())

train_csv = train_csv.dropna()

print(train_csv.isnull().sum())

x = train_csv.drop(['count','casual','registered'],axis =1)

y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle= True, train_size= 0.7, random_state= 7)

#2.모델구성

model = Sequential()
model.add(Dense(121,input_dim = 8))
model.add(Dense(110,activation = 'relu'))
model.add(Dense(99))
model.add(Dense(88))
model.add(Dense(77))
model.add(Dense(66))
model.add(Dense(355,activation = 'linear')) # 디폴트값. linear는 있으나 마나. #케라스 1번 문제
model.add(Dense(55,activation = 'relu'))
model.add(Dense(44))
model.add(Dense(33))
model.add(Dense(22))
model.add(Dense(11))
model.add(Dense(1))

#3. 컴파일

model.compile(loss = 'mse', optimizer ='adam')
model.fit(x_train,y_train, epochs = 4000, batch_size =400, verbose =1, validation_split = 0.2)

#4. 평가 예측

loss = model.evaluate(x_test,y_test)
print('loss : ',loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)

def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)

print('RMSE : ', rmse)

y_submit = model.predict(test_csv)

submission = pd.read_csv(path + 'sampleSubmission.csv', index_col = 0)

submission['count'] = y_submit

submission.to_csv(path_save + 'submit_val_0308_0924 .csv')
