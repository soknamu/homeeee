import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

#1. 데이터

path = './_data/ddarung/'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
# -> 'train_csv에 있는 index부분을 지움
# (실수하는 부분: train.csv 를 train_csv 씀 )

test_csv = pd.read_csv(path + 'test.csv', index_col = 0)

#print(train_csv) #[1459 rows x 10 columns]
#print(train_csv.shape) # (1459, 10)
#print(test_csv) #[715 rows x 9 columns]
#print(test_csv.shape) #(715, 9)
#-> 칼럼이 하나 차이남. 

#print(train_csv.info()) #데이터 종류를 보는 판다스 함수
#print(train_csv.describe())

######################################결측치 처리###############################################
#통 데이터일 때 결측치 처리를 한다. 분리 후 결측치 처리를 하게되면 데이터가 망가진다.

print(train_csv.isnull().sum()) #결측치 삭제전 개수

#isnull의 트루값이 몇개인지에 대한 합계(sum) ************자주 사용한다.

train_csv = train_csv.dropna() #결측치 삭제

print(train_csv.isnull().sum()) #결측치 삭제후 개수

#print(train_csv.info())

print(train_csv.shape) #(1328, 10)

## x, y 값

x = train_csv.drop(['count'], axis = 1)

#print(x) #[1328 rows x 9 columns]

y = train_csv['count']

print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,
    shuffle= True, random_state= 1234, train_size= 0.7)

print(x_train.shape, x_test.shape) #(929, 9) (399, 9)
print(y_train.shape, y_test.shape) #(929,) (399,)

#2. 모델구성

model = Sequential()
model.add(Dense(5, input_dim = 9))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(1))

#3. z컴파일, 훈련

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs =10, batch_size =100, verbose = 1)

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 =r2_score(y_test, y_predict)

print('r2 스코어는 : ', r2)

#4-1 함수

def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print('RMSE는 : ', rmse)


# 4-2 submission 만들기

#print(test_csv.isnull().sum()) 여기에도 결측치가 존재

y_submit = model.predict(test_csv)
print(y_submit)

submission = pd.read_csv(path + 'submission.csv', index_col = 0)
print(submission)

submission['count'] = y_submit
print(submission)

submission.to_csv(path + 'submission_name1.csv')

# loss :  6498.8984375 1/13 [=>............................] -13/13 [==============================] - 0s 2ms/step        
# r2 스코어는 :  0.10494093363002177      
# RMSE는 :  80.61574423914935