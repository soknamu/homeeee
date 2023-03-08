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

x_train, x_test, y_train, y_test = train_test_split(x,y, 
            shuffle= True, train_size= 0.7, random_state= 5352)

#2.모델구성

model = Sequential()
model.add(Dense(30,input_dim = 8))
model.add(Dense(20,activation = 'relu'))
model.add(Dense(15,activation = 'linear')) # 디폴트값. linear는 있으나 마나. #케라스 1번 문제
model.add(Dense(12))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일

from tensorflow.python.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor = 'val_loss', patience =5, mode = 'min',
              verbose=1,
              restore_best_weights=True) 

model.compile(loss = 'mse', optimizer= 'adam')
hist = model.fit(x_train, y_train, epochs = 100, batch_size =32, 
                verbose = 1, validation_split= 0.2,
                callbacks= [es])

print("===================발로스===================")
print(hist.history['val_loss'])
print("===================발로스====================")

#4. 평가 예측

loss = model.evaluate(x_test,y_test)
print('loss : ',loss)

y_predict = model.predict(x_test)  #4등분(trian, val, test, predict) 
# predict: Y를 구하고 싶어서 Y의 값이 존재하지 않음.

r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)

def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)

print('RMSE : ', rmse)

y_submit = model.predict(test_csv)

submission = pd.read_csv(path + 'sampleSubmission.csv', index_col = 0)

submission['count'] = y_submit

submission.to_csv(path_save + 'submit_val_0308_1612 .csv')

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize =(9,6))
#plt.plot(y) #-> x는 순서대로 가기때문에 x는 명시안해도됨
plt.plot(hist.history['loss'], marker = '.', c='red',label = 'loss') #-> 이것을 통해서 어느지점에서 loss가 줄어들고, 늘어나는지 알수 있음, 
plt.plot(hist.history['val_loss'], marker = '.', c='blue',label = 'val_loss') 
                               #또한 과적합 부분을 찾아서 줄일수도 있음.
plt.title('자전거 수요') #표제목
plt.xlabel('epochs') # x축 
plt.ylabel('로스, 검증로스') #y축
plt.legend() #범례표시 : 오른쪽 위에 뜨는 
plt.grid()  # 그래프에 오목판처럼 축이 생김
plt.show()
