
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler, MaxAbsScaler
from tensorflow.python.keras.callbacks import EarlyStopping
#1. 데이터 

path = './_data/ddarung/'
path_save = './_save/ddarung/'

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)

print(train_csv.shape) #(1459, 10)
print(test_csv.shape) #(715, 9)

print(train_csv.isnull().sum())

train_csv = train_csv.dropna()

print(train_csv.isnull().sum())

print(train_csv.shape) #(1328, 10)


x = train_csv.drop(['count'], axis= 1)

y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size= 0.7, shuffle= True, random_state= 942)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv) ##test_csv도 스케일러 해줘야됨.
print(np.min(x_test), np.max(x_test))

# print(x_train.shape, x_test.shape) #(929, 9) (399, 9)
# print(y_train.shape, y_test.shape) #(929,) (399,)

#2 모델구성

model = Sequential()
model.add(Dense(150, input_dim=9,activation= 'relu'))
model.add(Dense(135))
model.add(Dense(120,activation= 'relu'))
model.add(Dense(105))
model.add(Dense(90,activation= 'relu'))
model.add(Dense(75))
model.add(Dense(60,activation= 'relu'))
model.add(Dense(45))
model.add(Dense(30,activation= 'relu'))
model.add(Dense(13))
model.add(Dense(1,activation= 'linear'))

#3. 컴파일

es = EarlyStopping(monitor = 'val_loss', patience =300, mode = 'auto',
               verbose=1, restore_best_weights=True)

model.compile(loss = 'mse', optimizer= 'adam')
hist = model.fit(x_train, y_train, epochs = 15000, batch_size =22, 
                verbose = 1, validation_split= 0.2,
                callbacks= [es]
                )

# print("===================발로스===================")
# print(hist.history['val_loss'])
# print("===================발로스====================")p

#4 평가

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_predict, y_test)
print('r2 score :', r2)
def RMSE(y_predict, y_test) :
  return np.sqrt(mean_squared_error(y_predict, y_test))
rmse = RMSE(y_predict, y_test)
print('RMSE : ', rmse)

#print(test_csv.isnull().sum())
y_submit = model.predict(test_csv)
submission = pd.read_csv(path + 'submission.csv', index_col = 0)
submission['count'] = y_submit
submission.to_csv(path_save + 'submit_0313_1216 .csv')

# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.figure(figsize =(9,6))
# #plt.plot(y) #-> x는 순서대로 가기때문에 x는 명시안해도됨
# plt.plot(hist.history['loss'], marker = '.', c='red',label = 'loss') #-> 이것을 통해서 어느지점에서 loss가 줄어들고, 늘어나는지 알수 있음, 
# plt.plot(hist.history['val_loss'], marker = '.', c='blue',label = 'val_loss') 
#                                #또한 과적합 부분을 찾아서 줄일수도 있음.
# plt.title('따릉이 횟수') #표제목
# plt.xlabel('epochs') # x축 
# plt.ylabel('로스, 검증로스') #y축
# plt.legend() #범례표시 : 오른쪽 위에 뜨는 
# plt.grid()  # 그래프에 오목판처럼 축이 생김
# plt.show()

# loss :  1643.93310546875
# r2 score : 0.6457759826580607
# RMSE :  40.545445566623854 스케일러 x



