import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
import pandas as pd
# 1. 데이터

path = './_data/dacon_diabetes/'
path_save = './_save/dacon_diabetes/'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)

#print(train_csv.shape) #(652, 9)
#print(test_csv.shape) #(116, 8)

#print(train_csv.isnull().sum()) # 결측치 x
x = train_csv.drop(['Outcome'],axis =1)
y = train_csv['Outcome']

#print(x.shape) #(652, 8)
#print(y.shape) #(652,)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size= 0.7, shuffle= True, random_state= 942, stratify=y) 

#print(x_train.shape, y_train.shape)
#print(x_test.shape, y_test.shape)


#2.모델구성

model = Sequential()
model.add(Dense(150, input_dim =8, activation= 'relu'))
model.add(Dense(135, activation= 'relu'))
model.add(Dense(120, activation= 'linear'))
model.add(Dense(105, activation= 'relu'))
model.add(Dense(90, activation= 'relu'))
model.add(Dense(90, activation= 'linear'))
model.add(Dense(75,activation = 'relu'))
model.add(Dense(60, activation= 'linear'))
model.add(Dense(45, activation= 'relu'))
model.add(Dense(30, activation= 'linear'))
model.add(Dense(15, activation= 'relu'))
model.add(Dense(15, activation= 'relu'))
model.add(Dense(1, activation= 'sigmoid')) #새로운 코드 sigmoid

#1.마지막에 시그모이드 준다.
#2.'binary_crossentropy' 를 넣어준다.

#3. 컴파일

es = EarlyStopping(monitor= 'val_loss', restore_best_weights= True, 
                   mode= 'min', patience= 90)

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', #새로운 코드 'binary_crossentropy'
              metrics=['accuracy','mse']) #두개이상은 list       #새로운 코드metrics=['accuracy','mse']
model.fit(x_train, y_train, epochs =1600,
                 batch_size =30, verbose =1, 
                 validation_split= 0.2, callbacks=[es])

#4. 평가, 예측

result  = model.evaluate(x_test, y_test)
print('result : ', result)

y_predict = np.round(model.predict(x_test))  #새로운코드 np.round 반올림.

acc =accuracy_score(y_test, y_predict)       #sklearn.metrics 에서 퍼옴.
print('acc :', acc)


#파일저장.
y_submit = np.round(model.predict(test_csv))
submission = pd.read_csv(path + 'sample_submission.csv', index_col = 0)
submission['Outcome'] = y_submit
submission.to_csv(path_save + 'submit_acc_0309_1547 .csv')


# [0.5620990991592407, 0.7602040767669678, 0.18914617598056793]
# acc : 0.7602040816326531
# model = Sequential()
# model.add(Dense(150, input_dim =8, activation= 'relu'))
# model.add(Dense(135, activation= 'relu'))
# model.add(Dense(120, activation= 'linear'))
# model.add(Dense(105, activation= 'relu'))
# model.add(Dense(90, activation= 'relu'))
# model.add(Dense(75))
# model.add(Dense(60, activation= 'linear'))
# model.add(Dense(45, activation= 'relu'))
# model.add(Dense(30, activation= 'linear'))
# model.add(Dense(15, activation= 'relu'))
# model.add(Dense(1, activation= 'sigmoid')) #새로운 코드 sigmoid
# random_state= 942,batch_size =12, verbose =1, validation_split= 0.2, patience= 40