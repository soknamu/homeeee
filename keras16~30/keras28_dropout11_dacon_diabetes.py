import numpy as np
from tensorflow.python.keras.models import Sequential,Model 
from tensorflow.python.keras.layers import Dense,LeakyReLU,Input,Dropout
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
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
        train_size= 0.7, shuffle= True, random_state= 111, stratify=y) 


# # 스케일러 4가지
scaler = MinMaxScaler() #많이 퍼저있는 것
# scaler = StandardScaler() #표준분포가 모여 있으면 stand
# # scaler = MaxAbsScaler() 절대값.
# # scaler = RobustScaler()

x_tr = scaler.fit_transform(x_train) #ㅌ-train에 맞춰서 바뀌어짐.
x_test = scaler.transform(x_test) #x트레인의 변환범위에 맞춰야되서 변환해준다.
test_csv = scaler.transform(test_csv) ##test_csv도 스케일러 해줘야됨.
# print(np.min(x_test),np.max(x_test))

#print(x_train.shape, y_train.shape)
#print(x_test.shape, y_test.shape)


#2.모델구성

input1 =Input(shape=(8,))
dense1 = Dense(32, activation= LeakyReLU(0.95))(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(40, activation= LeakyReLU(0.95))(drop1)
drop2 = Dropout(0.4)(dense2)
dense3 = Dense(32, activation= LeakyReLU(0.80))(drop2)
drop3 = Dropout(0.32)(dense3)
dense4 = Dense(16, activation= LeakyReLU(0.8))(drop3)   
drop4 = Dropout(0.2)(dense4)
dense5 = Dense(8, activation= LeakyReLU(0.8))(drop4)
output1 = Dense(1,activation= 'sigmoid')(dense5)
model = Model(inputs = input1, outputs = output1)
model.summary()

#1.마지막에 시그모이드 준다.
#2.'binary_crossentropy' 를 넣어준다.

#3. 컴파일

es = EarlyStopping(monitor= 'val_acc', restore_best_weights= True, 
                     mode= 'max', patience= 200)

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', #새로운 코드 'binary_crossentropy'
              metrics=['acc','mse']) #두개이상은 list       #새로운 코드metrics=['accuracy','mse']
hist = model.fit(x_train, y_train, epochs =1550,
                 batch_size =22, verbose =1, 
                 validation_split= 0.2,
                 callbacks =[es]
                 )

#4. 평가, 예측
results = model.evaluate(x_test,y_test)
print('results :', results)

y_predict = np.round(model.predict(x_test))

acc = accuracy_score(y_test, y_predict)
print('accuary_score : ', acc)

#파일저장.
y_submit = np.round(model.predict(test_csv))
submission = pd.read_csv(path + 'sample_submission.csv', index_col = 0)
submission['Outcome'] = y_submit
submission.to_csv(path_save + 'submit_acc_0314_0945 .csv')

from matplotlib import pyplot as plt
plt.subplot(1,2,1)
plt.plot(hist.history['val_loss'])
plt.title('binary_crossentropy')
plt.subplot(1,2,2)
plt.plot(hist.history['val_acc'])
plt.title('val_acc')
plt.show()

# loss :  21199.791015625
# r2 score : 0.3431978302310429
# RMSE는 : 145.60148955814972