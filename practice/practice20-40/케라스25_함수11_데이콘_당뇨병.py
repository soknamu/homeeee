from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input,Dense,LeakyReLU
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import pandas as pd
#1. 데이터

path = './_data/dacon_diabetes/'
save_path = './_save/dacon_diabetes/'

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)

# print(train_csv.shape) #(652, 9)
# print(test_csv.shape) #(116, 8)

x = train_csv.drop(['Outcome'],axis = 1)

y = train_csv['Outcome']


x_train, x_test, y_train, y_test = train_test_split(
    x,y,shuffle=True, random_state=644, train_size= 0.7 
)
 
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

# print(x_train.shape, x_test.shape) #(456, 8) (196, 8)
# print(y_train.shape, y_test.shape) #(456,) (196,)

#2. 모델링

input1 = Input(shape=(8,))
dense1 = Dense(64, activation= LeakyReLU(0.9))(input1)
dense2 = Dense(32, activation= LeakyReLU(0.9))(dense1)
dense3 = Dense(16, activation= LeakyReLU(0.9))(dense2)
dense4 = Dense(8, activation= LeakyReLU(0.9))(dense3)
dense5 = Dense(4, activation= LeakyReLU(0.9))(dense4)
output1 = Dense(1, activation= LeakyReLU(0.9))(dense5)

model = Model(inputs = input1, outputs = output1)

#3.컴파일 훈련
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'Val_acc', patience= 300, 
                   mode= 'max', verbose= 1, restore_best_weights=True)
model.compile(loss = 'binary_crossentropy', optimizer ='adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs =10000, 
          batch_size = 45, verbose = 1, validation_split =0.2,callbacks=[es]
          )

#4. 평가 예측
import numpy as np
results = model.predict(x_test, y_test)
print('results : ', results)

y_predict = model.predict(x_test)

y_test_acc = np.argmax(y_test, axis =1)
y_predict = np.argmax(y_predict, axis =1)

from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test_acc, y_predict)
print('acc: ', acc)

#제출

y_submit = np.round(model.predict(test_csv))
submission = pd.read_csv(path + 'sample_submission.csv', index_col =0)
submission['Outcome'] =y_submit
submission.to_csv(path_save + 'submit_acc_0313_1904.csv')


