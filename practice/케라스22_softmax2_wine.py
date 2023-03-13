from sklearn.datasets import load_wine
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
#1.데이터
datasets = load_wine()

x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(178, 13) (178,)

#1-1 원핫 인코딩

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = y.reshape(-1,1)
y = ohe.fit_transform(y).toarray()

print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,
    random_state= 321, stratify= y, 
    train_size= 0.7,shuffle= True
)

#2. 모델구성

model = Sequential()
model.add(Dense(5, input_dim = 13))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(3,activation= 'softmax'))

model.summary() #149개

#3. 컴파일
es = EarlyStopping(monitor = 'acc', patience= 100, restore_best_weights=True, mode = 'max')
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
import time

model.fit(x_train, y_train, epochs = 10, batch_size = 32, verbose = 0, callbacks = [es] )

#4. 평가
start_time = time.time()
results = model.evaluate(x_test, y_test)
print('results : ',results)
end_start = time.time()
y_predict = model.predict(x_test)

print(y_predict.shape)

y_test_acc = np.argmax(y_test, axis = 1)
y_predict = np.argmax(y_predict , axis =1)

print(y_predict.shape)

acc = accuracy_score(y_test_acc, y_predict)
print('acc score : ',  acc)

