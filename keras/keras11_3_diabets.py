#[실습]
# R2 0.62 이상 데이터를 더 정확해야지 할수 있음

from sklearn.datasets import load_diabetes

#1.데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target

# print(x.shape, y.shape) #(442, 10) (442,) input_dim = 10

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state=67)

#2. 모델구성

model = Sequential()
#model.add(Dense(12,input_dim=10))
model.add(Dense(5,input_dim=10))
model.add(Dense(7))
model.add(Dense(8))
model.add(Dense(12))
model.add(Dense(21))
model.add(Dense(34))
model.add(Dense(46))
model.add(Dense(54))
model.add(Dense(64))
model.add(Dense(73))
model.add(Dense(83))
model.add(Dense(90))
model.add(Dense(103))
model.add(Dense(118))
model.add(Dense(118))
model.add(Dense(118))
model.add(Dense(118))
model.add(Dense(118))
model.add(Dense(128))
model.add(Dense(138))
model.add(Dense(136))
model.add(Dense(124))
model.add(Dense(120))
model.add(Dense(110))
model.add(Dense(95))
model.add(Dense(82))
model.add(Dense(79))
model.add(Dense(67))
model.add(Dense(53))
model.add(Dense(49))
model.add(Dense(38))
model.add(Dense(27))
model.add(Dense(20))
model.add(Dense(12))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

#3. compile

model.compile(loss = 'mse', optimizer = 'adam') 
model.fit(x_train, y_train, epochs = 1650, batch_size =15)

#4. 평가,예측

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)      
                                       

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)


