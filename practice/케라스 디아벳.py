from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#1.데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target

# print(x.shape, y.shape) #(442, 10) (442,) input_dim = 10

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9, shuffle=True, random_state=1137)

#2. 모델구성

model = Sequential()
#model.add(Dense(12,input_dim=10))
model.add(Dense(40,input_dim=10))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(1))

#3. compile

model.compile(loss = 'mae', optimizer = 'adam') 
model.fit(x_train, y_train, epochs = 100, batch_size =8)

#4. 평가,예측

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)      
                                       

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)