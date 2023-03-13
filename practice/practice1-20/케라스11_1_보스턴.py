import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1 data

from sklearn.datasets import load_boston

datasets = load_boston()

x = datasets.data  #(506, 13) (506,)
y = datasets.target


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size= 0.7, random_state= 425, shuffle= True)


print(x.shape, y.shape)
#2 모델구성

model = Sequential()
model.add(Dense(64,input_dim = 13))
model.add(Dense(128))
model.add(Dense(72))
model.add(Dense(56))
model.add(Dense(42))
model.add(Dense(28))
model.add(Dense(14))
model.add(Dense(1))

#3. compile

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs =10000, batch_size = 100)

#4. evaluate

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test) #test 붙히기

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)

print('r2스코어는 : ' ,r2)

'''
loss :  25.786338806152344
1/5 [=====>........................] -5/5 [==============================] - 0s 3ms/step
r2스코어는 :  0.6947780395631878 mse로 했을때

loss :  3.3013181686401367
1/5 [=====>........................] -5/5 [==============================] - 0s 878us/step
r2스코어는 :  0.7232553696411569 mae로 했을때 epochs 1000

loss :  3.381695508956909
1/5 [=====>........................] -5/5 [==============================] - 0s 3ms/step
r2스코어는 :  0.7228796564710808 epochs 2000 

loss :  3.211498975753784
1/5 [=====>........................] -5/5 [==============================] - 0s 3ms/step
r2스코어는 :  0.7278836082713226 
'''
