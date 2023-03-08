from sklearn.datasets import load_boston #load boston 을 가져오겠다.

#1. 데이터
datasets = load_boston() # 로드 보스턴을 데이터셋으로 부르겠다.
x = datasets.data      # 13개
y = datasets.target    # 1개

# print(x)
# print(y)
#FutureWarning: Function load_boston is deprecated; 
# `load_boston` is deprecated in 1.0 and will be removed in 1.2
# 우리는 로드보스턴 을 사용하지 않을 것이다. 1.0에서
# 4.9800e+00 정규화된 수 ex)1조 *1조는 오버클럭이 생김 그래서 수를 최댓값으로 나눠서 1을 못넘게함
# print(datasets)


#print(datasets.feature_names)

#['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT'] -> inputdim=13


# print(datasets.DESCR)

# instance : 506 예시
# attraribute 13개 y는 1000달러의 한개의 컬럼

# print(x.shape,y.shape)    #(506, 13) (506,) 스칼라 506 백터1개

###############[실습]##############
#1.train 0.7
#2. R2 0.8이상
###################################

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state= 58)
 
#2. 모델구성

model = Sequential()
model.add(Dense(125,input_dim=13))
model.add(Dense(115))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

#3. compile

model.compile(loss = 'mae', optimizer = 'adam') 
model.fit(x_train, y_train, epochs = 5000, batch_size =20, validation_split= 0.2)

# #학습시 test 데이터셋에서 지정한 수치만큼 검증용 데이터 분리
# 검증에 사용된 데이터는 학습하지 않음
# 예시 : train 75%, test 25%, validation_split = 0.2
# test 25%에서 test80%, validation 20%

print(x_test,y_test)
print(x_train,y_train)

#4. 평가,예측

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)


y_predict = model.predict(x_test)      
                                       
'''
import matplotlib.pyplot as plt

plt.scatter(x,y)

plt.show()
'''

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

# 5/5 [==============================] - 0s 1ms/step - loss: 3.0160
# loss :  3.0159571170806885
# 5/5 [==============================] - 0s 797us/step
# r2스코어 :  0.7781138267643735 epochs = 2000, batch_size =4,