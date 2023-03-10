# [과제]

# 3가지 원핫인코딩 방식을 비교할 것

#1. pandas의 get_dummies


import pandas as pd
y=pd.get_dummies(y)
print(y.shape)
#2. keras의 to_categorical
 
from tensorflow.keras.utils import to_categorical #tensorflow 빼도 가능.
y = to_categorical(y)
print(y.shape) #(178, 3)



#3. sklearn 의 OneHotEncoder

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = y.reshape(-1,1)
y = ohe.fit_transform(y).toarray()
print(y.shape)


# 미세한 차이를 정리하시오!!!
