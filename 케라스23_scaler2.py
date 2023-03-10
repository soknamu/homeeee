from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler,StandardScaler,RobustScaler
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

a=[]
for i in range(100):
    a.append(np.random.randint(-10,10))
a=np.array([a]).T
MMS=MinMaxScaler()
MAS=MaxAbsScaler()
SS=StandardScaler()
RS=RobustScaler()
MMS.fit(a)
MAS.fit(a)
SS.fit(a)
RS.fit(a)
aMMS=MMS.transform(a)
aMAS=MAS.transform(a)
aSS=SS.transform(a)
aRS=RS.transform(a)

plt.figure(0)
plt.plot(a)
plt.figure(1)
plt.subplot(2,2,1)
plt.plot(aMMS)
plt.title('aMMS',fontsize=10)
plt.subplot(2,2,2)
plt.plot(aMAS)
plt.title('aMAS',fontsize=10)
plt.subplot(2,2,3)
plt.plot(aSS)
plt.title('aSS',fontsize=10)
plt.subplot(2,2,4)
plt.plot(aRS)
plt.title('aRS',fontsize=10)
plt.legend()
plt.show()
print(aSS)
print(aRS)