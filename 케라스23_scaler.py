from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler,StandardScaler,RobustScaler
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

a=np.arange(-10,10,0.1)
a=a.reshape(1, -1)
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
aSS=SS.fit(a)
aRS=RS.fit(a)

plt.subplot(2,2,1)
plt.plot(a.T,aMMS.T,label='MMS')
plt.subplot(2,2,2)
plt.plot(a.T,aMAS.T,label='MAS')
# plt.subplot(2,2,3)
# plt.plot(aSS,label='SS')
# plt.subplot(2,2,4)
# plt.plot(aRS,label='RS')
plt.legend()
plt.show()
print(aSS)
print(aRS)