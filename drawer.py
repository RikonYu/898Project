import pickle
import matplotlib.pyplot as plt
import numpy
def smooth(x):
    return [numpy.average(x[:i]) for i in range(len(x)-30)]
#a=pickle.load(open("constrewards+1.txt",'rb'))
b=pickle.load(open('alldirrewards.txt','rb'))
c=pickle.load(open('alldirrewards-rsrg.txt','rb'))
#plt.plot(smooth(a),'r',label='1 as Value')
plt.plot(smooth(b[:2200]),'b',label='Fixed location')
plt.plot(smooth(c),'r',label='Random location')

plt.xlabel('#epoch')
plt.ylabel('episodic reward')
plt.legend()
plt.show()