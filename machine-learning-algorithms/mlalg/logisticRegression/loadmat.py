import scipy.io as sio  
import matplotlib.pyplot as plt  
import numpy as np  
  
#matlab filename 
matfn=u'ex3data1.mat'  
data=sio.loadmat(matfn)  
  
plt.close('all')  
xi=data['X']  
yi=data['y']
print(xi)
print(yi)
# ui=data['ui']  
# vi=data['vi']  
# plt.figure(1)  
# plt.quiver( xi[::5,::5],yi[::5,::5],ui[::5,::5],vi[::5,::5])  
# plt.figure(2)  
# plt.contourf(xi,yi,ui)  
# plt.show()  
  
# sio.savemat('saveddata.mat', {'xi': xi,'yi': yi,'ui': ui,'vi': vi})  