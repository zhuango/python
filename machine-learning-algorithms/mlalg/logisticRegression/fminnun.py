import numpy as np
import scipy
from scipy.optimize import minimize, rosen, rosen_der
def anotherfun(allparams):
    pass

def residue_alternative(allparams, inshape, xdata, ydata):
    m, n = inshape
    chi2 = 0.0
    anotherfun(allparams)
    for i in range(0,len(xdata)):
        x = xdata[i]
        y = 0
        for j in range(len(x)):
            idx = int(x[j]) * n +  j #Double check this to 
            y = y-allparams[idx]     #make sure it does what you want
            chi2 = chi2 + (ydata[i]-y)**2
    print(chi2)
    return chi2
def sigmoid(z):
    result = numpy.zeros(z.shape);
    result = 1.0 / (1.0 + numpy.exp(-z));
    return result;
def costFunction(theta,X, Y, lamda):
    m = len(Y);
    n = len(theta)
    J = 0;

    z = sigmoid(numpy.dot(X, theta));
    J = 1.0 / m * numpy.sum(-Y * numpy.log(z) - (1 - Y) * numpy.log(1 - z)) + lamda / (2.0 * m) * numpy.sum(theta * theta)
    #print(J)
    return J

numseq = ['0012000', '0112000', '0212000', '0312000', '1012000', '1112000',                                                                                   '1212000', '1312000', '2012000', '2112000', '2212000', '2312000', '3012000', '3112000',          '3212000', '3312000', '0002000', '0022000', '0032000', '1002000', '1022000', '1032000',     '2002000', '2022000', '2032000', '3002000', '3022000', '3032000', '0010000', '0011000', '0013000', '1010000', '1011000', '1013000', '2010000', '2011000', '2013000', '3010000', '3011000', '3013000', '0012100', '0012200', '0012300', '1012100', '1012200', '1012300', '2012100', '2012200', '2012300', '3012100']
prob = [-0.66474525640568083, -0.49518440694747212, -0.49518440694747212, -0.49518440694747212, -0.78361598908750163, -0.66474525640568083, -0.66474525640568083, -0.66474525640568083, -0.66474525640568083, -0.49518440694747212, -0.49518440694747212, -0.49518440694747212, -0.66474525640568083, -0.49518440694747212, -0.49518440694747212, -0.49518440694747212, -0.49518440694747212, -0.49518440694747212, -0.49518440694747212, -0.66474525640568083, -0.66474525640568083, -0.66474525640568083, -0.49518440694747212, -0.49518440694747212, -0.49518440694747212, -0.49518440694747212, -0.49518440694747212, -0.49518440694747212, -0.49518440694747212, -0.49518440694747212, -0.49518440694747212, -0.66474525640568083, -0.66474525640568083, -0.66474525640568083, -0.49518440694747212, -0.49518440694747212, -0.49518440694747212, -0.49518440694747212, -0.49518440694747212, -0.49518440694747212, -0.49518440694747212, -0.49518440694747212, -0.49518440694747212, -0.66474525640568083, -0.66474525640568083, -0.66474525640568083, -0.49518440694747212, -0.49518440694747212, -0.49518440694747212, -0.49518440694747212]
x0 = -0.6 * np.ones((28, 1), dtype=np.double)
# [xopt, fopt, iter, funcalls, warnflag] = \
#     scipy.fmin(residue_alternative, x0, args=(x0.shape, numseq, prob),
#        maxiter = 100000,
#        maxfun  = 100000,
#        full_output=True,
#        disp=True)

#res = minimize(residue_alternative,x0, (x0.shape, numseq, prob), method='BFGS', options={'gtol': 1e-3, 'disp': False})
res = minimize(residue_alternative,x0, (x0.shape, numseq, prob), method='BFGS', options={'gtol': 1e-3, 'disp': False})
#res = minimize(rosen,x0, method='BFGS',jac = rosen_der, options={'gtol': 1e-6, 'disp': False})
#print(rosen_der(x0[0]).shape)

print(res.x)