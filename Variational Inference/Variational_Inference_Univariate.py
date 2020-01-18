#!/usr/bin/env python
# coding: utf-8

# In[1256]:


import pylab as pb
import numpy as np
from math import pi
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm, gamma
import matplotlib.mlab as mlab
from scipy.stats import multivariate_normal
import math as math


# In[1258]:


def Likelihood(X,mu,tau):
    return (tau/(2*np.pi))**(N/2)*np.exp((-tau/2)*np.sum(np.square(X-mu)))


# In[1259]:


def jointprob(X,mu,tau,a_0,b_0,lambda_0,const):
    log_cp_mu = (1/2)*np.log(lambda_0*tau)-(lambda_0*tau/2)*np.square(mu-mu_0)
    log_cp_tau = (a_0-1)*np.log(tau)-b_0*tau
    return np.log(Likelihood(X,mu,tau))+log_cp_mu+log_cp_tau+const


# In[1260]:


def qmu(lambda_0,mu_0,X,a_N,b_N,N):
    E_tau = a_N/b_N
    mu_N = ((lambda_0*mu_0+np.mean(X)*N)/(lambda_0+N))
    lambda_N = (lambda_0+N)*E_tau
    return norm(mu_N,1/lambda_N).pdf(mu_0)


# In[1261]:


def qtau(a_N,b_N,tau):
    return gamma.pdf(tau,a_N,scale = 1/b_N)


# In[1262]:


def muN(X,lambda_0,mu_0,N):
    return (lambda_0*mu_0+np.mean(X)*N)/(lambda_0+N)


# In[1263]:


def tauN(lambda_0,N,a_N,b_N):
    return (lambda_0+N)*a_N/b_N


# In[1264]:


def aN(a_0,N):
    return a_0+(N+1)/2


# In[1265]:


def bN(X,b_0,lambda_0,mu_0,mu_N,tau_N):
    return b_0 + lambda_0*((1/tau_N) + mu_N**2+mu_0**2-2*mu_N*mu_0)+ 1/2*np.sum(X**2+(1/tau_N) + mu_N**2-2*mu_N*X)


# In[ ]:


np.random.seed(1)
mu_0 = 0
lambda_0 = 1
a_0 = 1
b_0 = 1
N = 10
tau = np.random.gamma(a_0,b_0)
mu = np.random.normal(mu_0,np.linalg.inv(lambda_0*tau*np.eye(1))).reshape(-1)
X = np.random.normal(mu, 1/lambda_0, N)


# In[1266]:


mu_N = muN(X,lambda_0,mu_0,N)
a_N = aN(a_0,N)
tau_N = 1
b_N = bN(X,b_0,lambda_0,mu_0,mu_N,tau_N)
tau_N = tauN(lambda_0,N,a_N,b_N)


# In[1267]:


for i in range(1):
    b_N = bN(X,b_0,lambda_0,mu_0,mu_N,tau_N)
    tau_N = tauN(lambda_0,N,a_N,b_N)


# In[1268]:


x = np.linspace(-1, 2, 100)
y = np.linspace(-1, 2, 100)
X, Y = np.meshgrid(x, y)

Z = np.zeros(100*100).reshape(100,100)
for i in range(100):
    for j in range(100):
        Z[j][i] = qmu(lambda_0,mu_0,x[i],a_N,b_N,N)*qtau(a_N,b_N,y[j])

a_actual = a_0 + (N+1)/2
lambda_0_actual = lambda_0 + N
mu_0_actual = (lambda_0*mu_0+N*x)/(lambda_0+N)
b_actual = b_0 + (N*np.var(x) + (lambda_0*N*(x - mu_0)**2)/(lambda_0 + N))
Z_actual = norm.pdf(x,scale=1/a_actual)*gamma.pdf(y,a_actual,scale = 1/(b_actual[99])).reshape(-1,1)
approx = plt.contour(X, Y, Z,cmap = 'Reds_r',z=1-0)
posterior = plt.contour(X, Y, Z_actual)
plt.ylim(0, 2)  
plt.xlim(-1, 1)  
plt.show()


# In[ ]:





# In[ ]:




