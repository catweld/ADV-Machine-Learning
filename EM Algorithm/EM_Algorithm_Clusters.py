#!/usr/bin/env python
# coding: utf-8

# In[92]:


import numpy as np
import math as math
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats
from scipy.stats import multivariate_normal, poisson, gamma, norm


def generate_data(n_data, means, covariances, weights, rates):
    n_clusters, n_features = means.shape
    data = np.zeros((n_data, n_features))
    poission_data = np.zeros(n_data)
    colors = np.zeros(n_data, dtype='str')
    for i in range(n_data):
        k = np.random.choice(n_clusters, size=1, p=weights)[0]
        x = np.random.multivariate_normal(means[k], covariances[k])
        data[i] = x
        poission_data[i] = np.random.poisson(rates[k])
        if k == 0:
            colors[i] = 'red'
        elif k == 1:
            colors[i] = 'blue'
        elif k == 2:
            colors[i] = 'green'

    return data, poission_data, colors


def plot_contours(X, S, means, covs, title, rates):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], s=S)

    delta = 0.025
    k = means.shape[0]
    x = np.arange(-2.0, 7.0, delta)
    y = np.arange(-2.0, 7.0, delta)
    X, Y = np.meshgrid(x, y)
    col = ['green', 'red', 'indigo']
    for i in range(k):
        mean = means[i]
        cov = covs[i]
        sigmax = np.sqrt(cov[0][0])
        sigmay = np.sqrt(cov[1][1])
        sigmaxy = cov[0][1] / (sigmax * sigmay)
        Z = mlab.bivariate_normal(X, Y, sigmax, sigmay, mean[0], mean[1], sigmaxy)
        plt.contour(X, Y, Z, colors=col[i], linewidths=rates[i], alpha=0.1)

    plt.title(title)
    plt.tight_layout()


class EM:

    def __init__(self, n_components, n_iter, tol, seed):
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.seed = seed
        self.responsibilities = np.zeros(300).reshape(100,3)

    def fit(self, X, S):

        # data's dimensionality
        self.n_row, self.n_col = X.shape

        # initialize parameters
        np.random.seed(self.seed)
        chosen = np.random.choice(self.n_row, self.n_components, replace=False)
        self.means = X[chosen]
        self.weights = np.full(self.n_components, 1 / self.n_components)
        if self.n_components == 3:
            self.rates = (np.mean(S) * np.ones(self.n_components) / np.array([1, 2, 3])[np.newaxis]).flatten()
        elif self.n_components == 2:
            self.rates = (np.mean(S) * np.ones(self.n_components) / np.array([1, 2])[np.newaxis]).flatten()
        shape = self.n_components, self.n_col, self.n_col
        self.covs = np.full(shape, np.cov(X, rowvar=False))
        new_covs = []
        for c in self.covs:
            new_covs = np.append(new_covs, np.diag(np.diag(c))) # making the covariances diagonal (question assumption)
        self.covs = np.array(new_covs).reshape(self.n_components, 2, 2)

        log_likelihood = -np.inf
        self.converged = False

        for i in range(self.n_iter):
            self._do_estep(X, S)
            self._do_mstep(X, S)
            log_likelihood_new = self._compute_log_likelihood(X, S)

            if (log_likelihood_new - log_likelihood) <= self.tol:
                self.converged = True
                break

            log_likelihood = log_likelihood_new

        return self
    
    def _do_estep(self, X, S):
        z = np.ones(100*3).reshape(100,3)
        e = .0000001
        tau = self.covs
        mu = self.means
        pi = self.weights
        lam = self.rates
        denom = np.zeros(100)
        num = np.zeros(300).reshape(100,3)
        print("components = ",self.n_components)
        #likelihood = np.log(norm.pdf()*poisson.pdf()*pi)
        for n in range(100):
            for k in range(self.n_components):
                num[n][k] = multivariate_normal(mu[k],tau[k]).pdf(X[n])*poisson(lam[k]).pmf(S[n])*pi[k]
                denom[n] += multivariate_normal(mu[k],tau[k]).pdf(X[n])*poisson(lam[k]).pmf(S[n])*pi[k]
        for n in range(100):
            for k in range(self.n_components):
                self.responsibilities[n,k] = num[n,k]/np.sum(denom[n])                                                                                         
        return self
            
    def _do_mstep(self, X, S):
        """M-step, update parameters"""
        print("means = ",self.means)
        mu = self.means
        pi = self.weights
        lam = self.rates
        N = np.zeros(self.n_components)
        resp = self.responsibilities
        for k in range(self.n_components):
            N[k] = np.sum(self.responsibilities[:,k])
            self.means[k,0] = (1/N[k])*(np.sum(resp[:,k]*(X[:,0])))
            self.means[k,1] = (1/N[k])*(np.sum(resp[:,k]*(X[:,1])))
            self.covs[k,0,0] = (1/N[k])*(np.sum(resp[:,k]*((X[:,0]-mu[k,0])*(X[:,0]-mu[k,0]).T)))
            self.covs[k,1,1] = (1/N[k])*(np.sum(resp[:,k]*((X[:,1]-mu[k,1])*(X[:,1]-mu[k,1]).T)))
            self.weights[k] = N[k]/100
            self.rates[k] = (1/N[k])*(np.sum(resp[:,k]*S)) 
    
    def _compute_log_likelihood(self, X, S):
        tau = self.covs
        mu = self.means
        pi = self.weights
        lam = self.rates
        for n in range(100):
            for k in range(self.n_components):
                num = multivariate_normal.pdf(X[n,:],mu[k,:],tau[k])*poisson(lam[k]).pmf(S[n])*pi[k]
                total = np.log(np.sum(num))
        """compute the log likelihood of the current parameter"""
        return total
        

# params for 3 clusters
means = np.array([
    [5, 0],
    [1, 1],
    [0, 5]
])

covariances = np.array([
    [[.5, 0], [0, .5]],
    [[.92, 0], [0, .91]],
    [[.5, 0.], [0, .5]]
])

weights = [1 / 4, 1 / 2, 1 / 4]

# params for 2 clusters
means_2 = np.array([
    [5, 0],
    [1, 1]
])

covariances_2 = np.array([
    [[.5, 0.], [0, .5]],
    [[.92, 0], [0, .91]]
])

weights_2 = [1 / 4, 3 / 4]

np.random.seed(3)

rates = np.random.uniform(low=.2, high=20, size=3)
print("Poisson rates for 3 components:")
print(rates)

rates_2 = np.random.uniform(low=.2, high=20, size=2)
print("Poisson rates for 2 components:")
print(rates_2)

# generate data
X, S, colors = generate_data(100, means, covariances, weights, rates)
plt.scatter(X[:, 0], X[:, 1], s=S, c=colors) 
#plt.show()

X_2, S_2, colors_2 = generate_data(100, means_2, covariances_2, weights_2, rates_2)
plt.scatter(X_2[:, 0], X_2[:, 1], s=S_2, c=colors_2)
#plt.show()


em = EM(n_components=3, n_iter=1, tol=1e-4, seed=1)
em.fit(X, S)

plot_contours(X, S, em.means, em.covs, '3 clusters, 1 iteration', em.rates)
plt.show()

em = EM(n_components=3, n_iter=50, tol=1e-4, seed=1)
em.fit(X, S)

plot_contours(X, S, em.means, em.covs, '3 clusters, 50 iterations', em.rates)
plt.show()

em_2 = EM(n_components=2, n_iter=1, tol=1e-4, seed=1)
em_2.fit(X_2, S_2)

plot_contours(X_2, S_2, em_2.means, em_2.covs, '2 clusters, 1 iteration', em_2.rates)
plt.show()

em_2 = EM(n_components=2, n_iter=100, tol=1e-4, seed=1)
em_2.fit(X_2, S_2)

plot_contours(X_2, S_2, em_2.means, em_2.covs, '2 clusters, 50 iterations', em_2.rates)
plt.show()


# In[ ]:




