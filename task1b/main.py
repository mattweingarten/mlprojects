import numpy as np
np.set_printoptions(precision=10,suppress=True)
#R^5 -> R
#f([x1,x2,x3,x4,x5]) -> y


#setups data into X matrix and y vector

def setup():
    raw = np.genfromtxt("./data/train.csv",delimiter=",",skip_header=1)
    X = raw[:,2:7]
    y = raw[:,1:2].flatten()
    return X,y

#phi function for input vector x e R^5
def phi(x):
    return np.insert( np.array([
        x,
        np.square(x),
        np.exp(x),
        np.cos(x),

    ]).flatten(), 20,1)


def k(xi,xj):
    return np.dot(phi(xi), phi(xj))

def computeKernel(X):
    print("Computing kernel")
    n = np.size(X,0)
    K = np.empty([n,n])
    for i in range(n):
        for j in range(n):
            K[i][j] = k(X[i],X[j])
    return K

def closedForm(K,lam,y):
    print("Computing closed form")
    n = np.size(K,0)
    nlI = (lam * n) * np.identity(n)
    M = np.linalg.inv(np.add(K, nlI ))
    return np.dot(M, y)


def computeY(X,y,K):
X, y = setup()
xtest = X[0]
K = computeKernel(X)
alphas = closedForm(K, 1,y)




print(y.shape)
print(alphas)
print(alphas.shape)
# print(xtest)
# print(phi(xtest))
# print(X)
# print(y)
