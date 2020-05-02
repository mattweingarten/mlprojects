import numpy as np
from sklearn import linear_model as lm

np.set_printoptions(precision=5,suppress=True, linewidth=125)



def set_up():
    raw_train = np.genfromtxt("./data/train.csv",delimiter=",",skip_header=1)[:,1:15]
    raw_train = raw_train[raw_train[:,0].argsort()]
    return raw_train



def fold(X,i):
    n = X.shape[0]
    m = n/10 + 1
    if( i <= 5 ):
        s = m
    else:
        s = m - 1

    vali = np.zeros((s,14))
    res = np.zeros((n-s,14))
    rescounter = 0
    valicounter = 0
    for l in range(n):
        choose = l % 10
        if(choose == i):
            vali[valicounter] = X[l]
            valicounter += 1
        else:
            res[rescounter] = X[l]
            rescounter += 1

    # np.savetxt("vali.csv",vali, delimiter=",",fmt='%1.10f',comments='')
    return res,vali
# def fold(X,i):
#     assert(i < 10)
#     n = X.shape[0]
#     width = X.shape[1]
#     s = n / 10
#     if(i == 9):
#         res= np.empty(((9 * s), width))
#         vali = X[(i * s):n]
#         for k in range(10):
#             if(k != i):
#                 res[(k*s):((k+1)*s)] = X[(k*s):((k+1)*s)]
#
#     else:
#         res= np.empty(((n - s), width))
#         vali = X[(i * s):n]
#         k = 0
#         for c in range(10):
#             if(c != i):
#                 if(k == 8):
#                     res[(k*s):(n-s)] = X[(c*s):n]
#                 else:
#                     res[(k*s):((k+1)*s)] = X[(c*s):((c+1)*s)]
#                 k += 1
#     print(res.shape, vali.shape)
#     return res,vali



#return weigths trained with ridge regression model
def closedformregression(train,l):
    y = train[:,0:1]
    X = train[:,1:15]
    XT = np.transpose(X)
    inv = np.linalg.inv(np.matmul(XT,X) + (np.identity(X.shape[1]) * l ))
    M = np.matmul(inv,XT)
    return np.matmul(M,y).flatten()


# def gradientregression(train,l):
#     y = train[:,0:1]
#     X = train[:,1:15]
#     XT

def skregression(train,l):
    y = train[:,0:1]
    X = train[:,1:15]
    reg = lm.Ridge(alpha=l)
    reg.fit(X,y)
    return reg.coef_[0]
#eval weigths of regression model with test data using RMSE
def eval(w,test):
    y = test[:,0:1]
    X = test[:,1:15]
    n = y.size
    sum = 0.0
    for i in range(n):
        sum += pow((y[i] - np.dot(w,X[i])) , 2)
    return pow((sum/n),0.5)[0]

# test Model with certain lambda on each fold and compute average
def testModel(X,l):
    sum = 0.0
    for i in range(10):
        train,test = fold(X,i)
        w = closedformregression(train,l)
        sum += eval(w,test)
    return sum/10


# X= set_up()
# res, vali = fold(X,5)
# print(vali)
def main():
    X= set_up()

    ls = np.array([0.01,0.1,1,10,100])
    for i in range(5):
        ls[i] =  testModel(X,ls[i])
    print(ls)
    np.savetxt("prediction.csv",ls, delimiter=",",fmt='%1.10f',comments='')


main()
