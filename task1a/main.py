import numpy as np
np.set_printoptions(precision=5,suppress=True)



def set_up():
    raw_train = np.genfromtxt("./data/train.csv",delimiter=",",skip_header=1)
    return raw_train[:,1:15]


def fold(X,i):
    assert(i < 10)
    n = X.shape[0]
    width = X.shape[1]
    s = n / 10
    if(i == 9):
        res= np.empty(((9 * s), width))
        vali = X[(i * s):n]
        for k in range(10):
            if(k != i):
                res[(k*s):((k+1)*s)] = X[(k*s):((k+1)*s)]

    else:
        res= np.empty(((n - s), width))
        vali = X[(i * s):n]
        k = 0
        for c in range(10):
            if(c != i):
                if(k == 8):
                    res[(k*s):(n-s)] = X[(c*s):n]
                else:
                    res[(k*s):((k+1)*s)] = X[(c*s):((c+1)*s)]
                k += 1

    return res,vali



#return weigths trained with ridge regression model
def regression(train,l):
    y = train[:,0:1]
    X = train[:,1:15]
    XT = np.transpose(X)
    inv = np.linalg.inv(np.matmul(XT,X) + (np.identity(X.shape[1]) * l ))
    M = np.matmul(inv,XT)
    return np.matmul(M,y).flatten()

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
        w = regression(train,l)
        sum += eval(w,test)
    return sum/10



def main():
    X= set_up()
    np.random.shuffle(X)
    ls = np.array([0.01,0.1,1,10,100])
    for i in range(5):
        ls[i] =  testModel(X,ls[i])
    print(ls)
    np.savetxt("prediction.csv",ls, delimiter=",",fmt='%1.10f',comments='')


main()
