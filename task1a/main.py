import numpy as np
np.set_printoptions(precision=5,suppress=True)



def set_up():
    raw_train = np.genfromtxt("./data/train.csv",delimiter=",",skip_header=1)
    X = raw_train[:,2:15]
    y = raw_train[:,1:2]
    w = np.zeros(y.size)
    return (X,y,w)

def fold(X):
    return np.array_split(X,10)

def gradient(X,w,y,l):
    return w

def validate_on(folds,index):
    for i in range(folds.size)

X,y,w = set_up()

folds = fold(X)

print(folds[0])
# print(X)
# print(y)
# print(w)
