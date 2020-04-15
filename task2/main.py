
#to produce a sample we get a confidence range of each LABEL field:

#[BaseExcess,


import numpy as np
import math
from sklearn import svm
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn import linear_model
import time
from sklearn.preprocessing import StandardScaler


np.set_printoptions(precision=5,suppress=True, linewidth=300)






def sigmoid(x):
    return 1/(1 + math.exp(-x))


def map_sigmoid(X):
    n,m = X.shape
    for i in range(n):
        for j in range(m):
            X[i][j] = sigmoid(X[i][j])
    return X
# add logic how to best fill in Nans
##add lots more to do (like add interpolation for missing in betweens
# and other criteria for a whole missing colums)
def missing_values(raw):
    n,m = raw.shape
    means = np.nanmean(raw,axis=0)
    stds = np.nanstd(raw,axis=0)

    for j in range(m):

        for i in range(n):
            if (math.isnan(raw[i][j])):
                # for mean:
                raw[i][j] = means[j]
                #for zero
                # raw[i][j] = 0
        scaler = StandardScaler(copy=False)
        scaler.fit(raw)
        raw = scaler.transform(raw)
    return raw


# turns the data into a 2 d array such that svm can take it in(each row is all data we have on 1 patient)
def flatten(raw_data):
    n,w = raw_data.shape
    res = np.zeros((n/12, 1 + 12 * (w - 2)))
    tw = w-2
    for i in range(res.shape[0]):
        res[i][0] = raw_data[i * 12][2]
        temp = np.zeros((12 * tw))
        for j in range(12):
            res[i][1 + j * tw] = raw_data[i*12 + j][1]
            res[i][(1 + j * tw + 1) :  (1 + (j+1) * tw)  ]  = raw_data[i*12+j][3:w]

    return res

def setup():
        raw_data = np.genfromtxt("./data/train_features.csv",delimiter=",",skip_header=1)
        raw_data = missing_values(raw_data)
        samples = flatten(raw_data)
        raw_labels = np.genfromtxt("./data/train_labels.csv",delimiter=",",skip_header=1)
        labels = raw_labels[:,1:11]
        return samples,labels


# def ksplit(k,samples,labels):
#     n = samples.shape[0]
#     for i in range(k):
#
#     return 0


def restructure_predict(raw_results):
    cols = len(raw_results)
    rows = raw_results[0].shape[0]
    result = np.zeros((rows,cols))
    for j in range(cols):
        for i in range(rows):
            result[i][j] = raw_results[j][i][0]
    return result

def train_model(samples,labels):
    return OneVsRestClassifier(svm.SVC(kernel='linear',decision_function_shape='ovr',probability=False,max_iter=10000)).fit(samples,labels)


def predict_csv(clf):
    raw_data = np.genfromtxt("./data/train_features.csv",delimiter=",",skip_header=1)
    raw_data = missing_values(raw_data)
    data = flatten(raw_data)
    return predict(data,clf)
    return restructure_predict(clf.predict_proba(data))

# def predict_proba(clf,data):
#     return restructure_predict(clf.predict_proba(data))

def predict_sigmoid(clf,data):
    return map_sigmoid(clf.decision_function(data))


def predict(clf,data):
    return clf.predict(data)

def score(clf,tests,labels):
    return clf.score(tests,labels)




def main():
    set_up_start = time.time()
    samples,labels = setup()
    print("done setup")
    train_start = time.time()
    setup_duration = train_start - set_up_start
    print "Preprocessing takes: %.1f second"  % setup_duration
    train_start = time.time()
    clf = train_model(samples[0:1000],labels[0:1000])
    print("done training")
    train_duration = time.time() - train_start
    print "Training takes: %.1f second"  % train_duration
    print(predict_sigmoid(clf,samples[10001:10010]))
    print(predict(clf,samples[10001:10010]))
    print(labels[10001:10010])
    print(score(clf,samples[15000:18000],labels[15000:18000]))






# def practice():
#      X = np.array([
#                         [1,3,4,120,29],
#                         [2,3,27,155,18],
#                         [4,1,11,110,28]
#                   ])
#      y = np.array([
#                     [1,0,1,0,1],
#                     [0,1,0,1,0],
#                     [0,0,0,0,1]
#                   ])
#
#      p = np.array([
#                         [0,1.5,4,131,19],
#                         [2,3,27,155,18],
#                         [4,1,11,110,28]
#                    ])
#      print(X)
#      scaler = StandardScaler(copy=False)
#      scaler.fit(X)
#      X = scaler.transform(X)
#      print(X)
#
#      inner_clf = svm.SVC(kernel='linear',probability=False,decision_function_shape='ovo')
#
#      clf = OneVsRestClassifier(inner_clf).fit(X,y)
#      raw_res = map_sigmoid(clf.decision_function(p))
#      print(raw_res)
#      # print(restructure_predict(raw_res))
#
#      print(clf.predict(p))

# practice()

main()
