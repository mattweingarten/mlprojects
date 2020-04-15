
#to produce a sample we get a confidence range of each LABEL field:

#[BaseExcess,


import numpy as np
import math
from sklearn import svm
from sklearn.multioutput import MultiOutputClassifier
import time

np.set_printoptions(precision=5,suppress=True, linewidth=300)









# add logic how to best fill in Nans
##add lots more to do (like add interpolation for missing in betweens
# and other criteria for a whole missing colums)
def missing_values(raw):
    n,m = raw.shape
    means = np.nanmean(raw,axis=0)
    for j in range(m):
        for i in range(n):
            if (math.isnan(raw[i][j])):
                # for mean:
                # raw[i][j] = means[j]
                #for zero
                raw[i][j] = 0
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
    return MultiOutputClassifier(svm.SVC(kernel='linear',decision_function_shape='ovr',probability=True)).fit(samples,labels)


def predict_csv(clf):
    raw_data = np.genfromtxt("./data/train_features.csv",delimiter=",",skip_header=1)
    raw_data = missing_values(raw_data)
    data = flatten(raw_data)
    return predict(data,clf)
    return restructure_predict(clf.predict_proba(data))

def predict(clf,data ):
    return restructure_predict(clf.predict_proba(data))







def main():
    samples,labels = setup()
    print("done setup")
    train_start = time.time()
    clf = train_model(samples[0:500],labels[0:500])
    print("done training")
    duration = time.time() - train_start
    print "Training takes: %.2f second"  % duration
    print(predict(clf,samples[1001:1005]))
    compare = labels[5001:5005]
    print(compare)





# def practice():
#      X = np.array([
#                         [1,3,4,120,29],
#                         [2,3,27,155,18],
#                         [4,1,11,110,28]
#                   ])
#      y = np.array([
#                     [1,0,1,0,1],
#                     [0,1,0,1,0],
#                     [0,1,0,1,0]
#                   ])
#
#      p = np.array([
#                         [0,1.5,4,131,19],
#                         [2,3,27,155,18],
#                         [4,1,11,110,28]
#                    ])
#      inner_clf = svm.SVC(kernel='linear',probability=True,decision_function_shape='ovo')
#
#      clf = MultiOutputClassifier(inner_clf).fit(X,y)
#      raw_res = clf.predict_proba(p)
#      print(raw_res)
#      print(restructure_predict(raw_res))
#
#      print(clf.predict(p))

# practice()

main()
