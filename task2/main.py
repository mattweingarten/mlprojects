
#to produce a sample we get a confidence range of each LABEL field:

#[BaseExcess,


import numpy as np
import math
from sklearn import svm

np.set_printoptions(precision=5,suppress=True, linewidth=200)



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
                raw[i][j] = means[j]
                #for zero
                # raw[i][j] = 0
    return raw


# turns the data into a 2 d array such that svm can take it in(each row is all data we have on 1 patient)
def flatten(raw_data):
    n,w = raw_data.shape
    res = np.zeros((n/12, 2 + 12 * (w - 2)))
    tw = w-2
    for i in range(res.shape[0]):
        res[i][0] = raw_data[i * 12][0]
        res[i][1] = raw_data[i*12][2]
        temp = np.zeros((12 * tw))
        for j in range(12):
            res[i][2 + j * tw] = raw_data[i*12 + j][1]
            res[i][(2 + j * tw + 1) :  (2 + (j+1) * tw)  ]  = raw_data[i*12+j][3:w]

    return res

def setup():
        raw_data = np.genfromtxt("./data/train_features.csv",delimiter=",",skip_header=1)
        raw_data = missing_values(raw_data)
        samples = flatten(raw_data)
        raw_labels = np.genfromtxt("./data/train_labels.csv",delimiter=",",skip_header=1)
        labels = raw_labels[:,1]
        return samples,labels


# def ksplit(k,samples,labels):
#     n = samples.shape[0]
#     for i in range(k):
#
#     return 0


def train_model(samples,labels):
    clf = svm.SVC(probability=True,gamma='auto')
    clf.fit(samples,labels)
    return clf


def predict(clf):
    raw_data = np.genfromtxt("./data/train_features.csv",delimiter=",",skip_header=1)
    raw_data = missing_values(raw_data)
    data = flatten(raw_data)
    return clf.predict_proba(data)






def main():
    samples,labels = setup()
    print("done setup")
    clf = train_model(samples[0:1000],labels[0:1000])
    print("done training")
    # res = predict(clf)
    res = clf.predict(samples[5001:18000])
    print(np.sum(res))
    compare = labels[5001:5020]
    print(compare)
    # print(np.sum(res))

main()
