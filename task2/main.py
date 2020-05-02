
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
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

np.set_printoptions(precision=5,suppress=True, linewidth=300)

# FOLLOWING FUNCTION ARE HELPER METHODS ASSIST PREPROCESSING:
# LINEAR INTERPOLATION (ZEROING IF NO VALUES)
# SIGMOID FUNCTION
# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
def check_all_nan(vector):
    checker = np.vectorize(np.isnan)
    return np.all(checker(vector))

def zeroize(vector):
    for i in range(vector.size):
        vector[i] = 0
    return vector


def get_next_val(vector,index):
    count = 0
    for j in range(vector.size - index):
        if(np.isnan(vector[index + j]) == False):
            return (count,vector[index + j])
        count += 1
    return (count,np.nan)

def interp(vector):
   if(check_all_nan(vector)):
       return zeroize(vector)
   prev_val = np.nan
   for i in range(vector.size):
        nans,next_val = get_next_val(vector, i)
        if(np.isnan(vector[i])):
            if(np.isnan(prev_val)):
                vector[i] = next_val
            elif(np.isnan(next_val)):
                vector[i] = prev_val
            else:
                temp = prev_val +  (next_val - prev_val)/ (nans + 1)
                vector[i] = temp
                prev_val = temp
        else:
            prev_val = vector[i]
   return vector




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

def patient_pca(data,c):
    pca = PCA(n_components=c).fit(data)
    return np.reshape(pca.components_,(pca.components_.size,1))[0]

def get_patient_matrix(raw_data,i):
    n,w = raw_data.shape
    return raw_data[(12 * i): (12 *(i+1))][:,3:w]

def flatten_pca(raw_data,c):
    n,w = raw_data.shape
    res = np.zeros((n/12,c*(w-2)))
    temp = np.zeros((12,w-3))
    for i in range(n/12):
        temp = get_patient_matrix(raw_data,i)
        for j in range (w-3):
            temp[:,j] = interp(temp[:,j])

        res[i][0] = raw_data[i * 12][2] /100
        # print(patient_pca(temp,c))
        res[i][1:c*w-2] = patient_pca(temp,c)
    return res


# we use four values for each column
def flatten_min_max_slope(raw_data):
    n,w =  raw_data.shape
    c = w - 3
    means = np.nanmean(raw_data,axis=0)
    res = np.zeros((n/12,1 + c * 3))
    temp = np.zeros((12,c))
    for i in range(n/12):
        temp = get_patient_matrix(raw_data,i)
        for j in range (c):
            temp[:,j] = interp(temp[:,j])
        res[i][0] = raw_data[i * 12][2]
        for j in range (c):
            min = np.min(temp[:,j])
            max = np.max(temp[:,j])
            # if(min == 0):
            #     min = means[j + 3]
            # if(max == 0):
            #     max = means[j + 3]
            res[i][j*3+1] = min
            res[i][j*3+2] = max
            # res[i][j*3 + 2] = 0
            res[i][j*3+3] = (max-min)/12
    # print(res[0:5])
    return res


def scale(data):
    n,w = data.shape
    scaler = StandardScaler(copy=False)
    scaler.fit(data)
    data = scaler.transform(data)
    return data
# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------

def setup():
        raw_data = np.genfromtxt("./data/train_features.csv",delimiter=",",skip_header=1)
        # raw_data = missing_values(raw_data)
        # samples = flatten_pca(raw_data,1)
        samples = scale(flatten_min_max_slope(raw_data))
        raw_labels = np.genfromtxt("./data/train_labels.csv",delimiter=",",skip_header=1)
        labels = raw_labels[:,1:12]
        return samples,labels



#
# parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10], 'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}
# searcher = GridSearchCV(clf, parameters,scoring='roc_auc', cv=10,return_train_score=True)
# searcher.fit(train_array,y_train)
#
# # Report the best parameters
# print("Best CV params", searcher.best_params_)
#
# y_test_label=searcher.predict(train_array)
# print(y_test_label)
# print(searcher.best_score_)


def model():
    return OneVsRestClassifier(svm.SVC(
                                kernel='linear',
                                decision_function_shape='ovr',
                                C=0.01,
                                max_iter=10000))


def predict_csv(clf):
    raw_data = np.genfromtxt("./data/train_features.csv",delimiter=",",skip_header=1)
    raw_data = missing_values(raw_data)
    data = flatten(raw_data)
    return predict(data,clf)
    return restructure_predict(clf.predict_proba(data))


def predict_sigmoid(clf,data):
    return map_sigmoid(clf.decision_function(data))

def get_roc(samples,labels):
    return cross_val_score(model(),samples,labels, scoring='roc_auc',cv=5)


def main():
    set_up_start = time.time()
    samples,labels = setup()
    n,m = samples.shape
    print("--------------------------------\ndone setup")
    train_start = time.time()
    setup_duration = train_start - set_up_start
    print "Preprocessing takes: %.1f seconds\n--------------------------------"  % setup_duration
    roc = get_roc(samples,labels)
    print roc

    print("--------------------------------\ndone training")
    scoring_start = time.time()
    train_duration = scoring_start - train_start
    print "Training takes: %.1f seconds\n--------------------------------"  % train_duration

main()
