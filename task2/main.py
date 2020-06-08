
#to produce a sample we get a confidence range of each LABEL field:

#[BaseExcess,
# Needed libraries
import sklearn #Sklearn
from sklearn import datasets, linear_model
from sklearn.datasets import make_regression


#Libraries needed for imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#Libraries needed for models
#subtasks 1 and 2
from sklearn.svm import SVR
from sklearn.svm import SVC
#subtask3
from sklearn.linear_model import RidgeCV

#Libraries needed for plotters
from matplotlib import pyplot as plt

#Libraries needed for scoring
import sklearn.metrics as metrics

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
#subtask3
from sklearn.linear_model import RidgeCV

#Libraries needed for plotters
from matplotlib import pyplot as plt
#Libraries needed for scoring
import sklearn.metrics as metrics
np.set_printoptions(precision=5,suppress=True, linewidth=300)



#OMAR HELPER FUNCTIONS
# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
def data_fold_10(nds, f_nr, task):
    # """
    # Given a dataset, outputs two subsets: training set and test set. Test sets is given by the f_nr-th partition of the dataset,
    # meanwhile the training set is the remaining of the dataset
    # Parameters:
    # Input ds - numpy dataset to partition
    # Input f_nr - number of the fold that will be the test set 1-10
    # Output (testset, trainingset) - tuple containing the test set and training set
    # """
    dss = np.array_split(nds,10,axis=0) #dataset split
    testset = dss[f_nr] #test set
    if task==2:
        trainingset = np.hstack(np.delete(dss, f_nr, 0)) #training set
    else:
        trainingset = np.vstack(np.delete(dss, f_nr, 0)) #training set
    return (testset, trainingset)

def fold10_predict(models, ndsx, ndsy, ndsy_L, task):
    ###Performing 10-fold Cross Validation for each Model
    mlen = len(models) #Number of models
    nscores = np.zeros((mlen,10)) #score of each fold
    for j in range(10):
        #Creating test set and training set from data set
        if task==2:
            (tes_x, trs_x) = data_fold_10(ndsx, j, task-1)
            (tes_y, trs_y) = data_fold_10(ndsy, j, task)
        else:
            (tes_x, trs_x) = data_fold_10(ndsx, j, task)
            (tes_y, trs_y) = data_fold_10(ndsy, j, task)

        #Perform fitting and predicting for each model
        for i in range(mlen):
            models[i].fit(trs_x,trs_y)
            if task==3:#if task is third we use predict
                tes_yp = models[i].predict(tes_x)

                #Transform into DataFrame for scoring
                df_y = pd.DataFrame(tes_y, columns=ndsy_L)
                df_yp = pd.DataFrame(tes_yp, columns=ndsy_L)
                nscores[i,j] = scores(df_y,df_yp, task)
            else:#else use predict_proba
                tes_yp = models[i].predict_proba(tes_x)

                #Transform into DataFrame for scoring
                df_y = pd.DataFrame(tes_y, columns=ndsy_L)
                if task==2:
                    df_yp = pd.DataFrame(tes_yp[:,0], columns=ndsy_L)
                else:
                    df_yp = pd.DataFrame(tes_yp, columns=ndsy_L)
                nscores[i,j] = scores(df_y,df_yp, task)
    return nscores


def scores(tes_y, tes_yp, task):
    if task==3:
        VITALS = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
        score = np.mean([0.5 + 0.5 * np.maximum(0, metrics.r2_score(tes_y[entry], tes_yp[entry])) for entry in VITALS])
    elif task==2:
        score = metrics.roc_auc_score(tes_y['LABEL_Sepsis'], tes_yp['LABEL_Sepsis'])
    else:
        TESTS = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
         'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
         'LABEL_Bilirubin_direct', 'LABEL_EtCO2']
        score = np.mean([metrics.roc_auc_score(tes_y[entry], tes_yp[entry]) for entry in TESTS])

    return score


def nan_imputer(nds,method):
    # """
    # Given a dataset removes NaNs using
    # Parameters:
    # Input nds - Numpy array: dataset
    # Input method - method of imputation to use
    # Output nds_xnan - Numpy array: dataset without NaNs
    # """
    if method==1:#Sklearn: IterativeImputer, removes NaN considering other features
        imp = IterativeImputer(max_iter=10, random_state=0)
        imp.fit(nds)
        IterativeImputer(random_state=0)
        nds_xnan = imp.transform(nds)
    return nds_xnan


def time_reduction(nds,labels, time, method):
    # """
    # Given a dataset containing data on consecutive hours outputs a row extracting information time features
    # Parameters:
    # Input nds - numpy dataset
    # Input labels - list labels of dataset
    # Input time - time in hours to compress data
    # Input method - method of reduction to use
    # Output nds_reduced - dataset compressed
    # """
    nds = pd.DataFrame(nds,columns=labels)
    datalen = len(nds)
    numpatients = datalen / time #number of patients

    if method==1:#average of values per patient
        #Reduce by taking mean of columns for each patient
        nds_reduced = nds.groupby('pid',sort=False,as_index=False).mean()

    elif method==2:#scoring method based on evolution of patient during stay
        dss = np.array_split(nds,datalen,axis=0) #dataset split for each patient
        nds_reduced = []
        flagr=True
        for k in range(datalen):#for each patient
            patient = dss[k]#select patient
            npat = patient.to_numpy()
            r_pat = []
            for i in range (np.size(npat,1)):#for each label
                cur_col = npat[:,i]
                temp=0
                ev = 0
                flagn = True
                for j in range(np.size(cur_col)):#for each row
                    this=cur_col[j]
                    if ~(np.isnan(this)):
                        ev = ev + (this-temp)*(j+1) #evolution increasing on time
                        temp=this
                        flagn= False
                if flagn:#if row is all NaN
                    r_pat = np.append(r_pat,np.NaN)#insert NaN
                else:#else
                    r_pat = np.append(r_pat,ev)#insert evolution
            if flagr:#if reduced set is empty
                nds_reduced = np.append(nds_reduced,r_pat)#insert patient
                flagr=False
            else:#if at least one patient has been added
                nds_reduced = np.vstack((nds_reduced, r_pat))#insert patient as row
        #Transform to pandas for compatibility
        nds_reduced=pd.DataFrame(nds_reduced,columns=labels)
    #Reduce considering patient evolution during stay
    return nds_reduced.to_numpy()


def clean_set(nds,labels,imp_method,time_method,sequence):
    ds_clean = nds
    if sequence:
        ds_clean = time_reduction(ds_clean, labels, 12,time_method)
        ds_clean = nan_imputer(ds_clean,imp_method)
        ds_clean = pd.DataFrame(ds_clean, columns=labels)
    else:
        ds_clean = nan_imputer(ds_clean,imp_method)
        ds_clean = time_reduction(ds_clean, labels, 12,time_method)
        ds_clean = pd.DataFrame(ds_clean, columns=labels)


    return ds_clean
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

def omar():

    #Data needed for all subtasks
    #Extracting training labels
    dataset_y = pd.read_csv("data/train_labels.csv")

    #Extracting training feature
    dataset_x = pd.read_csv("data/train_features.csv")

    #lists that contain labels of dataset
    dataset_x_L = list(dataset_x)

    #Standard Scaler of dataset
    scaler = StandardScaler()
    scaler.fit(dataset_x)
    scaled_data = scaler.transform(dataset_x)
    scaled_data = pd.DataFrame(scaled_data,columns=dataset_x_L)
    ##Model set used for training
    models3 = []
    #0 - Sklearn: Ridge regression function with alpha 1
    models3 = np.append(models3, linear_model.RidgeCV(alphas=[10*a for a in range(1,10)]))
    #1 - Sklearn: Multi Task Lasso Cross Validation on alphas
    #models3 = np.append(models3,linear_model.MultiTaskLassoCV(cv=10,
    #                                                          alphas=[10**a for a in range(-1,10)],
    #                                                          fit_intercept=False,
    #                                                          max_iter=1000))
    #2 Sklearn: Multitask Elastic Net with Cross Validation
    #models3 = np.append(models3,linear_model.MultiTaskElasticNetCV(cv=10,random_state=0))

    #Preparing datasets to feed models
    ##clean dataset using
    # scaled_data: dataset_x scaled through StandardScaler
    # dataset_x_L: dataset labels
    # imp method: 1 - impute with IterativeImputer
    # time method: 1 - reduce with average
    # sequence: True - first reduce then converge
    cs3 = clean_set(scaled_data,dataset_x_L,1,1,True)

    ##Division for the prediction, probabilities divided from the real values
    dataset_y3 = dataset_y.loc[:,"LABEL_RRate":"LABEL_Heartrate"] #Real valued labels
    ds_p3 = cs3.loc[:,"Time":"pH"] #reduced dataset for prediction, without pid

    #labels of datasets
    ds_y3_L = list(dataset_y3)
    ds_p3_L = list(ds_p3)

    #transform into numpy
    ndsy3 = dataset_y3.to_numpy()
    nds_p3 = ds_p3.to_numpy()

    #Get predictions
    s3 = fold10_predict(models3, nds_p3, ndsy3, ds_y3_L, 3)
        #This outputs the predictions for subtask3, in pandas format

    #extract dataset to predict
    testset_x = pd.read_csv("data/test_features.csv")
    testset_x_L = list(testset_x)
    test_x = testset_x.to_numpy()

    #Standard Scaler of dataset
    scaler = StandardScaler()
    scaler.fit(testset_x)
    scaled_test = scaler.transform(testset_x)
    scaled_test = pd.DataFrame(scaled_test,columns=dataset_x_L)

    #cleaning data
    test3_x = clean_set(scaled_test,testset_x_L,1,1,True)

    ctes3 = test3_x.loc[:,"Time":"pH"]
    best_3 = models3[0] #0 is ridge, fast but worse performing
    pred3 = best_3.predict(ctes3)
    pred3 = pd.DataFrame(pred3,columns=ds_y3_L)
    return pred3

print(omar())

# main()
