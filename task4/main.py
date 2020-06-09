import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import layers

import PIL

import pandas as pd



# Task 4: Image food labeling

DICT = {}

### Images in data/food/* in JPEG format
### test triplets in data/test_triplets
### train triplets in data/train_triplets


####################################################################
##################### FEATURE EXTRACTION PART(CNN)##################
####################################################################
def get_train_triplets():
    return np.genfromtxt("./data/train_triplets.txt", dtype="str")


def get_test_triplets():
    return np.genfromtxt("./data/test_triplets.txt", dtype="str")

def get_image_path(name):
    return './data/food/' + name + '.jpg'


# Here we use VGG16 pretrained deep CNN to extract features from Images
# TODO: Increase accuracy by training the model on our dataset?
def setup_pretrained_model():
    #Decisions:
        # - max or avg?
    model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    for layer in model.layers:
        layer.trainable = False
    return model


def feature_extraction(model,name):
    img = image.load_img(get_image_path(name),target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x)[0] ##[0] to change shape from (1,512) -> (512, )




def append(arr1,arr2):
    m = arr1.size
    res = np.zeros(2 * m)
    res[0:m] = arr1
    res[m:2*m] = arr2
    return res

def setup_data(model,min,max,train):
    if(train):
        triplets = get_train_triplets()[min:max]
    else:
        triplets = get_test_triplets()
    m,n = triplets.shape
    res = np.zeros((2 * m, 1024))
    labels = np.zeros(2 * m)
    for i in range(m):
        anchor = feature_extraction(model,triplets[i][0])
        pos = feature_extraction(model,triplets[i][1])
        neg = feature_extraction(model,triplets[i][2])
        res[2 * i ][0:512] = anchor
        res[2 * i + 1][0:512] = pos

        res[2 * i ][512:1024] = anchor
        res[2 * i + 1][512:1024] = neg

        labels[2 * i] = 1
        labels[2 * i + 1] = 0
    return res,labels



####################################################################
###############Classifier part(fully connected ANN)#################
####################################################################

def create_dict(model):

    triplets1 = get_train_triplets()
    triplets2 = get_test_triplets()
    strings1 = triplets1.flatten()
    strings2 = triplets2.flatten()
    strings = np.unique(append(strings1,strings2))
    for name in strings:
        DICT[name]  =  feature_extraction(model,name)


def get_feature(name):
    return DICT[name]
# create ndarry features of image 1 (vector 512), features of image 2 (vector 512)
# 0 or 1 for classificiation
# def data_transformer(cnn_model,data):
#     output_size = feature_extraction(cnn_model,data[0][0]).size
#     m,n = data.shape
#     new_data = np.zeros((2 * m,output_size))
#     labels = np.zeros(2 * m)
#
#     for i in range(m):
#         anchor_features = feature_extraction(cnn_model,data[i][0])
#         positive_features = feature_extraction(cnn_model,data[i][1])
#         negative_features = feature_extraction(cnn_model,data[i][2])
#
#         new_data[i * 2] = anchor_features - positive_features
#         labels[i * 2] = 1
#
#         new_data[i * 2] = anchor_features - negative_features
#         labels[i * 2] = 0
#     return new_data,labels


def check_distances(model,data):
    count = 0.0
    m,n = data.shape
    for i in range(m):
        d_first = get_distance(model,data[i][0], data[i][1])
        d_second = get_distance(model,data[i][0], data[i][2])
        if(d_first < d_second):
            count += 1
    return count/m

def get_distance(model,img1,img2):
    return distance(feature_extraction(model, img1), feature_extraction(model,img2))


#distance between two images vectors x and y
def distance(x,y):
    return np.linalg.norm((x-y), 1)

def create_fully_connect_model():
    inputs = keras.Input(shape=(1024,))
    x = layers.Dense(256,activation="relu")(inputs)
    x = layers.Dense(256,activation="relu")(inputs)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs,outputs=outputs,name="fully_connected")
    return model

# def pair_loss(y_true,y_pred):
#     n = tf.size(y_true)
#     print(n)
#     res = np.zeros(n)
#     for i in range(n):
#         if(y_true[i] == 1.0):
#             res[i] = -1.0 * np.linalg.norm(y_pred[i])
#         else:
#             res[i] = np.linalg.norm(y_pred[i])
#
#     return res


def parse_results(results):
    n = results.size /2
    ret = np.zeros(n)
    for i in range(n):
        if(results[i * 2] > results[i * 2 + 1]):
            ret[i] = 1
    return ret



####################################################################
##################### Main Part (execution)       ##################
####################################################################


def main():
    pretrained_model = setup_pretrained_model()
    # pretrained_model.summary()
    # create_dict(pretrained_model)
    # print(get_feature('01186'))
    fully_connected_model = create_fully_connect_model()
    fully_connected_model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])



    train_data,labels = setup_data(pretrained_model,0,5000,True)
    fully_connected_model.fit(train_data,labels)


    test_data, labels = setup_data(pretrained_model,0,0,False)
    results = fully_connected_model.predict(test_data)
    np.savetxt('submission.txt',parse_results(results),fmt='%d')


    # features1 = feature_extraction(model,'00000')
    # features2 = feature_extraction(model,'00001')
    # print(append(features1,features2).shape)
    # x = append(features1,features2)
    # x = np.expand_dims(x, axis=0)
    #
    # # fully_connected_model.summary()
    # print(fully_connected_model.predict(x))
    # print(features1.shape)
    # print(np.linalg.norm(features1 - features2))
    # print(np.sum(features1 - features2))
    # print(features2)
    # print(check_distances(model,get_train_triplets()[0:100]))
main()


####################################################################
####################################################################
####################################################################
