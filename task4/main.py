import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

import PIL


# Task 4: Image food labeling


### Images in data/food/* in JPEG format
### test triplets in data/test_triplets
### train triplets in data/train_triplets

def get_train_triplets():
    return np.genfromtxt("./data/train_triplets.txt", dtype="str")


def get_test_triplets():
    return np.genfromtxt("./data/test_triplets.txt", dtype="str")

def get_image_path(name):
    return './data/food/' + name + '.jpg'


def feature_extraction(model,name):
    img = image.load_img(get_image_path(name),target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x)

# Here we use VGG16 pretrained deep CNN to extract features from Images
# TODO: Increase accuracy by training the model on our dataset?
def setup_pretrained_model():
    #Decisions:
        # - max or avg?
    return VGG16(weights='imagenet', include_top=False, pooling='max')




def main():
    model = setup_pretrained_model()
    model.summary()
    features = feature_extraction(model,'00000')
    print(features.shape)

main()
