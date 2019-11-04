import cv2
import keras
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.applications import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.backend import argmax
from skin_detection import *
from config import *



def read_image(path):
    """
    Reads an image given by the path
    
    Arguments:
        path {str} -- path of image to be read
    
    Returns:
        np.array -- an array of image pixels
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # ! Conversion to color is permanent! Channels always 3
    if len(img.shape) != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return np.array(img)



def show_image(img):
    """
    Display an image on the screen
    
    Arguments:
        img {np.array} -- an array of image pixels
    """
    cv2.imshow('Image', img)
    cv2.waitKey()



def resize_image(img):
    """
    Resizes an image in (width, height, channels) format with padding to keep the aspect ratio

    Arguments:
        img {np.array} -- an array of image pixels
    
    Returns:
        np.array -- resized image of shape SHAPE✕SHAPE✕CHANNELS
    """
    # Resize the image
    old_shape = img.shape[1::-1]
    ratio = SHAPE / max(old_shape)
    new_shape = tuple([int(dim * ratio) for dim in old_shape])
    img = cv2.resize(img, new_shape)

    # Pad the image
    color = [0, 0, 0]
    w, h = [SHAPE - dim for dim in new_shape]
    top, bottom = h // 2, h - h // 2
    left, right = w // 2, w - w // 2
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img



def shuffle_in_unison(a, b):
	"""
    Shuffles 2 numpy array together inplace
    
    Arguments:
        a {np.array} -- an array containing the pixel values for each image
        b {np.array} -- an array containing the pixel values for each image
    """
	rng_state = np.random.get_state()
	np.random.shuffle(a)
	np.random.set_state(rng_state)
	np.random.shuffle(b)



def filter_skin(img):
	return extractSkin(img)



def evaluate(test_xs, test_ys, model):
    """
    Checking how well the model performed
    
    Arguments:
        test_xs {np.array} -- an array containing the pixel values for each image in test data
        test_ys {np.array} -- the label for each image in test data
    """
    score = model.evaluate(test_xs, test_ys, verbose=0)
    print('Test Loss:', score[0])
    print('Test Accuracy: ', score[1])



def create_pickle(data, path):
    """
    Creates a pickle object file for given data
    
    Arguments:
        data {object} -- data to be stored as pickle object
        path {str} -- path to output binary file
    """
    out_file = open(path, 'wb')
    pickle.dump(data, out_file, protocol=4)
    out_file.close()



def load_pickle(path):
    """
    Loads an object given by the pickle file
    
    Arguments:
        path {str} -- path to pickle object
    
    Returns:
        object -- loaded object from the binary file
    """
    in_file = open(path, 'rb')
    data = pickle.load(in_file)
    in_file.close()
    
    return data



def load_mobilenetv2():
    mobilenetv2 = MobileNetV2(include_top=False, weights='imagenet', input_shape=(SHAPE, SHAPE, CHANNELS), alpha=1.0, classes=NUM_CLASSES)
    layer = mobilenetv2.layers[-1]
    return keras.Model(inputs=mobilenetv2.inputs, outputs=layer.output)



def load_xception():
    xception = Xception(include_top=False, weights='imagenet', input_shape=(SHAPE, SHAPE, CHANNELS), classes=NUM_CLASSES)
    layer = xception.layers[-1]
    return keras.Model(inputs=xception.inputs, outputs=layer.output)



def load_inceptionv3():
    inceptionv3 = InceptionV3(include_top=False, weights='imagenet', input_shape=(SHAPE, SHAPE, CHANNELS), classes=NUM_CLASSES)
    layer = inceptionv3.layers[-1]
    return keras.Model(inputs=inceptionv3.inputs, outputs=layer.output)



def load_inceptionresnetv2():
    inceptionresnetv2 = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(SHAPE, SHAPE, CHANNELS), classes=NUM_CLASSES)
    layer = inceptionresnetv2.layers[-1]
    return keras.Model(inputs=inceptionresnetv2.inputs, outputs=layer.output)



def load_learning_model(model):
    """
    Loads the transfer learning model with input SHAPE✕SHAPE✕CHANNELS
    
    Arguments:
        model {str} -- One of the  4 categories ['mobilenetv2', 'inceptionv3', 'xception', 'inceptionresnetv2']
    
    Returns:
        keras.Model -- transfer learning model
    """
    return globals()['load_' + model]()