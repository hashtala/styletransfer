# -*- coding: utf-8 -*-
"""
Created on Sun May 12 00:01:42 2019

@author: gela
"""

from keras.layers import Input, Lambda, Dense, Flatten
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
import keras


#recreate VGG with average pooling 
def wrapper(im_vector):
    
    los, grad = los_grad([im_vector.reshape(*batch_shape)])
    return los.astype(np.float64), grad.flatten().astype(np.float64)

def VGG_AVERAGE(shape):
    vgg = VGG16(input_shape = shape, weights = 'imagenet', include_top = False)
    
    averge_vgg = keras.Sequential()
    '''
    for layer in vgg.layers:
        if layer.__class__ == MaxPooling2D:
            print(layer.__class__ )
            averge_vgg.add(AveragePooling2D)
        else:
            averge_vgg.add(layer)
    '''
    return vgg

def custom_vgg(shape, n):
    
    vgg = VGG_AVERAGE(shape)
    new_vgg = Sequential()
    
    x = 0
    
    for layer in vgg.layers:

        if layer.__class__ == Conv2D:
            x += 1
        new_vgg.add(layer)
        if x == n:
            break

    return new_vgg
            
        
    


def unprocess(img):
    img[...,0] += 103.939
    img[...,1] += 116.779
    img[...,2] += 126.68
    
    img = img[...,::-1]
    
    return img


def process(img):
    img = img - img.min()
    img = img/img.max()
    return img


if __name__ == '__main__':
    
    im = image.load_img('firo.jpg')
    im = image.img_to_array(im)
    im = np.expand_dims(im, axis = 0)
    im = preprocess_input(im)
    batch_shape = im.shape
    shape = im.shape[1:]
    
    
    #AQ UNDA AIRCHIO ROMELI CONV LAYERIDAN GINDA AGEBA
    
    vgg_model = custom_vgg(shape, 8)
    
    target = K.variable(vgg_model.predict(im)) #it is conv2d output
    
    
    loss = K.mean(K.square(target - vgg_model.output))
    grad = K.gradients(loss, vgg_model.input)
    los_grad = K.function(inputs =[vgg_model.input], outputs = [loss] + grad)
    print('aqame movedit')


    initial_guess = np.random.randn(np.prod(batch_shape))
    los_array = []
    for i in range(10):
        print(i)
        initial_guess, l, _ = fmin_l_bfgs_b(
                func = wrapper,
                x0 = initial_guess,
                maxfun = 20)
        initial_guess = np.clip(initial_guess, -127, 127)
        print(i, end = '')
        print(l)
        los_array.append(l)
        
        
    new_image = initial_guess.reshape(*batch_shape)
    real_im = unprocess(new_image)
    im = real_im[0]
    plt.imshow(process(im))
        
    


'''
wava ra
'''
    
    
    
    