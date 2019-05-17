# -*- coding: utf-8 -*-
"""
Created on Sun May 12 16:41:19 2019

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


from style_capture import minimize, gram_matrix, style_loss
from content_capture import custom_vgg, process, unprocess


def wrapper(im_vector):
    los, gradd = loss_and_grads([im_vector.reshape(*batch_shape)])
    return los.astype(np.float64), gradd.flatten().astype(np.float64)

def load_and_preprocess(path, shape =None):
    im = image.load_img(path, target_size = shape)
    im = image.img_to_array(im)
    im = np.expand_dims(im, axis = 0)
    im = preprocess_input(im)
    return im


content_image = load_and_preprocess('firo.jpg')

h,w = content_image.shape[1:3]

style_image = load_and_preprocess('starry.jpg', shape = (h,w))

batch_shape = content_image.shape
shape = content_image.shape[1:]

vgg = custom_vgg(shape, 13)

content_model = Model(vgg.input, vgg.layers[13].get_output_at(1))
content_target = K.variable(content_model.predict(content_image))

symbolic_conv_outputs = [layer.get_output_at(1) for layer in vgg.layers if layer.name.endswith('conv1')]

style_model = Model(vgg.input, symbolic_conv_outputs)
style_layer_outputs = [K.variable(k) for k in vgg.predict(style_image)]
style_weight = [1,2,3,4,5]
loss = K.mean(K.square(content_model.output - content_target))

for w, symbol, actual in zip(style_weight,symbolic_conv_outputs, style_layer_outputs):
    loss += style_loss(symbol[0], actual[0])

grad = K.gradients(loss, vgg.input)

loss_and_grads = K.function(
        inputs = [vgg.inputs],
        outputs = [loss] + grad)


train = minimize(wrapper, 10, batch_shape)

del grad, loss, vgg 

real_im = process(train)
real_im = process(real_im)

