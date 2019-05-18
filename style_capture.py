# -*- coding: utf-8 -*-
"""


Created on Sat May 11 23:51:30 2019

@author: gela
"""

#from keras.layers import Input, Lambda, Dense, Flatten
#from keras.layers import AveragePooling2D, MaxPooling2D
#from keras.layers.convolutional import Conv2D
from keras.models import Model, Sequential
#from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
import keras

from content_capture import process, unprocess, custom_vgg







def wrapper(im_vector):
    los, gradd = loss_grad([im_vector.reshape(*batch_shape)])
    return los.astype(np.float64), gradd.flatten().astype(np.float64)

def gram_matrix(img):
    #input is C H W and needs to be converted to
    #C H*W
    X = K.batch_flatten(K.permute_dimensions(img, (2,0,1)))
    #so now dimensoins has been changed
    
    Gram_matrix = K.dot(X, K.transpose(X))/img.get_shape().num_elements()
    # G = 1/N(X*X^T)
    return Gram_matrix

def style_loss(img1, img2):
    
    gram1 = gram_matrix(img1)
    gram2 = gram_matrix(img2)
    
    loss = K.mean(K.square(gram1 - gram2))
    
    return loss


def minimize(fun, epoch, batch_shape):
    
    losses = []
    
    img0 = np.random.randn(np.prod(batch_shape))
    
    for x in range(epoch):
        img0, l, _ = fmin_l_bfgs_b(func = fun, x0 = img0, maxfun = 20)
        losses.append(l)
        img0 = np.clip(img0, -127, 127)
        print(x, end =' iteration  ')
        print(l)
    return img0

    img = img0.reshape(*batch_shape)
    img = unprocess(img)
    return img[0]
    

if __name__ == '__main__':
    
    im = image.load_img('starry.jpg')
    im = image.img_to_array(im)
    im = np.expand_dims(im, axis = 0)
    im = preprocess_input(im)
    batch_shape = im.shape
    shape = im.shape[1:]
    
    
    #AQ UNDA AIRCHIO ROMELI CONV LAYERIDAN GINDA AGEBA
    
    vgg_model = custom_vgg(shape, 13)
    
    symbolic_conv_outputs = [layer.get_output_at(1) for layer in vgg_model.layers if layer.name.endswith('conv1')]
    



    #now are are doing actual model and it also has a good shortcut
    
    multi_model_output = Model(vgg_model.input, symbolic_conv_outputs)
    
    layer_outputs = [K.variable(k) for k in multi_model_output.predict(im)]
    loss = 0
    
    for symbolic, numerical in zip(symbolic_conv_outputs,layer_outputs):
        
        loss += style_loss(symbolic[0], numerical[0])
        
    grad = K.gradients(loss, multi_model_output.input)
    
    loss_grad = K.function(
            inputs = [multi_model_output.input],
            outputs = [loss] + grad)
    
    
    img = minimize(wrapper, 10, batch_shape)
    img = process(img)
    plt.imshow(img)
    
    
    
    








