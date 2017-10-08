#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

import random


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])
    #print params
    
        
    
    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))
    
    ### YOUR CODE HERE: forward propagation
    #-------------------------------------------------#
    num_data= data.shape[0]
    hidden_prime= np.transpose(np.dot(data,W1)+b1)# Dh * M
    z = np.dot(data,W1) + b1
    hidden_layer=sigmoid(hidden_prime)
    a = sigmoid(z)
    ## a matrix  dh * M 
    #-------------------------------------------------#
    output_prime= np.dot(W2.T,hidden_layer)+b2.T
    
    z2 = np.dot(a,W2)+b2
    output_layer=softmax(output_prime.T).T
    
    ##softmax is a function taking argeuments that a matrix M*n M is the number of data,which requires input have to be M * dy 
    a2 = softmax(z2)
    ###a matrix dy * m
    cost_temp=np.log (output_layer)*labels.T
    ###a matrix dy*m
    
    #-------------------------------------------------#
   
    cost=-np.sum(cost_temp)
    #-------------------------------------------------#
    ### END YOUR CODE
    

    ### YOUR CODE HERE: backward propagation
    dce_dz=(output_layer-np.transpose(labels))
    ###dce_dz is dy*m
    #labels is supposed to be m*dy, get a matrix dy*m
    ## matrix m * dy
    gradW2=np.dot(hidden_layer,np.transpose (dce_dz))
    ## dh*m  m*dy       dh * dy
    
    
    gradb2=np.transpose (np.sum(dce_dz,1))
    ### gradb2 1*dy
    
    dce_dh=np.dot(W2,dce_dz)
    ####matrix 
    dce_dsigmoid= dce_dh* sigmoid_grad(hidden_layer)
    ### matrix  Dh *m
    
    gradW1=np.dot(np.transpose(data),np.transpose(dce_dsigmoid))
    ### dx * dh
    
    gradb1=np.transpose( np.sum(dce_dsigmoid,1))
 
    ### END YOUR CODE
   
    
    
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )
    gradcheck_naive(lambda x:
        forward_backward_prop(data, labels, x, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    #raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()