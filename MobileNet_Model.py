

import numpy as np

def zero_padding(X, n):
    X_padded = np.pad(X,((0,0),(n,n),(n,n),(0,0)),'constant',constant_values=(0,0))
    return X_padded

def perform_convolution_(one_slice, w, b):
    '''
    one_slice = 
    w = weights/kernel
    b = bias
    '''
    p = np.multiply(one_slice, w)
    point = np.sum(p)
    point += float(b)
    return point


def conv_forward_layer(X, w, b, stride, padding):
    '''
    w = filters with dimensions of filter and number of filters
    '''
    #previous layer shapes
    (m, n_row_prev, n_col_prev, n_depth_prev) = X
    
    #filter dimensions, kernel size fxf, n=#of channels, n_depth = number of filters
    (f,f, n_depth_prev, n_depth) = w.shape
    
    #Add padding to X
    X_padded = zero_padding(X, padding)
    
    #determine output dimensions of layer
    n_row_out = int((n_row_prev - f + 2*padding)/stride) + 1
    n_col_out = int((n_col_prev - f + 2*padding)/stride) + 1
    
    #create output matrix for this layer
    out = np.zeros((m, n_row_out, n_col_out, n_depth))
    
    #iterate over all images in the batch (size of X)
    for i in range(m):
        for row in range(n_row_out):
            for col in range(n_col_out):
                for d in range(n_depth):
                    rs, re, cs, ce = (stride*row), (stride*row+f), (stride*col), (stride*col+f)
                    x_slice_prev = X_padded[i, rs:re, cs:ce, :] #apply filter across all channels
                    out[i, row, col, d] = perform_convolution(x_slice_prev, w[:,:,:,d], bias[:,:,:,d])
    alpha = (X, w, b, stride, padding) #save values to be used in backprop
    
    return out, alpha

def pooling_forward_layer(X, stride, ps, method="max"):
    '''
    ps = pool size 
    '''
    
    #Dimensions of input 
    (m, n_row_prev, n_col_prev, n_depth_prev) = X.shape
    
    #Dimensions of output
    n_row_out = int(1+(n_row_prev - ps)/stride)
    n_col_out = int(1+(n_col_prev - ps)/stride)
    n_depth = n_depth_prev
    
    #create output matrix for this layer
    out = np.zeros((m, n_row, n_col, n_depth))
    for i in range(m):
        for row in range(n_row_out):
            for col in range(n_col_out):
                for d in range(n_depth):
                    rs, re, cs, ce = (stride*row), (stride*row+f), (stride*col), (stride*col+f)
                    if method == "max":
                        out[i, row, col, d] = np.max(X[i, rs:re, cs:ce, d])
                    else: 
                        out[i, row, col, d] = np.mean(X[i, rs:re, cs:ce, d])
    #alpha of previous layer, save for backprop
    alpha = (X_prev, stride, ps)
    
    return out, alpha
    
def DWSep_conv_layer(X, w, b, stride, padding):
    #depthwise covolution
    
    #Dimensions of input
    (m, n_row_prev, n_col_prev, n_depth_prev) = X.shape
    
    #Dimensions of input
    (f, f, n_depth_prev, n_depth) = w.shape
    
    #Dimensions of output
    n_row_out = int((n_row_prev - f + 2*padding)/stride) + 1
    n_col_out = int((n_col_prev - f + 2*padding)/stride) + 1    
    
    #Apply padding to input to retain dimensions
    X_padded = zero_padding(X, padding)
    
    #Output Dimensions
    out = np.zeros((m, n_row_out, n_col_out, n_depth))
    
    for i in range(m):
        for d in range(n_depth):
            #for each image, apply to each of its channels seperately its corresponding filter 
            for row in range(n_row_out):
                for col in range(n_col_out):
                    rs, re, cs, ce = (stride*row), (stride*row + f), (stride*col), (stride*col + f)
                    x_slice =  X_padded[i, rs:re, cs:ce, d]
                    output[i, row, col, d] = perform_convolution(x_slice, w[:,:,:,d], bias[:,:,:,d])
    
    #1x1 seperable convolution 
    for i in range(m):
        for i in range(n_row_out):
            for j in range(n_col_out):
                for c in range(n_depth):
                    
            
    
    
def conv_backprop_layer():
    


    



            
    








    