

import numpy as np
import tensorflow as tf

def zero_padding(X, n):
    X_padded = np.pad(X,((0,0),(n,n),(n,n),(0,0)),'constant',constant_values=(0,0))
    return X_padded

def perform_convolution(one_slice, w, b):
    '''
    one_slice = 
    w = weights/kernel
    b = bias
    '''
    p = np.multiply(one_slice, w)
    point = np.sum(p)
    point += float(b)
    return point

def relu(x):
    return np.maximum(x, 0)


def conv_forward_layer(X, w, b, stride, padding):
    '''
    w = filters with dimensions of filter and number of filters
    '''
    #previous layer shapes
    (m, n_row_prev, n_col_prev, n_depth_prev) = X.shape
    
    #filter dimensions, kernel size fxf, n=#of channels, n_depth = number of filters
    (f,f, n_depth_prev, n_depth) = w.shape
    
    #Add padding to X
    X_padded = zero_padding(X, padding)
    
    #determine output dimensions of layer
    n_row_out = int((n_row_prev - f + 2*padding)/stride) + 1
    n_col_out = int((n_col_prev - f + 2*padding)/stride) + 1

    #create output matrix for this layer
    Z = np.zeros((m, n_row_out, n_col_out, n_depth))
    
    #iterate over all images in the batch (size of X)
    for i in range(m):
        for row in range(n_row_out):
            for col in range(n_col_out):
                for d in range(n_depth):
                    #perform convolution
                    rs, re, cs, ce = (stride*row), (stride*row+f), (stride*col), (stride*col+f)
                    x_slice_prev = X_padded[i, rs:re, cs:ce, :] #apply filter across all channels
                    #print((x_slice_prev.shape, w.shape))
                    Z[i, row, col, d] = perform_convolution(x_slice_prev, w[:,:,:,d], b[:,:,:,d]) 
                    #apply activation
                    Z[i, row, col, d] = relu(Z[i,row,col,d])
        
    A_prev_cache = (X, w, b, stride, padding) #save values to be used in backprop
    
    #Z is now g(input) after performing convlution on input
    return Z, A_prev_cache
#
#np.random.seed(1)
#A_prev = np.random.randn(10,4,4,3) #(#images, #rows, #cols, #channels_prev)
#W_prev = np.random.randn(2,2,3,8) #(#rows, #cols, #channels_prev, #channels_out)
#b_prev = np.random.randn(1,1,1,8) #(1,1,1, #channels_out)
#padding = 2
#stride = 2
#
#Z, alpha_cache = conv_forward_layer(A_prev, W_prev, b_prev, stride, padding)
#
#print("Z mean: ", np.mean(Z))
#print("Z [3,2,1]: ",Z[3,2,1])
#print("alpha_cache[0][1][2]: ", alpha_cache[0][2][3])

def pooling_forward_layer(X, stride, ps, method="max"):
    '''
    ps = pool-size/dimension 
    '''
    
    #Dimensions of input 
    (m, n_row_prev, n_col_prev, n_depth_prev) = X.shape
    
    #Dimensions of output
    n_row_out = int(1+(n_row_prev - ps)/stride)
    n_col_out = int(1+(n_col_prev - ps)/stride)
    n_depth = n_depth_prev
    
    #create output matrix for this layer
    
    A = np.zeros((m, n_row_out, n_col_out, n_depth))
    for i in range(m):
        for row in range(n_row_out):
            for col in range(n_col_out):
                for d in range(n_depth):
                    rs, re, cs, ce = (stride*row), (stride*row+ps), (stride*col), (stride*col+ps)
                    if method == "max":
                        A[i, row, col, d] = np.max(X[i, rs:re, cs:ce, d])
                    else: 
                        A[i, row, col, d] = np.mean(X[i, rs:re, cs:ce, d])
    #alpha of previous layer, save for backprop
    A_prev_cache = (X, stride, ps)
    
    return A,  A_prev_cache

#test pool forward
#np.random.seed(1)
#A_prev = np.random.randn(2,4,4,3) #(#images, #rows, #cols, #channels_prev)
#padding = 2
#stride = 2
#A, A_prev_cache = pooling_forward_layer(A_prev, stride, 3, method="avg")
               
def DW_conv_layer(X, w, b, stride, padding):
    
    #Dimensions of input
    (m, n_row_prev, n_col_prev, n_depth_prev) = X.shape
    
    #Dimensions of kernel
    #(f, f, n_depth_prev, n_depth) = w.shape
    (f, f, n_depth) = w.shape
     
    #Dimensions of output
    n_row_out = int((n_row_prev - f + 2*padding)/stride) + 1
    n_col_out = int((n_col_prev - f + 2*padding)/stride) + 1    
    
    #Apply padding to input to retain dimensions
    X_padded = zero_padding(X, padding)
    print("X_padded shape: ", X_padded.shape)

    
    #Output Dimensions
    out = np.zeros((m, n_row_out, n_col_out, n_depth))
    print("out shape: ", out.shape)
    
    #depthwise covolution
    for i in range(m):
        for d in range(n_depth_prev): #Each channel of the input
            #for each image, apply to each of its channels seperately its corresponding filter 
            for row in range(n_row_out):
                for col in range(n_col_out):
                    rs, re, cs, ce = (stride*row), (stride*row + f), (stride*col), (stride*col + f)
                        #print((i, rs,re, cs,ce, d))
                    x_slice =  X_padded[i, rs:re, cs:ce, d] #.reshape(f,f,1)
                        #print((x_slice.shape, w[:,:,d].shape))
                    out[i, row, col, d] = perform_convolution(x_slice, w[:,:,d], b[:,:,d])
    #need to add in relu and batch norm
    return out

#Test DW Layer
#np.random.seed(1)
#A_prev = np.random.randn(1,112,112,64) #(#images, #rows, #cols, #channels_prev)
#W_prev = np.random.randn(3,3,64) #(#rows, #cols, #channels_prev)
#b_prev = np.random.randn(1,1,64) #(1,1,1, #channels_out)
#padding = 1
#stride = 2
#
#Z = DW_conv_layer(A_prev, W_prev, b_prev, stride, padding)
#         

def pointwise_conv_layer(X, w, b, stride, padding):

    #Dimensions of input
    (m, n_row_prev, n_col_prev, n_depth_prev) = X.shape
    
    #Dimensions of kernel
    (f, f, n_depth_prev) = w.shape
     
    #Dimensions of output
    #n_row_out = int((n_row_prev - f + 2*padding)/stride) + 1
    #n_col_out = int((n_col_prev - f + 2*padding)/stride) + 1
    n_row_out = n_row_prev
    n_col_out = n_col_prev    
    n_depth = n_depth_prev
    
    #Apply padding to input to retain dimensions
    #X_padded = zero_padding(X, padding)
    #print("X_padded shape: ", X_padded.shape)

    #Output Dimensions
    out = np.zeros((m, n_row_out, n_col_out, n_depth))
    print("out shape: ", out.shape)
    
    #1x1 seperable convolution 
    for i in range(m):
        for d in range(n_depth): #N in the paper
            for row in range(n_row_out):
                for col in range(n_col_out):
                    rs, re, cs, ce = (stride*row), (stride*row + f), (stride*col), (stride*col +f)
                    x_slice = X[i, rs:re, cs:ce, :]
                    out[i, row, col, d] = perform_convolution(x_slice, w[:,:,:], b[:,:,d]) 
    
    #need to add in relu and batch norm
    return out


#Test Pointwise Layer
#np.random.seed(1)
#A_prev = np.random.randn(1,56,56,64) #(#images, #rows, #cols, #channels_prev)
#W_prev = np.random.randn(1,1,64) #(#rows, #cols, #channels_prev)
#b_prev = np.random.randn(1,1,64) #(1,1,1, #channels_out)
#padding = 0
#stride = 1
#
#Z = pointwise_conv_layer(A_prev, W_prev, b_prev, stride, padding)  


def conv_back_layer(dA, cache):
    #extract previous layer elements
    X_prev, weights, bias, stride, padding = cache
    m, n_row_prev, n_col_prev, n_depth_prev = X_prev.shape
    f, f, n_depth_prev, n_depth = w.shape
    
    #get dimensions of gradient cost function
    m, n_row, n_col, n_depth = dA.shape
    
    #output dimensions, going backward in the network so its the same dimensions of X_prev
    dA_prev = np.zeros((m, n_row_prev, n_col_prev, n_depth_prev))
    dW = np.zeros((f,f, n_depth_prev, n_depth))
    db = np.zeros((f,f,n_depth_prev, n_depth))
    
    #Add padding to previous layer
    X_prev_padded = zero_padding(X_prev, padding)
    dX_prev_padded = get_padding(dX_prev, padding)
    
    
    #backpropagation 
    for i in range(m):
        for row in range(n_row) :
            for col in range(n_col):
                for d in range(n_depth):
                    rs, re, cs, ce = (stride*row), (stride*row + f), (stride*col), (stride*col + f)
                    x_slice = X_prev_padded[i, rs:re, cs:ce, :]
                    dA_prev_padded[i, rs:re, cs:ce, :] += w[:,:,:,d]*dA[i, row, col, d]
                    dW[:,:,:,d] += x_slice*dA[i, row, col, d]
                    db[:,:,:,d] += dA[i, row, col, d]
        dA_prev[i,:,:,:] = dA_prev_padded[i, padding:-padding, padding:-padding, :]
    
    return dA_prev, dW, db
                    
    
    
    
def pool_back_layer(dA, cache, method="max"):
    X_prev, stride, ps = cache
    
    m, n_row_prev, n_col_prev, n_depth_prev = X_prev.shape
    m, n_row, n_col, n_depth = dA.shape
    
    #output dimensions, going backward in the network so its the same dimensions of X_prev
    dA_prev = np.zeros_like(X_prev)
    
    for i in range(m):
        for row in range(n_row):
            for col in range(n_col):
                for d in range(n_depth):
                    rs, re, cs, ce = (stride*row), (stride*row + f), (stride*col), (stride*col + f)
                    if method == "max":
                        x_prev_slice = X_prev[i, rs:re, cs:ce, d]
                        largest = (x_prev_slice == np.max(x_prev_slice))
                        dA_prev[i, rs:re, cs:ce, d] += largest*dA[i, rs, cs, d]
                    elif method == "avg":
                        da = dA[i, rs, cs, d]
                        avg = np.ones((ps,ps))*(da/(ps*ps))
                        dA_prev[i, rs:re, cs:ce, d] += avg
                        
    return dA_prev

#need to completed                        
def depthwise_back_layer():
    return None

#need to complete
def pointwise_back_layer():
    return None

def RMSprop():
    return None
    
def initalize_weights(dim):
    '''
    Randomly intalizes weights for input layer of CNN
    0.01 constant is used to ensure the weights are initally really small so that
    our Z values (where Z= Wx+b) arent initally large and GD learning isnt initally slow
    '''
    
    return np.random.randn(dim)*0.01

def training(images, n_classes, batch_size, n_epochs):
    
    #initalize weights
    
    
    #train
    for epoch in range(n_epochs):
        
        
        for batch in range(batch_size):
            
    

    #export model with pickle
    with open(path, 'wb') as file:
        pickle.dumb(params, file)
        
        
    return None
    


            
    








    