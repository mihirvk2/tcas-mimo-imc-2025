import numpy as np

def int_to_bits(x, nbits=8,flip=False):
    # Converts signed integer vector x to bits matrix (N x nbits)
    # Rightmost bit is LSB and leftmost is MSB if flip is False
    # Leftmost bit is LSB and rightmost is MSB if flip is True
        nrows = len(x)
        ncols = nbits
        y = np.zeros(shape=(nrows,ncols))
        for i in range(nrows):
            x_curr_int = int(x[i])
            temp = np.binary_repr(x_curr_int ,nbits)
            if(flip): y[i,:] = np.flip(np.array([int(char) for char in temp]))
            else: y[i,:] = np.array([int(char) for char in temp])
        return y

def quantize_real_scalar(x,nbits,input_range = 1):
# Quantizes a real floating point input scalar to real signed int scalar
# Input range always symmetric around 0
    scale = 2 * input_range / (2**nbits) # Calculate LSB
    x_clipped = np.clip(x, -input_range, input_range - scale)
    level = np.round(x_clipped / scale)
    return level, scale

def quantize_complex_scalar(x,nbits,input_range = 1):
# Quantizes a complex floating point input to complex signed int
# Input range always symmetric around 0
    scale = 2 * input_range / (2**nbits) # Calculate LSB
    x_real = np.real(x)
    x_real_level = quantize_real_scalar(x_real,nbits,input_range)[0]
    x_imag = np.imag(x)
    x_imag_level = quantize_real_scalar(x_imag,nbits,input_range)[0]
    level = complex(x_real_level, x_imag_level)
    return level, scale

def quantize_complex_vector(x,nbits,input_range = 1):
# Quantizes a complex floating point input vector to complex signed int vector
# Input range always symmetric around 0
    dim = len(x)
    scale = 2 * input_range / (2**nbits) # Calculate LSB
    level = np.zeros(dim, dtype= np.complex_)
    for i in range(dim):
        level[i] = quantize_complex_scalar(x[i],nbits,input_range)[0]
    return level, scale

def quantize_complex_matrix_one_sf(W,nbits,input_range = 1):
# Quantizes a complex floating point input matrix to complex signed int matrix
# Input range always symmetric around 0
# Same scaling factor for all values
    nrows = W.shape[0]
    ncols = W.shape[1]
    scale = 2 * input_range / (2**nbits) # Calculate LSB
    level = np.zeros((nrows, ncols), dtype= np.complex_)
    for i in range(nrows):
        for j in range(ncols):
            level[i, j] = quantize_complex_scalar(W[i, j],nbits,input_range)[0]
    return level, scale


def CMVM_quantized_one_sf(W,W_scale,x,x_scale):
# MVM of a quantized complex matrix a quantized complex vector
# W_scale is the scaling factor of weight matrix
# x_scale is scaling factor of input vector
    y_scale = W_scale*x_scale
    y = np.matmul(W,x)
    return y, y_scale

def quantize_complex_matrix_many_sf(W,nbits, input_ranges):
# Quantizes a complex floating point input matrix to complex signed int matrix
# Input range always symmetric around 0
# Input range dimension = num_rows
# Different scaling factors for different column vectors
    nrows = W.shape[0]
    ncols = W.shape[1]
    scaling_factors = 2 * input_ranges / (2**nbits) # Calculate LSB
    level = np.zeros((nrows, ncols), dtype= np.complex_)
    for i in range(nrows):
        for j in range(ncols):
            level[i, j] = quantize_complex_scalar(W[i, j],nbits,input_ranges[i])[0]
    return level, scaling_factors

def CMVM_quantized_many_sf(W,W_scale,x,x_scale):
# MVM of a quantized complex matrix a quantized complex vector
# W_scale is a vector of per UE scaling factors
# x_scale is scaling factor of input vector
    y = np.matmul(W,x)
    y_fl = np.zeros(y.shape[0], dtype = np.complex_)
    for i in range(y.shape[0]):
        y_fl[i] = W_scale[i]*y[i]*x_scale
    return y_fl, y

def occ_std(nbits):
# Optimal Clipping Criteria from Charbel's TSP paper
    k_occ = 0
    if(nbits==2):
        k_occ = 1.71
    elif(nbits==3):
        k_occ = 2.15
    elif(nbits==4):
        k_occ = 2.55
    elif(nbits==5):
        k_occ = 2.94
    elif(nbits==6):
        k_occ = 3.29
    elif(nbits==7):
        k_occ = 3.61
    elif(nbits==8):
        k_occ = 3.92
    elif(nbits==9):
        k_occ = 4.21
    elif(nbits==10):
        k_occ = 4.49
    return k_occ
