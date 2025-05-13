import numpy as np
import re

def str_to_complex(x):
# converts string format from csv file to complex number format
    real = ''
    img = ''
    if '+' in x:
        real, img = tuple(x.split('+'))
    else:
        if(x[0]!='-'):
            tmp = x.split('-')
            real = '-'.join(tmp[0:2])
            img = '-' + '-'.join(tmp[2:])
        else: 
            tmp = x.split('-')
            real = '-'.join(tmp[0:3])
            img = '-' + '-'.join(tmp[3:])
    # REAL
    real_exp_loc = real.find('e')
    num_real = 0
    if(real_exp_loc == -1):
        num_real = float(real)
    else:
        real_exp = int(real[real_exp_loc+1:])
        num_real = float(real[:real_exp_loc])*10**(real_exp)
    # IMAGINARY
    img_exp_loc = img.find('e')
    num_img = 0
    if(img_exp_loc== -1):
        num_img = float(img[:-1])
    else:
        img_exp = int(img[img_exp_loc+1:-1])
        num_img = float(img[:img_exp_loc])*10**(img_exp)

    num = complex(num_real, num_img)
    return num

def parse_H_csv(file):
# extracts channel matrix H from csv file 
    H_raw = np.genfromtxt(file,delimiter=',',dtype=str)
    nrows, ncols = H_raw.shape
    H = np.zeros((nrows,ncols),dtype=np.complex_)
    for i in range(nrows):
        for j in range(ncols):
            H[i,j] = str_to_complex(H_raw[i,j])
    return H

def extract_H(channel, nsnapshots=50):
# returns normalized, averaged channel matrix H based on string input
# Input can be WINNER_<N>_<K> or ARGOS_<N>_<K> 
    N, K = tuple(map(int, re.findall(r'\d+', channel)[-2:]))
    H = np.zeros(shape=(N,K), dtype = np.complex_)
    if(channel[:5]=='ARGOS'):
        userCSI = np.load('MIMO_channels/Argos_96_8.npy')
        ofdm_carrier = 0 
        frame = userCSI.shape[0] // 2 # middle frame
        H = userCSI[frame,:,:,ofdm_carrier].T
    elif(channel[:6]=='WINNER'):
        for i in range(nsnapshots):
            file_name = 'MIMO_channels/WINNER_UMa_C2_LOS_' + str(K) + '_' + str(N) + '_' + str(i+1) + '.csv'
            H_i = parse_H_csv(file_name)
            H += H_i
    else:
        print('Channel not listed')    
    return H/np.linalg.norm(H)

def generate_input_vector(K,mod):
# generates random input vector of dimension K x 1 
# modulation can be 'BPSK', 'QPSK', '16QAM', '64QAM'
# returns symbol vector and average symbol energy
    nsymbols = 0     
    symbols = []
    if(mod == 'BPSK'):
        nsymbols = 2
        symbols = [-1, 1]
    else:
        if(mod == 'QPSK'):
            nsymbols = 4
        elif(mod == '16QAM'):
            nsymbols = 16
        elif(mod == '64QAM'):
            nsymbols = 64
        else:
            print("Please choose modulation from 'BPSK', 'QPSK', '16QAM' or '64QAM' " )
            return
        nmax = int(np.sqrt(nsymbols))
        # Create list of all possible symbols for the given modulation
        for real in range(nmax):
            for img in range(nmax):
                symbols.append(complex(real*2 - nmax + 1, img*2 - nmax + 1))
    es = np.mean(np.abs(symbols)**2) # average symbol energy
    x = np.random.choice(symbols,size=K)
    return x, es

def slicer(x_tilde, mod):
# Decision slicer 
# Based on the modulation scheme, returns the symbol nearest to x_tilde
    nsymbols = 0     
    symbols = []
    if(mod == 'BPSK'):
        nsymbols = 2
        symbols = [-1, 1]
    else:
        if(mod == 'QPSK'):
            nsymbols = 4
        elif(mod == '16QAM'):
            nsymbols = 16
        elif(mod == '64QAM'):
            nsymbols = 64
        else:
            print("Please choose modulation from 'BPSK', 'QPSK', '16QAM' or '64QAM' " )
            return
        nmax = int(np.sqrt(nsymbols))
        # Create list of all possible symbols for the given modulation
        for real in range(nmax):
            for img in range(nmax):
                symbols.append(complex(real*2 - nmax + 1, img*2 - nmax + 1))
    symbols = np.array(symbols)
    K = x_tilde.shape[0]
    x_hat = np.zeros(K, dtype = np.complex_)
    for i in range(K):
        idx = (np.abs(symbols - x_tilde[i])).argmin() # pick the symbol closest to x_tilde
        x_hat[i] = symbols[idx]
    return x_hat

def SER(xt_hat,xt):
# Symbol error rate = (number of symbol errors)/(number of symbols) 
# xt and xt_hat are K x nsamples matrices
    nerrors = 0
    nsamples = xt.shape[1]
    K = xt.shape[0]
    for i in range(nsamples):
        for j in range(K):
            if(xt[j][i] != xt_hat[j][i]):
                nerrors += 1
    return (nerrors/(nsamples*K))

def EVM(xt_tilde, xt):
# EVM = sqrt(E(|xt - xt_tilde|^2)/E(|xt|^2))
    nsamples = xt.shape[1]
    K = xt.shape[0]
    err_sq_tot = 0
    sig_sq_tot = 0
    for i in range(nsamples):
        for j in range(K):
            err_sq = np.abs(xt[j][i] - xt_tilde[j][i])**2
            sig_sq = np.abs(xt[j][i])**2
            err_sq_tot += err_sq
            sig_sq_tot += sig_sq
    evm = np.sqrt(err_sq_tot/sig_sq_tot)
    # evm_percent = evm*100
    return evm