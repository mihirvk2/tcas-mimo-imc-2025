import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

def extract_H(channel, nsnapshots=1):
# returns normalized, averaged channel matrix H based on string input
# Input can be WINNER_<N>_<K> or ARGOS_<N>_<K> 
    N, K = tuple(map(int, re.findall(r'\d+', channel)[-2:]))
    H = np.zeros(shape=(N,K), dtype = np.complex_)
    if(channel[:5]=='Argos'):
        userCSI = np.load('Argos_96_8.npy')
        ofdm_carrier = 0 
        frame = userCSI.shape[0] // 2 # middle frame
        H = userCSI[frame,:,:,ofdm_carrier].T
    elif(channel[:6]=='WINNER'):
        for i in range(nsnapshots):
            file_name = 'WINNER_UMa_C2_LOS_' + str(K) + '_' + str(N) + '_' + str(i+1) + '.csv'
            H_i = parse_H_csv(file_name)
            H += H_i
    else:
        print('Channel not listed')    
    return H/np.linalg.norm(H)

def viz_channel(channel):
    N, K = tuple(map(int, re.findall(r'\d+', channel)[-2:]))
    # file_name = 'WINNER_UMa_C2_LOS_' + str(K) + '_' + str(N) + '_5.csv'
    # H = extract_H(file_name)
    H = extract_H(channel, 50)
    # userCSI = np.load('Argos_96_8.npy')
    # ofdm_carrier = 0 
    # frame = userCSI.shape[0] // 2 # middle frame
    # H = userCSI[frame,:,:,ofdm_carrier].T
    # H = np.random.randn(N, K) + 1j * np.random.randn(N, K)
    # H = H.astype(np.complex_) 
    # H = H/np.linalg.norm(H)
    H_H = np.matrix(H).H
    M = np.matmul(H_H,H)
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(np.abs(M), 
                xticklabels=np.arange(1, M.shape[1] + 1), 
                yticklabels=np.arange(1, M.shape[0] + 1))
    plt.show()
    
if __name__ == '__main__':

    # viz_channel('WINNER_64_8')
    # viz_channel('WINNER_128_16')
    # viz_channel('WINNER_256_16')
    viz_channel('Argos_96_8')