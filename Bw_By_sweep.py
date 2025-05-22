import numpy as np
import matplotlib.pyplot as plt
from mimo_detectors import *
from testing_functions import *
from IMC_model.imc_adc import *
from IMC_model.imc_dp_unit import *
from IMC_model.imc_mvm_unit import *
from IMC_model.imc_cmvm_unit import *
from IMC_model.quantization_functions import *
import logging
from tqdm import tqdm
import re
from matplotlib.colors import Normalize
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D



def Bw_By_sweep(channel, modulation, nsamples):
    # channel can be ['WINNER_64_8', 'WINNER_128_16', 'WINNER_256_16' 'ARGOS_96_8']
    # modulation can be ['BPSK', 'QPSK', '16QAM', '64QAM']

    N, K = tuple(map(int, re.findall(r'\d+', channel)[-2:]))
    H = extract_H(channel)
    
    nbits_min = 4
    nbits_max = 10
    npoints = nbits_max - nbits_min
    B_y_options = np.array(range(nbits_min, nbits_max))
    B_w_options = np.array(range(nbits_min, nbits_max))

    # EVM
    evm_lmmse_fx = np.zeros(shape = (npoints,npoints))
    evm_lmmse_fx_margin_dB = np.zeros(shape = (npoints,npoints))
    evm_lmmse_fp_margin_dB = np.zeros(shape = (npoints,npoints))


    # RX SNR
    SNRdB = 20
    if channel == 'WINNER_64_8' and modulation == 'QPSK':
        SNRdB = 12.39685
    elif channel == 'WINNER_128_16' and modulation == 'QPSK':
        SNRdB = 12.64686
    elif channel == 'WINNER_256_16' and modulation == 'QPSK':
        SNRdB = 9.41398
    elif channel == 'ARGOS_96_8' and modulation == 'QPSK':
        SNRdB = 10.65343
    elif channel == 'WINNER_64_8' and modulation == '16QAM':
        SNRdB = 15.36744
    elif channel == 'WINNER_128_16' and modulation == '16QAM':
        SNRdB = 15.63112
    elif channel == 'WINNER_256_16' and modulation == '16QAM':
        SNRdB = 12.42296
    elif channel == 'ARGOS_96_8' and modulation == '16QAM':
        SNRdB = 13.64712
    elif channel == 'WINNER_64_8' and modulation == '64QAM':
        SNRdB = 19.29103
    elif channel == 'WINNER_128_16' and modulation == '64QAM':
        SNRdB = 19.53417
    elif channel == 'WINNER_256_16' and modulation == '64QAM':
        SNRdB = 16.30254
    elif channel == 'ARGOS_96_8' and modulation == '64QAM':
        SNRdB = 17.58257

    # Average symbol energy for a modulation
    es = generate_input_vector(K,modulation)[1]
    # Noise power used in subsequent calculations
    N0 = ((np.linalg.norm(H)**2)/N)*es*10**(-SNRdB/10)
    # Standard deviation of received vector
    y_std = np.sqrt(0.5*N0*(1+10**(SNRdB/10)))
    # Floating point baseline
    A_H_LMMSE = LMMSE(H,N0,es)

    # Floating point baseline
    # Transmitted vector
    xt = np.zeros(shape=(K,nsamples), dtype = np.complex_)
    # Received vectors
    yt = np.zeros(shape=(N, nsamples), dtype = np.complex_)
    # Pre-slicer estimated vectors
    xt_tilde_lmmse = np.zeros(shape=(K,nsamples), dtype = np.complex_)
    print(f'({nsamples} samples, {K} UEs, {N} BS antennas, {modulation} modulation)')
    print(f"Starting simulation for floating point baseline.")
    for i in tqdm(range(nsamples)):
        xt[:,i] = generate_input_vector(K, modulation)[0]
        # AWGN with specified SNR
        awgn_2d = np.random.multivariate_normal(np.zeros(2),np.eye(2)*N0/2,size=N) 
        n = awgn_2d[:,0] + awgn_2d[:,1]*complex(0,1) 
        # Received vector at time instant i
        yt[:,i] = np.matmul(H,xt[:,i]) + n
        # Pre-slicer estimates
        xt_tilde_lmmse[:,i] = np.matmul(A_H_LMMSE, yt[:,i]).flatten()
    # Pre-slicer EVM results (baseline)
    evm_lmmse_fp = EVM(xt_tilde_lmmse,xt)

    # Fixed point precision sweep
    for B_y in B_y_options:
        # Input clipping range of received vector for quantization
        y_max = np.mean(y_std)*occ_std(B_y)
        y_scale = 2* y_max / (2**B_y)
        for B_w in B_w_options:
            # Received vectors
            yt_fx = np.zeros(shape=(N, nsamples), dtype = np.complex_)
            # Pre-slicer estimated vectors
            xt_tilde_lmmse_fx = np.zeros(shape=(K,nsamples), dtype = np.complex_)
            # Quantizing LMMSE matrix            
            A_H_LMMSE_ranges = np.array([max(np.max(np.abs(np.real(A_H_LMMSE[i,:]))),np.max(np.abs(np.imag(A_H_LMMSE[i,:])))) for i in range(K)])
            A_H_LMMSE_fx, A_H_LMMSE_scaling_factors = quantize_complex_matrix_many_sf(A_H_LMMSE, B_w, A_H_LMMSE_ranges) # quantize matrix
            print(f"Starting simulation for input precision = {B_y} and weight precision = {B_w}.")
            for i in tqdm(range(nsamples)):
                xt[:,i] = generate_input_vector(K, modulation)[0]
                # AWGN with specified SNR
                awgn_2d = np.random.multivariate_normal(np.zeros(2),np.eye(2)*N0/2,size=N) 
                n = awgn_2d[:,0] + awgn_2d[:,1]*complex(0,1) 
                # Received vector at time instant i
                yt[:,i] = np.matmul(H,xt[:,i]) + n
                yt_fx[:,i] = quantize_complex_vector(yt[:,i], B_y, y_max)[0]
                # Pre-slicer estimates
                xt_tilde_lmmse_fx[:,i] = CMVM_quantized_many_sf(A_H_LMMSE_fx, A_H_LMMSE_scaling_factors, yt_fx[:,i], y_scale)[0].flatten()

            # Pre-slicer EVM results
            evm_lmmse_fx[B_y-nbits_min, B_w-nbits_min] = EVM(xt_tilde_lmmse_fx,xt)
    
    qpsk_spec = -15.14
    qam16_spec = -18.06
    qam64_spec = -21.94
    
    # EVM degradation (dB)
    if(modulation=='QPSK'):
        evm_lmmse_fx_margin_dB = qpsk_spec - 20*np.log10(evm_lmmse_fx)
        evm_lmmse_fp_margin_dB = qpsk_spec - 20*np.log10(evm_lmmse_fp)
    elif(modulation=='16QAM'):
        evm_lmmse_fx_margin_dB = qam16_spec - 20*np.log10(evm_lmmse_fx)
        evm_lmmse_fp_margin_dB = qam16_spec - 20*np.log10(evm_lmmse_fp)
    elif(modulation=='64QAM'):
        evm_lmmse_fx_margin_dB = qam64_spec - 20*np.log10(evm_lmmse_fx)
        evm_lmmse_fp_margin_dB = qam64_spec - 20*np.log10(evm_lmmse_fp)
    
    np.save(f'Outputs/evm_By_Bw_{channel}_{modulation}.npy', evm_lmmse_fp)
    np.save(f'Outputs/evm_fx_By_Bw_{channel}_{modulation}.npy', evm_lmmse_fx)
    np.save(f'Outputs/evm_fx_margin_dB_By_Bw_{channel}_{modulation}.npy', evm_lmmse_fx_margin_dB)
    np.save(f'Outputs/evm_fp_margin_dB_By_Bw_{channel}_{modulation}.npy', evm_lmmse_fp_margin_dB)

    
    # Plotting the heatmap
    # Annotate each square with its value
    plt.figure(figsize=(8, 6))
    heatmap = plt.imshow(evm_lmmse_fx_margin_dB, origin='lower', extent=[nbits_min-0.5, nbits_max-0.5, nbits_min-0.5, nbits_max-0.5], 
                        cmap='viridis', aspect='auto', norm=Normalize(vmin=0, vmax=3))
    
    nrows, ncols = evm_lmmse_fx_margin_dB.shape
    for i in range(nrows):
        for j in range(ncols):
            plt.text(j + nbits_min, i + nbits_min, f"{evm_lmmse_fx_margin_dB[i, j]:.2f}", 
                    ha='center', va='center', color="white" if heatmap.norm(evm_lmmse_fx_margin_dB[i, j]) < 0.5 else "black", 
                    fontsize=14)
    
    # Add colorbar with label
    colorbar = plt.colorbar(heatmap)
    # colorbar.ax.set_ylabel("EVM degradation (dB)", fontsize=18)
    colorbar.ax.set_ylabel("EVM margin (dB)", fontsize=18)
    colorbar.ax.tick_params(labelsize=14)
    # Adjusting ticks to be centered on squares
    nbits_range = np.arange(nbits_min, nbits_max)  # Assuming integer values for ticks
    plt.xticks(ticks=nbits_range, labels=nbits_range, fontsize=16)  # Set x-axis ticks and font size
    plt.yticks(ticks=nbits_range, labels=nbits_range, fontsize=16)  # Set y-axis ticks and font size
    plt.title(f'EVM margin for digital FP baseline = {evm_lmmse_fp_margin_dB:.2f} dB', fontsize = 13)
    plt.xlabel("$B_y$ (bits)", fontsize = 18)
    plt.ylabel("$B_w$ (bits)", fontsize = 18)
    plt.grid(False)
    plt.savefig(f'Figures/evm_fx_margin_dB_By_Bw_{channel}_{modulation}.png', format='png', bbox_inches='tight')
    # plt.show()
    

def plot_Bw_By_sweep(channel, modulation):

    nbits_min = 4
    nbits_max = 10
    
    evm_lmmse_fp = np.load(f'Outputs/evm_By_Bw_{channel}_{modulation}.npy')
    evm_lmmse_fx = np.load(f'Outputs/evm_fx_By_Bw_{channel}_{modulation}.npy')
    evm_lmmse_fx_margin_dB = np.load(f'Outputs/evm_fx_margin_dB_By_Bw_{channel}_{modulation}.npy')
    evm_lmmse_fp_margin_dB = np.load(f'Outputs/evm_fp_margin_dB_By_Bw_{channel}_{modulation}.npy')
    
    # Plotting the heatmap
    # Annotate each square with its value
    plt.figure(figsize=(8, 6))
    heatmap = plt.imshow(evm_lmmse_fx_margin_dB, origin='lower', extent=[nbits_min-0.5, nbits_max-0.5, nbits_min-0.5, nbits_max-0.5], 
                        cmap='viridis', aspect='auto', norm=Normalize(vmin=0, vmax=3))
    
    nrows, ncols = evm_lmmse_fx_margin_dB.shape
    for i in range(nrows):
        for j in range(ncols):
            plt.text(j + nbits_min, i + nbits_min, f"{evm_lmmse_fx_margin_dB[i, j]:.2f}", 
                    ha='center', va='center', color="white" if heatmap.norm(evm_lmmse_fx_margin_dB[i, j]) < 0.5 else "black", 
                    fontsize=14)
    
    # Add colorbar with label
    colorbar = plt.colorbar(heatmap)
    # colorbar.ax.set_ylabel("EVM degradation (dB)", fontsize=18)
    colorbar.ax.set_ylabel("EVM margin (dB)", fontsize=18)
    colorbar.ax.tick_params(labelsize=14)
    # Adjusting ticks to be centered on squares
    nbits_range = np.arange(nbits_min, nbits_max)  # Assuming integer values for ticks
    plt.xticks(ticks=nbits_range, labels=nbits_range, fontsize=16)  # Set x-axis ticks and font size
    plt.yticks(ticks=nbits_range, labels=nbits_range, fontsize=16)  # Set y-axis ticks and font size
    plt.title(f'EVM margin for digital FP baseline = {evm_lmmse_fp_margin_dB:.2f} dB', fontsize = 13)
    plt.xlabel("$B_y$ (bits)", fontsize = 18)
    plt.ylabel("$B_w$ (bits)", fontsize = 18)
    plt.grid(False)
    plt.savefig(f'Figures/evm_fx_margin_dB_By_Bw_{channel}_{modulation}.png', format='png', bbox_inches='tight')
    # plt.show()

    M = 1
    if(modulation=='QPSK'):
        M = 2
    elif(modulation == '16QAM'):
        M = 4
    elif(modulation == '64QAM'):
        M = 8

    gamma = 500
    E_mac = 12.24
    E_read = 124.25
    E_write = 135
    N, K = tuple(map(int, re.findall(r'\d+', channel)[-2:]))
    E_bpsk_1b = 4*(E_mac+E_read)*N + 2*E_write*N/(gamma)

    normalized_ecmvm = np.zeros((nbits_max-nbits_min,nbits_max-nbits_min))
    for i in range(nbits_max-nbits_min):
        for j in range(nbits_max-nbits_min):
            normalized_ecmvm[i,j] = (4*(E_mac+E_read)*N*(nbits_min + i)*(nbits_min + j)/M + 2*E_write*N*(nbits_min + j)/(M*gamma))/E_bpsk_1b
    
    # Plotting the heatmap
    plt.figure(figsize=(8, 6))
    heatmap = plt.imshow(normalized_ecmvm, origin='lower', extent=[nbits_min-0.5, nbits_max-0.5, nbits_min-0.5, nbits_max-0.5], 
                        cmap='viridis', aspect='auto', norm=Normalize(vmin=0, vmax=20))
    
    # Annotate each square with its value
    nrows, ncols = normalized_ecmvm.shape
    for i in range(nrows):
        for j in range(ncols):
            plt.text(j + nbits_min, i + nbits_min, f"{normalized_ecmvm[i, j]:.2f}", 
                    ha='center', va='center', color="white" if heatmap.norm(normalized_ecmvm[i, j]) < 0.5 else "black", 
                    fontsize=14)
    
    # Add colorbar with label
    colorbar = plt.colorbar(heatmap)
    colorbar.ax.set_ylabel(r"$E_\mathrm{b}$ (normalized)", fontsize=18)
    colorbar.ax.tick_params(labelsize=14)
    # Adjusting ticks to be centered on squares
    nbits_range = np.arange(nbits_min, nbits_max)  # Assuming integer values for ticks
    plt.xticks(ticks=nbits_range, labels=nbits_range, fontsize=16)  # Set x-axis ticks and font size
    plt.yticks(ticks=nbits_range, labels=nbits_range, fontsize=16)  # Set y-axis ticks and font size
    # plt.title(f'{modulation}')
    plt.xlabel("$B_y$ (bits)", fontsize = 18)
    plt.ylabel("$B_w$ (bits)", fontsize = 18)
    plt.grid(False)
    plt.savefig(f'Figures/Eb_By_Bw_{modulation}.png', format='png', bbox_inches='tight')
    # plt.show()

if __name__ == '__main__':

    # Bw_By_sweep('WINNER_64_8', 'QPSK', 20000)
    # Bw_By_sweep('WINNER_128_16', 'QPSK', 10000)
    # Bw_By_sweep('WINNER_256_16', 'QPSK', 10000)
    # Bw_By_sweep('ARGOS_96_8', 'QPSK', 20000)
    # Bw_By_sweep('WINNER_64_8', '16QAM', 20000)
    # Bw_By_sweep('WINNER_128_16', '16QAM', 10000)
    # Bw_By_sweep('WINNER_256_16', '16QAM', 10000)
    # Bw_By_sweep('ARGOS_96_8', '16QAM', 20000)
    # Bw_By_sweep('WINNER_64_8', '64QAM', 20000)
    # Bw_By_sweep('WINNER_128_16', '64QAM', 10000)
    # Bw_By_sweep('WINNER_256_16', '64QAM', 10000)
    # Bw_By_sweep('ARGOS_96_8', '64QAM', 20000)

    plot_Bw_By_sweep('ARGOS_96_8', 'QPSK')
    plot_Bw_By_sweep('ARGOS_96_8', '16QAM')
    plot_Bw_By_sweep('ARGOS_96_8', '64QAM')
