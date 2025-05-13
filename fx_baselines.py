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

def fx_baselines(channel, modulation, nsamples):
    N, K = tuple(map(int, re.findall(r'\d+', channel)[-2:]))
    H = extract_H(channel)

    SNRdB_min = -4
    SNRdB_max = 30
    npoints = 35
    SNRdB_options = np.linspace(SNRdB_min, SNRdB_max, npoints)

    B_y = 6
    B_w = 6
    if modulation == 'QPSK':
        B_w = 5
        B_y = 5
    elif modulation == '16QAM':
        B_w = 6
        B_y = 6
    elif modulation == '64QAM':
        B_w = 7
        B_y = 7

    # EVM
    evm_lmmse_fx = np.zeros(npoints)
    # SER
    ser_lmmse_fx = np.zeros(npoints)
    
    print(f'({nsamples} samples, {K} UEs, {N} BS antennas, {modulation} modulation)')
    logging.basicConfig(filename=f'Outputs/fp_rx_snr_sweep_{channel}_{modulation}.log', level=logging.INFO, filemode='w')
    for idx in range(npoints):
        # Transmitted vectors
        xt = np.zeros(shape=(K,nsamples), dtype = np.complex_)
        # Received vectors
        yt = np.zeros(shape=(N, nsamples), dtype = np.complex_)
        yt_fx = np.zeros(shape=(N, nsamples), dtype = np.complex_)
        # Pre-slicer estimated vectors
        xt_tilde_lmmse_fx = np.zeros(shape=(K,nsamples), dtype = np.complex_)
        # Post-slicer estimated vectors
        xt_hat_lmmse_fx = np.zeros(shape=(K,nsamples), dtype = np.complex_)

        # Average symbol energy for a modulation
        es = generate_input_vector(K,modulation)[1]
        # Noise power used in subsequent calculations
        N0 = ((np.linalg.norm(H)**2)/N)*es*10**(-SNRdB_options[idx]/10)
        # Standard deviation of received vector
        y_std = np.sqrt(0.5*N0*(1+10**(SNRdB_options[idx]/10)))
        # Input clipping range of received vector for quantization
        y_max = np.mean(y_std)*occ_std(B_y)
        y_scale = 2* y_max / (2**B_y)
        # LMMSE matrix
        A_H_LMMSE = LMMSE(H,N0,es)
        # Quantizing LMMSE matrix            
        A_H_LMMSE_ranges = np.array([max(np.max(np.abs(np.real(A_H_LMMSE[i,:]))),np.max(np.abs(np.imag(A_H_LMMSE[i,:])))) for i in range(K)])
        A_H_LMMSE_fx, A_H_LMMSE_scaling_factors = quantize_complex_matrix_many_sf(A_H_LMMSE, B_w, A_H_LMMSE_ranges) # quantize matrix
        
        print(f"Starting simulation for RX SNR = {SNRdB_options[idx]}")
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
            # Post-slicer estimates
            xt_hat_lmmse_fx[:,i] = slicer(xt_tilde_lmmse_fx[:,i], modulation)
        # EVM
        evm_lmmse_fx[idx] = EVM(xt_tilde_lmmse_fx,xt) 
        # SER
        ser_lmmse_fx[idx] = SER(xt_hat_lmmse_fx,xt)
        logging.info("Finished simulation for RX SNR = {}".format(SNRdB_options[idx]))
        logging.info("EVM LMMSE FX = {}".format(evm_lmmse_fx[idx]))
        logging.info("SER LMMSE FX = {}".format(ser_lmmse_fx[idx]))

    np.save(f'Outputs/fx_baselines_rx_snr_options_{channel}_{modulation}.npy', SNRdB_options)
    np.save(f'Outputs/fx_baselines_evm_rx_snr_{channel}_{modulation}.npy', evm_lmmse_fx)
    np.save(f'Outputs/fx_baselines_ser_rx_snr_{channel}_{modulation}.npy', ser_lmmse_fx)
    
    plt.figure(figsize=(6,4))
    plt.plot(SNRdB_options, 20*np.log10(evm_lmmse_fx), '-^', label='FX', markerfacecolor='none', markeredgecolor='green', color = 'green')
    plt.grid()
    plt.xlabel('RX SNR (dB)',fontsize = 13)
    plt.ylabel('EVM (dB)', fontsize = 13)
    plt.legend(fontsize=8)
    plt.savefig(f'Figures/fx_baselines_evm_rx_snr_{channel}_{modulation}.png', format='png', bbox_inches='tight')

    plt.figure(figsize=(6,4))
    plt.semilogy(SNRdB_options, ser_lmmse_fx, '-^', label='FX', markerfacecolor='none', markeredgecolor='green', color = 'green')
    plt.grid()
    plt.xlabel('RX SNR (dB)',fontsize = 13)
    plt.ylabel('SER', fontsize = 13)
    plt.legend(fontsize=8)
    plt.savefig(f'Figures/fx_baselines_ser_rx_snr_{channel}_{modulation}.png', format='png', bbox_inches='tight')

def plot_fx_baselines():
    SNRdB_min = -4
    SNRdB_max = 30
    npoints = 35
    SNRdB_options = np.linspace(SNRdB_min, SNRdB_max, npoints)

    evm_WINNER_64_8_QPSK = np.load('Outputs/fp_baselines_evm_rx_snr_WINNER_64_8_QPSK.npy')
    evm_WINNER_64_8_16QAM = np.load('Outputs/fp_baselines_evm_rx_snr_WINNER_64_8_16QAM.npy')
    evm_WINNER_64_8_64QAM = np.load('Outputs/fp_baselines_evm_rx_snr_WINNER_64_8_64QAM.npy')
    evm_WINNER_128_16_QPSK = np.load('Outputs/fp_baselines_evm_rx_snr_WINNER_128_16_QPSK.npy')
    evm_WINNER_128_16_16QAM = np.load('Outputs/fp_baselines_evm_rx_snr_WINNER_128_16_16QAM.npy')
    evm_WINNER_128_16_64QAM = np.load('Outputs/fp_baselines_evm_rx_snr_WINNER_128_16_64QAM.npy')
    evm_WINNER_256_16_QPSK = np.load('Outputs/fp_baselines_evm_rx_snr_WINNER_256_16_QPSK.npy')
    evm_WINNER_256_16_16QAM = np.load('Outputs/fp_baselines_evm_rx_snr_WINNER_256_16_16QAM.npy')
    evm_WINNER_256_16_64QAM = np.load('Outputs/fp_baselines_evm_rx_snr_WINNER_256_16_64QAM.npy')
    evm_ARGOS_96_8_QPSK = np.load('Outputs/fp_baselines_evm_rx_snr_ARGOS_96_8_QPSK.npy')
    evm_ARGOS_96_8_16QAM = np.load('Outputs/fp_baselines_evm_rx_snr_ARGOS_96_8_16QAM.npy')
    evm_ARGOS_96_8_64QAM = np.load('Outputs/fp_baselines_evm_rx_snr_ARGOS_96_8_64QAM.npy')

    evm_WINNER_64_8_QPSK_fx = np.load('Outputs/fx_baselines_evm_rx_snr_WINNER_64_8_QPSK.npy')
    evm_WINNER_64_8_16QAM_fx = np.load('Outputs/fx_baselines_evm_rx_snr_WINNER_64_8_16QAM.npy')
    evm_WINNER_64_8_64QAM_fx = np.load('Outputs/fx_baselines_evm_rx_snr_WINNER_64_8_64QAM.npy')
    evm_WINNER_128_16_QPSK_fx = np.load('Outputs/fx_baselines_evm_rx_snr_WINNER_128_16_QPSK.npy')
    evm_WINNER_128_16_16QAM_fx = np.load('Outputs/fx_baselines_evm_rx_snr_WINNER_128_16_16QAM.npy')
    evm_WINNER_128_16_64QAM_fx = np.load('Outputs/fx_baselines_evm_rx_snr_WINNER_128_16_64QAM.npy')
    evm_WINNER_256_16_QPSK_fx = np.load('Outputs/fx_baselines_evm_rx_snr_WINNER_256_16_QPSK.npy')
    evm_WINNER_256_16_16QAM_fx = np.load('Outputs/fx_baselines_evm_rx_snr_WINNER_256_16_16QAM.npy')
    evm_WINNER_256_16_64QAM_fx = np.load('Outputs/fx_baselines_evm_rx_snr_WINNER_256_16_64QAM.npy')
    evm_ARGOS_96_8_QPSK_fx = np.load('Outputs/fx_baselines_evm_rx_snr_ARGOS_96_8_QPSK.npy')
    evm_ARGOS_96_8_16QAM_fx = np.load('Outputs/fx_baselines_evm_rx_snr_ARGOS_96_8_16QAM.npy')
    evm_ARGOS_96_8_64QAM_fx = np.load('Outputs/fx_baselines_evm_rx_snr_ARGOS_96_8_64QAM.npy')

    fig, ax = plt.subplots(figsize=(6,4))
    plt.plot(SNRdB_options, 20*np.log10(evm_WINNER_64_8_QPSK), linestyle = 'dashed', color = 'red', linewidth = 1)
    # plt.plot(SNRdB_options, 20*np.log10(evm_WINNER_64_8_16QAM), linestyle = 'dashed', color = 'red', linewidth = 1)
    # plt.plot(SNRdB_options, 20*np.log10(evm_WINNER_64_8_64QAM), linestyle = 'dashed', color = 'red', linewidth = 1)
    plt.plot(SNRdB_options, 20*np.log10(evm_WINNER_128_16_QPSK),linestyle = 'dashed', color = 'green', linewidth = 1)
    # plt.plot(SNRdB_options, 20*np.log10(evm_WINNER_128_16_16QAM), linestyle = 'dashed', color = 'green', linewidth = 1)  
    # plt.plot(SNRdB_options, 20*np.log10(evm_WINNER_128_16_64QAM),linestyle = 'dashed', color = 'green', linewidth = 1)
    plt.plot(SNRdB_options, 20*np.log10(evm_WINNER_256_16_QPSK), linestyle = 'dashed', color = 'blue', linewidth = 1)
    # plt.plot(SNRdB_options, 20*np.log10(evm_WINNER_256_32_16QAM),  linestyle = 'dashed', color = 'blue', linewidth = 1)    
    # plt.plot(SNRdB_options, 20*np.log10(evm_WINNER_256_32_64QAM), linestyle = 'dashed', color = 'blue', linewidth = 1)
    plt.plot(SNRdB_options, 20*np.log10(evm_ARGOS_96_8_QPSK), linestyle = 'dashed', color = 'orange', linewidth = 1)
    # plt.plot(SNRdB_options, 20*np.log10(evm_ARGOS_96_8_16QAM), linestyle = 'dashed', color = 'orange', linewidth = 1)
    # plt.plot(SNRdB_options, 20*np.log10(evm_ARGOS_96_8_64QAM),  linestyle = 'dashed', color = 'orange', linewidth = 1)
    
    plt.plot(SNRdB_options, 20*np.log10(evm_WINNER_64_8_QPSK_fx), '-o', label=r'WIN-64x8, QPSK ($B_y=B_w=5$)', markerfacecolor='none', markeredgecolor='red', color = 'red')
    plt.plot(SNRdB_options, 20*np.log10(evm_WINNER_64_8_16QAM_fx), '-s', label=r'WIN-64x8, 16-QAM ($B_y=B_w=6$)', markerfacecolor='none', markeredgecolor='red', color = 'red')
    plt.plot(SNRdB_options, 20*np.log10(evm_WINNER_64_8_64QAM_fx), '-^', label=r'WIN-64x8, 64-QAM ($B_y=B_w=7$)', markerfacecolor='none', markeredgecolor='red', color = 'red')
    plt.plot(SNRdB_options, 20*np.log10(evm_WINNER_128_16_QPSK_fx), '-o', label=r'WIN-128x16, QPSK ($B_y=B_w=5$)', markerfacecolor='none', markeredgecolor='green', color = 'green')
    plt.plot(SNRdB_options, 20*np.log10(evm_WINNER_128_16_16QAM_fx), '-s', label=r'WIN-128x16, 16-QAM ($B_y=B_w=6$)', markerfacecolor='none', markeredgecolor='green', color = 'green')  
    plt.plot(SNRdB_options, 20*np.log10(evm_WINNER_128_16_64QAM_fx), '-^', label=r'WIN-128x16, 64-QAM ($B_y=B_w=7$)', markerfacecolor='none', markeredgecolor='green', color = 'green')
    plt.plot(SNRdB_options, 20*np.log10(evm_WINNER_256_16_QPSK_fx), '-o', label=r'WIN-256x16, QPSK ($B_y=B_w=5$)', markerfacecolor='none', markeredgecolor='blue', color = 'blue')
    plt.plot(SNRdB_options, 20*np.log10(evm_WINNER_256_16_16QAM_fx), '-s', label=r'WIN-256x16, 16-QAM ($B_y=B_w=6$)', markerfacecolor='none', markeredgecolor='blue', color = 'blue')    
    plt.plot(SNRdB_options, 20*np.log10(evm_WINNER_256_16_64QAM_fx), '-^', label=r'WIN-256x16, 64-QAM ($B_y=B_w=7$)', markerfacecolor='none', markeredgecolor='blue', color = 'blue')
    plt.plot(SNRdB_options, 20*np.log10(evm_ARGOS_96_8_QPSK_fx), '-o', label=r'ARGOS-96x8, QPSK ($B_y=B_w=5$)', markerfacecolor='none', markeredgecolor='orange', color = 'orange')
    plt.plot(SNRdB_options, 20*np.log10(evm_ARGOS_96_8_16QAM_fx), '-s', label=r'ARGOS-96x8, 16-QAM ($B_y=B_w=6$)', markerfacecolor='none', markeredgecolor='orange', color = 'orange')
    plt.plot(SNRdB_options, 20*np.log10(evm_ARGOS_96_8_64QAM_fx), '-^', label=r'ARGOS-96x8, 64-QAM ($B_y=B_w=7$)', markerfacecolor='none', markeredgecolor='orange', color = 'orange')
    
    plt.axhline(y = -15.14, color = 'purple', linewidth = 0.8)
    plt.text(x=0.9*SNRdB_options[0], y=-15.14 + 0.3, s='QPSK', color='purple', fontsize=9)
    plt.axhline(y = -18.06,  color = 'purple', linewidth = 0.8)
    plt.text(x=0.9*SNRdB_options[0], y=-18.06 + 0.3, s='16-QAM', color='purple', fontsize=9)
    plt.axhline(y = -21.94,  color = 'purple', linewidth = 0.8)
    plt.text(x=0.9*SNRdB_options[0], y=-21.94 + 0.3, s='64-QAM', color='purple', fontsize=9)
    plt.grid()
    plt.xlim((-4,27))
    plt.ylim((-30,-8))
    plt.xlabel('RX SNR (dB)',fontsize = 13)
    plt.ylabel('EVM (dB)', fontsize = 13)
    # fl_legend = Line2D([0], [0], color='black', linestyle='--', linewidth=1, label="Digital FP baselines")
    # # Get existing legend handles and labels
    # handles, labels = plt.gca().get_legend_handles_labels()
    # # Append the new custom legend entry
    # handles.append(fl_legend)
    # labels.append("Digital FP baselines")
    # plt.legend(handles=handles, labels=labels,fontsize=5.8, ncol=2)
    # Get existing legend handles and labels
    fp_legend = Line2D([0], [0], color='black', linestyle='--', linewidth=1, label="Digital FP baselines")
    evm_legend = Line2D([0], [0], color='purple', linewidth=1, label="3GPP EVM specifications")


    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(fp_legend)
    labels.append("Digital FP baselines")
    handles.append(evm_legend)
    labels.append("3GPP EVM specifications")

    # Add the custom legend entry separately
    main_handles, main_labels = handles[:-2], labels[:-2]  # First 12 legends
    shared_handles, shared_labels = handles[-2:], labels[-2:]  # The last legend

    # First, plot the main legend (6 per column)
    legend1 = plt.legend(main_handles, main_labels, fontsize=5.2, ncol=2, loc='lower left')
    legend2 = plt.legend(shared_handles, shared_labels, fontsize=7, loc='upper right')

    # Then, add the shared legend entry centered below
    plt.gca().add_artist(legend1)  # Keep the first legend
    plt.gca().add_artist(legend2)  # Keep the second legend
    # plt.legend([shared_handle], [shared_label], fontsize=6, loc='upper right', frameon=True)
    # Draw the double-sided arrow
    ax.annotate(
        "", xy=(15.63112, -17.9), xytext=(15.63112, -21),
        arrowprops=dict(arrowstyle='<->', linewidth=1.4, color='black')
    )

    # Add the "EVM margin" text
    ax.text(16, (-21 + (-18.06)) / 2.01, "EVM margin",
            verticalalignment='center', fontsize=8.3)

    plt.savefig(f'Figures/fp_fx_baselines_evm_rx_snr_all.png', format='png', bbox_inches='tight')
    # plt.show()

if __name__ == '__main__':

    # fx_baselines('WINNER_64_8', 'QPSK', 40000)
    # fx_baselines('WINNER_128_16', 'QPSK', 20000)
    # fx_baselines('WINNER_256_16', 'QPSK', 20000)
    # fx_baselines('ARGOS_96_8', 'QPSK', 40000)
    # fx_baselines('WINNER_64_8', '16QAM', 40000)
    # fx_baselines('WINNER_128_16', '16QAM', 20000)
    # fx_baselines('WINNER_256_16', '16QAM', 20000)
    # fx_baselines('ARGOS_96_8', '16QAM', 40000)
    # fx_baselines('WINNER_64_8', '64QAM', 40000)
    # fx_baselines('WINNER_128_16', '64QAM', 20000)
    # fx_baselines('WINNER_256_16', '64QAM', 20000)
    # fx_baselines('ARGOS_96_8', '64QAM', 40000)

    plot_fx_baselines()

