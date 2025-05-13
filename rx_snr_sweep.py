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


def rx_snr_sweep(channel, modulation, nsamples):
    N, K = tuple(map(int, re.findall(r'\d+', channel)[-2:]))
    H = extract_H(channel)
    B_y = 6
    B_w = 6
    B_ADC = int(np.ceil(np.log2(N))) - 1
    sigma_ADC = 0.0005
    cap = 1
    VDD = 0.9
    SNRdB_min = 13 
    SNRdB_max = 26
    npoints = 15
    if channel == 'WINNER_64_8' and modulation == 'QPSK':
        SNRdB_min = -1
        SNRdB_max = 13
        B_w = 5
        B_y = 5
        B_ADC = 6
    elif channel == 'WINNER_128_16' and modulation == 'QPSK':
        SNRdB_min = -1
        SNRdB_max = 13
        B_w = 5
        B_y = 5
    elif channel == 'WINNER_256_16' and modulation == 'QPSK':
        SNRdB_min = -1
        SNRdB_max = 13
        B_w = 5
        B_y = 5
    elif channel == 'ARGOS_96_8' and modulation == 'QPSK':
        SNRdB_min = -1
        SNRdB_max = 13
        B_w = 5
        B_y = 5
    elif channel == 'WINNER_64_8' and modulation == '16QAM':
        SNRdB_min = 4
        SNRdB_max = 19
        B_w = 6
        B_y = 6
        B_ADC = 6
    elif channel == 'WINNER_128_16' and modulation == '16QAM':
        SNRdB_min = 4
        SNRdB_max = 19
        B_w = 6
        B_y = 6
    elif channel == 'WINNER_256_16' and modulation == '16QAM':
        SNRdB_min = 4
        SNRdB_max = 19
        B_w = 6
        B_y = 6
    elif channel == 'ARGOS_96_8' and modulation == '16QAM':
        SNRdB_min = 4
        SNRdB_max = 19
        B_w = 6
        B_y = 6
    elif channel == 'WINNER_64_8' and modulation == '64QAM':
        SNRdB_min = 9
        SNRdB_max = 24
        B_w = 7
        B_y = 7
        B_ADC = 6
    elif channel == 'WINNER_128_16' and modulation == '64QAM':
        SNRdB_min = 9
        SNRdB_max = 24
        B_w = 7
        B_y = 7
    elif channel == 'WINNER_256_16' and modulation == '64QAM':
        SNRdB_min = 9
        SNRdB_max = 24
        B_w = 7
        B_y = 7
    elif channel == 'ARGOS_96_8' and modulation == '64QAM':
        SNRdB_min = 9
        SNRdB_max = 24
        B_w = 7
        B_y = 7

    SNRdB_options = np.linspace(SNRdB_min, SNRdB_max, npoints)
    nsamples_options = np.logspace(np.log10(nsamples), np.log10(nsamples)+0.75, npoints, dtype=int)

    # EVM
    evm_lmmse = np.zeros(npoints)
    evm_lmmse_fx = np.zeros(npoints)
    evm_lmmse_IMC = np.zeros(npoints)
    # SER
    ser_lmmse = np.zeros(npoints)
    ser_lmmse_fx = np.zeros(npoints)
    ser_lmmse_IMC = np.zeros(npoints)

    print(f'({nsamples} samples, {K} UEs, {N} BS antennas, {modulation} modulation, {B_y} bit input, {B_w} bit weights, {cap} bit-cell capacitance, {B_ADC} bit ADC, {VDD} V supply, {sigma_ADC} ADC noise)')
    logging.basicConfig(filename=f'Outputs/rx_snr_sweep_{channel}_{modulation}.log', level=logging.INFO, filemode='w')

    for idx in range(npoints):
        # Transmitted vectors
        xt = np.zeros(shape=(K,nsamples_options[idx]), dtype = np.complex_)
        # Received vectors
        yt = np.zeros(shape=(N, nsamples_options[idx]), dtype = np.complex_)
        yt_fx = np.zeros(shape=(N, nsamples_options[idx]), dtype = np.complex_)
        # Pre-slicer estimated vectors
        xt_tilde_lmmse = np.zeros(shape=(K,nsamples_options[idx]), dtype = np.complex_)
        xt_tilde_lmmse_fx = np.zeros(shape=(K,nsamples_options[idx]), dtype = np.complex_)
        xt_tilde_lmmse_IMC = np.zeros(shape=(K,nsamples_options[idx]), dtype = np.complex_)
        # Post-slicer estimated vectors
        xt_hat_lmmse = np.zeros(shape=(K,nsamples_options[idx]), dtype = np.complex_)
        xt_hat_lmmse_fx = np.zeros(shape=(K,nsamples_options[idx]), dtype = np.complex_)
        xt_hat_lmmse_IMC = np.zeros(shape=(K,nsamples_options[idx]), dtype = np.complex_)

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
        
        mimo_imc_lmmse = mimo_imc(n_phy_rows=int(2**(np.ceil(np.log2(N)))), n_phy_cols=K*B_w, b_w = B_w, b_x=B_y, b_adc= B_ADC,c_qr_mean=cap, sigma_adc=sigma_ADC, v_dd= VDD)
        mimo_imc_lmmse.store_complex_weights(A_H_LMMSE, A_H_LMMSE_ranges)

        print(f"Starting simulation for RX SNR = {SNRdB_options[idx]}")
        for i in tqdm(range(nsamples_options[idx])):
            mimo_imc_lmmse.update_b_adc_mimo(N*0.25, B_ADC)
            xt[:,i] = generate_input_vector(K, modulation)[0]
            # AWGN with specified SNR
            awgn_2d = np.random.multivariate_normal(np.zeros(2),np.eye(2)*N0/2,size=N) 
            n = awgn_2d[:,0] + awgn_2d[:,1]*complex(0,1) 
            # Received vector at time instant i
            yt[:,i] = np.matmul(H,xt[:,i]) + n
            yt_fx[:,i] = quantize_complex_vector(yt[:,i], B_y, y_max)[0]
            # Pre-slicer estimates
            xt_tilde_lmmse[:,i] = np.matmul(A_H_LMMSE, yt[:,i]).flatten()
            xt_tilde_lmmse_fx[:,i] = CMVM_quantized_many_sf(A_H_LMMSE_fx, A_H_LMMSE_scaling_factors, yt_fx[:,i], y_scale)[0].flatten()
            xt_tilde_lmmse_IMC[:,i] = mimo_imc_lmmse.compute_complex_mvm(yt[:,i], y_max)
            # Post-slicer estimates
            xt_hat_lmmse[:,i] = slicer(xt_tilde_lmmse[:,i], modulation)
            xt_hat_lmmse_fx[:,i] = slicer(xt_tilde_lmmse_fx[:,i], modulation)
            xt_hat_lmmse_IMC[:,i] = slicer(xt_tilde_lmmse_IMC[:,i], modulation)
        # EVM
        evm_lmmse[idx] = EVM(xt_tilde_lmmse,xt) 
        evm_lmmse_fx[idx] = EVM(xt_tilde_lmmse_fx,xt) 
        evm_lmmse_IMC[idx] = EVM(xt_tilde_lmmse_IMC,xt) 
        # SER
        ser_lmmse[idx] = SER(xt_hat_lmmse, xt)
        ser_lmmse_fx[idx] = SER(xt_hat_lmmse_fx,xt) 
        ser_lmmse_IMC[idx] = SER(xt_hat_lmmse_IMC,xt)
        logging.info("Finished simulation for RX SNR = {}".format(SNRdB_options[idx]))
        logging.info("EVM LMMSE = {}".format(evm_lmmse[idx]))
        logging.info("EVM LMMSE FX = {}".format(evm_lmmse_fx[idx]))
        logging.info("EVM LMMSE IMC = {}".format(evm_lmmse_IMC[idx]))
        logging.info("SER LMMSE = {}".format(ser_lmmse[idx]))
        logging.info("SER LMMSE FX = {}".format(ser_lmmse_fx[idx]))
        logging.info("SER LMMSE IMC = {}".format(ser_lmmse_IMC[idx]))

    np.save(f'Outputs/rx_snr_options_{channel}_{modulation}.npy', SNRdB_options)
    np.save(f'Outputs/evm_rx_snr_{channel}_{modulation}.npy', evm_lmmse)
    np.save(f'Outputs/evm_fx_rx_snr_{channel}_{modulation}.npy', evm_lmmse_fx)
    np.save(f'Outputs/evm_imc_rx_snr_{channel}_{modulation}.npy', evm_lmmse_IMC)
    np.save(f'Outputs/ser_rx_snr_{channel}_{modulation}.npy', ser_lmmse)
    np.save(f'Outputs/ser_fx_rx_snr_{channel}_{modulation}.npy', ser_lmmse_fx)
    np.save(f'Outputs/ser_imc_rx_snr_{channel}_{modulation}.npy', ser_lmmse_IMC)
    
    plt.figure(figsize=(6,4))
    plt.plot(SNRdB_options, 20*np.log10(evm_lmmse_IMC), '-^', label='IMC', markerfacecolor='none', markeredgecolor='red', color = 'red')
    plt.plot(SNRdB_options, 20*np.log10(evm_lmmse), '-^', label='FP', markerfacecolor='none', markeredgecolor='green', color = 'green')
    plt.plot(SNRdB_options, 20*np.log10(evm_lmmse_fx), '-^', label='FX', markerfacecolor='none', markeredgecolor='blue', color = 'blue')
    plt.grid()
    plt.xlabel('RX SNR (dB)',fontsize = 13)
    plt.ylabel('EVM (dB)', fontsize = 13)
    plt.title(f'B_y = {B_y}, B_w = {B_w}')
    plt.legend(fontsize=8)
    plt.savefig(f'Figures/evm_rx_snr_{channel}_{modulation}.png', format='png', bbox_inches='tight')

    plt.figure(figsize=(6,4))
    plt.semilogy(SNRdB_options, ser_lmmse_IMC, '-^', label='IMC', markerfacecolor='none', markeredgecolor='red', color = 'red')
    plt.semilogy(SNRdB_options, ser_lmmse, '-^', label='FP', markerfacecolor='none', markeredgecolor='green', color = 'green')
    plt.semilogy(SNRdB_options, ser_lmmse_fx, '-^', label='FX', markerfacecolor='none', markeredgecolor='blue', color = 'blue')
    plt.grid()
    plt.xlabel('RX SNR (dB)',fontsize = 13)
    plt.ylabel('SER', fontsize = 13)
    plt.title(f'B_y = {B_y}, B_w = {B_w}')
    plt.legend(fontsize=8)
    plt.savefig(f'Figures/ser_rx_snr_sweep_{channel}_{modulation}.png', format='png', bbox_inches='tight')

def plot_rx_snr_sweep():

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

    ser_WINNER_64_8_QPSK_fx = np.load('Outputs/fx_baselines_ser_rx_snr_WINNER_64_8_QPSK.npy')
    ser_WINNER_64_8_16QAM_fx = np.load('Outputs/fx_baselines_ser_rx_snr_WINNER_64_8_16QAM.npy')
    ser_WINNER_64_8_64QAM_fx = np.load('Outputs/fx_baselines_ser_rx_snr_WINNER_64_8_64QAM.npy')
    ser_WINNER_128_16_QPSK_fx = np.load('Outputs/fx_baselines_ser_rx_snr_WINNER_128_16_QPSK.npy')
    ser_WINNER_128_16_16QAM_fx = np.load('Outputs/fx_baselines_ser_rx_snr_WINNER_128_16_16QAM.npy')
    ser_WINNER_128_16_64QAM_fx = np.load('Outputs/fx_baselines_ser_rx_snr_WINNER_128_16_64QAM.npy')
    ser_WINNER_256_16_QPSK_fx = np.load('Outputs/fx_baselines_ser_rx_snr_WINNER_256_16_QPSK.npy')
    ser_WINNER_256_16_16QAM_fx = np.load('Outputs/fx_baselines_ser_rx_snr_WINNER_256_16_16QAM.npy')
    ser_WINNER_256_16_64QAM_fx = np.load('Outputs/fx_baselines_ser_rx_snr_WINNER_256_16_64QAM.npy')
    ser_ARGOS_96_8_QPSK_fx = np.load('Outputs/fx_baselines_ser_rx_snr_ARGOS_96_8_QPSK.npy')
    ser_ARGOS_96_8_16QAM_fx = np.load('Outputs/fx_baselines_ser_rx_snr_ARGOS_96_8_16QAM.npy')
    ser_ARGOS_96_8_64QAM_fx = np.load('Outputs/fx_baselines_ser_rx_snr_ARGOS_96_8_64QAM.npy')

    evm_WINNER_64_8_QPSK_IMC = np.load('Outputs/evm_imc_rx_snr_WINNER_64_8_QPSK.npy')
    evm_WINNER_64_8_16QAM_IMC = np.load('Outputs/evm_imc_rx_snr_WINNER_64_8_16QAM.npy')
    evm_WINNER_64_8_64QAM_IMC = np.load('Outputs/evm_imc_rx_snr_WINNER_64_8_64QAM.npy')
    evm_WINNER_128_16_QPSK_IMC = np.load('Outputs/evm_imc_rx_snr_WINNER_128_16_QPSK.npy')
    evm_WINNER_128_16_16QAM_IMC = np.load('Outputs/evm_imc_rx_snr_WINNER_128_16_16QAM.npy')
    evm_WINNER_128_16_64QAM_IMC = np.load('Outputs/evm_imc_rx_snr_WINNER_128_16_64QAM.npy')
    evm_WINNER_256_16_QPSK_IMC = np.load('Outputs/evm_imc_rx_snr_WINNER_256_16_QPSK.npy')
    evm_WINNER_256_16_16QAM_IMC = np.load('Outputs/evm_imc_rx_snr_WINNER_256_16_16QAM.npy')
    evm_WINNER_256_16_64QAM_IMC = np.load('Outputs/evm_imc_rx_snr_WINNER_256_16_64QAM.npy')
    evm_ARGOS_96_8_QPSK_IMC = np.load('Outputs/evm_imc_rx_snr_ARGOS_96_8_QPSK.npy')
    evm_ARGOS_96_8_16QAM_IMC = np.load('Outputs/evm_imc_rx_snr_ARGOS_96_8_16QAM.npy')
    evm_ARGOS_96_8_64QAM_IMC = np.load('Outputs/evm_imc_rx_snr_ARGOS_96_8_64QAM.npy')

    ser_WINNER_64_8_QPSK_IMC = np.load('Outputs/ser_imc_rx_snr_WINNER_64_8_QPSK.npy')
    ser_WINNER_64_8_16QAM_IMC = np.load('Outputs/ser_imc_rx_snr_WINNER_64_8_16QAM.npy')
    ser_WINNER_64_8_64QAM_IMC = np.load('Outputs/ser_imc_rx_snr_WINNER_64_8_64QAM.npy')
    ser_WINNER_128_16_QPSK_IMC = np.load('Outputs/ser_imc_rx_snr_WINNER_128_16_QPSK.npy')
    ser_WINNER_128_16_16QAM_IMC = np.load('Outputs/ser_imc_rx_snr_WINNER_128_16_16QAM.npy')
    ser_WINNER_128_16_64QAM_IMC = np.load('Outputs/ser_imc_rx_snr_WINNER_128_16_64QAM.npy')
    ser_WINNER_256_16_QPSK_IMC = np.load('Outputs/ser_imc_rx_snr_WINNER_256_16_QPSK.npy')
    ser_WINNER_256_16_16QAM_IMC = np.load('Outputs/ser_imc_rx_snr_WINNER_256_16_16QAM.npy')
    ser_WINNER_256_16_64QAM_IMC = np.load('Outputs/ser_imc_rx_snr_WINNER_256_16_64QAM.npy')
    ser_ARGOS_96_8_QPSK_IMC = np.load('Outputs/ser_imc_rx_snr_ARGOS_96_8_QPSK.npy')
    ser_ARGOS_96_8_16QAM_IMC = np.load('Outputs/ser_imc_rx_snr_ARGOS_96_8_16QAM.npy')
    ser_ARGOS_96_8_64QAM_IMC = np.load('Outputs/ser_imc_rx_snr_ARGOS_96_8_64QAM.npy')

    SNRdB_options_all = np.linspace(-4, 30, 35)
    SNRdB_options_QPSK = np.linspace(-1, 13, 15)
    SNRdB_options_16QAM = np.linspace(4, 19, 15)
    SNRdB_options_64QAM = np.linspace(9, 24, 15)

    plt.figure(figsize=(6,4))
    plt.plot(SNRdB_options_QPSK, 20*np.log10(evm_WINNER_64_8_QPSK_IMC), '-o', label=r'WIN-64x8, QPSK', markerfacecolor='none', markeredgecolor='red', color = 'red')
    plt.plot(SNRdB_options_all, 20*np.log10(evm_WINNER_64_8_QPSK_fx), linestyle = 'dashed',  color='red', linewidth = 1)
    plt.plot(SNRdB_options_QPSK, 20*np.log10(evm_WINNER_128_16_QPSK_IMC), '-o', label=r'WIN-128x16, QPSK', markerfacecolor='none', markeredgecolor='green', color = 'green')
    plt.plot(SNRdB_options_all, 20*np.log10(evm_WINNER_128_16_QPSK_fx), linestyle = 'dashed',  color='green', linewidth = 1)  
    plt.plot(SNRdB_options_QPSK, 20*np.log10(evm_WINNER_256_16_QPSK_IMC), '-o', label=r'WIN-256x16, QPSK', markerfacecolor='none', markeredgecolor='blue', color = 'blue')
    plt.plot(SNRdB_options_all, 20*np.log10(evm_WINNER_256_16_QPSK_fx), linestyle = 'dashed',  color='blue', linewidth = 1)
    plt.plot(SNRdB_options_QPSK, 20*np.log10(evm_ARGOS_96_8_QPSK_IMC), '-o', label=r'ARGOS-96x8, QPSK', markerfacecolor='none', markeredgecolor='orange', color = 'orange')
    plt.plot(SNRdB_options_all, 20*np.log10(evm_ARGOS_96_8_QPSK_fx), linestyle = 'dashed',  color='orange', linewidth = 1)
    plt.axhline(y = -15.14, color = 'purple', linewidth = 0.8, label = '3GPP EVM specification')
    plt.text(x=-0.5, y=-15.14 + 0.3, s='QPSK', color='purple', fontsize=9)
    # Create a custom legend entry for FP baselines (dotted black line)
    fl_legend = Line2D([0], [0], color='black', linestyle='--', linewidth=1, label="Digital FX baselines")
    # Get existing legend handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    # Append the new custom legend entry
    handles.append(fl_legend)
    labels.append("Digital FX baselines")
    # Add the legend with both existing and new entries
    plt.legend(handles=handles, labels=labels,fontsize=8)
    plt.grid()
    plt.xlim((-1, 13))
    plt.ylim((-21.5,-7.5))
    plt.xlabel('RX SNR (dB)',fontsize = 13)
    plt.ylabel('EVM (dB)', fontsize = 13)
    # plt.legend(fontsize=8)
    plt.savefig(f'Figures/evm_rx_snr_sweep_all_QPSK.png', format='png', bbox_inches='tight')
    # plt.show()

    plt.figure(figsize=(6,4))
    plt.plot(SNRdB_options_16QAM, 20*np.log10(evm_WINNER_64_8_16QAM_IMC), '-s', label=r'WIN-64x8, 16QAM', markerfacecolor='none', markeredgecolor='red', color = 'red')
    plt.plot(SNRdB_options_all, 20*np.log10(evm_WINNER_64_8_16QAM_fx), linestyle = 'dashed',  color='red', linewidth = 1)
    plt.plot(SNRdB_options_16QAM, 20*np.log10(evm_WINNER_128_16_16QAM_IMC), '-s', label=r'WIN-128x16, 16QAM', markerfacecolor='none', markeredgecolor='green', color = 'green')
    plt.plot(SNRdB_options_all, 20*np.log10(evm_WINNER_128_16_16QAM_fx), linestyle = 'dashed',  color='green', linewidth = 1)  
    plt.plot(SNRdB_options_16QAM, 20*np.log10(evm_WINNER_256_16_16QAM_IMC), '-s', label=r'WIN-256x16, 16QAM', markerfacecolor='none', markeredgecolor='blue', color = 'blue')
    plt.plot(SNRdB_options_all, 20*np.log10(evm_WINNER_256_16_16QAM_fx), linestyle = 'dashed',  color='blue', linewidth = 1)
    plt.plot(SNRdB_options_16QAM, 20*np.log10(evm_ARGOS_96_8_16QAM_IMC), '-s', label=r'ARGOS-96x8, 16QAM', markerfacecolor='none', markeredgecolor='orange', color = 'orange')
    plt.plot(SNRdB_options_all, 20*np.log10(evm_ARGOS_96_8_16QAM_fx), linestyle = 'dashed',  color='orange', linewidth = 1)
    plt.axhline(y = -18.06, color = 'purple', linewidth = 0.8, label = '3GPP EVM specification')
    plt.text(x=4.5, y=-18.06 + 0.3, s='16QAM', color='purple', fontsize=9)
    # Create a custom legend entry for FP baselines (dotted black line)
    fl_legend = Line2D([0], [0], color='black', linestyle='--', linewidth=1, label="Digital FX baselines")
    # Get existing legend handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    # Append the new custom legend entry
    handles.append(fl_legend)
    labels.append("Digital FX baselines")
    # Add the legend with both existing and new entries
    plt.legend(handles=handles, labels=labels,fontsize=8)
    plt.grid()
    plt.xlim((4, 19))
    plt.ylim((-24.5,-10.5))
    plt.xlabel('RX SNR (dB)',fontsize = 13)
    plt.ylabel('EVM (dB)', fontsize = 13)
    # plt.legend(fontsize=8)
    plt.savefig(f'Figures/evm_rx_snr_sweep_all_16QAM.png', format='png', bbox_inches='tight')
    # plt.show()

    plt.figure(figsize=(6,4))
    plt.plot(SNRdB_options_64QAM, 20*np.log10(evm_WINNER_64_8_64QAM_IMC), '-^', label=r'WIN-64x8, 64QAM', markerfacecolor='none', markeredgecolor='red', color = 'red')
    plt.plot(SNRdB_options_all, 20*np.log10(evm_WINNER_64_8_64QAM_fx), linestyle = 'dashed',  color='red', linewidth = 1)
    plt.plot(SNRdB_options_64QAM, 20*np.log10(evm_WINNER_128_16_64QAM_IMC), '-^', label=r'WIN-128x16, 64QAM', markerfacecolor='none', markeredgecolor='green', color = 'green')
    plt.plot(SNRdB_options_all, 20*np.log10(evm_WINNER_128_16_64QAM_fx), linestyle = 'dashed',  color='green', linewidth = 1)  
    plt.plot(SNRdB_options_64QAM, 20*np.log10(evm_WINNER_256_16_64QAM_IMC), '-^', label=r'WIN-256x16, 64QAM', markerfacecolor='none', markeredgecolor='blue', color = 'blue')
    plt.plot(SNRdB_options_all, 20*np.log10(evm_WINNER_256_16_64QAM_fx), linestyle = 'dashed',  color='blue', linewidth = 1)
    plt.plot(SNRdB_options_64QAM, 20*np.log10(evm_ARGOS_96_8_64QAM_IMC), '-^', label=r'ARGOS-96x8, 64QAM', markerfacecolor='none', markeredgecolor='orange', color = 'orange')
    plt.plot(SNRdB_options_all, 20*np.log10(evm_ARGOS_96_8_64QAM_fx), linestyle = 'dashed',  color='orange', linewidth = 1)
    plt.axhline(y = -21.94, color = 'purple', linewidth = 0.8, label = '3GPP EVM specification')
    plt.text(x=9.5, y=-21.94 + 0.3, s='64QAM', color='purple', fontsize=9)
    # Create a custom legend entry for FP baselines (dotted black line)
    fl_legend = Line2D([0], [0], color='black', linestyle='--', linewidth=1, label="Digital FX baselines")
    # Get existing legend handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    # Append the new custom legend entry
    handles.append(fl_legend)
    labels.append("Digital FX baselines")
    # Add the legend with both existing and new entries
    plt.legend(handles=handles, labels=labels,fontsize=8)
    plt.grid()
    plt.xlim((9, 24))
    plt.ylim((-27.5,-13.5))
    plt.xlabel('RX SNR (dB)',fontsize = 13)
    plt.ylabel('EVM (dB)', fontsize = 13)
    # plt.legend(fontsize=8)
    plt.savefig(f'Figures/evm_rx_snr_sweep_all_64QAM.png', format='png', bbox_inches='tight')
    # plt.show()

    plt.figure(figsize=(6,4))
    plt.semilogy(SNRdB_options_QPSK, ser_WINNER_64_8_QPSK_IMC, '-o', label=r'WIN-64x8, QPSK', markerfacecolor='none', markeredgecolor='red', color = 'red')
    plt.semilogy(SNRdB_options_all, ser_WINNER_64_8_QPSK_fx, linestyle = 'dashed',  color='red', linewidth = 1)
    plt.semilogy(SNRdB_options_QPSK, ser_WINNER_128_16_QPSK_IMC, '-o', label=r'WIN-128x16, QPSK', markerfacecolor='none', markeredgecolor='green', color = 'green')
    plt.semilogy(SNRdB_options_all, ser_WINNER_128_16_QPSK_fx, linestyle = 'dashed',  color='green', linewidth = 1)
    plt.semilogy(SNRdB_options_QPSK, ser_WINNER_256_16_QPSK_IMC, '-o', label=r'WIN-256x16, QPSK', markerfacecolor='none', markeredgecolor='blue', color = 'blue')
    plt.semilogy(SNRdB_options_all, ser_WINNER_256_16_QPSK_fx, linestyle = 'dashed',  color='blue', linewidth = 1)    
    plt.semilogy(SNRdB_options_QPSK, ser_ARGOS_96_8_QPSK_IMC, '-o', label=r'ARGOS-96x16, QPSK', markerfacecolor='none', markeredgecolor='orange', color = 'orange')
    plt.semilogy(SNRdB_options_all, ser_ARGOS_96_8_QPSK_fx, linestyle = 'dashed',  color='orange', linewidth = 1)    
    # Create a custom legend entry for FL baselines (dotted black line)
    fl_legend = Line2D([0], [0], color='black', linestyle='--', linewidth=1, label="Digital FX baselines")
    # Get existing legend handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    # Append the new custom legend entry
    handles.append(fl_legend)
    labels.append("Digital FX baselines")
    # Add the legend with both existing and new entries
    plt.legend(handles=handles, labels=labels,fontsize=8)
    plt.grid()
    plt.xlim((-1, 13))
    plt.ylim((0.0001,0.2))
    plt.xlabel('RX SNR (dB)',fontsize = 13)
    plt.ylabel('SER', fontsize = 13)
    # plt.legend(fontsize=8)
    plt.savefig(f'Figures/ser_rx_snr_sweep_all_QPSK.png', format='png', bbox_inches='tight')
    # plt.show()

    plt.figure(figsize=(6,4))
    plt.semilogy(SNRdB_options_16QAM, ser_WINNER_64_8_16QAM_IMC, '-s', label=r'WIN-64x8, 16QAM', markerfacecolor='none', markeredgecolor='red', color = 'red')
    plt.semilogy(SNRdB_options_all, ser_WINNER_64_8_16QAM_fx, linestyle = 'dashed',  color='red', linewidth = 1)
    plt.semilogy(SNRdB_options_16QAM, ser_WINNER_128_16_16QAM_IMC, '-s', label=r'WIN-128x16, 16QAM', markerfacecolor='none', markeredgecolor='green', color = 'green')
    plt.semilogy(SNRdB_options_all, ser_WINNER_128_16_16QAM_fx, linestyle = 'dashed',  color='green', linewidth = 1)
    plt.semilogy(SNRdB_options_16QAM, ser_WINNER_256_16_16QAM_IMC, '-s', label=r'WIN-256x16, 16QAM', markerfacecolor='none', markeredgecolor='blue', color = 'blue')
    plt.semilogy(SNRdB_options_all, ser_WINNER_256_16_16QAM_fx, linestyle = 'dashed',  color='blue', linewidth = 1)
    plt.semilogy(SNRdB_options_16QAM, ser_ARGOS_96_8_16QAM_IMC, '-s', label=r'ARGOS-96x8, 16QAM', markerfacecolor='none', markeredgecolor='orange', color = 'orange')
    plt.semilogy(SNRdB_options_all, ser_ARGOS_96_8_16QAM_fx, linestyle = 'dashed',  color='orange', linewidth = 1)
    # Create a custom legend entry for FL baselines (dotted black line)
    fl_legend = Line2D([0], [0], color='black', linestyle='--', linewidth=1, label="Digital FX baselines")
    # Get existing legend handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    # Append the new custom legend entry
    handles.append(fl_legend)
    labels.append("Digital FX baselines")
    # Add the legend with both existing and new entries
    plt.legend(handles=handles, labels=labels,fontsize=8)
    plt.grid()
    plt.xlim((4, 19))
    plt.ylim((0.0001, 0.2))
    plt.xlabel('RX SNR (dB)',fontsize = 13)
    plt.ylabel('SER', fontsize = 13)
    # plt.legend(fontsize=8)
    plt.savefig(f'Figures/ser_rx_snr_sweep_all_16QAM.png', format='png', bbox_inches='tight')
    # plt.show()

    plt.figure(figsize=(6,4))
    plt.semilogy(SNRdB_options_64QAM, ser_WINNER_64_8_64QAM_IMC, '-^', label=r'WIN-64x8, 64QAM', markerfacecolor='none', markeredgecolor='red', color = 'red')
    plt.semilogy(SNRdB_options_all, ser_WINNER_64_8_64QAM_fx, linestyle = 'dashed',  color='red', linewidth = 1)
    plt.semilogy(SNRdB_options_64QAM, ser_WINNER_128_16_64QAM_IMC, '-^', label=r'WIN-128x16, 64QAM', markerfacecolor='none', markeredgecolor='green', color = 'green')
    plt.semilogy(SNRdB_options_all, ser_WINNER_128_16_64QAM_fx, linestyle = 'dashed',  color='green', linewidth = 1)
    plt.semilogy(SNRdB_options_64QAM, ser_WINNER_256_16_64QAM_IMC, '-^', label=r'WIN-256x16, 64QAM', markerfacecolor='none', markeredgecolor='blue', color = 'blue')
    plt.semilogy(SNRdB_options_all, ser_WINNER_256_16_64QAM_fx, linestyle = 'dashed',  color='blue', linewidth = 1)
    plt.semilogy(SNRdB_options_64QAM, ser_ARGOS_96_8_64QAM_IMC, '-^', label=r'ARGOS-96x8, 64QAM', markerfacecolor='none', markeredgecolor='orange', color = 'orange')
    plt.semilogy(SNRdB_options_all, ser_ARGOS_96_8_64QAM_fx, linestyle = 'dashed',  color='orange', linewidth = 1)
    # Create a custom legend entry for FL baselines (dotted black line)
    fl_legend = Line2D([0], [0], color='black', linestyle='--', linewidth=1, label="Digital FX baselines")
    # Get existing legend handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    # Append the new custom legend entry
    handles.append(fl_legend)
    labels.append("Digital FX baselines")
    # Add the legend with both existing and new entries
    plt.legend(handles=handles, labels=labels,fontsize=8)
    plt.grid()
    plt.xlim((9, 24))
    plt.ylim((0.0001, 0.2))
    plt.xlabel('RX SNR (dB)',fontsize = 13)
    plt.ylabel('SER', fontsize = 13)
    # plt.legend(fontsize=8)
    plt.savefig(f'Figures/ser_rx_snr_sweep_all_64QAM.png', format='png', bbox_inches='tight')
    # plt.show()

if __name__ == '__main__':

    # rx_snr_sweep('WINNER_64_8', 'QPSK', 5000)
    # rx_snr_sweep('WINNER_64_8', '16QAM', 5000)
    # rx_snr_sweep('WINNER_64_8', '64QAM', 5000)
    # rx_snr_sweep('WINNER_128_16', 'QPSK', 3000)
    # rx_snr_sweep('WINNER_128_16', '16QAM', 3000)
    # rx_snr_sweep('WINNER_128_16', '64QAM', 3000)
    # rx_snr_sweep('WINNER_256_16', 'QPSK', 1250)
    # rx_snr_sweep('WINNER_256_16', '16QAM', 1250)
    # rx_snr_sweep('WINNER_256_16', '64QAM', 1500)
    # rx_snr_sweep('ARGOS_96_8', '64QAM', 5000)
    # rx_snr_sweep('ARGOS_96_8', '16QAM', 5000)
    # rx_snr_sweep('ARGOS_96_8', 'QPSK', 5000)
    plot_rx_snr_sweep()