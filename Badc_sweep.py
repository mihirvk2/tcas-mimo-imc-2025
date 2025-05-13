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

def Badc_sweep(channel, modulation, nsamples):
    N, K = tuple(map(int, re.findall(r'\d+', channel)[-2:]))
    H = extract_H(channel)
    B_y = 6
    B_w = 6
    B_ADC = np.array(range(3,10))
    npoints = len(B_ADC)
    sigma_ADC = 0.0005
    SNRdB = 20 
    cap = 1
    VDD = 0.9
    if channel == 'WINNER_64_8' and modulation == 'QPSK':
        SNRdB = 12.39685
        B_w = 5
        B_y = 5
    elif channel == 'WINNER_128_16' and modulation == 'QPSK':
        SNRdB = 12.64686
        B_w = 5
        B_y = 5
    elif channel == 'WINNER_256_16' and modulation == 'QPSK':
        SNRdB = 9.41398
        B_w = 5
        B_y = 5
    elif channel == 'ARGOS_96_8' and modulation == 'QPSK':
        SNRdB = 10.65343
        B_w = 5
        B_y = 5
    elif channel == 'WINNER_64_8' and modulation == '16QAM':
        SNRdB = 15.36744
        B_w = 6
        B_y = 6
    elif channel == 'WINNER_128_16' and modulation == '16QAM':
        SNRdB = 15.63112
        B_w = 6
        B_y = 6
    elif channel == 'WINNER_256_16' and modulation == '16QAM':
        SNRdB = 12.42296
        B_w = 6
        B_y = 6
    elif channel == 'ARGOS_96_8' and modulation == '16QAM':
        SNRdB = 13.64712
        B_w = 6
        B_y = 6
    elif channel == 'WINNER_64_8' and modulation == '64QAM':
        SNRdB = 19.29103
        B_w = 7
        B_y = 7
    elif channel == 'WINNER_128_16' and modulation == '64QAM':
        SNRdB = 19.53417
        B_w = 7
        B_y = 7
    elif channel == 'WINNER_256_16' and modulation == '64QAM':
        SNRdB = 16.30254
        B_w = 7
        B_y = 7
    elif channel == 'ARGOS_96_8' and modulation == '64QAM':
        SNRdB = 17.58257
        B_w = 7
        B_y = 7

    # EVM
    evm_lmmse_fp = 0
    evm_lmmse_fx = 0
    evm_lmmse_IMC = np.zeros(npoints)

    # Average symbol energy for a modulation
    es = generate_input_vector(K,modulation)[1]
    # Noise power used in subsequent calculations
    N0 = ((np.linalg.norm(H)**2)/N)*es*10**(-SNRdB/10)
    # Standard deviation of received vector
    y_std = np.sqrt(0.5*N0*(1+10**(SNRdB/10)))
    # Input clipping range of received vector for quantization
    y_max = np.mean(y_std)*occ_std(B_y)
    y_scale = 2* y_max / (2**B_y)
    # LMMSE matrix
    A_H_LMMSE = LMMSE(H,N0,es)
    # Quantizing LMMSE matrix            
    A_H_LMMSE_ranges = np.array([max(np.max(np.abs(np.real(A_H_LMMSE[i,:]))),np.max(np.abs(np.imag(A_H_LMMSE[i,:])))) for i in range(K)])
    A_H_LMMSE_fx, A_H_LMMSE_scaling_factors = quantize_complex_matrix_many_sf(A_H_LMMSE, B_w, A_H_LMMSE_ranges) # quantize matrix
         
    # Transmitted vectors
    xt = np.zeros(shape=(K,nsamples), dtype = np.complex_)
    # Received vectors
    yt = np.zeros(shape=(N, nsamples), dtype = np.complex_)
    yt_fx = np.zeros(shape=(N, nsamples), dtype = np.complex_)
    # Pre-slicer estimated vectors
    xt_tilde_lmmse = np.zeros(shape=(K,nsamples), dtype = np.complex_)
    xt_tilde_lmmse_fx = np.zeros(shape=(K,nsamples), dtype = np.complex_)
    xt_tilde_lmmse_IMC = np.zeros(shape=(K,nsamples), dtype = np.complex_)

    print(f"Starting simulation for floating point and fixed point baselines")
    for i in tqdm(range(nsamples)):
        xt[:,i] = generate_input_vector(K, modulation)[0]
        # AWGN with specified SNR
        awgn_2d = np.random.multivariate_normal(np.zeros(2),np.eye(2)*N0/2,size=N) 
        n = awgn_2d[:,0] + awgn_2d[:,1]*complex(0,1) 
        # Received vector at time instant i
        yt[:,i] = np.matmul(H,xt[:,i]) + n
        yt_fx[:,i] = quantize_complex_vector(yt[:,i], B_y, y_max)[0]
        # Pre-slicer estimates
        xt_tilde_lmmse[:,i] = np.matmul(A_H_LMMSE, yt[:,i]).flatten()
        # Pre-slicer estimates
        xt_tilde_lmmse_fx[:,i] = CMVM_quantized_many_sf(A_H_LMMSE_fx, A_H_LMMSE_scaling_factors, yt_fx[:,i], y_scale)[0].flatten()

    # Pre-slicer EVM results (baselines)
    evm_lmmse_fp = EVM(xt_tilde_lmmse,xt)
    evm_lmmse_fx = EVM(xt_tilde_lmmse_fx,xt)
    logging.basicConfig(filename=f'Outputs/Badc_sweep_{channel}_{modulation}.log', level=logging.INFO, filemode='w')
    logging.info("EVM LMMSE = {}".format(evm_lmmse_fp))
    logging.info("EVM LMMSE FX = {}".format(evm_lmmse_fx))

    mimo_imc_lmmse = mimo_imc(n_phy_rows=int(2**(np.ceil(np.log2(N)))), n_phy_cols=K*B_w, b_w = B_w, b_x=B_y, b_adc= 8, sigma_adc=sigma_ADC,c_qr_mean=cap, v_dd= VDD)
    mimo_imc_lmmse.store_complex_weights(A_H_LMMSE, A_H_LMMSE_ranges)

    for idx in range(npoints):
        print(f"Starting simulation for ADC precision = {B_ADC[idx]}")
        for i in tqdm(range(nsamples)):
            mimo_imc_lmmse.update_b_adc_mimo(N*0.25, B_ADC[idx])
            xt[:,i] = generate_input_vector(K, modulation)[0]
            # AWGN with specified SNR
            awgn_2d = np.random.multivariate_normal(np.zeros(2),np.eye(2)*N0/2,size=N) 
            n = awgn_2d[:,0] + awgn_2d[:,1]*complex(0,1) 
            # Received vector at time instant i
            yt[:,i] = np.matmul(H,xt[:,i]) + n
            xt_tilde_lmmse_IMC[:,i] = mimo_imc_lmmse.compute_complex_mvm(yt[:,i], y_max)
        evm_lmmse_IMC[idx] = EVM(xt_tilde_lmmse_IMC,xt)        
        logging.info("Finished simulation for ADC precision = {}".format(B_ADC[idx]))
        logging.info("EVM LMMSE IMC = {} \n".format(evm_lmmse_IMC[idx]))

    np.save(f'Outputs/Badc_options_{channel}_{modulation}.npy', B_ADC)
    np.save(f'Outputs/evm_fp_Badc_{channel}_{modulation}.npy', evm_lmmse_fp)
    np.save(f'Outputs/evm_fx_Badc_{channel}_{modulation}.npy', evm_lmmse_fx)
    np.save(f'Outputs/evm_imc_Badc_{channel}_{modulation}.npy', evm_lmmse_IMC)
    
    plt.figure(figsize=(6,4))
    plt.plot(B_ADC, 20*np.log10(evm_lmmse_IMC), '-^', label='IMC', markerfacecolor='none', markeredgecolor='red', color = 'red')
    plt.axhline(y = 20*np.log10(evm_lmmse_fp), linestyle = 'dashed', label='FP',color = 'green', linewidth = 1)
    plt.axhline(y = 20*np.log10(evm_lmmse_fx), linestyle = 'dashed', label = 'FX', color = 'blue', linewidth = 1)
    plt.grid()
    plt.xlabel('$B_{\mathrm{ADC}}$ (bits)',fontsize = 13)
    plt.ylabel('EVM (dB)',fontsize = 13)
    plt.legend(fontsize=8)
    plt.savefig(f'Figures/Badc_sweep_{channel}_{modulation}.png', format='png', bbox_inches='tight')

def plot_evm_badc_all():

    evm_WINNER_64_8_64QAM_IMC = np.load('Outputs/evm_imc_Badc_WINNER_64_8_64QAM.npy')
    evm_WINNER_64_8_64QAM_FX = np.load('Outputs/evm_fx_Badc_WINNER_64_8_64QAM.npy')
    evm_WINNER_128_16_64QAM_IMC = np.load('Outputs/evm_imc_Badc_WINNER_128_16_64QAM.npy')
    evm_WINNER_128_16_64QAM_FX = np.load('Outputs/evm_fx_Badc_WINNER_128_16_64QAM.npy')
    evm_WINNER_256_16_64QAM_IMC = np.load('Outputs/evm_imc_Badc_WINNER_256_16_64QAM.npy')
    evm_WINNER_256_16_64QAM_FX = np.load('Outputs/evm_fx_Badc_WINNER_256_16_64QAM.npy')
    evm_ARGOS_96_8_64QAM_IMC = np.load('Outputs/evm_imc_Badc_ARGOS_96_8_64QAM.npy')
    evm_ARGOS_96_8_64QAM_FX = np.load('Outputs/evm_fx_Badc_ARGOS_96_8_64QAM.npy')

    B_ADC = np.array(range(3,10))

    plt.figure(figsize=(6,4))
    plt.plot(B_ADC, 20*np.log10(evm_WINNER_64_8_64QAM_IMC), '-^', label=r'WIN-64x8, 64-QAM', markerfacecolor='none', markeredgecolor='red', color = 'red')
    plt.axhline(y = 20*np.log10(evm_WINNER_64_8_64QAM_FX), linestyle = 'dashed', color = 'red', linewidth = 1)
    plt.plot(B_ADC, 20*np.log10(evm_WINNER_128_16_64QAM_IMC), '-^', label=r'WIN-128x16, 64-QAM', markerfacecolor='none', markeredgecolor='green', color = 'green')
    plt.axhline(y = 20*np.log10(evm_WINNER_128_16_64QAM_FX), linestyle = 'dashed', color = 'green', linewidth = 1.2)
    plt.plot(B_ADC, 20*np.log10(evm_WINNER_256_16_64QAM_IMC), '-^', label=r'WIN-256x16, 64-QAM', markerfacecolor='none', markeredgecolor='blue', color = 'blue')
    plt.axhline(y = 20*np.log10(evm_WINNER_256_16_64QAM_FX), linestyle = 'dashed', color = 'blue', linewidth = 1, alpha = 0.6)
    plt.plot(B_ADC, 20*np.log10(evm_ARGOS_96_8_64QAM_IMC), '-^', label=r'ARGOS-96x8, 64-QAM', markerfacecolor='none', markeredgecolor='orange', color = 'orange')
    plt.axhline(y = 20*np.log10(evm_ARGOS_96_8_64QAM_FX), linestyle = 'dashed', color = 'orange', linewidth = 1)
    # plt.axhline(y = -21.94, linestyle = 'dashed', color = 'grey', linewidth = 1)
    # plt.text(x=0.005, y=-21.94 + 0.5, s='64-QAM', color='grey', fontsize=9)
    # plt.axhline(y = -24.5, label='FX baseline', linestyle = 'dashed', color = 'purple', linewidth = 1)
    # plt.axhline(y = -25, label='FL baseline', linestyle = 'dashed', color = 'purple', linewidth = 1)
    # plt.axhline(y = 20*np.log10(evm_lmmse_fx), linestyle = 'dashed', label = 'FX', color = 'blue', linewidth = 1)
    # Create a custom legend entry for FL baselines (dotted black line)
    fl_legend = Line2D([0], [0], color='black', linestyle='--', linewidth=1, label="Digital FX baselines")
    # Get existing legend handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    # Append the new custom legend entry
    handles.append(fl_legend)
    labels.append("Digital FX baselines")
    plt.legend(handles=handles, labels=labels,fontsize=8)
    plt.grid()
    plt.xlabel('$B_{\mathrm{ADC}}$ (bits)',fontsize = 13)
    plt.ylabel('EVM (dB)',fontsize = 13)
    plt.ylim(-26.5, -0.2)
    # plt.legend(fontsize=8)
    plt.savefig(f'Figures/evm_badc_sweep_all.png', format='png', bbox_inches='tight')
    plt.show()

def plot_eb_badc_all():
    N = np.array([64, 128, 256, 96])
    B_ADC = np.array(range(3,10))
    k1 = 0.1 # pJ
    k2 = 10**(-6) # pJ
    alpha = 1.244
    vdd = 0.9
    cqr = 1 # fF
    M = 8
    B_w = 7
    B_y = 7
    E_ADC = k1*B_ADC + k2*4**(B_ADC)
    E_IA = alpha*cqr*0.001*N*vdd**2 # pJ
    E_write = 0.135 # pJ
    gamma = 500
    npoints = len(B_ADC)
    Eb_badc = np.zeros((npoints,4))
    for i in range(npoints):
        # print(E_IA)
        Eb_badc[i,:] = 4*(E_ADC[i] + E_IA)*B_w*B_y/(M) + 2*E_write*N*B_w/gamma# pJ
    plt.figure(figsize=(6,4))
    plt.plot(B_ADC, Eb_badc[:,0], '-^', label=r'WIN-64x8, 64-QAM', markerfacecolor='none', markeredgecolor='red', color = 'red')
    plt.plot(B_ADC, Eb_badc[:,1], '-^', label=r'WIN-128x16, 64-QAM', markerfacecolor='none', markeredgecolor='green', color = 'green')
    plt.plot(B_ADC, Eb_badc[:,2], '-^', label=r'WIN-256x16, 64-QAM', markerfacecolor='none', markeredgecolor='blue', color = 'blue')
    plt.plot(B_ADC, Eb_badc[:,3], '-^', label=r'ARGOS-96x8, 64-QAM', markerfacecolor='none', markeredgecolor='orange', color = 'orange')
    plt.grid()
    plt.xlabel('$B_{\mathrm{ADC}}$ (bits)',fontsize = 13)
    plt.ylabel('$E_{\mathrm{b}} \ \mathrm{(pJ/b)}$',fontsize = 13)
    plt.legend(fontsize=8)
    plt.savefig(f'Figures/Eb_badc_sweep_all.png', format='png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':

    # Badc_sweep('WINNER_64_8', 'QPSK', 20000)
    # Badc_sweep('WINNER_128_16', 'QPSK', 10000)
    # Badc_sweep('WINNER_256_16', 'QPSK', 10000)
    # Badc_sweep('ARGOS_96_8', 'QPSK', 20000)
    # Badc_sweep('WINNER_64_8', '16QAM', 20000)
    # Badc_sweep('WINNER_128_16', '16QAM', 10000)
    # Badc_sweep('WINNER_256_16', '16QAM', 10000)
    # Badc_sweep('ARGOS_96_8', '16QAM', 20000)
    # Badc_sweep('WINNER_64_8', '64QAM', 2500)
    # Badc_sweep('WINNER_128_16', '64QAM', 1250)
    # Badc_sweep('WINNER_256_16', '64QAM', 1250)
    # Badc_sweep('ARGOS_96_8', '64QAM', 2500)

    # Badc_sweep('ARGOS_96_8', '64QAM', 5)
    # plot_evm_badc_all()
    plot_eb_badc_all()