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

def fp_baselines(channel, modulation, nsamples):
    # channel can be ['WINNER_64_8', 'WINNER_128_16', 'WINNER_256_32' 'ARGOS_96_8']
    # modulation can be ['QPSK', '16QAM', '64QAM']
    N, K = tuple(map(int, re.findall(r'\d+', channel)[-2:]))
    H = extract_H(channel)

    SNRdB_min = -4
    SNRdB_max = 30
    npoints = 35
    SNRdB_options = np.linspace(SNRdB_min, SNRdB_max, npoints)

    # EVM
    evm_lmmse = np.zeros(npoints)
    # SER
    ser_lmmse = np.zeros(npoints)

    print(f'({nsamples} samples, {K} UEs, {N} BS antennas, {modulation} modulation)')

    for idx in range(npoints):
        # Transmitted vectors
        xt = np.zeros(shape=(K,nsamples), dtype = np.complex_)
        # Received vectors
        yt = np.zeros(shape=(N, nsamples), dtype = np.complex_)
        # Pre-slicer estimated vectors
        xt_tilde_lmmse = np.zeros(shape=(K,nsamples), dtype = np.complex_)
        # Post-slicer estimated vectors
        xt_hat_lmmse = np.zeros(shape=(K,nsamples), dtype = np.complex_)

        # Average symbol energy for a modulation
        es = generate_input_vector(K,modulation)[1]
        # Noise power used in subsequent calculations
        N0 = ((np.linalg.norm(H)**2)/N)*es*10**(-SNRdB_options[idx]/10)
        # LMMSE matrix
        A_H_LMMSE = LMMSE(H,N0,es)
        print(f"Starting simulation for RX SNR = {SNRdB_options[idx]}")
        for i in tqdm(range(nsamples)):
            # mimo_imc_lmmse.clip_adc_discrete_occ(N*0.25, B_ADC)
            xt[:,i] = generate_input_vector(K, modulation)[0]
            # AWGN with specified SNR
            awgn_2d = np.random.multivariate_normal(np.zeros(2),np.eye(2)*N0/2,size=N) 
            n = awgn_2d[:,0] + awgn_2d[:,1]*complex(0,1) 
            # Received vector at time instant i
            yt[:,i] = np.matmul(H,xt[:,i]) + n
            # Pre-slicer estimates
            xt_tilde_lmmse[:,i] = np.matmul(A_H_LMMSE, yt[:,i]).flatten()
            # Post-slicer estimates
            xt_hat_lmmse[:,i] = slicer(xt_tilde_lmmse[:,i], modulation)
        # EVM
        evm_lmmse[idx] = EVM(xt_tilde_lmmse,xt) 
        # SER
        ser_lmmse[idx] = SER(xt_hat_lmmse, xt)
        logging.basicConfig(filename=f'Outputs/fp_baselines_sweep_{channel}_{modulation}.log', level=logging.INFO, filemode='w')
        logging.info("Finished simulation for RX SNR = {}".format(SNRdB_options[idx]))
        logging.info("EVM LMMSE = {}".format(evm_lmmse[idx]))
        logging.info("SER LMMSE = {}\n".format(ser_lmmse[idx]))

    np.save(f'Outputs/fp_baselines_rx_snr_options_{channel}_{modulation}.npy', SNRdB_options)
    np.save(f'Outputs/fp_baselines_evm_rx_snr_{channel}_{modulation}.npy', evm_lmmse)
    np.save(f'Outputs/fp_baselines_ser_rx_snr_{channel}_{modulation}.npy', ser_lmmse)

    plt.figure(figsize=(6,4))
    plt.plot(SNRdB_options, 20*np.log10(evm_lmmse), '-^', label='FP', markerfacecolor='none', markeredgecolor='green', color = 'green')
    plt.grid()
    plt.xlabel('RX SNR (dB)',fontsize = 13)
    plt.ylabel('EVM (dB)', fontsize = 13)
    if(modulation=='QPSK'):
        plt.axhline(y = -15.14, color = 'purple', linewidth = 0.8, label = 'QPSK spec')
        plt.axhline(y = -18.14, color = 'brown', linewidth = 0.8, label = 'QPSK spec - 3dB')
    elif(modulation=='16QAM'):
        plt.axhline(y = -18.06,  color = 'purple', linewidth = 0.8, label = '16QAM spec')
        plt.axhline(y = -21.06,  color = 'brown', linewidth = 0.8, label = '16QAM spec - 3dB')
    elif(modulation=='64QAM'):
        plt.axhline(y = -21.94,  color = 'purple', linewidth = 0.8, label = '64QAM spec')
        plt.axhline(y = -24.94,  color = 'brown', linewidth = 0.8, label = '64QAM spec - 3dB')
    plt.legend(fontsize=8)
    # plt.show()
    plt.savefig(f'Figures/fp_baselines_evm_rx_snr_{channel}_{modulation}.png', format='png', bbox_inches='tight')

    plt.figure(figsize=(6,4))
    plt.semilogy(SNRdB_options, ser_lmmse, '-^', label='FP', markerfacecolor='none', markeredgecolor='green', color = 'green')
    plt.grid()
    plt.xlabel('RX SNR (dB)',fontsize = 13)
    plt.ylabel('SER', fontsize = 13)
    plt.legend(fontsize=8)
    # plt.show()
    plt.savefig(f'Figures/fp_baselines_ser_rx_snr_{channel}_{modulation}.png', format='png', bbox_inches='tight')

if __name__ == '__main__':

    fp_baselines('WINNER_64_8', 'QPSK', 20000)
    fp_baselines('WINNER_128_16', 'QPSK', 10000)
    fp_baselines('WINNER_256_16', 'QPSK', 10000)
    fp_baselines('ARGOS_96_8', 'QPSK', 20000)

    fp_baselines('WINNER_64_8', '16QAM', 20000)
    fp_baselines('WINNER_128_16', '16QAM', 10000)
    fp_baselines('WINNER_256_16', '16QAM', 10000)
    fp_baselines('ARGOS_96_8', '16QAM', 20000)

    fp_baselines('WINNER_64_8', '64QAM', 20000)
    fp_baselines('WINNER_128_16', '64QAM', 10000)
    fp_baselines('WINNER_256_16', '64QAM', 10000)
    fp_baselines('ARGOS_96_8', '64QAM', 20000)