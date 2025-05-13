import numpy as np
import re

def rx_snr_at_3dB_evm_margin(channel, modulation):
    evm = np.load(f'Outputs/fp_baselines_evm_rx_snr_{channel}_{modulation}.npy')
    evm_dB = 20*np.log10(evm)
    rx_snr_dB = np.load(f'Outputs/fp_baselines_rx_snr_options_{channel}_{modulation}.npy')
    evm_spec_minus_3dB = 0
    if(modulation=='QPSK'):
        evm_spec_minus_3dB = -18.14
    elif(modulation=='16QAM'):
        evm_spec_minus_3dB = -21.06
    elif(modulation=='64QAM'):
        evm_spec_minus_3dB = -24.94
    else:
        print(f'Unsupported modulation: {modulation}')
    # Find indices where evm_dB crosses evm_spec_minus_3dB
    diffs = evm_dB - evm_spec_minus_3dB
    sign_changes = np.where(np.diff(np.sign(diffs)))[0]
    if len(sign_changes) == 0:
        print("Interpolation not possible")
        return None  # or raise an error if interpolation is not possible
    idx = sign_changes[0]
    x0, x1 = rx_snr_dB[idx], rx_snr_dB[idx + 1]
    y0, y1 = evm_dB[idx], evm_dB[idx + 1]
    # Linear interpolation
    slope = (y1 - y0) / (x1 - x0)
    rx_snr_interp = x0 + (evm_spec_minus_3dB - y0) / slope
    print(f"For channel {channel} with modulation {modulation}, RX SNR @ EVM spec - 3dB : {rx_snr_interp}")
    return rx_snr_interp

if __name__ == '__main__':
    rx_snr_at_3dB_evm_margin('WINNER_64_8', 'QPSK')
    rx_snr_at_3dB_evm_margin('WINNER_128_16', 'QPSK')
    rx_snr_at_3dB_evm_margin('WINNER_256_16', 'QPSK')
    rx_snr_at_3dB_evm_margin('ARGOS_96_8', 'QPSK')
    rx_snr_at_3dB_evm_margin('WINNER_64_8', '16QAM')
    rx_snr_at_3dB_evm_margin('WINNER_128_16', '16QAM')
    rx_snr_at_3dB_evm_margin('WINNER_256_16', '16QAM')
    rx_snr_at_3dB_evm_margin('ARGOS_96_8', '16QAM')
    rx_snr_at_3dB_evm_margin('WINNER_64_8', '64QAM')
    rx_snr_at_3dB_evm_margin('WINNER_128_16', '64QAM')
    rx_snr_at_3dB_evm_margin('WINNER_256_16', '64QAM')
    rx_snr_at_3dB_evm_margin('ARGOS_96_8', '64QAM')

    