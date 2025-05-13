import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from IMC_model.imc_adc import *
from IMC_model.imc_dp_unit import *
from IMC_model.imc_mvm_unit import *

class mimo_imc:
    def __init__(self,n_phy_rows, n_phy_cols, b_w, b_x, b_adc, sigma_adc, c_qr_mean=1, v_dd=0.9, analog_pots = False, heterogenous_adc = False, adc_config = None, n_banks=2):
        self.n_banks = n_banks
        self.imc_mvm_unit_array = [qr_bpbs_mvm_unit(n_phy_rows, n_phy_cols, b_w, b_x, b_adc, sigma_adc, c_qr_mean, v_dd, analog_pots = False, heterogenous_adc = False, adc_config = None) for i in range(self.n_banks)]

    def store_complex_weights(self, w_mat, w_ranges):
        # assumes 2-bank IMC architecture
        w_mat_nrows = w_mat.shape[0]
        w_mat_ncols = w_mat.shape[1]
        w_mat_real = np.zeros(shape=(w_mat_nrows, w_mat_ncols))
        w_mat_imag = np.zeros(shape=(w_mat_nrows, w_mat_ncols))
        for i in range(w_mat_nrows):
            for j in range(w_mat_ncols):
                w_mat_real[i,j] = w_mat[i,j].real
                w_mat_imag[i,j] = w_mat[i,j].imag
        self.imc_mvm_unit_array[0].store_weights(w_mat_real, w_ranges)
        self.imc_mvm_unit_array[1].store_weights(w_mat_imag, w_ranges)
        return
    
    def compute_complex_mvm(self, x_vec, x_range):
        # assumes 2-bank IMC architecture
        x_vec_len = len(x_vec)
        x_vec_real = np.zeros(x_vec_len)
        x_vec_imag = np.zeros(x_vec_len)
        for i in range(x_vec_len):
            x_vec_real[i] = x_vec[i].real
            x_vec_imag[i] = x_vec[i].imag
        y_vec_real = self.imc_mvm_unit_array[0].compute_mvm(x_vec_real,x_range) - self.imc_mvm_unit_array[1].compute_mvm(x_vec_imag, x_range)
        y_vec_imag = self.imc_mvm_unit_array[0].compute_mvm(x_vec_imag,x_range) + self.imc_mvm_unit_array[1].compute_mvm(x_vec_real, x_range)
        y_vec = y_vec_real + 1j*y_vec_imag
        return y_vec
    
    def update_c_qr(self, c_qr_mean_new):
        for i in range(self.n_banks):
            self.imc_mvm_unit_array[i].update_c_qr(c_qr_mean_new)

    def update_b_adc(self, b_adc):
        for i in range(self.n_banks):
            self.imc_mvm_unit_array[i].update_b_adc(b_adc)

    def update_b_adc_mimo(self, mu, b_adc):
        for i in range(self.n_banks):
            self.imc_mvm_unit_array[i].update_b_adc_mimo(mu, b_adc)
        return
    
    def update_sigma_adc(self, sigma_adc):
        for i in range(self.n_banks):
            self.imc_mvm_unit_array[i].update_sigma_adc(sigma_adc)

if __name__ == '__main__':
    mimo_accelerator = mimo_imc(n_phy_rows=128, n_phy_cols=128, b_w=16, b_x=16, b_adc=7, sigma_adc=0.0005, c_qr_mean=1, v_dd=0.9)
    w_real = np.random.uniform(-1,1,size=(8,100))
    w_imag = np.random.uniform(-1,1,size=(8,100))
    w = w_real + 1j*w_imag
    # print(f"w = {w}")
    x_real = np.random.uniform(-1,1,size=(100))
    x_imag = np.random.uniform(-1,1,size=(100))
    x = x_real + 1j*x_imag
    # print(f"x = {x}")
    y = np.matmul(w,x)
    mimo_accelerator.store_complex_weights(w,  w_ranges=np.ones(16))
    mimo_accelerator.update_b_adc_mimo(32, 6)
    y_imc = mimo_accelerator.compute_complex_mvm(x, x_range=1)
    print(f"Ideal result = {y}")
    print(f"IMC result = {y_imc}")