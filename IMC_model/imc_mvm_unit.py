import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from IMC_model.imc_adc import *
from IMC_model.imc_dp_unit import *

class qr_bpbs_mvm_unit:
    def __init__(self, n_phy_rows, n_phy_cols, b_w, b_x, b_adc, sigma_adc, c_qr_mean=1, v_dd=0.9, analog_pots = False, heterogenous_adc = False, adc_config = None):
        self.n_dp_unit = n_phy_cols//b_w # no. of dp units
        self.dp_unit_array = [qr_bpbs_dp_unit(n_phy_rows, b_w, b_x, b_adc, sigma_adc, c_qr_mean, v_dd, analog_pots, heterogenous_adc, adc_config) for i in range(self.n_dp_unit)]
    
    def store_weights(self, w_mat, w_ranges):
        # store the transpose of weight matrix
        # w_mat should be a 2-D numpy array 
        w_mat_n_rows = w_mat.shape[0]
        w_mat_n_cols = w_mat.shape[1]
        if(w_mat_n_rows > self.n_dp_unit):
            print("Weight matrix cannot be mapped in this IMC bank")
            return
        elif(w_mat_n_rows < self.n_dp_unit):
            print("Some IMC columns not utilized")
        for i in range(w_mat_n_rows):
            self.dp_unit_array[i].store_weights(w_mat[i,:], w_ranges[i])
        return    
    
    def compute_mvm(self, x_vec, x_range):
        y_vec = np.zeros(self.n_dp_unit)
        for i in range(self.n_dp_unit):
            y_vec[i] = self.dp_unit_array[i].compute_dp(x_vec, x_range)
        return y_vec
    
    def update_c_qr(self, c_qr_mean_new):
        for i in range(self.n_dp_unit):
            self.dp_unit_array[i].update_c_qr(c_qr_mean_new)
        return
    
    def update_b_adc(self, b_adc):
        for i in range(self.n_dp_unit):
            self.dp_unit_array[i].update_b_adc(b_adc)
        return

    def update_b_adc_mimo(self, mu, b_adc):
        for i in range(self.n_dp_unit):
            self.dp_unit_array[i].update_b_adc_mimo(mu, b_adc)
        return

    def update_sigma_adc(self, sigma_adc):
        for i in range(self.n_dp_unit):
            self.dp_unit_array[i].update_sigma_adc(sigma_adc)
        return
    
if __name__ == '__main__':
    mvm_unit = qr_bpbs_mvm_unit(n_phy_rows=128, n_phy_cols=128, b_w=8, b_x=8, b_adc=7, sigma_adc=0.001, c_qr_mean=1, v_dd=0.9)
    w = np.random.randint(-8,8,size=(16,100))
    # print(f"w = {w}")
    x = np.random.randint(-8,8,size=100)
    # print(f"x = {x}")
    y = np.matmul(w,x)
    mvm_unit.store_weights(w,  w_ranges=np.ones(16)*2**7)
    y_imc = mvm_unit.compute_mvm(x, x_range=2**7)
    print(f"Ideal result = {y}")
    print(f"IMC result = {y_imc}")
    