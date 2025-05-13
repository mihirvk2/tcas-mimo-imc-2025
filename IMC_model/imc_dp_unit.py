import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from IMC_model.imc_adc import uniform_adc_imc
from IMC_model.imc_adc import non_uniform_adc_imc
from IMC_model.quantization_functions import *
import logging

class qr_bpbs_dp_unit:
    # Dot product unit for QR SRAM-based In-Memory Computing with BPBS (Bit-Parallel weight Bit-Serial input) scheme

    # Parameters:
    # n_phy_rows     : Number of physical rows in each IMC column
    # b_w            : Bit precision of weights
    # b_x            : Bit precision of input activations
    # b_adc          : Bit precision of ADCs
    # sigma_adc      : Standard deviation of ADC thermal noise
    # c_qr_mean      : Mean unit cell capacitance (in femtofarads, fF)
    # c_qr_sigma     : Standard deviation of unit cell capacitance, derived from c_qr_mean
    # c_qr_array     : Matrix representing actual cell capacitances, sampled from a normal distribution
    # c_par          : Parasitic capacitance on the capacitance line (depends on c_qr_mean and n_phy_rows)
    # v_dd           : Supply voltage (default 0.9V)
    # delta_imc      : Voltage spacing between adjacent pre-ADC dot product levels
    # v_cl           : Array of possible capacitance line voltages
    # w_array        : Weight matrix (bit-parallel form), initially all zeros
    # w_scale        : Weight scaling factor for normalization
    # analog_pots    : Boolean flag to enable analog power-of-two-summing (not implemented)
    # heterogenous_adc : Boolean flag for enabling different ADC precision per column (not implemented)
    # adc_config     : Mapping of ADC precision to the corresponding ADC index
    # n_adc          : Total number of unique ADCs based on adc_config
    # adc_array      : List of ADC instances (uniform_adc_imc), each configured with thresholds and noise.

    # Note:
    # - Analog POTS and heterogeneous precision ADCs are not implemented in this work.

    def __init__(self, n_phy_rows, b_w, b_x, b_adc, sigma_adc, c_qr_mean=1, v_dd=0.9, analog_pots = False, heterogenous_adc = False, adc_config = None):
        # Instantiate IMC DP unit with all the necessary parameters
        self.n_phy_rows = n_phy_rows
        self.b_w = b_w
        self.b_x = b_x
        self.b_adc = b_adc
        self.sigma_adc = sigma_adc
        self.c_qr_mean = c_qr_mean
        self.c_qr_sigma = 2.1*10**(-2.5)*np.sqrt(c_qr_mean) # k-model value for 28nm technology
        self.c_qr_array = np.random.normal(self.c_qr_mean, self.c_qr_sigma, (self.n_phy_rows, self.b_w))
        self.w_scale = 1
        self.w_array = np.zeros((self.n_phy_rows, self.b_w))
        self.v_dd = 0.9
        self.c_par = self.c_qr_mean*self.n_phy_rows*0.3 + 2.04278 
        self.analog_pots = analog_pots
        self.heterogenous_adc = heterogenous_adc
        self.delta_imc = self.c_qr_mean*self.v_dd/(self.n_phy_rows*self.c_qr_mean + self.c_par)
        self.v_cl = np.arange(self.n_phy_rows+1)*self.delta_imc
        if self.analog_pots == False:
            # [0, 1, 2, 3] : ADC_0 for column 0, ADC_1 for column 1, ...
            self.adc_config = np.array(range(b_w)) 
        else:
            if adc_config is None: 
                print("ADC configuration not provided")
                return
            else:
                # can be something like [0, 1, 2, 2] : ADC_0 for col 0, ADC_1 for col 1, ADC_2 for col 2 and 3
                self.adc_config = adc_config
                self.n_adc =  len(set(self.adc_config))
        self.n_adc = len(self.adc_config)
        if self.heterogenous_adc == False:
            self.adc_array = [uniform_adc_imc(b_adc=b_adc, t1=0.5*self.delta_imc*self.n_phy_rows/(2**b_adc), tM = (2**b_adc - 1.5)*self.delta_imc*self.n_phy_rows/(2**b_adc), sigma_adc=self.sigma_adc, v_cl= self.v_cl) for i in range(self.n_adc)]
        else:
            self.adc_array = [uniform_adc_imc(b_adc=b_adc[i], t1=0.5*self.delta_imc*self.n_phy_rows/(2**b_adc[i]), tM = (2**b_adc[i] - 1.5)*self.delta_imc*self.n_phy_rows/(2**b_adc[i]), sigma_adc=self.sigma_adc, v_cl= self.v_cl) for i in range(self.n_adc)]

    def pots(self, x):
        # Power of two summation
        x_len = len(x)
        x_pots = 0
        for i in range(x_len):
            if(i==0):
                x_pots -= x[i]*2**(x_len-1)
            else:
                x_pots += x[i]*2**(x_len-i-1)
        return x_pots
    
    def store_weights(self, w_vec, w_range):
        # Store weights in memory assuming uniform quantization in the range [-w_range, w_range)
        w_vec_int, self.w_scale = quantize_real_scalar(w_vec, self.b_w, w_range)
        w_vec_bits = int_to_bits(w_vec_int, self.b_w)
        w_rows = w_vec_bits.shape[0]
        w_cols = self.b_w 
        if(w_rows>self.n_phy_rows):
            print("Weight vector cannot be mapped in this IMC dot product unit")
        else:
            for i in range(w_rows):
                for j in range(w_cols):
                    self.w_array[i,j] =  w_vec_bits[i,j]
        return
    
    def compute_dp(self, x_vec, x_range):
        # Computes 2's complement dot product between input activation and stored weights
        # Assumes uniform quantization for input activations in the range [-x_range, x_range)
        dp_dim = len(x_vec)
        if(dp_dim>self.n_phy_rows): 
            print("Size of input vector larger than number of IMC rows")
            return
        x_vec_int, x_scale = quantize_real_scalar(x_vec, self.b_x, x_range)
        x_vec_bits = int_to_bits(x_vec_int, self.b_x)
        bpbs_pots_result = 0
        col_pots_result = np.zeros(self.b_x)
        for k in range(self.b_x):
            v_cl = np.zeros(self.b_w)
            r_points = np.zeros(self.n_adc)
            for j in range(self.b_w):
                cap_tot = 0
                sum = 0 
                for i in range(self.n_phy_rows):
                    cap_tot += self.c_qr_array[i,j]
                    if(i<dp_dim):
                        sum += self.w_array[i,j]*x_vec_bits[i,k]*self.c_qr_array[i,j]
                cap_tot += self.c_par
                v_cl[j] = sum*self.v_dd/cap_tot
                r_points[j] = self.adc_array[j].convert(v_cl[j])
            col_pots_result[k] = self.pots(r_points)
        bpbs_pots_result = self.pots(col_pots_result)
        return bpbs_pots_result*x_scale*self.w_scale
    
    def compute_snr(self, n_samples):
        # Computes SNR of the IMC with respect to fixed point baseline
        y_ideal = np.zeros(n_samples)
        y_imc = np.zeros(n_samples)
        n_errors = 0
        tolerance = 1e-10 
        for i in tqdm(range(n_samples)):
            w_vec = np.random.randint(-2**(self.b_w-1),2**(self.b_w-1),size=self.n_phy_rows)
            x_vec = np.random.randint(-2**(self.b_x-1),2**(self.b_x-1),size=self.n_phy_rows)
            y_ideal[i] = np.dot(w_vec, x_vec)
            self.store_weights(w_vec, 2**(self.b_w-1))
            y_imc[i] = self.compute_dp(x_vec, 2**(self.b_x-1))
            # Fix floating point errors: if y_imc is within tolerance of y_ideal, update it
            if abs(y_imc[i] - y_ideal[i]) < tolerance:
                y_imc[i] = y_ideal[i]
            if(y_imc[i]!=y_ideal[i]):
                n_errors += 1
        print(f"Number of errors = {n_errors}")
        logging.info(f"Number of errors = {n_errors}")
        if(n_errors==0):
            print(f"NO ERRORS DETECTED IN {n_samples} SAMPLES")
            logging.info(f"NO ERRORS DETECTED IN {n_samples} SAMPLES")
            return 100 # large number
        return 10*np.log10(np.var(y_ideal)/np.var(y_ideal-y_imc))
        
    
    def update_c_qr(self, c_qr_mean_new):
        # Updates c_qr and all the other parameters derived from it
        self.c_qr_mean = c_qr_mean_new
        self.c_qr_sigma = 2.1*10**(-2.5)*np.sqrt(c_qr_mean_new) # k-model value for 28nm technology
        self.c_par = self.c_qr_mean*self.n_phy_rows*0.3 + 2.04278
        self.c_qr_array = np.random.normal(self.c_qr_mean, self.c_qr_sigma, (self.n_phy_rows, self.b_w))
        self.delta_imc = self.c_qr_mean*self.v_dd/(self.n_phy_rows*self.c_qr_mean + self.c_par)
        self.v_cl = self.delta_imc/2 + np.arange(self.n_phy_rows+1)*self.delta_imc
        for i in range(self.n_adc):
            self.adc_array[i].v_cl = self.v_cl
            self.adc_array[i].t1 =  0.5*self.delta_imc*self.n_phy_rows/(2**self.adc_array[i].b_adc) 
            self.adc_array[i].tM = (2**self.adc_array[i].b_adc - 1.5)*self.delta_imc*self.n_phy_rows/(2**self.adc_array[i].b_adc)
            self.adc_array[i].t_vec = np.linspace(self.adc_array[i].t1, self.adc_array[i].tM, self.adc_array[i].M)
            self.adc_array[i].v_lsb = (self.adc_array[i].tM-self.adc_array[i].t1)/(self.adc_array[i].M-1)
            self.adc_array[i].r_vec = np.zeros(self.adc_array[i].M+1)
            self.adc_array[i].r_vec[0] = (self.adc_array[i].t1 - self.adc_array[i].v_lsb/2 - self.v_cl[0])/self.delta_imc
            self.adc_array[i].r_vec[1:] = (self.adc_array[i].t_vec + self.adc_array[i].v_lsb/2 - self.v_cl[0])/self.delta_imc
        return
    
    def update_b_adc(self, b_adc):
        # Updates b_adc and all the other parameters derived from it
        for i in range(self.n_adc):
            self.adc_array[i].b_adc = b_adc
            self.adc_array[i].M = 2**self.adc_array[i].b_adc -1
            self.adc_array[i].t1 =  0.5*self.delta_imc*self.n_phy_rows/(2**self.adc_array[i].b_adc) 
            self.adc_array[i].tM = (2**self.adc_array[i].b_adc- 1.5)*self.delta_imc*self.n_phy_rows/(2**self.adc_array[i].b_adc)
            self.adc_array[i].t_vec = np.linspace(self.adc_array[i].t1, self.adc_array[i].tM, self.adc_array[i].M)
            self.adc_array[i].v_lsb = (self.adc_array[i].tM-self.adc_array[i].t1)/(self.adc_array[i].M-1)
            self.adc_array[i].r_vec = np.zeros(self.adc_array[i].M+1)
            self.adc_array[i].r_vec[0] = (self.adc_array[i].t1 - self.adc_array[i].v_lsb/2 - self.v_cl[0])/self.delta_imc
            self.adc_array[i].r_vec[1:] = (self.adc_array[i].t_vec + self.adc_array[i].v_lsb/2 - self.v_cl[0])/self.delta_imc
        return

    def update_b_adc_mimo(self, mu, b_adc):
        
        t1 = 0.5*self.delta_imc
        tM = 126.5*self.delta_imc
        if(b_adc==9):
            if(self.n_phy_rows==64):
                t1 = 0.0625*self.delta_imc
                tM = 63.8125*self.delta_imc
            elif(self.n_phy_rows==128):
                t1 = 0.125*self.delta_imc
                tM = 127.625*self.delta_imc
            elif(self.n_phy_rows==256):
                t1 = 0.25*self.delta_imc
                tM = 255.25*self.delta_imc
        elif(b_adc==8):
            if(self.n_phy_rows==64):
                t1 = 0.125*self.delta_imc
                tM = 63.625*self.delta_imc
            elif(self.n_phy_rows==128):
                t1 = 0.25*self.delta_imc
                tM = 127.25*self.delta_imc
            elif(self.n_phy_rows==256):
                t1 = 0.5*self.delta_imc
                tM = 254.5*self.delta_imc
        elif(b_adc==7):
            if(self.n_phy_rows==64):
                t1 = 0.25*self.delta_imc
                tM = 63.25*self.delta_imc
            elif(self.n_phy_rows==128):
                t1 = 0.5*self.delta_imc
                tM = 126.5*self.delta_imc
            elif(self.n_phy_rows==256):
                t1 = 0.5*self.delta_imc
                tM = 126.5*self.delta_imc
        elif(b_adc==6):
            if(self.n_phy_rows==64):
                t1 = 0.5*self.delta_imc
                tM = 62.5*self.delta_imc
            elif(self.n_phy_rows==128):
                t1 = 0.5*self.delta_imc
                tM = 62.5*self.delta_imc
            elif(self.n_phy_rows==256):
                t1 = (mu - 31 - 0.5)*self.delta_imc
                tM = (mu + 31 - 0.5)*self.delta_imc
        elif(b_adc==5):
            if(self.n_phy_rows==64):
                t1 = 0.5*self.delta_imc
                tM = 30.5*self.delta_imc
            elif(self.n_phy_rows==128):
                t1 = (mu - 15 - 0.5)*self.delta_imc
                tM = (mu + 15 - 0.5)*self.delta_imc
            elif(self.n_phy_rows==256):
                t1 = (mu - 15 - 0.5)*self.delta_imc
                tM = (mu + 15 - 0.5)*self.delta_imc
        elif(b_adc==4):
            t1 = (mu - 7 - 0.5)*self.delta_imc
            tM = (mu + 7 - 0.5)*self.delta_imc
        elif(b_adc==3):
            t1 = (mu - 3 - 0.5)*self.delta_imc
            tM = (mu + 3 - 0.5)*self.delta_imc

        for i in range(self.n_adc):
            self.adc_array[i].delta_imc = self.delta_imc
            self.adc_array[i].b_adc = b_adc
            self.adc_array[i].M = 2**self.adc_array[i].b_adc -1
            self.adc_array[i].t1 = t1
            self.adc_array[i].tM = tM
            self.adc_array[i].t_vec = np.linspace(self.adc_array[i].t1, self.adc_array[i].tM, self.adc_array[i].M)
            self.adc_array[i].v_lsb = (self.adc_array[i].tM-self.adc_array[i].t1)/(self.adc_array[i].M-1)
            self.adc_array[i].r_vec = np.zeros(self.adc_array[i].M+1)
            self.adc_array[i].r_vec[0] = (self.adc_array[i].t1 - self.adc_array[i].v_lsb/2)/self.delta_imc
            self.adc_array[i].r_vec[1:] = (self.adc_array[i].t_vec + self.adc_array[i].v_lsb/2)/self.delta_imc

        return

    
    def update_sigma_adc(self, sigma_adc):
        # Updates sigma_adc and all the other parameters derived from it
        for i in range(self.n_adc):
            self.adc_array[i].sigma_adc = sigma_adc
        return

if __name__ == '__main__':
    # Example usage

    dp_unit = qr_bpbs_dp_unit(n_phy_rows=256, b_w=8, b_x=8, b_adc=8, sigma_adc=0.0004, c_qr_mean=1, v_dd=0.9)
    print(f"Compute SNR = {dp_unit.compute_snr(1000)} dB")
    dp_unit.update_b_adc_mimo(64, 4)
    print(f"Compute SNR = {dp_unit.compute_snr(1000)} dB")