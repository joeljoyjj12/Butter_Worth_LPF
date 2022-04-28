import numpy as np

def calculate_min_order(fc):
    Ap=3
    As=40
    Fs=100
    Td=1/Fs
    
    stop_band_limit = 2*fc
    Omega_s_limit = (2/Td)*np.tan(2*np.pi*stop_band_limit/Fs/2) # Stop band limit
                                                                 # in analog domain
    
    Omega_c = (2/Td)*np.tan(2*np.pi*fc/Fs/2)
    
    N=1
    while True:
        Omg_p=( (10**(0.1*Ap)-1)**(0.5/N) ) * Omega_c
        Omg_s=( (10**(0.1*As)-1)**(0.5/N) ) * Omega_c
        print(Omg_s)
        if Omg_s < Omega_s_limit:
            print(f"Minimum order required = {N}")
            break
        N=N+1    
Fc=20
calculate_min_order(Fc)