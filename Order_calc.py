import numpy as np

eps=3
As=40
fc=20
Omg_c=2*np.pi*fc

N=4
Omg_p=( (10**(0.1*eps)-1)**(0.5/N) ) * Omg_c
Omg_s=( (10**(0.1*As)-1)**(0.5/N) ) * Omg_c

print(f'Omega_p = {Omg_p/2/np.pi}\n Omega_s = {Omg_s/2/np.pi}')