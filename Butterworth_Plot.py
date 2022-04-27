import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import buttord,butter,freqz,bilinear
from scipy.signal import TransferFunction as tf
#%%
PCF = 1.25; # PassBand Corner Frequency
SCF = 2.2; # StopBand Corner Frequency
Td = 0.01; # Sampling Time
fs = 1/Td; # Sampling Frequency
alpha_p = 3; # Converting into Db
alpha_s = 40;
PCF_A = (2/Td)*np.tan(PCF/2); # Using formula for Bilinear Transformation [(2/Td)*tan(Omega/2)]
SCF_A = (2/Td)*np.tan(SCF/2);
[n,Wn] = buttord(PCF_A,SCF_A,alpha_p,alpha_s,analog=True);

fc=20;
Omega_c_w=(2/Td)*np.tan(2*np.pi*fc/2/fs);

# [Bn,An] = butter(1,Omega_c_w,analog=True); 
# Hsn = tf(Bn,An); # H(s) 

order=6 #Order input

[b,a] = butter(order,Omega_c_w,analog=True);
Hs = tf(b,a); # Normalized
[num,den] = bilinear(b,a,fs);

Hz = tf(num, den, dt=Td); # Filter Function

# Magnitude Response
w = np.arange(0,np.pi,np.pi/128)
_,Hw = freqz(num,den,worN=w)
Hw_mag = np.absolute(Hw);
plt.figure(1)
#plt.grid()
plt.plot(w/np.pi*fs/2,Hw_mag) 
plt.title('Magnitude Response');