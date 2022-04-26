import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import buttord,butter,freqz,bilinear,lfilter
from scipy.signal import TransferFunction as tf
from scipy.io import loadmat

# plt.style.use('dark_background')
# plt.style.use('seaborn-dark')
# matplotlib.rcParams['lines.linewidth'] = 2
#mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use('dark_background')

colors=['#7A0BC0','#270082','#4E9F3D','#D8E9A8','#E2703A','#590995','#C62A88','#8A2BE2','#ff1493',
        '#7fff00','#3a90e5','#ffcd00']

#--------------------    Part 1 -----------------#
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

order=int(input('Enter order of filter\n'))

[b,a] = butter(order,Omega_c_w,analog=True);
Hs = tf(b,a); # Normalized
[num,den] = bilinear(b,a,fs);

Hz = tf(num, den, dt=Td); # Filter Function

# Magnitude Response
w = np.arange(0,np.pi,np.pi/128)
_,Hw = freqz(num,den,worN=w)
Hw_mag = np.absolute(Hw);
plt.figure(1)
plt.grid()
plt.plot(w/np.pi*fs/2,Hw_mag) 
plt.title('Magnitude Response');

#%%
#--------------------- Loading data -----------------------------------#
files=['PPG_5sec_clean_100Hz.mat','PPG_10sec_clean_100Hz.mat','PPG_120sec_clean_100Hz.mat',
       'ppg_baseline.mat','ppg_baseline_awgn.mat','ppg_baseline_powerline.mat','ppg_random_noise.mat']

data = loadmat(f'Data/{files[6]}')
#data=data.data;
ppg_100hz=data[list(data.keys())[-1]][0];
#r_num=randi(53);
#disp(r_num);
#ppg_data=data(r_num).ppg;

#fs = ppg_data.fs;
fs=100;
#ppg_sig = ppg_data.v;
plt.figure(1);
#stem(ppg_sig(1:fs));
#plot(ppg_sig(1:10*fs));

#ppg_100hz = resample(ppg_sig,4,5);
#stem(ppg_100hz(1:100));
ppg_sig_final=ppg_100hz[0:200];
plt.plot(ppg_sig_final);

#%%
#----------------------- Signal filtering ------------------------------#
time_sig=np.linspace(0,len(ppg_sig_final)/100,len(ppg_sig_final))

fft_len=len(ppg_sig_final)<<1;
plt.figure(17);
fft_bef_filt=np.absolute(np.fft.fft(ppg_sig_final,2^fft_len));
plt.figure()
plt.plot(fft_bef_filt);
#stem(fft_bef_filt);
plt.title('FFT Before Filtering')

##
y=lfilter(num,den,ppg_sig_final);
plt.figure(figsize=(8,6));
fft_aft_filt=np.absolute(np.fft.fft(y,2^fft_len));
plt.plot(fft_aft_filt);
#stem(fft_aft_filt);
plt.title('FFT After Filtering')

plt.figure(figsize=(8,6))
plt.plot(time_sig,ppg_sig_final,linewidth=2,c='red');
plt.title('Signal before filtering')
plt.tight_layout()
plt.savefig('irst.png',dpi=400,transparent=1)

plt.figure(figsize=(8,6))
plt.plot(time_sig,y,color=colors[10],lw=2);
plt.title('Signal after filtering')
plt.tight_layout()
# plt.grid()
plt.savefig('lmno.png',dpi=400,transparent=1)

