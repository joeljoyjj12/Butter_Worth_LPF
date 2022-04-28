import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz,bilinear,lfilter
from scipy.signal import TransferFunction as tf
from scipy.io import loadmat

plt.style.use('dark_background')

colors=['#7A0BC0','#270082','#4E9F3D','#D8E9A8','#E2703A','#590995','#C62A88','#8A2BE2','#ff1493',
        '#7fff00','#3a90e5','#ffcd00']

#--------------------    Part 1 -----------------#
#%%

def tf_scaled(o,w):
    polynomial={1:([1*w],[1,1*w]), 2:([1*w**2],[1,1.4142*w,1*w**2]), 
                3:([1*w**3],[1,2*w,2*w**2,1*w**3]),
                4:([1*w**4],[1,2.6131*w,3.4142*w**2,2.6131*w**3,1*w**4]),
                5:([1*w**5],[1,3.2361*w,5.2361*w**2,5.2361*w**3,3.2361*w**4,1*w**5]),
                6:([1*w**6],[1,3.8637*w,7.4641*w**2,9.1416*w**3,7.4641*w**4,
                   3.8637*w**5,1*w**6])}
    return polynomial[o]

fc=20;    # Cutoff Freq
Td = 0.01; # Sampling Time
Fs = 1/Td; # Sampling Frequency
Ap = 3; # Passband Attenuation in dB
As = 40; # stopband Attenuation in dB

Omega_c=(2/Td)*np.tan(2*np.pi*fc/Fs/2);

order = 4

butterworth_polynomial={1:([1],[1,1]), 2:([1],[1,1.4142,1]), 3:([1],[1,2,2,1]),
                        4:([1],[1,2.6131,3.4142,2.6131,1]),
                        5:([1],[1, 3.2361, 5.2361, 5.2361, 3.2361,1]),
                        6:([1],[1,3.8637,7.4641,9.1416,7.4641,
                               3.8637,1])}

[b,a] = butterworth_polynomial[order]

H_s_normalized = tf(b,a)

[b_s,a_s] = tf_scaled(order,Omega_c)

H_z=bilinear(b_s,a_s,Fs)

num=H_z[0]
den=H_z[1]

w = np.arange(0,np.pi,np.pi/256)
_,Hw = freqz(H_z[0],H_z[1],worN=w)
Hw_mag=np.absolute(Hw)

plt.figure(1)
plt.scatter(20, 0.70711, marker='x',s=20,c='w',zorder=3,label='3dB cutoff')
plt.plot(w/np.pi*Fs/2,Hw_mag,color=colors[8],zorder=2)  # To change limits from 0 50Hz
plt.title('Magnitude Response',fontsize=9);
plt.xlabel('Frequency (Hz)',fontsize=7)
plt.yticks(fontsize=6)
plt.xticks(fontsize=6)
plt.ylabel('Magnitude',fontsize=7)
plt.axvline(x=20,ymin=0,ymax=0.7,lw=0.5,zorder=1,c=colors[10])
plt.axhline(y=0.707,xmin=0, xmax=20.5/50,lw=0.5,zorder=1,c=colors[10])
plt.tight_layout()
plt.legend(fontsize=6)
# plt.savefig('Mag.png',dpi=400,transparent=0)

plt.figure(2)
plt.plot(w/np.pi*Fs/2,20*np.log10(Hw_mag),color=colors[8])  # To change limits from 0 50Hz
plt.scatter(20, -3, marker='x',s=20,c='w',zorder=3,label='3dB cutoff')
plt.title('Magnitude Response in dB',fontsize=9)
plt.xlabel('Frequency (Hz)',fontsize=7)
plt.ylabel('Magnitude (dB)',fontsize=7)
plt.yticks(fontsize=6)
plt.xticks(fontsize=6)
plt.axvline(x=20,ymin=0,ymax=1,lw=0.5,c=colors[10])
plt.axhline(y=-3,xmin=0, xmax=1,lw=0.5,c=colors[10])
plt.tight_layout()
plt.legend(fontsize=6)
# plt.savefig('Mag_dB.png',dpi=400,transparent=0)

#%%
#--------------------- Loading data -----------------------------------#
files=['PPG_5sec_clean_100Hz.mat','PPG_10sec_clean_100Hz.mat','PPG_120sec_clean_100Hz.mat',
       'ppg_baseline.mat','ppg_baseline_awgn.mat','ppg_baseline_powerline.mat','ppg_random_noise.mat']

data = loadmat(f'Data/{files[5]}')
ppg_100hz=data[list(data.keys())[-1]][0];
fs=100;
ppg_sig_final=ppg_100hz[:] if len(ppg_100hz)<3000 else ppg_100hz[:3000]

#%%
#----------------------- Signal filtering using filt------------------------------#
def plot_fft(x,y,color,title):
    plt.figure(figsize=(8,6))
    plt.plot(x,y,c=color);
    plt.xlabel('Frequency (Hz)',fontsize=9)
    plt.ylabel('X(k)',fontsize=9)
    plt.title(title)
    plt.tight_layout()
    
def plt_signal(x,y,color,title):
    plt.figure(figsize=(8,6))
    plt.plot(x,y,linewidth=1.5,c=color);
    plt.title(title)
    plt.tight_layout()
    plt.xlabel('time (sec)',fontsize=10)
    plt.ylabel('PPG',fontsize=10)
    plt.tight_layout()

time_sig=np.linspace(0,len(ppg_sig_final)/100,len(ppg_sig_final))

fft_len=int(np.ceil(np.log2(len(ppg_sig_final))));
plt.figure(17);
fft_bef_filt=np.absolute(np.fft.fft(ppg_sig_final,2**fft_len));

res_fft=fs/len(fft_bef_filt) # Resolution
x_fft=np.arange(0,len(fft_bef_filt))*res_fft

# FIltering using lfilter
y=lfilter(num,den,ppg_sig_final)
fft_aft_filt=np.absolute(np.fft.fft(y,2**fft_len));

# FFT before filtering
plot_fft(x_fft,fft_bef_filt,colors[10],'FFT Before Filtering using function')

# FFT after filtering
plot_fft(x_fft,fft_aft_filt,colors[10],'FFT After Filtering using function')

# Signal before filtering
plt_signal(time_sig,ppg_sig_final,'red','Signal before filtering using function')

# Signal after filtering
plt_signal(time_sig,y,colors[10],'Signal after filtering using function')


#%%
#---------------------------- Manual Signal filtering -----------------------------#
x_m=np.zeros(len(ppg_sig_final)+4)
x_m[4:]=ppg_sig_final
l=len(ppg_sig_final)
y_m=np.zeros(len(x_m))

b0=num[0]  #b0
b1=num[1]  #b1
b2=num[2]  #b2
b3=num[3]  #b3
b4=num[4]  #b4
a1=-den[1] #a1
a2=-den[2] #a2
a3=-den[3] #a3
a4=-den[4] #a4

for n in range(4,len(y_m)):
    y_m[n]=a1*y_m[n-1]+a2*y_m[n-2]+a3*y_m[n-3]+a4*y_m[n-4]+b0*x_m[n]+b1*x_m[n-1]+b2*x_m[n-2]+b3*x_m[n-3]+b4*x_m[n-4]

y_m=y_m[4:] # Extracting the output signal

fft_aft_filt=np.absolute(np.fft.fft(y_m,2**fft_len));

# FFT before filtering
plot_fft(x_fft,fft_bef_filt,colors[10],'FFT Before Filtering using design approach')

# FFT after filteringPU
plot_fft(x_fft,fft_aft_filt,colors[10],'FFT After Filtering using design approach')

# Signal before filtering
plt_signal(time_sig,ppg_sig_final,'red','Signal before filtering using design approach')

# Signal after filtering
plt_signal(time_sig,y_m,colors[10],'Signal after filtering using design approach')

#---------------------------------------------------------------------------#
#%%
#----------   Closer look at freq from 20 to 50Hz  ---------------
start_ind=int(20/res_fft)
end_ind=int(50/res_fft)

plt.figure(figsize=(8,6))

plt.plot(x_fft[start_ind+1:end_ind],fft_bef_filt[start_ind+1:end_ind],label='FFT Original',c=f'{colors[1]}')
plt.title('FFT 20 to 50')

plt.plot(x_fft[start_ind+1:end_ind],fft_aft_filt[start_ind+1:end_ind],c='r',label='FFT Filtered')
plt.title('FFT 20 Hz to 50 Hz')
plt.xlabel('Frequency (Hz)',fontsize=10)
plt.ylabel('X(k)',fontsize=10)
plt.tight_layout()
plt.legend()
# plt.savefig('fft_comparison_20_50.png',dpi=400,transparent=0)