import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import buttord,butter,freqz,bilinear,lfilter
from scipy.signal import TransferFunction as tf
from scipy.io import loadmat
import time
import os

plt.style.use('dark_background')
colors=['#7A0BC0','#270082','#4E9F3D','#D8E9A8','#E2703A','#590995','#C62A88','#8A2BE2','#ff1493',
        '#7fff00','#3a90e5','#ffcd00']

#--------------------------------------------------------------------------#
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
#---------------------  Streamlit --------------------------------
st.title(r'Filtering PPG signals using Butterworth Low Pass Filter')
placeholder=st.empty()

def display(order,signal):
    placeholder.empty()  # Clearing elements
    
    [b,a] = butter(order,Omega_c_w,analog=True);
    Hs = tf(b,a); # Normalized Analog TF  H(s)
    
    [num,den] = bilinear(b,a,fs);
    Hz = tf(num, den, dt=Td); # Dggital Filter Function H(z)

    # Magnitude Response
    w = np.arange(0,np.pi,np.pi/128)
    _,Hw = freqz(num,den,worN=w)
    Hw_mag = np.absolute(Hw);

    #Plot 1 - Filter Transfer Function
    (fig1,ax1)=plt.subplots(figsize=(6,4))
    ax1.plot(w/np.pi*fs/2,Hw_mag,label=f'N={order}',lw=2,c=colors[3]) 
    ax1.set(title='Magnitude Response of Butterworth LPF',xlabel='frequency (Hz)',
            ylabel='Magnitude');
    ax1.legend()
    placeholder.pyplot(fig1,clear_figure=True,dpi=400)
    
    
    data = loadmat(f'Data\{signal}')
    ppg_100hz=data[list(data.keys())[-1]][0];
    fig2,ax2=plt.subplots();
    ppg_sig_final=ppg_100hz[:];
    ax2.plot(ppg_sig_final);
    ax2.set(title='PPG Signal',xlabel='frequency (Hz)',
            ylabel='Magnitude');
    ax2.legend()
    placeholder.pyplot(fig2,clear_figure=True,dpi=400)

with st.sidebar:
    with st.form("my_form"):
        signal = st.selectbox(
             'Select the PPG Signal to be filtered',
             ('PPG_5sec_clean', 'PPG_10sec_clean', 'PPG_120sec_clean','ppg_baseline',
              'ppg_baseline_awgn','ppg_baseline_powerline','ppg_random_noise'))
        
        order = st.slider('Select Filter Order', n, 14, 6)
        submitted = st.form_submit_button("Submit")
        
if submitted:
    with st.spinner('Loading Plots !!'):
        display(order,signal)
        