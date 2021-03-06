import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import buttord,butter,freqz,bilinear,lfilter
from scipy.signal import TransferFunction as tf
from scipy.io import loadmat
from pathlib import Path

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

signal_dict={'PPG_5sec_clean':0, 'PPG_10sec_clean':1, 'PPG_120sec_clean':2,'ppg_baseline':3,
 'ppg_baseline_awgn':4,'ppg_baseline_powerline':5,'ppg_random_noise':6}

#data = loadmat(f'Data/{files[6]}')
#data=data.data;
#ppg_100hz=data[list(data.keys())[-1]][0];
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
#ppg_sig_final=ppg_100hz[:] if len(ppg_100hz)<1000 else ppg_100hz[:1000];
#plt.plot(ppg_sig_final);
    
    
    



#%%                   
#---------------------  Streamlit --------------------------------
#st.title(r'Filtering PPG signals using Butterworth Low Pass Filter')
st.markdown(f'<p style="color:{colors[9]};font-size:38px;font-weight:800;margin-top:10px;">Filtering PPG signals using Butterworth Low Pass Filter</p>', unsafe_allow_html=True)
placeholder1=st.empty()
placeholder2=st.empty()
placeholder3=st.empty()

def display(order,signal):
    placeholder1.empty()  # Clearing elements
    placeholder2.empty()
    placeholder3.empty()
    
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
    ax1.axvline(x=20,ymin=0,ymax=0.69,lw=0.5)
    ax1.axhline(y=0.707,xmin=0, xmax=20.5/50,lw=0.5)
    with placeholder1.container():
        st.markdown(f'<p style="color:{colors[4]};font-size:24px;margin-top:30px;padding-top:40px;padding-bottom:10px;border-top:1px solid white">Butterworth Filter of Order {order}</p>', unsafe_allow_html=True)
        st.pyplot(fig1,dpi=400)
    
    file_number=int(signal_dict[f'{signal}'])
    file=files[file_number]
    filename=Path(file).parents[0] / f'Data/{file}'
    #st.write(filename)
    data = loadmat(filename)
    #data = loadmat(filename)
    ppg_100hz=data[list(data.keys())[-1]][0];
    ppg_100hz=ppg_100hz[:] if len(ppg_100hz)<3000 else ppg_100hz[:3000]
    
    # Plotting PPG
    time_sig=np.linspace(0,len(ppg_100hz)/100,len(ppg_100hz))
    
    fig2,ax2=plt.subplots(figsize=(6,4))
    ax2.plot(time_sig,ppg_100hz,linewidth=1,c='red')
    ax2.set(title=f'{signal} : Original',xlabel='time (sec)')
    fig2.tight_layout()
    with placeholder2.container():
        st.markdown('<p style="color:#E2703A;font-size:24px;margin-top:40px;padding-top:40px;padding-bottom:10px;border-top:1px solid white">PPG Signal before filtering</p>', unsafe_allow_html=True)
        st.pyplot(fig2,dpi=400)
    
    y=lfilter(num,den,ppg_100hz);
    fig3,ax3=plt.subplots(figsize=(6,4))
    ax3.plot(time_sig,y,color=colors[10],lw=1);
    ax3.set(title=f'{signal} : filtered',xlabel='time (sec)')
    fig3.tight_layout()
    with placeholder3.container():
        st.markdown(f'<p style="color:#E2703A;font-size:24px;margin-top:40px;padding-top:40px;padding-bottom:10px;border-top:1px solid white">PPG Signal after filtering</p>', unsafe_allow_html=True)
        st.pyplot(fig3,dpi=400)


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
    