import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import buttord,butter,freqz,bilinear,lfilter
from scipy.signal import TransferFunction as tf
from scipy.io import loadmat
from pathlib import Path

st.set_page_config(layout="wide")

plt.style.use('dark_background')
colors=['#7A0BC0','#270082','#4E9F3D','#D8E9A8','#E2703A','#590995','#C62A88','#8A2BE2','#ff1493',
        '#7fff00','#3a90e5','#ffcd00']

#--------------------------------------------------------------------------#
Fc=20
Ap=3
As=40
Fs=100
Td=1/Fs

def calculate_min_order(Ap,As,Fs,fc):
    Td=1/Fs
    stop_band_limit = 2*fc
    Omega_s_limit = (2/Td)*np.tan(2*np.pi*stop_band_limit/Fs/2) # Stop band limit
                                                                 # in analog domain
    Omega_c = (2/Td)*np.tan(2*np.pi*fc/Fs/2)    
    N=1
    while True:
        Omg_p=( (10**(0.1*Ap)-1)**(0.5/N) ) * Omega_c
        Omg_s=( (10**(0.1*As)-1)**(0.5/N) ) * Omega_c
        if Omg_s < Omega_s_limit:
            return N
        N=N+1    
        
# N=calculate_min_order(Ap,As,Fs,Fc)
# Omega_c=(2/Td)*np.tan(2*np.pi*Fc/Fs/2);
#----------------------------------------------------------------------------#


#%%
#--------------------- Loading data -----------------------------------#
files=['PPG_5sec_clean_100Hz.mat','PPG_10sec_clean_100Hz.mat','PPG_120sec_clean_100Hz.mat',
       'ppg_baseline.mat','ppg_baseline_awgn.mat','ppg_baseline_powerline.mat','ppg_random_noise.mat']

signal_dict={'PPG_5sec_clean':0, 'PPG_10sec_clean':1, 'PPG_120sec_clean':2,'ppg_baseline':3,
 'ppg_baseline_awgn':4,'ppg_baseline_powerline':5,'ppg_random_noise':6}

#%%                   
#---------------------  Streamlit --------------------------------
st.markdown(f'<p style="color:{colors[9]};font-size:38px;font-weight:800;margin-top:10px;">Filtering PPG signals using Butterworth Low Pass Filter and Bilinear Transform</p>', unsafe_allow_html=True)
placeholder1=st.empty()
placeholder2=st.empty()
placeholder3=st.empty()


def display(N,signal,Fc):
    placeholder1.empty()  # Clearing elements
    placeholder2.empty()
    placeholder3.empty()
    
    with placeholder2.container():
        col1, col2 = st.columns(2)
        
    with placeholder1.container():
        col4, col5, col6 = st.columns([2,6,2])
        
    with placeholder3.container():
        col7, col8 = st.columns(2)
    
    Omega_c=(2/Td)*np.tan(2*np.pi*Fc/Fs/2);
    
    [b,a] = butter(order,Omega_c,analog=True);
    
    [num,den] = bilinear(b,a,Fs);
    Hz = tf(num, den, dt=Td); # Dggital Filter Function H(z)

    # Magnitude Response
    w = np.arange(0,np.pi,np.pi/128)
    _,Hw = freqz(num,den,worN=w)
    Hw_mag = np.absolute(Hw);

    #Plot 1 - Filter Transfer Function -----------------------------#
    (fig1,ax1)=plt.subplots(figsize=(6,4))
    ax1.scatter(Fc, 0.70711, marker='x',s=20,c='w',zorder=3,label='3dB cutoff')
    ax1.plot(w/np.pi*Fs/2,Hw_mag,label=f'N={order}',lw=2,c=colors[8]) 
    ax1.set_title('Magnitude Response',fontsize=9);
    ax1.set_xlabel('Frequency (Hz)',fontsize=7)
    ax1.tick_params(axis='x', labelsize=6)
    ax1.tick_params(axis='y', labelsize=6)
    ax1.set_ylabel('Magnitude',fontsize=7)
    ax1.legend(fontsize=6)
    ax1.axvline(x=Fc,ymin=0,lw=0.5)
    ax1.axhline(y=0.707,xmin=0,lw=0.5)
    with placeholder1.container():
        with col4:
            pass
    with placeholder1.container():
        with col5:
            st.markdown(f'<p style="color:{colors[2]};font-size:24px;margin-top:30px;padding-top:40px;padding-bottom:10px;">Butterworth Filter of Order {order}</p>', unsafe_allow_html=True)
            st.pyplot(fig1,dpi=400)
    with placeholder1.container():
        with col6:
            pass
    
    #--------------- Plots 2 Signals ---------------#
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
        with col1:
            st.markdown(f'<p style="color:{colors[2]};font-size:24px;margin-top:40px;padding-top:40px;text-align:center;padding-bottom:5px;">PPG Signal before filtering</p>', unsafe_allow_html=True)
            st.pyplot(fig2,dpi=400)
    
    y=lfilter(num,den,ppg_100hz);
    fig3,ax3=plt.subplots(figsize=(6,4))
    ax3.plot(time_sig,y,color=colors[10],lw=1);
    ax3.set(title=f'{signal} : filtered',xlabel='time (sec)')
    fig3.tight_layout()
    with placeholder2.container():
        with col2:
            st.markdown(f'<p style="color:{colors[2]};font-size:24px;margin-top:40px;padding-top:40px;text-align:center;padding-bottom:5px;">PPG Signal after filtering</p>', unsafe_allow_html=True)
            st.pyplot(fig3,dpi=400)
            
    #--------- Plots 3 FFT -------------------------------#
    
    fft_len=int(np.ceil(np.log2(len(ppg_100hz))));
    plt.figure(17);
    fft_bef_filt=np.absolute(np.fft.fft(ppg_100hz,2**fft_len));

    res_fft=Fs/len(fft_bef_filt) # Resolution
    x_fft=np.arange(0,len(fft_bef_filt))*res_fft
    
    #Output
    y=lfilter(num,den,ppg_100hz)
    fft_aft_filt=np.absolute(np.fft.fft(y,2**fft_len));
    
    (fig4,ax4)=plt.subplots(figsize=(6,4))
    ax4.plot(x_fft,fft_bef_filt,c=f'{colors[11]}',lw=1) 
    ax4.set_title('FFT before filtering',fontsize=9)
    ax4.set_xlabel('Frequency (Hz)',fontsize=7)
    ax4.tick_params(axis='x', labelsize=6)
    ax4.tick_params(axis='y', labelsize=6)
    ax4.set_ylabel('X(k)',fontsize=7)
    
    with placeholder3.container():
        with col7:
            st.markdown(f'<p style="color:{colors[2]};font-size:24px;margin-top:40px;padding-top:40px;text-align:center;padding-bottom:5px;">FFT of signal before filtering</p>', unsafe_allow_html=True)
            st.pyplot(fig4)
    
    (fig5,ax5)=plt.subplots(figsize=(6,4))
    ax5.plot(x_fft,fft_aft_filt,c=f'{colors[7]}',lw=1) 
    ax5.set_title('FFT after filtering',fontsize=9)
    ax5.set_xlabel('Frequency (Hz)',fontsize=7)
    ax5.tick_params(axis='x', labelsize=6)
    ax5.tick_params(axis='y', labelsize=6)
    ax5.set_ylabel('X(k)',fontsize=7)
    
    with placeholder3.container():
        with col8:
            st.markdown(f'<p style="color:{colors[2]};font-size:24px;margin-top:40px;padding-top:40px;text-align:center;padding-bottom:5px;">FFT of signal after filtering</p>', unsafe_allow_html=True)
            st.pyplot(fig5)
    


with st.sidebar:
    Fc = st.radio("Select a Cut off frequency",(20, 15, 10,5))
    with st.form("my_form"):
        signal = st.selectbox(
             'Select the PPG Signal to be filtered',
             ('PPG_5sec_clean', 'PPG_10sec_clean', 'PPG_120sec_clean','ppg_baseline',
              'ppg_baseline_awgn','ppg_baseline_powerline','ppg_random_noise'))
        
        N=calculate_min_order(Ap,As,Fs,Fc)
        order = st.slider('Select Filter Order', N, 10, N)
        
        submitted = st.form_submit_button("Submit")


if submitted:
    with st.spinner('Loading Plots !!'):
        display(order,signal,Fc)
    