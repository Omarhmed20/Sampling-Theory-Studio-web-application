import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st
from plotly.subplots import make_subplots
import plotly.io as pio
from PyPDF2 import PdfFileReader, PdfFileMerger
import PyPDF2
from scipy.interpolate import interp1d
from scipy import signal
import scipy as sc
import numpy as np
from scipy import signal
from math import ceil,floor


#--------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
st. set_page_config(layout="wide")
pio.templates.default = "simple_white"
side=st.sidebar
with open(r"C:\\Files\DSP\\mystyle.css") as design:
    st.markdown(f"<style>{design.read()}</style>", unsafe_allow_html=True)

st.title("Signal Viewer")

Mixer= side.checkbox('Signal Mixer', value=False)

submitMixer=False
frames =[]
option='none'
if 'max_frequency' not in st.session_state:
    st.session_state['max_frequency'] =1


with side:

    if Mixer == False:
        file = side.file_uploader("Upload Files", type={"csv", "txt", "xlsx"})

    Add_noise = side.checkbox("Add_Noise")
    if Add_noise:
        snr = side.slider('Select SNR', 1, 50, key=0, value=50)
        
   
Ts=[]
Recontructed_Signal=[]
Rescontructed_Time=[]
Amp=[]
Sampling_Value=[]
Difference=[]

if 'noise' not in st.session_state:
            st.session_state['noise'] = 0
if 'noise2' not in st.session_state:
            st.session_state['noise2'] = 0
if 'freqsample' not in st.session_state:
    st.session_state['freqsample'] = 0
if 'Tmax' not in st.session_state:
    st.session_state['Tmax'] = 0
if 'time' not in st.session_state:
    st.session_state['time'] = 0
if 'Noise_power' not in st.session_state:
    st.session_state['Noise_power'] = []
if 'Noise_power2' not in st.session_state:
    st.session_state['Noise_power2'] = []
if 'Signal_Array' not in st.session_state:
            st.session_state['Signal_Array'] = []

if 'x' not in st.session_state:
        st.session_state['x'] = []

if 'y' not in st.session_state:
        st.session_state['y'] = []


#-----------------------------------------------------------------------------------------
if Mixer ==False:
    if file is not None :
        File = pd.read_csv(file)
        #---------------------------------------------
        if Add_noise == True:
            power=np.array(File.iloc[1:1000+1,1])**2
            MeanPower_DB=10*np.log10(np.mean(power))
            noise_W= 10**((MeanPower_DB - snr)/10)
            st.session_state['noise']=np.random.normal(0,np.sqrt(noise_W),len(np.array(File.iloc[0:1001 ,1])))
        #-----------------------------------------------
        # st.experimental_rerun()
        
        st.session_state['x'] = np.array(File.iloc[0:1000+1,0])
        st.session_state['Tmax'] = floor(st.session_state['x'][1000])
        print(st.session_state['Tmax'])
        st.session_state['y'] = np.array(File.iloc[0:1000,1])
        if Add_noise == True:
            st.session_state['y']=st.session_state['y']+st.session_state['noise'][0:1000]

        import pandas as pd
        import numpy as np

        from scipy.fft import fft, fftfreq

        signal=np.array(File.iloc[:,1])
        # Step 3: Remove the DC component
        signal -= np.mean(signal)
        spectrum =  fft(signal)
        sampling_rate = 500  # Assuming the signal is uniformly sampled
        freq = fftfreq(signal.size, d=1/sampling_rate)
        max_frequency = freq[np.argmax(np.abs(spectrum))]
        max_magnitude = np.max(np.abs(spectrum))
        Sampling= side.checkbox('Sampling with fmax', value=False)


        if Sampling==True:
            S_Fmax = side.slider('Sampling  (in fmax)', min_value= 0.1, max_value =4.0, step =0.1)  #Select Sampling Rate, (From 0xFmax to 4xFmax) 
            S_Fm=S_Fmax
            NSamples = ceil(S_Fm*20)
        else:   
            Norm = side.slider('Sampling', min_value= float(20), max_value =float(max_frequency*4), step =0.1)  #Select Sampling Rate, (From 0xFmax to 4xFmax) 
            S_Fm=Norm
            NSamples = ceil(S_Fm )



        print("Maximum frequency:", max_frequency)


                        
        Ts = []                                # the list which will carry the values of the samples of the time
        Sampling_Value = []                              # the list which will carry the values of the samples of the amplitude
        


        Ts = []                              
        Sampling_Value = []             
        for i in range(0, 1000, (1000//(NSamples*8))):        #Calculating sampling pointsdf
            Ts.append(File.iloc[:,0][i])         # take the value of the time
            Sampling_Value.append(st.session_state['y'][i])       #take the value of the amplitude


        #----------------------------------------------------------------------------------------------------------------------------------------
            
        Rescontructed_Time = np.linspace(0,st.session_state['Tmax'], 1000)  # the domain we want to draw the recounstructed signal in it
        tot_time = np.resize(Rescontructed_Time, (len(Ts), len(Rescontructed_Time)))    
        t_nT = (tot_time.T - Ts) / (Ts[1] - Ts[0])        #calculating((t-nT)/Sampling Rate)
        interpolation = Sampling_Value * np.sinc(t_nT)        
        Recontructed_Signal = np.sum(interpolation, axis = 1)
        #----------------------------------------------------------------------------------------------------------------------------------------
        Difference = st.session_state['y']- Recontructed_Signal
        
        
        



    #-------------------------------------------------------------------------------------------------------------------------
if Mixer == True:
    S_Fmax = side.slider('Sampling  (in fmax)', min_value= 0.1, max_value =4.0, step =0.1)  #Select Sampling Rate, (From 0xFmax to 4xFmax) 
    Generated_Signals = pd.read_csv("C:\\Files\\DSP\\Generated_Signals.csv")    #Importing the CSV file that contains generated Signals' Data
    c=[]
    signal_values = [] 
    st.session_state['y']=[]
    st.session_state['Signal_Array']=[]

    with side.form("Generated Signals"):                                              #Choose the Amp,Freq of the generated signal
        shift = st.slider("Shift", min_value=0, max_value = 360, value=0,step=15 )  
        frequency = st.slider("Frequency", min_value=1, max_value = 50, value=1 )  
        amplitude= st.slider("Amplitude", min_value=1, max_value=50, value=1 )
        submitMixer= st.form_submit_button(label = "Add")  

        #---------------------------------------------------------------------------------------------------------

        if submitMixer: 
            
            Samp = Generated_Signals['Amplitude']    #importing Previous Generated Signals' Data
            n="signal"+str(len(Samp))                   
            new_data = {'Name':n,'Shift': shift, 'Frequency': frequency, 'Amplitude': amplitude}
            Generated_Signals = Generated_Signals.append(new_data, ignore_index=True)    
            Generated_Signals.to_csv("C:\\Files\DSP\\Generated_Signals.csv", index=False)
            
        Sname = Generated_Signals['Name']
        SShift = Generated_Signals['Shift']              
        Sfreq = Generated_Signals['Frequency']
        Samp = Generated_Signals['Amplitude']                

        for i in range(0, len(Sfreq)):     #Extracting Freqency and Amplitude values of the Generated Signals
            Sfreq[i] = int(Sfreq[i])   
        for i in range(0, len(Samp)):      
            Samp[i] = int(Samp[i]) 
        for i in range(0, len(SShift)):      
            SShift[i] = int(SShift[i]) 
        print(SShift)
        print(Samp)
        

        st.session_state['x']=[]
        for t in np.arange(0, 2, 0.005):                                 #Generating the Time (from 0 to 2), Point every 0.005 second
            st.session_state['x'].append(t)
        for n in range(0,len(Samp)):
            if SShift[n]==0:    
                Sh =0
            else:
                Sh=(np.pi/(180/SShift[n]))
            for t in np.arange(0, 2, 0.005):                                 #Generating the Time (from 0 to 2), Point every 0.005 second
                st.session_state['y'].append(Samp[n]*np.sin(2*np.pi*Sfreq[n]*t+Sh))

            signal_values.append(st.session_state['y'])
            st.session_state['y']=[]
    #print(signal_values)

    if len(Sname)!= 0: 
        Amp=np.zeros(len(signal_values[0]))    
        for i in range(len(signal_values)): 
            Amp+=np.array(signal_values[i])
            
            
    
    #-------------------------------------------------------------------------
    if Add_noise == True:                                                   #Calculating Noise(Noise required in DB = PSignal - SNR)
        Noise_power2=Amp**2
        Mean_power_DB=10*np.log10(np.mean(Noise_power2))
        noise_W= 10**((Mean_power_DB-snr)/10)
        noise2= np.random.normal(0,np.sqrt(noise_W),400)
        Amp=Amp+noise2
    #---------------------------------------------------------------------------------------------------------
    if len(Sfreq)!=0:    
        NSamples = ceil(S_Fmax *max(Sfreq))

        Ts = []                              
        Sampling_Value = []                            
        if st.session_state['x'] !=[]:
            for i in range(0, 400, (400//(NSamples*2))):        #Calculating sampling points
                Ts.append(st.session_state['x'][i])       
                Sampling_Value.append(Amp[i])    

        #------------------------------------------------------------------------------------------
            Rescontructed_Time = np.linspace(0, 2, 400)         #Restore signal using Nyquist–Shannon sampling theorem
            tot_time = np.resize(Rescontructed_Time, (len(Ts), len(Rescontructed_Time)))    
            t_nT = (tot_time.T - Ts) / (Ts[1] - Ts[0])        #calculating((t-nT)/Sampling Rate)
            interpolation = Sampling_Value * np.sinc(t_nT)
            Recontructed_Signal = np.sum(interpolation, axis=1)
        #--------------------------------------------------------------------------

            Difference = Amp - Recontructed_Signal

            #--------------------------------------------------------------------------
            delete = side.selectbox("Select signal",Sname)    #Function to Remove Generated Signals
            Delete=side.button(label="Delete")
            if Delete:
                Generated_Signals =  Generated_Signals[Generated_Signals.Name != delete] 
                Generated_Signals.to_csv("C:\\Files\\DSP\\Generated_Signals.csv", index=False)  




    #-------------------------------------------------------------------------------------------------------------------------
figure = go.Figure(
    )
import pydeck as pdk

figure.update_layout(autosize=True,
width=180000,
height=600)
#figure.add_trace(go.Scatter(x =st.session_state['x_int'], y = st.session_state['y_int'],mode = "lines",line=dict(color=color1)),)



if Mixer== True:
     y=Amp
else:
     y= st.session_state['y']

figure.add_trace(go.Scatter(x =st.session_state['x'], y = y,mode = "lines",line=dict(color='green'),name='Orignal Signal'),)
figure.add_trace(go.Scatter(x =Ts, y = Sampling_Value,mode = "markers",line=dict(color='blue'),name='Sample points'),)
figure.add_trace(go.Scatter(x =Rescontructed_Time, y = Recontructed_Signal,mode = "lines",line=dict(color='red'),name='Reconstructed'),)
figure.add_trace(go.Scatter(x =st.session_state['x'], y = Difference,mode = "lines",line=dict(color='yellow'),name='Difference'),)
st.plotly_chart(figure,use_container_width=True,use_container_height=True)
    
    #if Mixer == False:
    #    if file is not None:
    #        with container1:
    #            col2.write(figure)
    #            plot=go.Figure(go.Scatter(x =st.session_state['x'], y = st.session_state['y'],mode = "lines",line=dict(color=color1)))
    #        # col1.button('Save as PDF file', on_click=plot.write_image("D:\HEM\DSP\Signal-viewer-main/fig1.pdf"))
    #            df = pd.DataFrame(st.session_state['y'])
    #            table = go.Figure(data=[go.Table(header=dict(values=['Plot', 'Mean','std','duration','max','min']),
    #                            cells=dict(values=['figure 1',df.mean(),df.std(),max(st.session_state['x']),max(df),min(df)]))
    #            ])
            #  table.write_image("D:\HEM\DSP\Signal-viewer-main/Stats1.pdf")
            #  merger = PyPDF2.PdfFileMerger(strict=True)
            #   f1 = PdfFileReader(open('fig1.pdf', 'rb'))
            #   f2 = PdfFileReader(open('Stats1.pdf', 'rb'))
                
            #   merger = PdfFileMerger(strict=True)

            #  merger.append(f1)
            #  merger.append(f2)
            ##  

            #  merger.write('CompleteGraph1.pdf')
#--------------------------------------------------------------------------------------

            # table2.write_image("D:\HEM\DSP\Signal-viewer-main/Stats2.pdf")
            # merger = PyPDF2.PdfFileMerger(strict=True)
            # f1 = PdfFileReader(open('fig2.pdf', 'rb'))
            # f2 = PdfFileReader(open('Stats2.pdf', 'rb'))
                
            # merger = PdfFileMerger(strict=True)

            # merger.append(f1)
            # merger.append(f2)
                

            # merger.write('CompleteGraph2.pdf')


#--------------------------------------------------------------------------------------

       # table.write_image("D:\HEM\DSP\Signal-viewer-main/StatsSubplots.pdf")
       # merger = PyPDF2.PdfFileMerger(strict=True)
       # f1 = PdfFileReader(open('figSubplots.pdf', 'rb'))
       # f2 = PdfFileReader(open('StatsSubplots.pdf', 'rb'))
        
      #  merger = PdfFileMerger(strict=True)

      #  merger.append(f1)
      #  merger.append(f2)
        

      #  merger.write('CompleteGraphSubplots.pdf')

