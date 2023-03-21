import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
#values of used components
R1 = 10000 #Ohm
R2 = 100000 #Ohm
C = 5.6*10**(-9) #Farad
V_in = 1 #Volt

#function to find the difference in amplitude and phase of a wave current from before and after the low-pass
def ampandphasediff(frequency):
    #read data
    filename = 'NewFile1' + str(frequency) + 'hz.csv'
    (ch1, ch2, t_start, t_step, element) = read_data(filename)

    #time array (correct timestep added later)
    t = element

    #invert ch2, since OPV inverts signal
    ch2 =  (-1)* ch2

    #find amplitude difference between ch1 and ch2
    ch1_max = ch1.max()
    ch2_max = ch2.max()
    a = R2/R1 # amplification factor
    ampdiff = 1 - (ch1_max - abs(ch2_max)/a)/ch1_max #decrease in amplitude normalised to 1

    #find phase difference between ch1 and ch2
    peaks1 = find_peaks(ch1, width = 130)
    peaks2 = find_peaks(ch2, width = 130)

    peak1 = peaks1[0][0]
    peak2 = peaks2[0][0]
    

    tdiff = (peak1 - peak2) * t_step
    phasediff = (2 * np.pi * tdiff * frequency)/ np.pi #phasediff with norm to steps of pi

    #Uncertainties:
    #amp error not useful, since log scale estimated to be 0.04 Volt
    #phase error:
    #peak error estimated to be 5 measurement steps, frequency with no significant error
    pherr = (2 * np.pi * (np.sqrt(2)*5*t_step) * frequency)/ np.pi 

    return ampdiff, phasediff, pherr


#expected cut off frequency
fg_c = 1/(2*np.pi*R2*C)

#function to calculate the theoretical amplitude for a given frequency
def A_c(f,c):
   V_out = V_in*abs(1/(2*np.pi*f*c))/np.sqrt((1/(2*np.pi*f*c))**2 + R2**2)
   return V_out

#function to calculate the theoratical phase for a given frequency
def phi_c(f, c):
    phi = -np.arctan(2*np.pi*f*c*R2)/np.pi
    return phi

#get computed points from data
#array of used frequencies
f = np.array([20,50,100,200,500,1000,2000,5000,10000,20000,50000,100000])
x1 = np.ones(12) 
x2 = np.ones(12)
x2err = np.ones(12)
for i in range(0,12):
    x1[i], x2[i], x2err[i]= ampandphasediff(f[i])

#find a fit for C for our amplitude data
OptVal, CovarianceMatrix = sp.optimize.curve_fit(A_c, f,x1, sigma = 0.04*np.ones(12))
Cf = OptVal
Cf_err = np.sqrt(CovarianceMatrix[0][0])
print('fit reslut:', Cf, Cf_err)

#cut off frequency for fit
fg_f = 1/(2*np.pi*R2*Cf)



fig, axs=plt.subplots(1, 2, figsize=(12, 6)) #defining figure object
axs[0].plot(f, x1, 'kx') #amplitude values
#axs[0].errorbar(f, x1, yerr = np.ones(12)*0.04 ,linestyle='none',capsize = 4, label = 'Fehler $\mathregular{\omega}$')
axs[0].plot(np.arange(20,100000), A_c(np.arange(20,100000), C))
axs[0].plot(np.arange(20,100000), A_c(np.arange(20,100000), Cf), color = 'green')
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_xlabel('Frequenz f [Hz]')
axs[0].set_ylabel('normierte Ausgangsspannung')
axs[0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
axs[0].axvline(x = fg_c, color = 'grey', label = '$\mathregular{f_g\ theoretisch}$', ls = '--')
axs[0].axvline(x = fg_f, color = 'grey', label = '$\mathregular{f_g\ für\ Fit}$', ls= ':')
axs[0].legend(loc = 'lower left')
axs[0].set_title('(A)')
axs[1].errorbar(f, x2, yerr = x2err ,linestyle='none',capsize = 4, label = 'Fehler $\mathregular{\phi}$', color = 'grey')
axs[1].plot(f, x2, 'kx') #phase values
axs[1].plot(np.arange(20,100000), phi_c(np.arange(20,100000), C), label = 'theoretisch berechnet')
axs[1].plot(np.arange(20,100000), phi_c(np.arange(20,100000), Cf), label = 'Fit mit $\mathregular{C = 3,57(8) * 10^{-9}}$', color = 'green')
axs[1].set_xscale('log')
axs[1].set_xlabel('Frequenz f [Hz]')
axs[1].set_ylabel('normierte Phasenverschiebung $\mathregular{\Phi}$')
axs[1].set_title('(B)')
plt.legend()
plt.show()