import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend

def read_data(filename):
    return np.genfromtxt(filename, delimiter=',')

def read_files(signal_list, time_list):
    spectrum = (read_data(filename) for filename in spectrum_list)
    wavelengths = (read_data(filename) for filename in wavelength_list)
    return spectrum, wavelengths

def filter_signal(signal, time, threshold):
    # Detrend the signal to remove linear trend
    detrended_signal = detrend(signal)
    
    # Perform Fourier transform of detrended signal to obtain frequency domain
    signal_fft = np.fft.fft(detrended_signal)
    
    # Define amplitude of signal in frequency domain
    magnitude = np.abs(signal_fft)
    frequency = np.fft.fftfreq(len(detrended_signal), d=(time[1] - time[0]))
    
    # Apply threshold and filter signal by multiplying with thresholded magnitude and inverting Fourier transform to obtain filtered signal in time domain
    signal_fft[magnitude <= threshold] = 0
    filtered_signal = np.fft.ifft(signal_fft).real
    
    # Compute residuals by subtracting filtered signal from original signal in time domain
    residuals = detrended_signal - filtered_signal
    
    return detrended_signal, residuals, magnitude, frequency

def plot_signal(axs, x, y, xlabel, ylabel, colour):
    axs.plot(x, y, color=colour)
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)

def main():
    # Define file names
    spectrum_csv = 'Sig_para_Spec_Novo.csv'
    wavelength_csv = 'Sig_para_WL_Novo.csv'

    # Create subplots
    fig, axs = plt.subplots(len(spectrum_csv), 2, figsize=(8, 8))
    axs[0, 0].set_title("Magnitude vs Frequency")
    axs[0, 1].set_title("Filtered Signal and Residuals")
    plt.subplots_adjust(wspace=0.3, hspace=0.5)

    # Filter signals and plot them
    for i, (spectrum_csv, wavelength_csv) in enumerate(zip(spectrum_csv, wavelength_csv)):
        detrended_signal, residuals, magnitude, frequency= filter_signal(spectrum_csv, wavelength_csv, 1000)
        
        # Plot filtered signal and residuals
        plot_signal(axs[i][0], frequency, magnitude, "Frequency (Hz)", "Magnitude", "blue")
        axs[i][0].set_xlim(0, 10000)
        plot_signal(axs[i][1], time, detrended_signal, "Time (s)", "Signal", "blue")
        plot_signal(axs[i][1], time, residuals, "Time (s)", "Signal", "red")
    plt.show()

if __name__ == "__main__":
    main()