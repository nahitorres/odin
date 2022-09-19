from scipy.fft import fft, ifft, fftfreq
import numpy as np
from matplotlib import pyplot as plt

def get_main_period(xf, yf):
    '''
    Given the Fourier transform power spectrum, find the highest peak
    
    Args:
        xf: the frequencies of the power spectrum
        yf: the powers associated with the frequencies in xf, in the same order
        
    Returns:
        period: the period corresponding to the highest power peak
        power: the power of 'period'
    '''
    max_idx = np.argmax(np.abs(yf)) # get the maximum power in the FFT
    period = 1/xf[max_idx] # get the period (1/frequency) associated with the maximum power
    power = np.max(np.abs(yf))
    
    return period, power


def get_fft_periodicity(time_series):   
    '''
    Given a time series, returns its period and the standard deviation of the highest power peak with respect to the Fourier transform mean
    
    Args:
        time_series: a pandas Series containing the y values of a time series
    
    Returns:
        period: the period of the time series, obtained using the Fourier transform
        number_sigma: the number of standard deviation the peak associated with the main period differs from the power spectrum mean
    '''
    
    time_series = time_series - time_series.mean() # subtract the mean from the time series values (used to perform FFT)

    # Compute the FFT of the test set
    yf = fft(time_series.values)
    xf = fftfreq(len(time_series), 1)

    # The Fourier transform is assumed to be symmetric due to the nature of the signal
    mask = [idx for idx, val in enumerate(xf) if val >= 0] # create a mask to take only the positive frequencies
    yf = yf[mask] # consider only the positive frequencies values
    xf = xf[mask]
    
    # Estimate the main period from the highest peak in the Fourier transform
    period, power = get_main_period(xf, yf)
    
    # Get the std of the Fourier transform (yf)
    std = np.std(yf)
    number_sigma = power/std
    
    return period, number_sigma


def plot_fft(time_series):
    '''
    Given a time series, computes the Fast Fourier Transform and plots it using a log-log scale
    
    Args:
        time_series: a pandas Series containing the y values of a time series
        
    Returns:
        -
    '''
    time_series = time_series - time_series.mean() # subtract the mean from the time series values (used to perform FFT)

    # Compute the FFT of the test set
    yf = fft(time_series.values)
    xf = fftfreq(len(time_series), 1)

    # The Fourier transform is assumed to be symmetric due to the nature of the signal
    mask = [idx for idx, val in enumerate(xf) if val >= 0] # create a mask to take only the positive frequencies
    yf = yf[mask] # consider only the positive frequencies values
    xf = xf[mask] # convert to Hz (1/sample -> Hz)


    ## Plot FFT
    fig = plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(xf, np.abs(yf)) # plot only the absolute value of yf

    # Use a logarithmic scale
    plt.yscale('log')
    plt.xscale('log')

    # Set plot description
    plt.title('Power spectrum')
    plt.xlabel('Frequency')
    plt.ylabel('Power')

    # Plot
    plt.show()  