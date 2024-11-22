import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def wiener_filter(noisy_signal, noise_power): #, signal_power
    # Compute the Fourier transform of the noisy signal
    Y = np.fft.fft(noisy_signal)
    
    # Compute the power spectral densities
    S_yy = np.abs(Y) ** 2
    S_nn = noise_power
    #S_ss = signal_power
    #S_ss = np.maximum(S_yy - S_nn, 0)
    S_ss = signal.savgol_filter(S_yy, window_length=51, polyorder=5) - S_nn
    S_ss = np.maximum(S_ss, 0)  # Ensure non-negativity
    
    # Compute the Wiener filter in the frequency domain
    H = S_ss / (S_ss + S_nn)
    
    # Apply the filter to the noisy signal in the frequency domain
    S_hat = H * Y
    
    # Compute the inverse Fourier transform to get the filtered signal
    filtered_signal = np.fft.ifft(S_hat)
    
    return np.real(filtered_signal)

# Example usage
#np.random.seed(0)
t = np.linspace(0, 1, 500, endpoint=False)
original_signal = np.sin(2 * np.pi * 5 * t) # 5 hz signal
noise = np.random.normal(0, 0.5, t.shape)
noisy_signal = original_signal + noise

# We can assume that this is a scalar because we use white noise -> have equal power over all frequencies
# This is equal to 0.5^2 because its the var of noise which has std of 0.5 and var = std^2
noise_power = 0.5 ** 2 

# Commenting out because if we have the og signal we would just use that
# signal_power = np.var(original_signal)

filtered_signal = wiener_filter(noisy_signal, noise_power) #, signal_power

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, original_signal, label='Original Signal')
plt.plot(t, noisy_signal, label='Noisy Signal')
plt.plot(t, filtered_signal, label='Filtered Signal')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Wiener Filter')
plt.show()