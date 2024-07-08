import uRAD_USB_SDK11  # import uRAD library
import serial
import numpy as np
from numpy.fft import fft
from time import time, sleep
import csv
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import os
import pandas as pd

# Initialize live plot for x_fil and x_conv
plt.ion()
fig, ax = plt.subplots()
line_dc, = ax.plot([], [], 'b-', label='x_dc')
line_fil, = ax.plot([], [], 'r-', label='x_fil')
line_conv, = ax.plot([], [], 'g-', label='x_conv')
ax.set_title('Real-time Phase Data')
ax.set_xlabel('Samples')
ax.set_ylabel('Phase (radians) / Distance (mm)')
ax.legend()
ax.grid(True)

# Define timeSleep here
timeSleep = 5e-3

# input parameters
mode = 2  # sawtooth mode
f0 = 5  # starting at 24.005 GHz
BW = 240  # using all the BW available = 240 MHz
Ns = 200  # 200 samples
Ntar = 1  # 1 target of interest
Rmax = 100  # searching along the full distance range
MTI = 0  # MTI mode disable because we want information of static and moving targets
Mth = 0  # parameter not used because "movement" is not requested
Alpha = 10  # signal has to be 10 dB higher than its surrounding
distance_true = False  # request distance information
velocity_true = False  # mode 2 does not provide velocity information
SNR_true = True  # Signal-to-Noise-Ratio information requested
I_true = True  # In-Phase Component (RAW data) not requested
Q_true = True  # Quadrature Component (RAW data) not requested
movement_true = False  # not interested in boolean movement detection

# True if USB, False if UART
usb_communication = True
# Serial Port configuration
ser = serial.Serial()

# Method to correctly turn OFF and close uRAD
def closeProgram():
    try:
        # switch OFF uRAD
        return_code = uRAD_USB_SDK11.turnOFF(ser)
        if return_code != 0:
            print("Error turning OFF uRAD")
    except Exception as e:
        print(f"Exception turning OFF uRAD: {e}")

try:
    if usb_communication:
        ser.port = 'COM5'
        ser.baudrate = int(1e6)
    else:
        ser.port = '/dev/serial0'
        ser.baudrate = 115200

    # Other serial parameters
    ser.bytesize = serial.EIGHTBITS
    ser.parity = serial.PARITY_NONE
    ser.stopbits = serial.STOPBITS_ONE

    # Open serial port
    ser.open()

    try:
        # switch ON uRAD
        return_code = uRAD_USB_SDK11.turnON(ser)
        if return_code != 0:
            closeProgram()
    except Exception as e:
        print(f"Exception turning ON uRAD: {e}")
        closeProgram()

    try:
        # loadConfiguration uRAD
        return_code = uRAD_USB_SDK11.loadConfiguration(ser, mode, f0, BW, Ns, Ntar, Rmax, MTI, Mth, Alpha, distance_true, velocity_true, SNR_true, I_true, Q_true, movement_true)
        if return_code != 0:
            closeProgram()
    except Exception as e:
        print(f"Exception loading configuration: {e}")
        closeProgram()

    hasil_i = []
    hasil_q = []
    hasil_phase = []
    hasil_magnitude = []

    phase_file_path = "./hasil_phase.csv"
    if os.path.exists(phase_file_path):
        os.remove(phase_file_path)

    x_dc_file_path = "./hasil_x_dc.csv"
    if os.path.exists(x_dc_file_path):
        os.remove(x_dc_file_path)

    x_fil_file_path = "./hasil_x_fil.csv"
    if os.path.exists(x_fil_file_path):
        os.remove(x_fil_file_path)

    x_conv_file_path = "./hasil_x_conv.csv"
    if os.path.exists(x_conv_file_path):
        os.remove(x_conv_file_path)

    x_w_file_path = "./hasil_x_w.csv"
    if os.path.exists(x_w_file_path):
        os.remove(x_w_file_path)

    i_file = None
    q_file = None
    magnitude_file = None
    x_dc_file = None
    x_fil_file = None
    x_conv_file = None
    x_w_file = None

    try:
        i_file = open(f"./hasil_i.csv", "w+", newline="")
        q_file = open(f"./hasil_q.csv", "w+", newline="")
        magnitude_file = open(f"./hasil_magnitude.csv", "w+", newline="")

        i_csv = csv.writer(i_file)
        q_csv = csv.writer(q_file)
        magnitude_csv = csv.writer(magnitude_file)

        x_dc_file = open(x_dc_file_path, "w+", newline="")
        x_dc_csv = csv.writer(x_dc_file)

        x_fil_file = open(x_fil_file_path, "w+", newline="")
        x_fil_csv = csv.writer(x_fil_file)

        x_conv_file = open(x_conv_file_path, "w+", newline="")
        x_conv_csv = csv.writer(x_conv_file)

        x_w_file = open(x_w_file_path, "w+", newline="")
        x_w_csv = csv.writer(x_w_file)

        # Initialize x_fil with an empty list
        x_fil = []

        # Start time
        start_time = time()

        x_mean = 0

        minute = 1

        # Run indefinitely until manually stopped
        while True:

            # target detection request
            return_code, results, raw_results = uRAD_USB_SDK11.detection(ser)
            if return_code != 0:
                closeProgram()

            # Extract results from outputs
            NtarDetected = results[0]
            distance = results[1]
            SNR = results[3]
            I = raw_results[0]
            Q = raw_results[1]

            # Iterate through desired targets
            for i in range(NtarDetected):
                # If SNR is big enough
                if SNR[i] > 0:
                    # FFT Processing
                    datai = np.array(I)
                    dataq = np.array(Q)
                    i_csv.writerow(I)
                    q_csv.writerow(Q)

                    max_voltage = 3.3  # Maximum voltage
                    ADC_intervals = 4096  # ADC intervals (12-bit)
                    Ns = 200  # Number of samples
                    fs = 200000  # Sampling frequency (200 kHz)
                    c = 3e8  # Speed of light in m/s
                    fc = 24.005e9  # Center frequency in Hz

                    # Calculate mean of I and Q
                    mean_I = np.mean(I)
                    mean_Q = np.mean(Q)

                    # Subtract mean and scale to voltage
                    data_i = (I - mean_I) * (max_voltage / ADC_intervals)
                    data_q = (Q - mean_Q) * (max_voltage / ADC_intervals)

                    # Form complex vector
                    ComplexVector = data_i + 1j * data_q

                    # Perform FFT
                    ComplexVectorFFT = fft(ComplexVector)

                    # Calculate frequency axis
                    freq = np.fft.fftfreq(Ns, 1 / fs)

                    # Calculate magnitude and phase
                    Magnitude = np.absolute(ComplexVectorFFT)
                    magnitude_csv.writerow(Magnitude)
                    hasil_magnitude.append(Magnitude)

                    # Find maximum magnitude and corresponding index
                    max_fft = np.max(Magnitude)
                    index_fft = np.where(Magnitude == max_fft)[0][0]
                    print(index_fft)

                    # Get the phase at the peak index
                    Phase = np.angle(ComplexVectorFFT)
                    phase_value = Phase[index_fft]

                    # Append phase value to list and write to CSV
                    hasil_phase.append(phase_value)
                    with open(f"./hasil_phase.csv", "a", newline="") as f:
                        phase_csv = csv.writer(f)
                        phase_csv.writerow([phase_value])
                        f.flush()  # Ensure immediate write

                    # Assuming phase_data is a numpy array
                    phase_data = np.array(hasil_phase)  # Replace with your actual data

                    for i in range(1, len(phase_data) + 1):
                        x_w = np.unwrap(phase_data[:i])

                    # Save unwrapped phase data to CSV
                    with open(f"./hasil_x_w.csv", "w+", newline="") as f:
                        x_w_csv = csv.writer(f)
                        for value in x_w:
                            x_w_csv.writerow([value])

                    if len(x_w) == 100:
                        x_mean = np.mean(x_w)
                    else:
                        print("")

                    # DC Removal
                    x_dc = x_w - x_mean

                    # Save x_dc to CSV
                    with open(f"./hasil_x_dc.csv", "w+", newline="") as f:
                        x_dc_csv = csv.writer(f)
                        for value in x_dc:
                            x_dc_csv.writerow([value])

                    # Check if the length of x_dc is greater than the required padding length
                    padlen = 15
                    if len(x_dc) > padlen:
                        # Convert phase to distance (in millimeters)
                        x_lambda = c / fc  # Wavelength in meters
                        x_conv = 1000 * x_dc * x_lambda / (2 * np.pi)  # Convert to millimeters
                        
                        # Butterworth filter
                        b, a = butter(4, 0.01)  # Butterworth filter with normalized cutoff frequency
                        x_fil = lfilter(b, a, x_conv)  # Apply filter using filtfilt for zero-phase filtering

                        # Save x_fil to CSV
                        with open(f"./hasil_x_fil.csv", "w+", newline="") as f:
                            x_fil_csv = csv.writer(f)
                            for value in x_fil:
                                x_fil_csv.writerow([value])

                        # Save x_conv to CSV
                        with open(f"./hasil_x_conv.csv", "w+", newline="") as f:
                            x_conv_csv = csv.writer(f)
                            for value in x_conv:
                                x_conv_csv.writerow([value])

                        # for i in range (0, len(x_fil)): 
                        perubahan_fasa = x_fil[i-15]/2
                        print(f"Perubahan Fasa {i-15}: {perubahan_fasa}")
                        hasil_pengukuran = index_fft*450+perubahan_fasa
                        print(f"Hasil Pengukuran Ke-{i+1}: {hasil_pengukuran} mm")

                    # Update live plot for x_dc, x_fil, and x_conv
                        line_dc.set_xdata(np.arange(len(x_dc)))
                        line_dc.set_ydata(x_dc)
                        line_fil.set_xdata(np.arange(len(x_fil)))
                        line_fil.set_ydata(x_fil)
                        line_conv.set_xdata(np.arange(len(x_conv)))
                        line_conv.set_ydata(x_conv)
                        ax.set_xlim(0, max(len(x_dc), len(x_fil), len(x_conv)) - 1)
                        ax.set_ylim(min(min(x_dc), min(x_fil), min(x_conv)), max(max(x_dc), max(x_fil), max(x_conv)))
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                    else:
                        print(f"Skipping filtering, insufficient length: {len(x_dc)}")
                        # Ensure x_fil is the same length as x_w by appending None
                        x_fil.extend([None] * (len(x_w) - len(x_fil)))

            if NtarDetected > 0:
                pass

            # Sleep during specified time
            if not usb_communication:
                sleep(timeSleep)

    except Exception as e:
        print(f"Error in main loop: {e}")

    finally:
        # Close serial port
        try:
            ser.close()
        except Exception as e:
            print(f"Error closing serial port: {e}")

        # Close files
        try:
            if i_file:
                i_file.close()
            if q_file:
                q_file.close()
            if magnitude_file:
                magnitude_file.close()
            if x_dc_file:
                x_dc_file.close()
            if x_fil_file:
                x_fil_file.close()
            if x_conv_file:
                x_conv_file.close()
            if x_w_file:
                x_w_file.close()
        except Exception as e:
            print(f"Error closing files: {e}")

except Exception as e:
    print(f"Error initializing program: {e}")
