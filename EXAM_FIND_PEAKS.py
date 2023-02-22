import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
# ------------------------------------------------------------------------------------------------------------------
# Reading the files into a NumPy array:
WL = np.loadtxt("Data/Sig_para_WL_Novo.csv", delimiter=",")
# WL = Wavelength!!
spec = np.loadtxt("Data/Sig_para_Spec_Novo.csv", delimiter=",")
#Spec = Spectral Lines!!


# split the spec array into five parts of size 10000 and name each split array
spec_1, spec_2, spec_3, spec_4, spec_5 = np.split(spec, 5, axis=1)

# creating new arrays with the shape (,10000), like WL
spec_1 = spec[0]
spec_2 = spec[1]
spec_3 = spec[2]
spec_4 = spec[3]
spec_5 = spec[4]

# ------------------------------------------------------------------------------------------------------------------
#Finding Peaks Spec_1
threshold = 365

def find_peaks(spec_1, threshold):
    peaks = []
    for i in range(1, len(spec_1) - 1):
        if spec_1[i] < threshold:
            #if spec_1[i] > spec_1[i-1] and spec_1[i] > spec_1[i+1]:
            peaks.append(i)
    return peaks

peaks = find_peaks(spec_1, threshold)
print(peaks)

#plot the peaks which are found
# plt.plot(WL, spec_1, label='Original Data (Spec_1)')
# plt.plot([WL[i] for i in peaks], [spec_1[i] for i in peaks], 'ro', label='Peaks')
# plt.legend()
# plt.show()

#replacing peaks by None
def replace_peaks(spec_1, threshold):
    peaks = find_peaks(spec_1, threshold)
    new_spec_1 = spec_1.copy()
    for i in peaks:
        new_spec_1[i] = None
    return new_spec_1

new_spec_1 = replace_peaks(spec_1, threshold)

# Filter out None values
mask = np.logical_not(np.isnan(new_spec_1))
x = WL[mask]
y = new_spec_1[mask]

degree = 4 # choose the degree of the polynomial fit
poly_fit = np.polyfit(x, y, degree)
# evaluate the polynomial for each value of WL:
fitted_spec_1 = np.polyval(poly_fit, WL)

plt.plot(WL, spec_1, label= "Data (Spec_1)")
plt.plot(WL, fitted_spec_1, label= "Polyfit_1")
plt.xlabel('Wavelength')
plt.ylabel('Spectral Lines')
plt.legend()
plt.show()

# ------------------------------------------------------------------------------------------------------------------
#Finding Peaks Spec_2
threshold = 365

def find_peaks(spec_2, threshold):
    peaks = []
    for i in range(1, len(spec_2) - 1):
        if spec_2[i] < threshold:
            #if spec_2[i] > spec_2[i-1] and spec_2[i] > spec_2[i+1]:
            peaks.append(i)
    return peaks

peaks = find_peaks(spec_2, threshold)
print(peaks)

#plot the peaks which are found
# plt.plot(WL, spec_2, label='Original Data (Spec_2)')
# plt.plot([WL[i] for i in peaks], [spec_2[i] for i in peaks], 'ro', label='Peaks')
# plt.legend()
# plt.show()

#replacing peaks by None
def replace_peaks(spec_2, threshold):
    peaks = find_peaks(spec_2, threshold)
    new_spec_2 = spec_2.copy()
    for i in peaks:
        new_spec_2[i] = None
    return new_spec_2

new_spec_2 = replace_peaks(spec_2, threshold)

# Filter out None values
mask = np.logical_not(np.isnan(new_spec_2))
x = WL[mask]
y = new_spec_2[mask]

degree = 4 # choose the degree of the polynomial fit
poly_fit = np.polyfit(x, y, degree)
# evaluate the polynomial for each value of WL:
fitted_spec_2 = np.polyval(poly_fit, WL)

plt.plot(WL, spec_2, label= "Cutted Data (Spec_2)")
plt.plot(WL, fitted_spec_2, label= "Polyfit_1")
 # Set the axis labels and title
plt.xlabel('Wavelength')
plt.ylabel('Spectral Lines')
plt.legend()
plt.show()
# ------------------------------------------------------------------------------------------------------------------
#Finding Peaks Spec_3
threshold = 365

def find_peaks(spec_3, threshold):
    peaks = []
    for i in range(1, len(spec_3) - 1):
        if spec_3[i] < threshold:
            #if spec_3[i] > spec_3[i-1] and spec_3[i] > spec_3[i+1]:
            peaks.append(i)
    return peaks

peaks = find_peaks(spec_3, threshold)
print(peaks)

#plot the peaks which are found
# plt.plot(WL, spec_3, label='Original Data (Spec_3)')
# plt.plot([WL[i] for i in peaks], [spec_3[i] for i in peaks], 'ro', label='Peaks')
# plt.legend()
# plt.show()

#replacing peaks by None
def replace_peaks(spec_3, threshold):
    peaks = find_peaks(spec_3, threshold)
    new_spec_3 = spec_3.copy()
    for i in peaks:
        new_spec_3[i] = None
    return new_spec_3

new_spec_3 = replace_peaks(spec_3, threshold)

# Filter out None values
mask = np.logical_not(np.isnan(new_spec_3))
x = WL[mask]
y = new_spec_3[mask]

degree = 4 # choose the degree of the polynomial fit
poly_fit = np.polyfit(x, y, degree)
# evaluate the polynomial for each value of WL:
fitted_spec_3 = np.polyval(poly_fit, WL)

plt.plot(WL, spec_3, label= "Cutted Data (Spec_3)")
plt.plot(WL, fitted_spec_3, label= "Polyfit_1")
plt.xlabel('Wavelength')
plt.ylabel('Spectral Lines')
plt.legend()
plt.show()

# ------------------------------------------------------------------------------------------------------------------
#Finding Peaks Spec_4
threshold = 365

def find_peaks(spec_4, threshold):
    peaks = []
    for i in range(1, len(spec_4) - 1):
        if spec_4[i] < threshold:
            #if spec_4[i] > spec_4[i-1] and spec_4[i] > spec_4[i+1]:
            peaks.append(i)
    return peaks

peaks = find_peaks(spec_4, threshold)
print(peaks)

#plot the peaks which are found
# plt.plot(WL, spec_4, label='Original Data (Spec_4)')
# plt.plot([WL[i] for i in peaks], [spec_4[i] for i in peaks], 'ro', label='Peaks')
# plt.legend()
# plt.show()

#replacing peaks by None
def replace_peaks(spec_4, threshold):
    peaks = find_peaks(spec_4, threshold)
    new_spec_4 = spec_4.copy()
    for i in peaks:
        new_spec_4[i] = None
    return new_spec_4

new_spec_4 = replace_peaks(spec_4, threshold)

# Filter out None values
mask = np.logical_not(np.isnan(new_spec_4))
x = WL[mask]
y = new_spec_4[mask]

degree = 4 # choose the degree of the polynomial fit
poly_fit = np.polyfit(x, y, degree)
# evaluate the polynomial for each value of WL:
fitted_spec_4 = np.polyval(poly_fit, WL)

plt.plot(WL, spec_4, label= "Cutted Data (Spec_4)")
plt.plot(WL, fitted_spec_4, label= "Polyfit_1")
 # Set the axis labels and title
plt.xlabel('Wavelength')
plt.ylabel('Spectral Lines')
plt.legend()
plt.show()

# ------------------------------------------------------------------------------------------------------------------
#Finding Peaks Spec_5
threshold = 360

def find_peaks(spec_5, threshold):
    peaks = []
    for i in range(1, len(spec_5) - 1):
        if spec_5[i] < threshold:
            #if spec_5[i] > spec_5[i-1] and spec_5[i] > spec_5[i+1]:
            peaks.append(i)
    return peaks

peaks = find_peaks(spec_5, threshold)
print(peaks)

# plt.plot(WL, spec_5, label='Original Data (Spec_5)')
# plt.plot([WL[i] for i in peaks], [spec_5[i] for i in peaks], 'ro', label='Peaks')
# plt.legend()
# plt.show()

# ------------------------------------------------------------------------------------------------------------------
#replacing peaks by None
def replace_peaks(spec_5, threshold):
    peaks = find_peaks(spec_5, threshold)
    new_spec_5 = spec_5.copy()
    for i in peaks:
        new_spec_5[i] = None
    return new_spec_5

new_spec_5 = replace_peaks(spec_5, threshold)

# Filter out None values
mask = np.logical_not(np.isnan(new_spec_5))
x = WL[mask]
y = new_spec_5[mask]

degree = 4 # choose the degree of the polynomial fit
poly_fit = np.polyfit(x, y, degree)
# evaluate the polynomial for each value of WL:
fitted_spec_5 = np.polyval(poly_fit, WL)

plt.plot(WL, spec_5, label= "Cutted Data (Spec_5)")
plt.title("Modified Dataset together with the Fit")
plt.plot(WL, fitted_spec_5, label= "Polyfit_1")
plt.xlabel('Wavelength')
plt.ylabel('Spectral Lines')
plt.legend()
plt.show()

# # Create a 2x2 grid of subplots
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

# Plot Spec_1 and Spec_2 in the top row
axs[0, 0].plot(WL, spec_1, label="Original Data (Spec_1)")
axs[0, 0].plot(WL, fitted_spec_1, label="Polyfit_1")
axs[0, 0].set_ylabel('Spectral Lines')
axs[0, 0].legend()

axs[0, 1].plot(WL, spec_2, label="Original Data (Spec_2)")
axs[0, 1].plot(WL, fitted_spec_2, label="Polyfit_2")
axs[0, 1].set_ylabel('Spectral Lines')
axs[0, 1].legend()

# Plot Spec_3 and Spec_4 in the bottom row
axs[1, 0].plot(WL, spec_3, label="Original Data (Spec_3)")
axs[1, 0].plot(WL, fitted_spec_3, label="Polyfit_3")
axs[1, 0].set_xlabel('Wavelength')
axs[1, 0].set_ylabel('Spectral Lines')
axs[1, 0].legend()

axs[1, 1].plot(WL, spec_4, label="Original Data (Spec_4)")
axs[1, 1].plot(WL, fitted_spec_4, label="Polyfit_4")
axs[1, 1].set_xlabel('Wavelength')
axs[1, 1].set_ylabel('Spectral Lines')
axs[1, 1].legend()
fig.suptitle('Fitted Spectra for Spec_1 to Spec_4', fontsize=16) #Title
plt.show()