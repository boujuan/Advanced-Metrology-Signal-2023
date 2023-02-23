clear 
close all
clc

%% Load data
%==========================================================================
load("Data/Sig_para_Novo.mat", "ds_spectrum", "ds_wl_range");
raw_spectra = ds_spectrum;
wavelength = ds_wl_range;

%% Analyze the signal
%==========================================================================
num_curves = size(raw_spectra, 1);
% detrending the signal by substracting the means
spectrum = bsxfun(@minus, raw_spectra, mean(raw_spectra, 2));

% define timestep (wavelength_resolution in this case) and sampling frequency
wavelength_resolution = wavelength(2) - wavelength(1);
frequency = 1/wavelength_resolution;

% Display variance of each spectrum
spectrum_var = var(spectrum(1:5,:), 0, 2); 
disp("Variance of each spectrum:");
disp(spectrum_var)

%% Plotting the signal
%==========================================================================
figure( 'Name', "Initial Plot" );
plot(wavelength, spectrum);
xlabel('Wavelegth (nm)');
ylabel('Amplitude (a. u.)');
title("Signal");
legend('Spectrum 1', 'Spectrum 2', 'Spectrum 3', 'Spectrum 4', 'Spectrum 5');

%% Find the peaks in the spectrum and plot them
%==========================================================================
% Initialize the peak locations and values arrays
numSpectra = 4;
maxPeaksPerSpectrum = 10;
peaks = zeros(numSpectra, maxPeaksPerSpectrum);
peaksLoc = zeros(numSpectra, maxPeaksPerSpectrum);
negPeaks = zeros(numSpectra, maxPeaksPerSpectrum);
negPeaksLoc = zeros(numSpectra, maxPeaksPerSpectrum);

hold on
i = 3;
%smooth the curve for better peak finding
spectrum_smooth = smoothdata(spectrum, 2, 'movmean', 5);
plot(wavelength, spectrum_smooth, 'b');


[peaks,peaksLoc] = findpeaks(spectrum_smooth(i,:),'MinPeakProminence',0.2,'MinPeakHeight',1.5,'Threshold',0.01);
plot(wavelength(peaksLoc),peaks,'rv','MarkerFaceColor','r');

%% Plotting the Polynomial Fit
%==========================================================================




