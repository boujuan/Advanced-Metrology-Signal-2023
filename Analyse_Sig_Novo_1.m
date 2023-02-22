clear 
close all
clc

%% Load data
%==========================================================================
load("Data/Sig_para_Novo.mat", "ds_spectrum", "ds_wl_range");
spectrum_off = ds_spectrum;
wavelength = ds_wl_range;

%% Analyze the signal
%==========================================================================
% detrending the signal by substracting the means
spectrum = bsxfun(@minus, spectrum_off, mean(spectrum_off, 2));

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
% plot(wavelength, spectrum, wavelength, spectrum_smooth, '-x');
xlabel('Wavelegth (nm)');
ylabel('Amplitude (a. u.)');
title("Signal");
legend('Spectrum 1', 'Spectrum 2', 'Spectrum 3', 'Spectrum 4', 'Spectrum 5');

%% Find the peaks in the spectrum and plot them
%==========================================================================
% Find positive peaks in the spectrum
[peaks,peaksLoc] = findpeaks(spectrum(1,:),'MinPeakProminence',2, 'MinPeakDistance', 0.8, 'MinPeakWidth', 8);
hold on;
plot(wavelength(peaksLoc),peaks,'rv','MarkerFaceColor','r');

% Find negative peaks in the spectrum
[negPeaks,negPeaksLoc] = findpeaks(-spectrum(1,:),'MinPeakProminence',2, 'MinPeakDistance', 0.8, 'MinPeakWidth', 8);
hold on;
plot(wavelength(negPeaksLoc),-negPeaks,'g^','MarkerFaceColor','g');


%% Plotting the Polynomial Fit
%==========================================================================




