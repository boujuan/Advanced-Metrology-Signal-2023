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
% plot(wavelength, spectrum, wavelength, spectrum_smooth, '-x');
xlabel('Wavelegth (nm)');
ylabel('Amplitude (a. u.)');
title("Signal");
legend('Spectrum 1', 'Spectrum 2', 'Spectrum 3', 'Spectrum 4', 'Spectrum 5');

%% Find the peaks in the spectrum and plot them
%==========================================================================
% Initialise the peak locations and values arrays
peaks = [];
peaksLoc = [];
negPeaks = [];
negPeaksLoc = [];

%% Spectrum 1
hold on;
% Find positive peaks in the spectrum
[peaks(1,:),peaksLoc(1,:)] = findpeaks(spectrum(1,:),'MinPeakProminence',2, 'MinPeakDistance', 0.8, 'MinPeakWidth', 8, 'Threshold',0.1);
plot(wavelength(peaksLoc(1,:)),peaks(1,:),'rv','MarkerFaceColor','r');

% Find negative peaks in the spectrum
[negPeaks(1,:),negPeaksLoc(1,:)] = findpeaks(-spectrum(1,:),'MinPeakProminence',2, 'MinPeakDistance', 0.8, 'MinPeakWidth', 8,'Threshold',0.1);
plot(wavelength(negPeaksLoc(1,:)),-negPeaks(1,:),'g^','MarkerFaceColor','g');

%% Spectrum 2
hold on;
% Find positive peaks in the spectrum
[peaks,peaksLoc] = findpeaks(spectrum(2,:),'MinPeakProminence',2, 'MinPeakDistance', 0.8, 'MinPeakWidth', 8, 'Threshold',0.2);
plot(wavelength(peaksLoc),peaks,'rv','MarkerFaceColor','r');

% Find negative peaks in the spectrum
[negPeaks,negPeaksLoc] = findpeaks(-spectrum(2,:),'MinPeakProminence',2, 'MinPeakDistance', 0.8, 'MinPeakWidth', 8,'Threshold',0.1);
plot(wavelength(negPeaksLoc),-negPeaks,'g^','MarkerFaceColor','g');

%% Spectrum 3
hold on;
% Find positive peaks in the spectrum
[peaks,peaksLoc] = findpeaks(spectrum(3,:),'MinPeakProminence',2, 'MinPeakDistance', 0.8, 'MinPeakWidth', 8, 'Threshold',0.2);
plot(wavelength(peaksLoc),peaks,'rv','MarkerFaceColor','r');

% Find negative peaks in the spectrum
[negPeaks,negPeaksLoc] = findpeaks(-spectrum(3,:),'MinPeakProminence',2, 'MinPeakDistance', 0.8, 'MinPeakWidth', 8,'Threshold',0.1);
plot(wavelength(negPeaksLoc),-negPeaks,'g^','MarkerFaceColor','g');

%% Spectrum 4
hold on;
% Find positive peaks in the spectrum
[peaks,peaksLoc] = findpeaks(spectrum(4,:),'MinPeakProminence',2, 'MinPeakDistance', 0.8, 'MinPeakWidth', 8, 'Threshold',0.2);
plot(wavelength(peaksLoc),peaks,'rv','MarkerFaceColor','r');

% Find negative peaks in the spectrum
[negPeaks,negPeaksLoc] = findpeaks(-spectrum(4,:),'MinPeakProminence',2, 'MinPeakDistance', 0.8, 'MinPeakWidth', 8,'Threshold',0.1);
plot(wavelength(negPeaksLoc),-negPeaks,'g^','MarkerFaceColor','g');


hold off;


%% Plotting the Polynomial Fit
%==========================================================================




