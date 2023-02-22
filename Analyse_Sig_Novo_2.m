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

% smooth outh the signal
% spectrum_smooth = smoothdata(spectrum, 2, 'movmean', 5);

%% Plotting the signal
%==========================================================================
figure( 'Name', "Initial Plot" );
plot(wavelength, spectrum);
% plot(wavelength, spectrum, wavelength, spectrum_smooth, '-x');
xlabel('Wavelegth (nm)');
ylabel('Amplitude (a. u.)');
title("Signal");
legend('Spectrum 1', 'Spectrum 2', 'Spectrum 3', 'Spectrum 4', 'Spectrum 5');

%% Find peaks of the curve for substracting the continuum
%==========================================================================
% Parameters for peak detection
threshold = 1;
minPeakWidth = 0.1;
peaks_array=zeros(5,10000);
continuum=zeros(10000);

% Find the peaks
[peaks,peaks_loc,peaks_width] = findpeaks(spectrum(1,:),wavelength,'MinPeakHeight',threshold,'MinPeakWidth',minPeakWidth);

% Set values of the continuum belonging to the peaks to NaN
for i = 1:length(peaks)
    excluded_x = (x >= peaks(i)-widths(i)) & (x <= peaks(i)+widths(i));
    continuum(excluded_x) = NaN;
end

%% Calculate the polynomial fit of the continuum curve
%==========================================================================
poly_degree = 2; % Set the order of the polynomial to fit
num_curves = size(spectrum, 1);
coeffs = zeros(num_curves, poly_degree+1);



% Plot the fit over the continuum without the peaks and the original signal
figure;
hold on;
plot(wavelength, spectrum(1,:), 'k');
plot(wavelength, continuum(1,:), 'b');
plot(wavelength, poly_eval(1,:), 'r');
xlabel('Wavelength');
ylabel('Signal');
legend('Original', 'Polynomial Fit');
hold off;

