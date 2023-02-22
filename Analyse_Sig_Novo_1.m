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

%% Calculate the polynomial fit of the continuum curve
%==========================================================================
poly_degree = 2; % Set the order of the polynomial to fit
num_curves = size(spectrum, 1);
coeffs = zeros(num_curves, poly_degree+1);

% Parameters for peak detection
threshold = 1;
peaks_array=zeros(5,10000);
continuum_array=zeros(5,10000);

for i = 1:num_curves
    [peaks,continuum_array(i,:)] = peakdetector(wavelength,spectrum(i,:),threshold); % Detect peaks from the smooth spectrum and exclude regions
    valid_idx = ~isnan(continuum_array(i,:)); % exclude NaN regions from polynomial fit
    coeffs(i,:) = polyfit(wavelength(valid_idx), continuum_array(i,valid_idx), poly_degree); % Fit a polynomial to the continuum data
end

poly_eval = zeros(num_curves, length(wavelength));
for i = 1:num_curves
    poly_eval(i,:) = polyval(coeffs(i,:), wavelength); % Evaluate the fitted curve at the original wavelength values
end

figure;
hold on;
for i = 1:num_curves
    plot(wavelength, spectrum(i,:), 'k');
    plot(wavelength, poly_eval(i,:), 'r');
end
xlabel('Wavelength');
ylabel('Signal');
legend('Original', 'Polynomial Fit');
hold off;

%% calculate SNR
%==========================================================================
signal_power = rms(spectrum(:))^2;
noise_power = rms(spectrum - mean(spectrum, 2), 'all')^2;
snr = signal_power / noise_power;
snr_db = 10*log10(snr);

fprintf("Signal power = %.2f\n", signal_power);
fprintf("Noise power = %.2f\n", noise_power);
fprintf("SNR = %.2f (%.2f dB)\n", snr, snr_db);

%% [Function] Peak detection method for substracting them from the continuum
%==========================================================================
function [peaks,continuum] = peakdetector(x,y,threshold)

    % Inputs:
    % x: x-axis values
    % y: y-axis values
    % threshold: peak detection threshold

    % Outputs:
    % peaks: index of peak values
    % continuum: y-axis values with peaks and excluded regions set to NaN

    % Find the positive and negative peaks and their widths
    [pos_pks,pos_locs,pos_widths] = findpeaks(y,x,'MinPeakHeight',threshold,'MinPeakWidth',0.5);
    [neg_pks,neg_locs,neg_widths] = findpeaks(-y,x,'MinPeakHeight',threshold,'MinPeakWidth',0.5);

    % Combine the positive and negative peaks
    pos_peaks = pos_locs(pos_pks>=threshold);
    neg_peaks = neg_locs(neg_pks>=threshold);
    peaks = sort([pos_peaks neg_peaks]);
    widths = sort([pos_widths neg_widths]);

    % Exclude regions around the peaks
    continuum = y;
    for i = 1:length(peaks)
        excluded_x = (x >= peaks(i)-widths(i)) & (x <= peaks(i)+widths(i));
        continuum(excluded_x) = NaN;
    end

    disp(size(peaks))
    disp(size(continuum))

end
