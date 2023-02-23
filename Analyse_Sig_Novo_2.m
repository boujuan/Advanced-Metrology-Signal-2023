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
spectrogram = raw_spectra; % !!!!!!!!!!!! Left it without detrending due to weird asymptotes happening with correction later

% define timestep (wavelength_resolution in this case) and sampling frequency
wavelength_resolution = wavelength(2) - wavelength(1);
frequency = 1/wavelength_resolution;

% Display variance of each spectrum
spectrum_var = var(spectrogram(1:5,:), 0, 2); 
disp("Variance of each spectrum:");
disp(spectrum_var)

%% Plotting the signal
%==========================================================================
figure( 'Name', "Initial Plot" );
plot(wavelength, spectrogram);
xlabel('Wavelegth (nm)');
ylabel('Amplitude (a. u.)');
title("Signal");
legend('Spectrum 1', 'Spectrum 2', 'Spectrum 3', 'Spectrum 4', 'Spectrum 5');

%% Find the peaks in the spectrum and plot them
%==========================================================================
% Initialize the peak locations and values arrays
numSpectra = 4; % Only 4 spectra are being corrected because the 5th doesn't follow the curve
maxPeaksPerSpectrum = 10;
peaks = zeros(numSpectra, maxPeaksPerSpectrum);
peaksLoc = zeros(numSpectra, maxPeaksPerSpectrum);
negPeaks = zeros(numSpectra, maxPeaksPerSpectrum);
negPeaksLoc = zeros(numSpectra, maxPeaksPerSpectrum);

%smooth the curve slightly for better peak finding
spectrum_smooth = smoothdata(spectrogram, 2, 'movmean', 3);

hold on
for i = 1:numSpectra
    % Find positive peaks in the spectrum
    [peaksFound, peaksLocFound] = findpeaks(spectrum_smooth(i,:), 'MinPeakProminence', 0.5,'MinPeakHeight', 1.5,'Threshold', 0.01);
    numPeaksFound = numel(peaksFound);
    peaks(i, 1:numPeaksFound) = peaksFound;
    peaksLoc(i, 1:numPeaksFound) = peaksLocFound;
    plot(wavelength(peaksLoc(i,1:numPeaksFound)), peaks(i,1:numPeaksFound), 'rv', 'MarkerFaceColor', 'r');
    
    % Find negative peaks in the spectrum
    [negPeaksFound, negPeaksLocFound] = findpeaks(-spectrum_smooth(i,:), 'MinPeakProminence', 0.2, 'Threshold', 0.001);
    numNegPeaksFound = numel(negPeaksFound);
    negPeaks(i, 1:numNegPeaksFound) = -negPeaksFound;
    negPeaksLoc(i, 1:numNegPeaksFound) = negPeaksLocFound;
    plot(wavelength(negPeaksLoc(i,1:numNegPeaksFound)), negPeaks(i,1:numNegPeaksFound), 'g^', 'MarkerFaceColor', 'g');
end
hold off

%% Create the calibration curve
%==========================================================================
% Initialize calibration_curve as a copy of spectrogram
calibration_curve = spectrogram;

% Define the region around the peaks to set to NaN (0.5 wavelength values left and right)
region_size = round(0.5 / wavelength_resolution);

for i = 1:numSpectra
    % Get the positive and negative peak locations for the current spectrum
    peak_locs = peaksLoc(i,:);
    peak_locs = peak_locs(peak_locs~=0); % Remove zero entries
    neg_peak_locs = negPeaksLoc(i,:);
    neg_peak_locs = neg_peak_locs(neg_peak_locs~=0); % Remove zero entries

    % Set values around the positive peaks to NaN in calibration_curve
    for j = 1:length(peak_locs)
        peak_loc = peak_locs(j);
        lower_bound = max(1, peak_loc - region_size);
        upper_bound = min(size(calibration_curve, 2), peak_loc + region_size);
        calibration_curve(i, lower_bound:upper_bound) = NaN;
    end
    
    % Set values around the negative peaks to NaN in calibration_curve
    for j = 1:length(neg_peak_locs)
        neg_peak_loc = neg_peak_locs(j);
        lower_bound = max(1, neg_peak_loc - region_size);
        upper_bound = min(size(calibration_curve, 2), neg_peak_loc + region_size);
        calibration_curve(i, lower_bound:upper_bound) = NaN;
    end
end

figure( 'Name', "Calibration Curve" );
plot(wavelength,calibration_curve);
title("Calibration Curve");

%% Interpolate the missing values in Calibration Curve
%==========================================================================
% Interpolate each row of calibration_curve to fill in the missing NaN values
calibration_curve_interpolated = calibration_curve;
for i = 1:num_curves
    % Get the current row of calibration_curve
    current_row = calibration_curve_interpolated(i,:);
    
    % Find the indices of the NaN values
    nan_indices = find(isnan(current_row));
    
    % If there are any NaN values, interpolate them
    if ~isempty(nan_indices)
        % Find the indices of the non-NaN values
        non_nan_indices = find(~isnan(current_row));
        
        % Interpolate the NaN values using a cubic spline
        current_row(nan_indices) = interp1(non_nan_indices, current_row(non_nan_indices), nan_indices, 'linear');
        
        % Replace the current row in calibration_curve with the interpolated values
        calibration_curve_interpolated(i,:) = current_row;
    end
end

% Plot the interpolated calibration curve
figure( 'Name', "Interpolated Calibration Curve" );
plot(wavelength, calibration_curve_interpolated);
xlabel('Wavelength (nm)');
ylabel('Amplitude (a.u.)');
title("Interpolated Calibration Curve");

%% Find the polynomial fit of the calibration currve
%==========================================================================
x = wavelength;
y = calibration_curve_interpolated(1,:);
p = polyfit(x(~isnan(y)), y(~isnan(y)), 5); % fit a 4th-degree polynomial, ignoring NaN values
y_fit = polyval(p, x); % evaluate the polynomial at each wavelength

figure( 'Name', "Polynomial Fit" );
plot(wavelength, spectrogram);
hold on
plot(x, y_fit, 'LineWidth',2, 'Color', 'r', 'LineStyle', '--');
title("Polynomial Fit");
legend('Spectrum 1', 'Spectrum 2', 'Spectrum 3', 'Spectrum 4', 'Spectrum 5', 'Polynomial Fit');

%% Divide the signal by the polynomial fit to get the corrected signal
%==========================================================================
corrected_signal = spectrogram ./ y_fit;

figure( 'Name', "Corrected Signal" );
plot(wavelength, corrected_signal);
xlabel('Wavelength (nm)');
ylabel('Amplitude (a.u.)');
title("Corrected Signal");
legend('Spectrum 1', 'Spectrum 2', 'Spectrum 3', 'Spectrum 4', 'Spectrum 5');

%%% Weird asimptoptes centered around wavelength(1775)=435.48 nm ???
%%% Caused by polynomial fit approaching 0. => Corrected by removing
%%% detrend of the signal at the start of the code

%% Detrend Now??
% detrending the signal by substracting the means
corrected_signal_detrended = bsxfun(@minus, corrected_signal, mean(corrected_signal, 2));
% corrected_signal_detrended = bsxfun(@minus, corrected_signal, 1);
figure( 'Name', "Corrected Signal Detrended" );
plot(wavelength, corrected_signal_detrended);
xlabel('Wavelength (nm)');
ylabel('Amplitude (a.u.)');
title("Corrected Signal Detrended");
legend('Spectrum 1', 'Spectrum 2', 'Spectrum 3', 'Spectrum 4', 'Spectrum 5');

