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
maxPeaksPerSpectrum = 10; % Pre-allocate space for performance (more than needed)
peaks = zeros(numSpectra, maxPeaksPerSpectrum);
peaksLoc = zeros(numSpectra, maxPeaksPerSpectrum);
negPeaks = zeros(numSpectra, maxPeaksPerSpectrum);
negPeaksLoc = zeros(numSpectra, maxPeaksPerSpectrum);

%smooth the curve slightly for better peak finding -currently 3-points moving average-
spectrum_smooth = smoothdata(spectrogram, 2, 'movmean', 3);

hold on
for i = 1:numSpectra
    % Find positive peaks in the spectrum
    [peaksFound, loc_posPeaks] = findpeaks(spectrum_smooth(i,:), 'MinPeakProminence', 0.5,'MinPeakHeight', 1.5,'Threshold', 0.01);
    numPeaksFound = numel(peaksFound);
    peaks(i, 1:numPeaksFound) = peaksFound;
    peaksLoc(i, 1:numPeaksFound) = loc_posPeaks;
    plot(wavelength(peaksLoc(i,1:numPeaksFound)), peaks(i,1:numPeaksFound), 'rv', 'MarkerFaceColor', 'r');
    
    % Find negative peaks in the spectrum
    [negPeaksFound, loc_negPeaks] = findpeaks(-spectrum_smooth(i,:), 'MinPeakProminence', 0.2, 'Threshold', 0.001);
    numNegPeaksFound = numel(negPeaksFound);
    negPeaks(i, 1:numNegPeaksFound) = -negPeaksFound;
    negPeaksLoc(i, 1:numNegPeaksFound) = loc_negPeaks;
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
        
        % Interpolate the NaN values using a linear interpolation
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
p = polyfit(x(~isnan(y)), y(~isnan(y)), 4); % fit a 4th-degree polynomial, ignoring NaN values
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

%% Find a gaussian fit for the peaks
%==========================================================================
% THIS_WL = 400;
% loc = [THIS_WL-400]./wavelength_resolution;

% Defines the ranges where the gaussian peaks are pressent manually
ranges = [424.62 425.44; 425.66 426.42; 426.8 427.54; 427.9 428.6; 439.88 440.56; 446.48 447.06; 523.06 523.38; 583.46 583.96];

% Function to find the indices of these range values
for i = 1:size(ranges, 1)
    range_start = ranges(i, 1);
    range_end = ranges(i, 2);
    
    indices = [];
    for j = 1:length(wavelength)
        if range_start <= wavelength(j) && wavelength(j) <= range_end
            indices = [indices j];
        end
    end
    
    values = wavelength(indices);
    first_index = indices(1);
    last_index = indices(end);

    disp("  wavelength(" + first_index + ":" + (last_index+1) + ")")
end

% Extract x and y values around the peak
x_data = wavelength(1233:1274);
y_data = corrected_signal_detrended(1233:1274);

figure('Name',"Corrected Signal detrended 1 with gaussian fit for peak1");
plot(wavelength, corrected_signal_detrended(1,:));

% Define Gaussian function
gaussian = fittype('a*exp(-((x-b)/c)^2)', 'independent', 'x', 'coefficients', {'a', 'b', 'c'});
% gauss = @(coeff, x_data) coeff(1) * exp(-((x_data - coeff(2)) / coeff(3)).^2);

% Initial guess for parameters
% guess = [max_val, center, width];
% lb = [0, center - width, 0];
% ub = [2 * max_val, center + width, 2 * width];
a0 = max(y_data);
b0 = x_data(y_data==a0);
c0 = 1;

% Fit Gaussian function to data
% [coeff, ~, ~, ~, ~, ~] = lsqcurvefit(gauss, guess, xdata, ydata, lb, ub);
% fit_result = gauss(coeff, xdata);
fit_result = fit(x_data.', y_data.', 'gauss2');

% Plot original data and fitted Gaussian function
hold on
plot(fit_result)
hold off

