%%=========================================================================
% ■■■■■■■■■ BEGINNING OF THE MATLAB SPECTRAL SIGNAL ANALYSIS TOOL ■■■■■■■■■
% =========================================================================
% For the exam of the class of 'Advanced Metrology' (5.04.4660)
% Professor/Examiner: Prof. Dr. Philipp Huke
% Date: 20/02/2023 to 27/02/2023
% Institution: University of Oldenburg
% Study: Master Engineering Physics
% Student Name: Juan Manuel Boullosa Novo
% Student ID: 2481927
%
% Description:
% The following code was made individually with the purpose to load and 
% analyse the data provided belonging to a spectral information from an
% alledged "Prism-spectrograph" tool. This code begins by doing a basic
% signal analysis of the properties of the data, including the plotting,
% to then finding the interesting peaks in the spectrum, taking them
% apart to focus on the continuum line, which is fitted to a polynomial
% curve in order to flatten it. Then it is detrended, and the peaks
% belonging to the interesting spectral lines are located against the
% spurious ones and they are also fitted to a gaussian curve.
% This allow us to characterise them and compare them with common 
% wavelength values found in spectrography database such as the NIST one.
%
% All this information is encapsulated in the attached report pdf.
% =========================================================================
%% Init Clearance
clear 
close all
clc
%==========================================================================
%% Load data
%==========================================================================
load("Data/Sig_para_Novo.mat", "ds_spectrum", "ds_wl_range");
raw_spectra = ds_spectrum;
wavelength = ds_wl_range;
%==========================================================================
%% Access NIST API and download the DATA
%==========================================================================
url = 'https://data.nist.gov/rmm/resourceApi';
filename = 'Data/image.jpg';
websave(filename,url);
%==========================================================================
%% Analyze the signal
%==========================================================================
num_curves = size(raw_spectra, 1);
spectrogram = raw_spectra; % !Left it without detrending due to
%  weird asymptotes happening with correction later

% define timestep (wavelength_resolution in this case) 
% and sampling frequency
wavelength_resolution = wavelength(2) - wavelength(1);
frequency = 1/wavelength_resolution;
disp("Wavelength resolution:");
disp(wavelength_resolution)

% Display variance of each spectrum
spectrum_var = var(spectrogram(1:5,:), 0, 2); 
disp("Variance of each spectrum:");
disp(spectrum_var)
%==========================================================================
%% Plotting the signal
%==========================================================================
figure( 'Name', "Initial Plot" );
plot(wavelength, spectrogram);
xlabel('Wavelegth (nm)');
ylabel('Amplitude (a. u.)');
title("Signal");
legend('Spectrum 1', 'Spectrum 2', 'Spectrum 3', 'Spectrum 4', 'Spectrum 5');
%==========================================================================
%% Find the peaks in the spectrum and plot them
%==========================================================================
% Initialize the peak locations and values arrays
% Only 4 spectra are being corrected because the 5th doesn't follow the curve
numSpectra = 4; 
maxPeaksPerSpectrum = 10; % Pre-allocate space for performance
peaks = zeros(numSpectra, maxPeaksPerSpectrum);
peaksLoc = zeros(numSpectra, maxPeaksPerSpectrum);
negPeaks = zeros(numSpectra, maxPeaksPerSpectrum);
negPeaksLoc = zeros(numSpectra, maxPeaksPerSpectrum);

% Smooth the curve slightly for better 
% peak finding -currently 3-points moving average-
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
%==========================================================================
%% Create the calibration curve
%==========================================================================
% Initialize calibration_curve as a copy of spectrogram
calibration_curve = spectrogram;

% Define the region around the peaks to set to NaN 
% (0.5 wavelength values left and right)
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
%==========================================================================
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
%==========================================================================
%% Find the polynomial fit of the calibration currve
%==========================================================================
x = wavelength;
y = calibration_curve(1,:);
% fit a 4th-degree polynomial, ignoring NaN values
degree = 4;
warning('off', 'MATLAB:polyfit:RepeatedPointsOrRescale');
p = polyfit(x(~isnan(y)), y(~isnan(y)), degree);
y_fit = polyval(p, x); % evaluate the polynomial at each wavelength

figure( 'Name', "Polynomial Fit" );
plot(wavelength, spectrogram);
hold on
plot(x, y_fit, 'LineWidth',2, 'Color', 'r', 'LineStyle', '--');
title("Polynomial Fit");
legend('Spectrum 1', 'Spectrum 2', 'Spectrum 3', 'Spectrum 4', 'Spectrum 5', 'Polynomial Fit');
%==========================================================================
%% Divide the signal by the polynomial fit to get the corrected signal
%==========================================================================
corrected_signal = spectrogram ./ y_fit;

figure( 'Name', "Corrected Signal" );
plot(wavelength, corrected_signal);
xlabel('Wavelength (nm)');
ylabel('Amplitude (a.u.)');
title("Corrected Signal");
legend('Spectrum 1', 'Spectrum 2', 'Spectrum 3', 'Spectrum 4', 'Spectrum 5');
%==========================================================================
%% Detrend The signal
%==========================================================================
% detrending the signal by substracting the means
corrected_signal_detrended = bsxfun(@minus, corrected_signal, mean(corrected_signal, 2));

figure( 'Name', "Corrected Signal Detrended" );
plot(wavelength, corrected_signal_detrended);
xlabel('Wavelength (nm)');
ylabel('Amplitude (a.u.)');
title("Corrected Signal Detrended");
legend('Spectrum 1', 'Spectrum 2', 'Spectrum 3', 'Spectrum 4', 'Spectrum 5');
%==========================================================================
%% Find a gaussian fit for the peaks
%==========================================================================
% Defines the ranges where the gaussian peaks of the spectral lines are
% pressent in wavelength values (nm)
ranges = [424.66 425.44; 425.66 426.42; 426.8 427.54; 427.9 428.6; 439.88 440.56; 446.48 447.06; 523.04 523.5; 583.46 583.96];
% Find indices of these range values using the created function
% find_indices. Returns a x by 2 array.
sl_indices = find_indices(wavelength, ranges);

% Find the gaussian fit for each spectral line(sl_indices height) of each
% spectrum (numSpectra=4, as we don't count the broken number 5 one)
% Calculate the number of subplots needed
num_subplots = size(sl_indices, 1) * numSpectra;

% Create the figure with subplots in each column
figure('Name', "Gaussian Fit");
set(gcf, 'Position', get(0, 'Screensize')); % Fullscreen plot
for i = 1:num_subplots
    % Calculate the spectrum and spectral line indices for this subplot
    spectrum_index = mod(i-1, numSpectra) + 1;
    sl_index = ceil(i/numSpectra);
    
    % Create the subplot
    subplot(size(sl_indices,1), numSpectra, (sl_index-1)*numSpectra + spectrum_index);
    
    % Extract x and y values around the peak
    x_data = wavelength(sl_indices(sl_index, 1):sl_indices(sl_index, 2));
    y_data = corrected_signal_detrended(spectrum_index,sl_indices(sl_index, 1):sl_indices(sl_index, 2));
    
    % Fit the data with a gaussian function
    % startPoint is the initial guess for the algorithm.
    % startPoint = [0.00160161352429178 mean(x_data) 0.118834175367943];
    startPoint = [std(y_data) mean(x_data) max(y_data)-min(y_data)];
    [fitresult, gof] = gaussianFit(x_data, y_data, startPoint);

    % Plot fit with data.
    plot( fitresult, x_data, y_data );
    legend('off');
    
    % Label axes and title, grid
    xlabel( 'Wavelength', 'Interpreter', 'none' );
    ylabel( 'Signal', 'Interpreter', 'none' );
    title(sprintf("Spectrum %d - Spectral Line %d", spectrum_index, sl_index));
    grid on
end
% ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■%
%% =================== FUNCTIONS ======================================= %%
% ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■%
%% - GAUSSIAN FIT -
%==========================================================================
function [fitresult, gof] = gaussianFit(x_data, y_data, startPoint)
% gaussianFit(X_DATA,Y_DATA)
%  Create a gaussain fit:
%
%  Input:
%      X Input: x_data
%      Y Output: y_data
%  Output:
%      fitresult : a fit object representing the fit.
%      gof : structure with goodness-of fit info.

[xData, yData] = prepareCurveData( x_data, y_data);

% Set up fit type and parameters
ft = fittype( 'gauss1' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.Lower = [-Inf -Inf 0];
opts.StartPoint = startPoint;

% Fit model to the data using the previous parameters
[fitresult, gof] = fit( xData, yData, ft, opts );
end
%% - FIND WAVELENGTH INDICES -
%==========================================================================
function sl_indices = find_indices(wavelength_axis, ranges)
% Fanction to find the indices of the spectral lines within specified ranges
%   
%   Input:
%       wavelength_axis: a vector of wavelength values
%       ranges: a matrix of range values, with each row representing the start
%       and end points of a range
%   Output:
%       sl_indices: a matrix of the starting and ending indices of the spectral
%       lines within the specified ranges

sl_indices = zeros(size(ranges, 1), 2); % Preallocate the output matrix

    for i = 1:size(ranges, 1)
        % Find indices within the range
        sl_sub_indices = find((wavelength_axis >= ranges(i, 1)) & (wavelength_axis <= ranges(i, 2)));
        
        % Assign the starting and ending indices
        sl_indices(i, :) = [sl_sub_indices(1), sl_sub_indices(end)];
    end
end
%