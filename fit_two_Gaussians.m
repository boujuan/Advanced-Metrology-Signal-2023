% Uses fitnlm() to fit a non-linear model (sum of two gaussians on a ramp) through noisy data.
% Requires the Statistics and Machine Learning Toolbox, which is where fitnlm() is contained.
% Initialization steps.
clc;    % Clear the command window.
close all;  % Close all figures (except those of imtool.)
clear;  % Erase all existing variables. Or clearvars if you want.
workspace;  % Make sure the workspace panel is showing.
format long g;
format compact;
fontSize = 20;

% Create the X coordinates from 0 to 20 every 0.5 units.
X = linspace(0, 80, 400);
mu1 = 10; % Mean, center of Gaussian.
sigma1 = 3; % Standard deviation.
mu2 = 50; % Mean, center of Gaussian.
sigma2 = 9; % Standard deviation.

% Define function that the X values obey.
a = 6 % Arbitrary sample values I picked.
b = 35
c = 25
m = 0.1
Y = a + m * X + b * exp(-(X - mu1) .^ 2 / sigma1)+ c * exp(-(X - mu2) .^ 2 / sigma2); % Get a vector.  No noise in this Y yet.

% Add noise to Y.
Y = Y + 1.2 * randn(1, length(Y));
% Now we have noisy training data that we can send to fitnlm().
% Plot the noisy initial data.
plot(X, Y, 'b.', 'LineWidth', 2, 'MarkerSize', 15);
grid on;

% Convert X and Y into a table, which is the form fitnlm() likes the input data to be in.
tbl = table(X', Y');
% Define the model as Y = a + b*x + c*exp(-(x-d)^2/e) + d * exp(-(x-f)^2/g)
% Note how this "x" of modelfun is related to big X and big Y.
% x(:, 1) is actually X and x(:, 2) is actually Y - the first and second columns of the table.
modelfun = @(b,x) b(1) + b(2) * x(:, 1) + b(3) * exp(-(x(:, 1) - b(4)).^2/b(5)) + b(6) * exp(-(x(:, 1) - b(7)).^2/b(8));  
beta0 = [6, 0.1, 35, 10, 3, 25, 50, 9]; % Guess values to start with.  Just make your best guess.
% Now the next line is where the actual model computation is done.
mdl = fitnlm(tbl, modelfun, beta0);
% Now the model creation is done and the coefficients have been determined.
% YAY!!!!

% Extract the coefficient values from the the model object.
% The actual coefficients are in the "Estimate" column of the "Coefficients" table that's part of the mode.
coefficients = mdl.Coefficients{:, 'Estimate'}

% Let's do a fit, but let's get more points on the fit, beyond just the widely spaced training points,
% so that we'll get a much smoother curve.
X = linspace(min(X), max(X), 1920); % Let's use 1920 points, which will fit across an HDTV screen about one sample per pixel.

% Create smoothed/regressed data using the model:
yFitted = coefficients(1) + coefficients(2) * X + coefficients(3) * exp(-(X - coefficients(4)).^2 / coefficients(5)) + ...
	coefficients(6) * exp(-(X - coefficients(7)).^2 / coefficients(8));
% Now we're done and we can plot the smooth model as a red line going through the noisy blue markers.
hold on;
plot(X, yFitted, 'r-', 'LineWidth', 2);
grid on;
title('Exponential Regression with fitnlm()', 'FontSize', fontSize);
xlabel('X', 'FontSize', fontSize);
ylabel('Y', 'FontSize', fontSize);
legendHandle = legend('Noisy Y', 'Fitted Y', 'Location', 'northeast');
legendHandle.FontSize = 25;

% Set up figure properties:
% Enlarge figure to full screen.
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
% Get rid of tool bar and pulldown menus that are along top of figure.
% set(gcf, 'Toolbar', 'none', 'Menu', 'none');
% Give a name to the title bar.
set(gcf, 'Name', 'Demo by ImageAnalyst', 'NumberTitle', 'Off') 


