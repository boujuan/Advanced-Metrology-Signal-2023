% Define x and y data
x = linspace(-10, 10, 1000);
y = sin(x) + 0.1*randn(size(x));

% Define parameters for Gaussian peak detection
threshold = 0.1; % Peak detection threshold
widths = [0.5 1 2]; % Gaussian widths to test

% Detect positive peaks
[pk_pos, loc_pos, width_pos] = deal([]);
for i = 1:length(widths)
    [p, l, w] = findpeaks(y, x, 'MinPeakHeight', threshold, 'WidthReference', 'halfheight', 'MinPeakWidth', widths(i));
    pk_pos = [pk_pos; p];
    loc_pos = [loc_pos; l];
    width_pos = [width_pos; w];
end

% Detect negative peaks
[pk_neg, loc_neg, width_neg] = deal([]);
for i = 1:length(widths)
    [p, l, w] = findpeaks(-y, x, 'MinPeakHeight', threshold, 'WidthReference', 'halfheight', 'MinPeakWidth', widths(i));
    pk_neg = [pk_neg; -p];
    loc_neg = [loc_neg; l];
    width_neg = [width_neg; w];
end

% Plot the original curve and detected peaks
figure;
plot(x, y, 'b');
hold on;
plot(loc_pos, pk_pos, 'ro');
plot(loc_neg, pk_neg, 'go');
legend('Original curve', 'Positive peaks', 'Negative peaks');

% Subtract peaks from the original curve
y_no_peaks = y;
for i = 1:length(loc_pos)
    for j = 1:length(widths)
        y_no_peaks = y_no_peaks - pk_pos((i-1)*length(widths)+j)*exp(-(x-loc_pos(i)).^2/(2*width_pos((i-1)*length(widths)+j)^2));
    end
end
for i = 1:length(loc_neg)
    for j = 1:length(widths)
        y_no_peaks = y_no_peaks - pk_neg((i-1)*length(widths)+j)*exp(-(x-loc_neg(i)).^2/(2*width_neg((i-1)*length(widths)+j)^2));
    end
end

% Plot the resulting curve
figure;
plot(x, y_no_peaks, 'b');
legend('Curve with peaks subtracted');

% Perform polynomial fit on the resulting curve
order = 6; % Polynomial order
coeffs = polyfit(x, y_no_peaks, order);
y_fit = polyval(coeffs, x);

% Plot the polynomial fit on the resulting curve
figure;
plot(x, y_no_peaks, 'b');
hold on;
plot(x, y_fit, 'r');
legend('Curve with peaks subtracted', 'Polynomial fit');
