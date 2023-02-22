% Example data
x = 0:0.1:10;
y = sin(x) + randn(size(x));
continuum = smoothdata(y, 'movmean', 10); % Smooth the data to remove noise

% Find peaks
[~,locs,~,proms] = findpeaks(continuum);

% Set peak width and create a Gaussian filter
peakWidth = 5;
gaussFilter = fspecial('gaussian', [1 peakWidth*2+1], peakWidth);

% Create a mask for each peak and apply the Gaussian filter
mask = false(size(x));
for ii = 1:length(locs)
    mask = mask | x > locs(ii)-peakWidth & x < locs(ii)+peakWidth;
end
gaussMask = gaussFilter / max(gaussFilter);
gaussMask = gaussMask(:);
gaussMask = gaussMask / sum(gaussMask);
mask = conv(mask, gaussMask, 'same');

% Subtract peaks from continuum curve
yWithoutPeaks = continuum - mask.*proms;

% Perform polynomial fit
p = polyfit(x, yWithoutPeaks, 3);
fitCurve = polyval(p, x);

% Plot the results
figure;
plot(x, y, 'b-', x, continuum, 'r-', x, yWithoutPeaks, 'g-', x, fitCurve, 'k--');
legend('Data', 'Continuum', 'Without Peaks', 'Polynomial Fit');
