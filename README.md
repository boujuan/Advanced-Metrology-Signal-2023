# Advanced Metrology Project

This program was developed by Juan Manuel Boullosa Novo as part of a class project for Advanced Metrology at the University of Oldenburg. It takes in spectrograph noisy data provided by the professor, cleans and analyzes it, and plots the resulting spectra. In addition, the program uses a python script that connects to the NIST Atomic Spectra Database to download relevant atomic spectra and identify which elements correspond to some of the lines in the data.
The program is written in Matlab and includes several functions that perform different tasks, such as data cleaning, peak detection, and line fitting. The cleaning function uses a combination of smoothing and background subtraction techniques to remove noise and improve signal-to-noise ratio. The peak detection function identifies the peaks in the cleaned spectra and calculates their heights and positions. The line fitting function fits Gaussian curves to the peaks and calculates their parameters, such as peak area and full width at half maximum.
After analyzing the data, the program generates several plots that display the cleaned spectra, the identified peaks, and the fitted Gaussian curves. It also includes a summary table that lists the positions, heights, and parameters of the peaks.
To identify which elements correspond to some of the lines in the data, the program uses the python script to query the NIST Atomic Spectra Database and retrieve relevant atomic spectra. It then compares the positions of the lines in the data with the positions of the lines in the atomic spectra and identifies the closest match. The program outputs a table that lists the identified elements and their corresponding lines in the data.
To install and use this project, you need to have Matlab and Python installed on your computer. Follow these steps:

1. Clone this repository to your local machine or download latest release.

2. Open Matlab and navigate to the folder where you cloned this repository.

3. Please make sure at the beginning of the matlab file (lines 43 and 44), the path to your local python executable is correctly stablished.
   If using anaconda installed in the default location, leave it as follows:
   ```
    userprofile = getenv('USERPROFILE');
    python_location = [userprofile, '\anaconda3\'];
   ```
   However, if you have python install anywhere else in your system, modify these two lines appropiatly. For example, if you have python installed in C:/Python, set it as follows:
   ```
    python_location = 'C:\Python\';
   ```

4. Run Main_AM_SignalAnalysis_Boullosa.m file in Matlab. This will load the noisy data file Sig_para_Novo.mat and perform multiple operations.
   Furthermore, this will also run API_NIST_v3.py provided the python location is set up correctly. This will query NIST database and dowload 
   relevant Spectral data in the /DATA folder as .csv files. Then the main Matlab script will resume and perform the rest of the analysis.
   
5. This repository also includes a Report in PDF format in the folder /Report. Please check it for more information.
