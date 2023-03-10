import argparse
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup as bs
import io
import urllib.parse
import sys

# ====================== IMPORTANT NOTE: ========================================== #
# You might need to install pip_system_certs (apart from all the other dependencies) 
# (pip install pip_system_certs) to fix SSL certificate error on Windows 10/11. 
# Not sure about other OS.

########################################### BEGIN CODE ###################################################################
#
# Example usage:
#
# python.exe .\API_NIST_v3.py 426.5 data --element Fe --n 10 --low_w 425 --high_w 584 --ion_num 1 2 --min_intensity 10
#
# This will retrieve data for Iron (Fe element), filter out lines with wavelength outside the range of 425-584 nm,
# filter out lines with ionization stages other than 1 or 2, filter out lines with intensity below 10,
# find the 10 nearest lines to a wavelength of 426.5 nm, and save the results to a file named 'data.csv'.
#
# THIS SCRIPT WAS CREATED BY: Juan Manuel Boullosa Novo, Studentnummer: 2481927
#
# DATE: 2023-02-26
#
# PURPOSE: To retrieve data from NIST website and save it to a csv file to be used in conjunction with the main
# Matlab Script "Main_AM_SignalAnalysis_Boullosa.m" for the project of Advanced Metrology in the University of Oldenburg
# in the winter semester of 2022/2023. For analysing Spectrographic Data of Atomic Emission Spectroscopy.
# This also goes together with a Report with further explanations and results.
#
# REFERENCES:
#               https://physics.nist.gov/cgi-bin/ASD/lines1.pl
#               https://github.com/SirJohnFranklin/nist-asd
#               https://github.com/MKastek/NIST-database-webscraping
#               https://github.com/astro-data/nist_atomic_lines_package
#
##########################################################################################################################

# Parses arguments from command line
def initParser():
    parser = argparse.ArgumentParser()    
    parser.add_argument('line', type=float) # Line wavelength (e.g. 393.4)    
    parser.add_argument('filename', type=str) # Name of the file to save data to (e.g. 'data')
    parser.add_argument('--element', type=str, default="") # Element name (e.g. 'Fe', search any by default)
    parser.add_argument('--n', type=int, default=5) # Number of nearest lines to search for (default: 5)
    parser.add_argument('--low_w', type=float, default=200) # Lower wavelength limit (default: 200)
    parser.add_argument('--high_w', type=float, default=900) # Upper wavelength limit (default: 900)
    parser.add_argument('--ion_num', nargs='+', type=int, default=[1,2]) # List of ionization stages values to filter out (default: 1,2)
    parser.add_argument('--min_intensity', type=int, default=0) # Intensity minimum min_intensity (default: 0)

    return parser

# Retrieves data from NIST website as a pandas dataframe
def retrieve_data(element, low_w, high_w, min_intensity):
    url_params = {
        'spectra': element,
        'limits_type': '0',
        'low_w': str(low_w),
        'upp_w': str(high_w),
        'unit': '1',
        'submit': 'Retrieve Data',
        'de': '0',
        'format': '3',
        'line_out': '0',
        'remove_js': 'on',
        'en_unit': '0',
        'output': '0',
        'bibrefs': '1',
        'page_size': '15',
        'show_obs_wl': '1',
        'show_calc_wl': '1',
        'unc_out': '1',
        'order_out': '0',
        'max_low_enrg': '',
        'show_av': '2',
        'max_upp_enrg': '',
        'tsb_value': '0',
        'min_str': '',
        'A_out': '1',
        'intens_out': 'on',
        'max_str': '',
        'allowed_out': '1',
        'forbid_out': '1',
        'min_accur': '',
        'min_intens': str(min_intensity),
        'conf_out': 'on',
        'term_out': 'on',
        'enrg_out': 'on',
        'J_out': 'on'
    }

    # NIST website url with parameters from url_params
    site = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl?" + urllib.parse.urlencode(url_params)
    # print(site)
    # uncomment upper line to get URL of the website with parameters for access in browser

    site = site.format(element, low_w, high_w, min_intensity)
    respond = requests.get(site)
    soup = bs(respond.content, 'lxml')
    html_data = soup.get_text()
    html_data = html_data.replace('"', "")
    data = io.StringIO(html_data)
    drop_columns=['gA(s^-1)','Ei(cm-1)','conf_i','term_i','J_i','conf_k','term_k','J_k','Type','tp_ref','line_ref','Ek(cm-1)','Acc','Unnamed: 20']
    data_frame = pd.read_csv(data, sep="\t").drop(drop_columns, axis=1)
    return data_frame

# Cleans intensity column from non-numeric values and converts it to float type
def clean_intensity(data_frame):
    data_frame['intens'] = data_frame['intens'].apply(lambda item: re.sub('[^0-9]', '', str(item)))
    data_frame = data_frame[data_frame['intens'] != '']
    data_frame['intens'] = pd.to_numeric(data_frame['intens'])
    return data_frame

# Filters out lines with intensity lower than min_intensity
def line_threshold(data_frame, value):
    data_frame = data_frame[data_frame['intens'] > value]
    return data_frame

# Filters out lines with wavelength equal to 0 (NaN)
def filter_nan_values(data_frame, column='obs_wl_air(nm)'):
    data_frame = data_frame[data_frame[column] > 0]
    return data_frame

# Filters out lines with ion_num not in ion_num list (default: 1,2)
def filter_ion(data_frame, ion_num):
    data_frame = data_frame[data_frame['sp_num'].isin(ion_num)]
    return data_frame

# Searches for n nearest lines to given line and returns them as a dataframe sorted by intensity in descending order
def search_n_nearest_lines(data_frame, line, number_of_lines):
    sorted_by_lines_df = data_frame.iloc[(data_frame['obs_wl_air(nm)'] - line).abs().argsort()]
    sorted_by_lines_df = sorted_by_lines_df[sorted_by_lines_df['intens'].notna()]
    sorted_by_intens_df = sorted_by_lines_df[:number_of_lines].sort_values(by=['intens'], ascending=False)
    sorted_by_intens_df.reset_index(inplace=True, drop=True)
    return sorted_by_intens_df

# Main function that parses arguments, retrieves data, cleans it and saves it to a csv file
if __name__ == '__main__':
    # @@@@@ Up to Here
    
    parser = initParser()
    args = parser.parse_args()    
    data_frame = retrieve_data(args.element, args.low_w, args.high_w, args.min_intensity)
    data_frame = clean_intensity(data_frame)
    data_frame = line_threshold(data_frame, args.min_intensity)
    data_frame = filter_nan_values(data_frame)
    data_frame = filter_ion(data_frame, args.ion_num)
    lines_df = search_n_nearest_lines(data_frame, args.line, args.n)
    lines_df.to_csv('Data/'+args.filename+'.csv') # Saves data to a csv file in Data folder
    
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # Uncomment upper lines and comment lower lines to use arguments from command line
    # (Command Line mode currently the ONLY working one)
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # @ Takes arguments from matlab "pyrunfile" function:
    # low_w2 = float(low_w)
    # high_w2 = float(high_w)
    # min_intensity2 = float(min_intensity)
    # ion_num = [ion_num]
    # ion_num2 = [int(i) for i in ion_num]
    # line2 = float(line)
    # n2 = float(n)
    
    # data_frame = retrieve_data(element, low_w2, high_w2, min_intensity2)
    # data_frame = clean_intensity(data_frame)
    # data_frame = line_threshold(data_frame, min_intensity2)
    # data_frame = filter_nan_values(data_frame)
    # data_frame = filter_ion(data_frame, ion_num2)
    # lines_df = search_n_nearest_lines(data_frame, line2, n2)
    # lines_df.to_csv('Data/'+filename+'.csv') # Saves data to a csv file in Data folder
    # @@@@@ Down to here
