import argparse
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup as bs
import io

# Parses arguments from command line
def initParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('element', type=str) # Element name (e.g. 'Ca')
    parser.add_argument('line', type=float) # Line wavelength (e.g. 393.4)
    parser.add_argument('n', type=int) # Number of nearest lines to search for (e.g. 10)
    parser.add_argument('filename', type=str) # Name of the file to save data to (e.g. 'data')
    parser.add_argument('--low_w', type=int, default=200) # Lower wavelength limit (default: 200)
    parser.add_argument('--upper_w', type=int, default=900) # Upper wavelength limit (default: 900)
    parser.add_argument('--sp_num', nargs='+', type=int, default=[1,2]) # List of sp_num values to filter out (default: 1,2)
    parser.add_argument('--threshold', type=float, default=0) # Intensity minimum threshold (default: 0)

    return parser

# Retrieves data from NIST website as a pandas dataframe
def retrieve_data(element, low_w, upper_w):
    site =  "https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra={}" \
            "&limits_type=0&low_w={}" \
            "&upp_w={}" \
            "&unit=1&submit=Retrieve+Data&de=0&format=3&line_out=0&remove_js=on&en_unit=0&output=0&bibrefs=1&page_size=15&show_obs_wl=1&show_calc_wl=1&unc_out=1&order_out=0&max_low_enrg=&show_av=2&max_upp_enrg=&tsb_value=0&min_str=&A_out=1&intens_out=on&max_str=&allowed_out=1&forbid_out=1&min_accur=&min_intens=&conf_out=on&term_out=on&enrg_out=on&J_out=on"

    site = site.format(element, low_w, upper_w)
    respond = requests.get(site)
    soup = bs(respond.content, 'lxml')
    html_data = soup.get_text()
    html_data = html_data.replace('"', "")
    data = io.StringIO(html_data)
    data_frame = pd.read_csv(data, sep="\t").drop('Unnamed: 20', axis=1)
    return data_frame

# Cleans intensity column from non-numeric values and converts it to float type
def clean_intensity(data_frame):
    data_frame['intens'] = data_frame['intens'].apply(lambda item: re.sub('[^0-9]', '', str(item)))
    data_frame = data_frame[data_frame['intens'] != '']
    data_frame['intens'] = pd.to_numeric(data_frame['intens'])
    return data_frame

# Filters out lines with intensity lower than threshold
def line_threshold(data_frame, value=10 ** 2):
    data_frame = data_frame[data_frame['intens'] > value]
    return data_frame

# Filters out lines with wavelength equal to 0 (NaN)
def filter_nan_values(data_frame, column='obs_wl_air(nm)'):
    data_frame = data_frame[data_frame[column] > 0]
    return data_frame

# Filters out lines with sp_num not in sp_num list (default: 1,2)
def filter_sp(data_frame, sp_num):
    data_frame = data_frame[data_frame['sp_num'].isin(sp_num)]
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
    parser = initParser()
    args = parser.parse_args()
    data_frame = retrieve_data(args.element, args.low_w, args.upper_w)
    data_frame = clean_intensity(data_frame)
    data_frame = line_threshold(data_frame, args.threshold)
    data_frame = filter_nan_values(data_frame)
    data_frame = filter_sp(data_frame, args.sp_num)
    lines_df = search_n_nearest_lines(data_frame, args.line, args.n)
    lines_df.to_csv(args.filename+'.csv')
