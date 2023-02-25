import argparse
from bs4 import BeautifulSoup
import requests
import pandas as pd
import io
import re
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from pandas.core.common import SettingWithCopyWarning

def initParser():
    parser = argparse.ArgumentParser(description="NIST Database webscraping searches for closest lines.")
    parser.add_argument('element', help='Name of element from NIST Database',type=str)
    parser.add_argument('line', help='Value of wavelength to be searched', type=float)
    parser.add_argument('n', help='Number of wavelengths to be searched', type=int)
    parser.add_argument('filename', help='Name of the file with output', type=str)
    parser.add_argument('--low_w', help='Minimum wavelength value', type=int, default=200)
    parser.add_argument('--upper_w', help='Maximum wavelength value', type=int, default=900)
    parser.add_argument('--sp_num', help='List of ionization stages', nargs='+', type=int, default=[1,2])
    parser.add_argument('--threshold', help='Intensity threshold', type=float, default=0)
    
    return parser

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

class NIST_Line():

    def __init__(self, element, low_w, upper_w, sp_num, threshold=0):

        self.element = element
        self.low_w = low_w
        self.upper_w = upper_w
        self.data_frame = pd.DataFrame()

        self.retrieve_data()
        self.clean_intensity()
        self.line_threshold(threshold)

        self.filter_nan_values()
        self.filter_sp(sp_num)
        self.reset_index()

    def retrieve_data(self):
        site = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra={}" \
               "&limits_type=0&low_w={}" \
               "&upp_w={}" \
               "&unit=1&submit=Retrieve+Data&de=0&format=3&line_out=0&remove_js=on&en_unit=0&output=0&bibrefs=1&page_size=15&show_obs_wl=1&show_calc_wl=1&unc_out=1&order_out=0&max_low_enrg=&show_av=2&max_upp_enrg=&tsb_value=0&min_str=&A_out=1&intens_out=on&max_str=&allowed_out=1&forbid_out=1&min_accur=&min_intens=&conf_out=on&term_out=on&enrg_out=on&J_out=on"

        site = site.format(self.element, self.low_w, self.upper_w)
        respond = requests.get(site)
        soup = BeautifulSoup(respond.content, 'lxml')
        html_data = soup.get_text()
        html_data = html_data.replace('"', "")
        data = io.StringIO(html_data)
        self.data_frame = pd.read_csv(data, sep="\t").drop('Unnamed: 20', axis=1)

    def clean_intensity(self):
        self.data_frame['intens'] = self.data_frame['intens'].apply(lambda item: re.sub('[^0-9]', '', str(item)))
        self.data_frame = self.data_frame[self.data_frame['intens'] != '']
        self.data_frame['intens'] = pd.to_numeric(self.data_frame['intens'])

    def line_threshold(self, value=10 ** 2):
        # strength line gA > 10**8
        self.data_frame = self.data_frame[self.data_frame['intens'] > value]

    def filter_nan_values(self, column='obs_wl_air(nm)'):
        self.data_frame = self.data_frame[self.data_frame[column] > 0]

    def reset_index(self):
        self.data_frame.reset_index(inplace=True, drop=True)

    def filter_columns(self):
        if self.element != 'H':
            self.data_frame = self.data_frame[['element', 'sp_num', 'obs_wl_air(nm)', 'intens', 'gA(s^-1)']]
        else:
            self.data_frame = self.data_frame[['obs_wl_air(nm)', 'intens', 'gA(s^-1)']]

    def filter_sp(self, sp_num):
        self.data_frame = self.data_frame[self.data_frame['sp_num'].isin(sp_num)]

    def search_n_nearest_lines(self, line, number_of_lines):
        sorted_by_lines_df = self.data_frame.iloc[(self.data_frame['obs_wl_air(nm)'] - line).abs().argsort()]
        sorted_by_lines_df = sorted_by_lines_df[sorted_by_lines_df['intens'].notna()]
        sorted_by_intens_df = sorted_by_lines_df[:number_of_lines].sort_values(by=['intens'], ascending=False)
        sorted_by_intens_df.reset_index(inplace=True, drop=True)
        return sorted_by_intens_df

    def save_to_csv(self, filename):
        self.data_frame.to_csv(Path.cwd() / filename / '.csv')

if __name__ == '__main__':
    parser = initParser()
    args = parser.parse_args()
    if (args.low_w in range(199, 901)) & (args.upper_w in range(199, 901)):
        line = NIST_Line(element=args.element, low_w=args.low_w, upper_w=args.upper_w, sp_num=args.sp_num)
        lines_df = line.search_n_nearest_lines(args.line, args.n)
        lines_df.to_csv(args.filename+'.csv')
    else:
        print('Wavelength must be in range 200-900 nm, check low_w, upper_w parameters')