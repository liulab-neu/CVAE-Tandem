# Global setting for project
# import this file into your code: from user_setting import *

# CST python interfact location and cst model filename
# lib_path = "C:\Program Files (x86)\CST Studio Suite 2023\AMD64\python_cst_libraries"  # cst library location on your computer
lib_path = "/opt/cst/CST_Studio_Suite_2026/LinuxAMD64/python_cst_libraries"
cst_file = 'model2021_nk_model_multi_peaks.cst'     # CST model file, version need match CST software version
# cst_file = 'model2021.cst'

# Default CST parameter tables. ALL STRING FORMAT
parameter_values = [
    {"R": 4.5},
    {"w1": 800.0},
    {"w2": 380.0},
    {"wavelength": 532.0},
    {"Alpha_Z": "wavelength/16.0"},
    {"f2": "3e8/w2"},
    {"f1": "3e8/w1"},
    {"theta": 0},
    {"USize": "Alpha_Z"},
    {"f0": 3e8 / 439.0},
    {"T_host": "2*R+40.0"},
    {"n_host": 2.3},
    {"n_host1": 2.3},
    {"n_host2": 2.3}
]


# Item in CST to be exported, can use list to define multiple items.
# Legacy data files are still saved as .b, .c, etc.; plot-friendly .txt
# copies are saved in plot_txt_dir when save_txt_for_plot is True.
treeItems = [r'1D Results\T_ZminTE(0,0),ZmaxTE(0,0)', r'1D Results\R_ZmaxTE(0,0),ZmaxTE(0,0)']      # Tree item in CST to export.
# treeItems = ['1D Results\T_ZminTE(0,0),ZmaxTE(0,0)']
xaxis = 'lambda'            # plot x-axis
yaxis = 'T'                 # plot y-axis
save_txt_for_plot = True
plot_txt_dir = './data/txt'




