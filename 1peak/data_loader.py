# data_loader.py
import os
import pandas as pd

def load_a(filename):
    """
    Load data from a .a file.
    """
    values = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # skip empty lines
                continue
            var, expr = line.split(":")
            var = var.strip()
            expr = expr.strip()
            values[var] = eval(expr, None, values)
    return values

def load_b(filename):
    """
    Load data from a .b file.
    """
    return pd.read_csv(filename, sep='\t', header=None, skiprows=1, usecols=[0, 1])

def load_data(directory):
    """
    Load data from .a and .b files in the 'datasets' directory.
    Returns a tuple containing two lists: data_a_list and data_b_list.
    """
    files = os.listdir(directory)
    a_files = sorted([f for f in files if f.endswith('.a')])
    b_files = sorted([f for f in files if f.endswith('.b')])

    data_a_list = []
    data_b_list = []

    for a_file, b_file in zip(a_files, b_files):
        #data_a = load_a(os.path.join('C:/Users/liyux/Desktop/traindata/batch1', a_file))
        #data_b = load_b(os.path.join('C:/Users/liyux/Desktop/traindata/batch1', b_file))
        data_a = load_a(os.path.join(directory, a_file))
        #print(data_a['R'])
        data_b = load_b(os.path.join(directory, b_file))
        data_a_list.append(data_a)
        # print("\n")
        # print(f'list:{data_a_list}')
        data_b_list.append(data_b)

    return data_a_list, data_b_list

def display_first_three_groups():
    """
    Display the first three groups of files.
    """
    data_a_list, data_b_list = load_data('/home/yuxiao/Yuxiao Li/batch3')

    if len(data_a_list) < 3 or len(data_b_list) < 3:
        print("Not enough data groups to display.")
        return

    print("First three groups of files:")
    for i in range(3):
        print(f"Group {i + 1}")
        print("Data from .a file:")
        print(data_a_list[i])
        print("Data from .b file:")
        print(data_b_list[i])
        print("------------------------")

if __name__ == "__main__":
    # Execute code only if the file is run as the main module
    display_first_three_groups()
