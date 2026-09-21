import sys, os
import numpy as np, matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from user_setting import *


# Set to False by bulk generators: one PNG per export is ~24k matplotlib
# renders over a full dataset sweep, for files nothing downstream reads.
make_preview_plots = True


def delete_files_in_result_folder(cst_file):
    cst_file_main = os.path.splitext(cst_file)[0]

    # Build the full path to the Result folder
    result_folder_path = os.path.join(cst_file_main, 'Result')

    # Check if the Result folder exists
    if os.path.exists(result_folder_path):
        # Iterate over the directory tree rooted at result_folder_path
        for root, dirs, files in os.walk(result_folder_path, topdown=False):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                os.remove(file_path)
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                os.rmdir(dir_path)

        print("All files and subdirectories in the Result folder have been deleted.")
    else:
        print("The Result folder does not exist.")
    



def save_parameters(fid, parameter_values, data_dir='./data', plot_dir=None):
    creatresultFolder(data_dir)
    with open(os.path.join(data_dir, f'{fid}.a'), "w") as file:
        for parameter in parameter_values:
            for key, value in parameter.items():
                file.write(f"{key}: {value}\n")
    if save_txt_for_plot:
        effective_plot_dir = plot_dir or plot_txt_dir
        creatresultFolder(effective_plot_dir)
        with open(os.path.join(effective_plot_dir, f'{fid}_input_parameters.txt'), "w") as file:
            for parameter in parameter_values:
                for key, value in parameter.items():
                    file.write(f"{key}\t{value}\n")


def safe_filename(text):
    safe_chars = []
    for char in text:
        if char.isalnum() or char in ('-', '_'):
            safe_chars.append(char)
        else:
            safe_chars.append('_')
    return ''.join(safe_chars).strip('_')


def update_value_in_list_of_dicts(data_list, item_name, new_value):
    for data_dict in data_list:
        if item_name in data_dict:
            data_dict[item_name] = new_value
            # print(f'updated {item_name}={new_value}')
            break



def generate_command(parameter_values):
    num = len(parameter_values)
    # Extract keys and values into separate lists
    names = []
    values = []

    for param in parameter_values:
        for p, v in param.items():
            names.append(str(p))
            values.append(str(v))
    quoted_names = ['"' + item + '"' for item in names]
    quoted_values = ['"' + item + '"' for item in values]

    # VBA code line by line
    line1 = 'Sub Main'
    line2 = f'Dim names(1 To {num}) As String, values (1 To {num}) As String'
    lines = [line1, line2]

    for i in range(num):
        name = quoted_names[i]
        value = quoted_values[i]
        line = f'names({i + 1}) = {name}'
        lines.append(line)
        line = f'values({i + 1}) = {value}'
        lines.append(line)

    last2 = 'StoreParameters(names, values)'
    lines.append(last2)
    last = 'End Sub'
    lines.append(last)

    joined_lines = '\n'.join(lines)

    return joined_lines


def cst_parameter_sweep_and_export(
    current_project,
    fid,
    parameter_values,
    data_dir='./data',
    plot_dir=None,
    project_file=None,
):
    vba_command = generate_command(parameter_values)
    # print(vba_command)
    current_project.schematic.execute_vba_code(vba_command)
    # Run the solver
    current_project.modeler.run_solver()

    creatresultFolder(data_dir)
    effective_plot_dir = plot_dir or plot_txt_dir
    effective_project_file = project_file or cst_file
    for index, treeItem in enumerate(treeItems):
        letter = chr(ord('b') + index)  # Convert index to corresponding letter (b, c, d, ...)
        txt_file = None
        if save_txt_for_plot:
            creatresultFolder(effective_plot_dir)
            item_name = treeItem[treeItem.rfind('\\') + 1:]
            txt_file = os.path.join(effective_plot_dir, f'{fid}_{letter}_{safe_filename(item_name)}.txt')
        exported = export1D(
            cst_file=effective_project_file,
            treeItem=treeItem,
            data_file=os.path.join(data_dir, f'{fid}.{letter}'),
            txt_file=txt_file,
        )
        if not exported:
            if index == 0:
                raise RuntimeError(
                    f'Required CST transmission result was not found: '
                    f'{treeItem}'
                )
            print(
                f'Optional CST result was not found and was skipped: '
                f'{treeItem}'
            )


def export1D(cst_file, treeItem, data_file, txt_file=None):
    ################################################################
    # Modify settings according to your case
    # lib_path = "C:\Program Files (x86)\CST Studio Suite 2023\AMD64\python_cst_libraries"  # cst library location on your computer
    # cst_file = os.path.abspath(r'./model.cst')              # CST model file
    cst_file = os.path.abspath(cst_file)
    data_file = os.path.abspath(data_file)
    txt_file = os.path.abspath(txt_file) if txt_file is not None else None
    # data_file = os.path.abspath(r'./data_export_example.txt')       # filename save to
    # treeItem = '1D Results\S-Parameters'                               # which cst result to export
    nSamples = 1001
    Real_part_plot = False
    Imag_part_plot = False
    # Rendering one PNG per export costs both time and disk on a large sweep,
    # so callers doing bulk generation can turn it off via this module flag.
    Magnitude_plot = make_preview_plots
    ###############################################################
    # Don't change below code if you are not sure
    # add path
    sys.path.append(lib_path)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # readout data
    n = 0  # no. of parameters to readout
    spara = None
    headtxt = f'{xaxis}  '
    import cst.results as re
    result_file = re.ProjectFile(cst_file, allow_interactive=True)
    result = result_file.get_3d()

    l = result.get_tree_items('0D/1D')

    for i, content in enumerate(l):
        if treeItem in content:
            n += 1
            item_name = content[content.rfind('\\') + 1:]  # readout the name of data
            headtxt += f'Re({item_name})    Im({item_name}), '

            spara_freq = result.get_result_item(content).get_xdata()  # frequency
            freq = np.linspace(min(spara_freq), max(spara_freq), nSamples, endpoint=True)

            spara_re = np.array(result.get_result_item(content).get_ydata()).real
            spara_im = np.array(result.get_result_item(content).get_ydata()).imag

            f_re = interp1d(spara_freq, spara_re.tolist())
            f_im = interp1d(spara_freq, spara_im.tolist())

            spara_temp = np.hstack((f_re(freq).reshape(-1, 1), f_im(freq).reshape(-1, 1)))
            # plot Re or Im part of S-parameter
            if Real_part_plot:
                plt.plot(spara_freq, spara_re, 'bo', freq, f_re(freq), 'r-')
                plt.legend(['raw_data', 'resampled_data'], loc='best')
                plt.title(f'Real part of {item_name}')
                # plt.show()
                plt.close()
            if Imag_part_plot:
                plt.plot(spara_freq, spara_re, 'bo', freq, f_re(freq), 'r-')
                plt.legend(['raw_data', 'resampled_data'], loc='best')
                plt.title(f'Real part of {item_name}')
                # plt.show()
                plt.close()
            # Plot magnitude of S-parameter
            if Magnitude_plot:
                Mag = (spara_temp[:, 0] ** 2. + spara_temp[:, 1] ** 2.) ** 0.5
                plt.plot(freq, Mag, 'r-')
                plt.legend([f'{os.path.basename(data_file)}'], loc='best')
                plt.title(f'{item_name}')
                plt.savefig(f'{data_file}.png')
                plt.close()

            if n == 1:
                spara_freq = freq.reshape(-1, 1)  # add freq. column to data
                spara = np.hstack((spara_freq, spara_temp))
            else:
                spara = np.append(spara, spara_temp, axis=1)

    if spara is None:
        # Do not leave an older result in place when the current CST project
        # does not create this optional result tree (the single-peak project,
        # for example, creates T but not the legacy R table).
        for stale_path in (
            data_file,
            f'{data_file}.png',
            txt_file,
        ):
            if stale_path is not None and os.path.isfile(stale_path):
                os.remove(stale_path)
        return False

    # save legacy data file and plot-friendly txt copy
    np.savetxt(data_file, spara, delimiter='\t', header=headtxt)  # header can be removed if needed.
    if txt_file is not None:
        np.savetxt(txt_file, spara, delimiter='\t', header=headtxt)
    return True


def creatresultFolder(dirname):
    # Check whether the specified path exists or not
    isExist = os.path.exists(dirname)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(dirname)
        print("The new directory is created!")


def read_save_parameters():
    from cst import CST
    # Connect to CST Studio Suite
    cst = CST()

    # Open an existing project
    project_file = "./model.cst"
    project = cst.open_project(project_file)

    # Get the current design
    design = project.get_design()

    # Get the parameters of the current design
    parameters = design.get_current_parameters()

    # Iterate over the parameters and print their names and values
    for parameter in parameters:
        name = parameter.get_name()
        value = parameter.get_value()
        print(f"Parameter: {name} = {value}")

    # Disconnect from CST Studio Suite
    cst.disconnect()


if __name__ == "__main__":
    for index, treeItem in enumerate(treeItems):
        letter = chr(ord('b') + index )  # Convert index to corresponding letter (b, c, d, ...)
        fid=1
        print(f'./data/{fid}.{letter}')

