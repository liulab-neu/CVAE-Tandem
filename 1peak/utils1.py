import os
import torch
import numpy as np
import pandas as pd
from data_loader import load_data
import matplotlib.pyplot as plt


import shutil
import os

import shutil
import os

def merge_folders(source_folders, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    i = 0
    for folder in source_folders:
        for root, _, files in os.walk(folder):

            for file in files:
                source_file_path = os.path.join(root, file)
                destination_file_path = os.path.join(destination_folder, file)

                # 检查目标文件夹中是否已存在同名文件
                while os.path.exists(destination_file_path):
                    file_name, file_extension = os.path.splitext(file)
                    file_name = file_name + "_theta_{}".format(i)
                    file = file_name + file_extension
                    destination_file_path = os.path.join(destination_folder, file)

                shutil.copy2(source_file_path, destination_file_path)
            i += 30
    print("文件夹合并完成！")





def load(filename):
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


def process_data(dataset):
    # print(dataset)
    # dataset = dataset.numpy()
    # num = dataset.shape[0]
    # index = np.arange(num)
    # print(index)
    # np.random.shuffle(index)
    # print(index)
    # train_idx = index[:int(20)]
    # print(train_idx)

    txt_folder = 'C:/Users/liyux/Desktop/traindata/theta'  # 指定包含txt文件的文件夹路径
    tensor_seq = torch.empty((0, 3, len(dataset[0])), dtype=torch.float32)
    for i in range(dataset.shape[0]):
        matching_files = []  # 存储匹配文件的列表
        for file_name in os.listdir(txt_folder):
            if file_name.endswith('.b'):  # 仅考虑以.b结尾的文件
                file_path = os.path.join(txt_folder, file_name)

                # 读取txt文件的数据（假设第二列数据以制表符分隔）
                with open(file_path, 'r') as file:
                    file_data = pd.read_csv(file, sep='\t', header=None, skiprows=1, usecols=[1])
                    file_data = file_data.values.flatten().astype(np.float32)
                    # print(f'dataset[i]:{dataset[i]},file_data:{file_data}')
                # 检查第一维数据是否与文件中的第二列数据匹配
                if np.array_equal(dataset[i], file_data):

                    matching_files.append(file_name)
                    base_filename = os.path.splitext(file_name)[0]
                    a_filename = base_filename + '.a'
                    matching_data = load(os.path.join(txt_folder, a_filename))
                    matching_R = matching_data.get('R')
                    matching_n_host = matching_data.get('n_host')
                    # print(f'matching_R:{matching_R}')

                    matching_a_files_with_parameters = []

                    files = os.listdir(txt_folder)
                    a_files = sorted([f for f in files if f.endswith('.a')])
                    for a_file_name in a_files:
                        data = load(os.path.join(txt_folder, a_file_name))
                        R = data.get('R')
                        n_host = data.get('n_host')
                        if R == matching_R and n_host == matching_n_host:
                            # 如果相同，那么这个.a文件符合条件
                            matching_a_files_with_parameters.append(a_file_name)

                    matching_b_files = []

                    # 遍历匹配的.a文件
                    for a_file_path in matching_a_files_with_parameters:
                        # 提取.a文件的基本部分（假设文件名为 file.a）
                        a_file_name = os.path.basename(a_file_path)
                        a_file_base = os.path.splitext(a_file_name)[0]  # 去除文件扩展名，得到基本部分（file）

                        # 找到与.a文件基本部分相同的.b文件
                        matching_b_files.extend([b_file for b_file in os.listdir(txt_folder)  if
                                                 os.path.splitext(os.path.basename(b_file))[0] == a_file_base and b_file.endswith('.b')])


                    # 创建一个空的二维 NumPy 数组，用于保存第二列数据
                    # data_array = np.empty((0, len(matching_b_files)), dtype=np.float32)
                    data_array = np.empty((0, len(dataset[i])), dtype=np.float32)

                    # 遍历匹配的.b文件
                    for b_file_name in matching_b_files:
                        b_file_path = os.path.join(txt_folder, b_file_name)
                        # 加载.b文件数据，假设数据保存在名为 'data' 的变量中
                        data = np.loadtxt(b_file_path)

                        # 提取.b文件数据的第二列
                        b_file_column = data[:, 1]

                        # 将第二列数据转换为行向量并添加到数据数组中
                        # print(b_file_column.shape,data_array.shape)
                        data_array = np.vstack((data_array, b_file_column.reshape(1, -1)))

                    tensor_array = torch.tensor(data_array, dtype=torch.float32).unsqueeze(0)
                    # print(tensor_seq.size(),tensor_array.size())
                    tensor_seq = torch.cat((tensor_seq, tensor_array), dim=0)

    print("tensor_seq:", tensor_seq)

    return tensor_seq

if __name__ == '__main__':
    # # 指定要合并的三个文件夹的路径列表
    # source_folders = [
    #     "C:/Users/liyux/Desktop/traindata/batch3/batch3",
    #     "C:/Users/liyux/Desktop/traindata/batch4/batch4",
    #     "C:/Users/liyux/Desktop/traindata/batch5/batch5"
    # ]
    #
    # # 指定目标文件夹的路径
    # destination_folder = "C:/Users/liyux/Desktop/traindata/theta"
    #
    # # 合并文件夹
    # merge_folders(source_folders, destination_folder)

    # data_a_list, data_b_list = load_data('C:/Users/liyux/Desktop/traindata/batch3/batch3')
    # data_a_array = []
    # for data in data_a_list:
    #     # Assuming the dictionary keys are 'R', 'theta', and 'n_host'
    #     values = [data['R'], data['n_host']]
    #     data_a_array.append(values)
    #
    # # reshape data_b_list elements and remove first column
    # reshaped_data_b_list = [np.array(data_b)[:, 1].reshape(-1) for data_b in data_b_list]
    #
    # # create input and output data
    # X = torch.tensor(data_a_array, dtype=torch.float32)  # FNN
    # # X = torch.tensor(reshaped_data_b_list, dtype = torch.float32) #tandem
    # y = torch.tensor(reshaped_data_b_list, dtype=torch.float32)
    # # # print(y[0])
    # # #print(y)
    # y_related = process_data(y)
    # file_path = './data/y_related.pt'
    # os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # torch.save(y_related,file_path)
    data_a_list, data_b_list = load_data('C:/Users/liyux/Desktop/traindata/batch3/batch3')
    reshaped_data_b_list = [np.array(data_b)[:, 1].reshape(-1) for data_b in data_b_list]
    y = torch.tensor(reshaped_data_b_list, dtype=torch.float32)
    y_related = torch.load('./data/y_related.pt')
    print(y_related)
    y_mean = torch.mean(y_related, dim=1)
    print(y_mean)
    plt.figure()
    plt.plot(y[750],label = 'y')
    # plt.plot(wl,target,label='target')
    plt.plot(y_mean[750],label = 'y_mean')
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.legend()
    plt.show()
