import numpy as np
import tandem
import os
import torch
from data_loader import load_data
from sklearn.model_selection import train_test_split
import utils1
import matplotlib.pyplot as plt

# Select training mode
# mode = 'FNN'  # Train forward neural network
#mode = 'INN'
mode = 'tandem' # Train inverse part of the tandem network

# network parameters
n_input = 2  # 12
n_classes = 1001  # 3
# fnn_size = [n_input, 32,64, 128, 256, 512, n_classes]
# fnn_size = [n_input,64, 128, 256, 512, 1024, n_classes]
fnn_size = [n_input, 64, 128, 256, 512, 1024, 2048,1024, n_classes]
inn_size = [n_classes,1024,2048,1024,512,256,128,64, n_input]

# load the data
data_a_list, data_b_list = load_data('/home/yuxiao/Yuxiao Li/batch3')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# convert data_a_list to a numerical format
data_a_array = []
for data in data_a_list:
    # Assuming the dictionary keys are 'R', 'theta', and 'n_host'
    values = [data['R'], data['n_host']]
    data_a_array.append(values)

# reshape data_b_list elements and remove first column
reshaped_data_b_list = [np.array(data_b)[:, 1].reshape(-1) for data_b in data_b_list]

# create input and output data
X = torch.tensor(data_a_array, dtype=torch.float32) #FNN
#X = torch.tensor(reshaped_data_b_list, dtype = torch.float32) #tandem
y = torch.tensor(reshaped_data_b_list, dtype=torch.float32)

# Preprocessing
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# y_train_relate = utils1.process_data(y_train)
# y_test_relate = utils1.process_data(y_test)
# y_train_relate = torch.zeros(3, 3)
# y_test_relate = torch.zeros(3,3)

num = len(y)
idx = np.arange(num)
np.random.shuffle(idx)
r = 0.8
train_idx = idx[:int(num*r)]
val_idx = idx[int(num*r):int(num*r)+int(num*0.1)]
test_idx = idx[int(num*r)+int(num*0.1):]
X_train = X[train_idx]
X_val = X[val_idx]
X_test = X[test_idx]
y_train = y[train_idx]
y_val = y[val_idx]
y_test = y[test_idx]
y_related = torch.load('./data/y_related.pt')
y_mean = torch.mean(y_related, dim=1)
y_train_relate = y_mean[train_idx]
y_val_relate = y_mean[val_idx]
y_test_relate = y_mean[test_idx]
# y_train_relate = y[train_idx]
# y_val_relate = y[val_idx]
# y_test_relate = y[test_idx]



# Set training parameters
# For forward network.
if mode == 'FNN':
    batch_size = 64
    training_epochs = 2000
    start_lr = 0.001  # learning rate
    decay_rate = 0.97  # learning rate decay rate
    #decay_step = 400 * 300  # learning rate decay steps
    decay_step = 40 * 30
    structure_min = torch.tensor([2.5,1.5])
    structure_max = torch.tensor([6.5,3.05])
# For inverse network
elif mode == 'INN':
    batch_size = 2000
    training_epochs = 3000
    start_lr = 0.0015  # learning rate
    decay_rate = 0.95  # learning rate decay rate
    #decay_step = 400 * 300  # learning rate decay steps
    decay_step = 40 * 3
    structure_min = torch.tensor([2.5,1.5])
    structure_max = torch.tensor([6.5,3.05])
# For tandem network
elif mode == 'tandem':
    batch_size = 128
    training_epochs = 800
    start_lr = 0.001
    decay_rate = 0.97
    #decay_step = 400 * 100
    decay_step = 50 * 30
    structure_min = torch.tensor([2.5,1.5])
    structure_max = torch.tensor([6.5,3.05])


def lowest_wavelength(x):
    # Delegates to the network so the reported metric matches the trained
    # objective.  The previous version reloaded the whole dataset from disk on
    # every call, which dominated the epoch time.
    return tandem_net.lowest_wavelength(x)

# Build network
tandem_net = tandem.tandem_network(INN_size=inn_size, FNN_size=fnn_size, starter_learning_rate=start_lr,
                                   decay_step=decay_step, decay_rate=decay_rate,
                                   structure_max=structure_max,structure_min=structure_min).to(device)

# Create a directory to save models
os.makedirs("./1peak/model", exist_ok=True)
# Restore model
if mode == 'tandem':
    tandem_net.restore_FNN('./1peak/model/DNN_tandem_FNN_label.ckpt')

#tandem_net.reset_global_step()
import time

t = time.time()
output = open("./1peak/DNN_" + mode + "-t.txt", "w+")

#Training
# for epoch in range(training_epochs):
#     num = X_train.numpy().shape[0]
#     idx = np.arange(num)
#     np.random.shuffle(idx)
#     X_train = X_train[idx]
#     y_train = y_train[idx]
#     y_train_relate = y_train_relate[idx]
#     for i in range(len(X_train) // batch_size):
#         batch_x = X_train[i * batch_size: i * batch_size + batch_size]
#         batch_y = y_train[i * batch_size: i * batch_size + batch_size]
#         batch_y_relate = y_train_relate[i * batch_size: i * batch_size + batch_size]
#         tandem_net.train(batch_x, batch_y_relate,batch_y_relate, mode)
#         if mode == 'FNN':
#             loss = tandem_net.show_loss(batch_x, batch_y_relate,batch_y_relate, mode)
#             lr = tandem_net.show_lr()
#             print(f"Epoch: {epoch}, Loss of banch: {loss}, lr: {lr}")
#             output.write(str(loss) + ' ' + str(lr) + "\n")
#         else:
#             loss1, loss2,mean_loss,var_loss = tandem_net.show_loss(batch_x, batch_y_relate, batch_y_relate, mode)
#             lr = tandem_net.show_lr()
#             print(f"Epoch: {epoch}, Loss1 of banch: {loss1}, Loss2 of banch:{loss2}, mean_loss of banch:{mean_loss}, var_loss of banch:{var_loss}, lr: {lr}")
#             output.write(str(loss1) + ' ' + str(loss2) + ' ' + str(mean_loss) + ' ' +str(var_loss) + ' '+ str(lr) + "\n")
#
#     tandem_net.train(X_train, y_train_relate, y_train_relate,mode)
#     if mode == 'FNN':
#         loss = tandem_net.show_loss(X_train, y_train_relate,y_train_relate, mode)
#         lr = tandem_net.show_lr()
#         print(f"Epoch: {epoch}, Loss of total: {loss}, lr: {lr}")
#
#     else:
#         loss1,loss2,mean_loss,var_loss = tandem_net.show_loss(X_train, y_train_relate, y_train_relate,mode)
#         lr = tandem_net.show_lr()
#         print(f"Epoch: {epoch}, Loss1 of total: {loss1}, Loss2 of total:{loss2}, mean_loss of total:{mean_loss}, var_loss of total:{var_loss}, lr: {lr}")
#         output.write(str(loss1)+' '+str(loss2)+' '+str(lr)+"\n")

for epoch in range(training_epochs):
    loss_total = 0
    label_loss_total = 0
    lowest_wave_loss_total = 0
    error_total = 0
    num = X_train.size(0)
    idx = np.arange(num)
    np.random.shuffle(idx)
    # Shuffled CPU tensors; move per-batch to device
    X_train_shuf = X_train[idx]
    y_train_shuf = y_train[idx]
    y_train_relate_shuf = y_train_relate[idx]
    for i in range(len(X_train_shuf) // batch_size):
        batch_x = X_train_shuf[i * batch_size: i * batch_size + batch_size].to(device)
        batch_y = y_train_shuf[i * batch_size: i * batch_size + batch_size].to(device)
        batch_y_relate = y_train_relate_shuf[i * batch_size: i * batch_size + batch_size].to(device)
        tandem_net.train(batch_x, batch_y_relate, mode)
        if mode == 'FNN':
            loss,label_loss,lowest_wave_loss,error = tandem_net.show_loss(batch_x, batch_y_relate, mode)
            lr = tandem_net.show_lr()
            loss_total += loss
            label_loss_total += label_loss
            lowest_wave_loss_total += lowest_wave_loss
            error_total += error
            # print(f"Epoch: {epoch}, Loss of banch: {loss}, error: {error*100}%,label_loss:{label_loss}, lowest_wave:{lowest_wave_loss},lr: {lr}")
            # output.write(str(loss) + ' ' + str(lr) + "\n")
        else:
            loss1,label_loss,lowest_wave_loss,error = tandem_net.show_loss(batch_x, batch_y_relate,  mode)
            lr = tandem_net.show_lr()
            loss_total += loss1
            label_loss_total += label_loss
            lowest_wave_loss_total += lowest_wave_loss
            error_total += error
            # print(f"Epoch: {epoch}, Loss1 of banch: {loss1},error:{error*100}%, label_loss:{label_loss}, lowest_wave:{lowest_wave_loss}, lr: {lr}")
            #output.write(str(loss1) + ' ' + str(loss2) + ' ' + str(lr) + "\n")

    #tandem_net.train(X_train, y_train, mode)
    if mode == 'FNN':
        # loss,label_loss,lowest_wave_loss,error = tandem_net.show_loss(X_train.to(device), y_train_relate.to(device), mode)
        lr = tandem_net.show_lr()
        loss = loss_total/(len(X_train) // batch_size)
        label_loss = label_loss_total/(len(X_train) // batch_size)
        lowest_wave_loss = lowest_wave_loss_total/(len(X_train) // batch_size)
        error = error_total/(len(X_train) // batch_size)
        print(f"Epoch: {epoch}, Loss of total: {loss},error:{error*100}%, label_loss:{label_loss}, lowest_wave:{lowest_wave_loss},lr: {lr}")
        # output.write(str(loss) + ' ' + str(lr) + "\n")
    else:
        # loss1,label_loss,lowest_wave_loss,error = tandem_net.show_loss(X_train.to(device), y_train_relate.to(device),mode)
        lr = tandem_net.show_lr()
        loss1 = loss_total/(len(X_train) // batch_size)
        label_loss = label_loss_total/(len(X_train) // batch_size)
        lowest_wave_loss = lowest_wave_loss_total/(len(X_train) // batch_size)
        error = error_total/(len(X_train) // batch_size)
        print(f"Epoch: {epoch}, Loss1 of total: {loss1},error:{error*100}%, label_loss:{label_loss}, lowest_wave:{lowest_wave_loss}, lr: {lr}")
        # output.write(str(loss1)+' '+str(lr)+"\n")


    num_val = X_val.size(0)
    idx_val = np.arange(num_val)
    np.random.shuffle(idx_val)
    X_val = X_val[idx_val]
    y_val = y_val[idx_val]
    y_val_relate = y_val_relate[idx_val]
    err_rms = 0
    label_loss = 0
    error = 0
    lowest_wavelength_loss = 0
    for i in range(len(X_val) // batch_size):
        batch_x = X_val[i * batch_size: i * batch_size + batch_size].to(device)
        batch_y = y_val[i * batch_size: i * batch_size + batch_size].to(device)
        batch_y_relate = y_val_relate[i * batch_size: i * batch_size + batch_size].to(device)
        tandem_net.test(batch_x, batch_y_relate, mode)
        if mode == 'FNN':
            response = tandem_net.test(batch_x, batch_y, "FNN")
            response_tensor = torch.tensor(response, dtype=torch.float32, device=device)
            err_rms += tandem_net.loss_fn(response_tensor,batch_y_relate)
            error += tandem_net.calculate_rse(batch_y_relate, response_tensor)
            min_index, c_hat = lowest_wavelength(batch_y_relate)
            min_index, c = lowest_wavelength(response_tensor)
            label_loss += tandem_net.loss_fn(c_hat, c)
            lowest_intensity_x = response_tensor[torch.arange(min_index.shape[0]), min_index[:, 0]]
            lowest_intensity_recon_x = batch_y_relate[torch.arange(min_index.shape[0]), min_index[:, 0]]
            lowest_wavelength_loss += tandem_net.loss_fn(lowest_intensity_x, lowest_intensity_recon_x)

            # print(f'Test set Mse:{err_rms},label loss:{label_loss},error:{(1-error)*100}%,lowest_wavelength:{lowest_wavelength_loss}')
        else:
            layer_m,response = tandem_net.test(batch_x, batch_y_relate, 'tandem')
            response_tensor = torch.tensor(response, dtype=torch.float32, device=device)
            err_rms += tandem_net.loss_fn(response_tensor,batch_y_relate)
            error += tandem_net.calculate_rse(batch_y_relate,response_tensor)
            min_index, c_hat = lowest_wavelength(batch_y_relate)
            min_index, c = lowest_wavelength(response_tensor)
            label_loss += tandem_net.loss_fn(c_hat, c)
            lowest_intensity_x = response_tensor[torch.arange(min_index.shape[0]), min_index[:, 0]]
            lowest_intensity_recon_x = batch_y_relate[torch.arange(min_index.shape[0]), min_index[:, 0]]
            lowest_wavelength_loss += tandem_net.loss_fn(lowest_intensity_x, lowest_intensity_recon_x)

            # print(f'Test set loss1:{err_rms},label_loss:{label_loss},error:{(1-error)*100}%,lowest_wavelength:{lowest_wavelength_loss}')
            #output.write(str(loss1) + ' ' + str(loss2) + ' ' + str(lr) + "\n")

    #tandem_net.train(X_train, y_train, mode)
    if mode == 'FNN':
        val_loss = err_rms/(len(X_val) // batch_size)
        label_loss = label_loss/(len(X_val) // batch_size)
        lowest_wave_loss = lowest_wavelength_loss/(len(X_val) // batch_size)
        error = error/(len(X_val) // batch_size)
        print(f"Epoch: {epoch}, Loss of val total: {val_loss},error:{(1-error)*100}%, label_loss:{label_loss}, lowest_wave:{lowest_wave_loss}")
        output.write(
            f"Epoch {epoch} | Train Loss: {loss:.6f} | Val Loss: {val_loss:.6f}\n"
        )
    else:
        val_loss1 = err_rms/(len(X_val) // batch_size)
        label_loss = label_loss/(len(X_val) // batch_size)
        lowest_wave_loss = lowest_wavelength_loss/(len(X_val) // batch_size)
        error = error/(len(X_val) // batch_size)
        print(f"Epoch: {epoch}, Loss1 of val total: {val_loss1},error:{(1-error)*100}%, label_loss:{label_loss}, lowest_wave:{lowest_wave_loss}")
  
        output.write(
            f"Epoch {epoch} | Train Loss: {loss1:.6f} | Val Loss: {val_loss1:.6f}\n"
        )

    #output.write(str(loss) + ' ' + str(lr) + "\n")



    # save every 100 epochs
    if (epoch + 1) % 100 == 0:
        if mode == 'FNN':
            #tandem_net.copy_FNN()
            tandem_net.save_FNN('./1peak/model/DNN_tandem_FNN_label.ckpt')
        elif mode == 'tandem':
            tandem_net.save_INN('./1peak/model/DNN_tandem_INN_label.ckpt')

# output.write('%training time: ' + str(time.time() - t))
output.close()
print('training time: ' + str(time.time() - t))

# y_pred = tandem_net(X_test)
# test_loss = tandem_net.show_loss(y_pred, y_test,mode)
# print(f"Test set MSE: {test_loss.item()}")

# Testing network
tandem = tandem.tandem_network(INN_size=inn_size, FNN_size=fnn_size,structure_max=structure_max,structure_min=structure_min).to(device)


# test FNN
if mode == 'FNN':
    tandem.restore_FNN('./1peak/model/DNN_tandem_FNN_label.ckpt')
    X_test = X_test.to(device)
    y_test_relate = y_test_relate.to(device)
    response = tandem.test(X_test, y_test_relate, "FNN")
    response_tensor = torch.tensor(response, dtype=torch.float32, device=device)
    err_rms = tandem.loss_fn(response_tensor,y_test_relate.to(device))
    error = tandem.calculate_rse(y_test_relate.to(device), response_tensor)
    min_index, c_hat = lowest_wavelength(y_test_relate.to(device))
    min_index, c = lowest_wavelength(response_tensor)
    label_loss = tandem.loss_fn(c_hat, c)
    lowest_intensity_x = response_tensor[torch.arange(min_index.shape[0], device=device), min_index[:, 0]]
    lowest_intensity_recon_x = y_test_relate.to(device)[torch.arange(min_index.shape[0], device=device), min_index[:, 0]]
    lowest_wavelength_loss = tandem.loss_fn(lowest_intensity_x, lowest_intensity_recon_x)

    print(f'Test set Mse:{err_rms},label loss:{label_loss},error:{(1-error)*100}%,lowest_wavelength:{lowest_wavelength_loss}')

# test INN
# design = tandem.test(test_x, test_y, 'INN')
# response = tandem.test(design, test_y, 'FNN')
# err_rms = err(response,test_y)
# print(np.concatenate((test_y, response),axis=1)[0:5])
# print(err_rms)
# import scipy.io
# s = response[:,0:3]
# c = response[:,3:6]
# scipy.io.savemat('prediction.mat', {'pred_design':design, 'pred_siny':s, 'pred_cosy':c})


# test INN+cpFNN
elif mode == 'tandem':
    tandem.restore_INN('./1peak/model/DNN_tandem_INN_label.ckpt')
    tandem.restore_FNN('./1peak/model/DNN_tandem_FNN_label.ckpt')

    X_test = X_test.to(device)
    y_test_relate = y_test_relate.to(device)
    layer_m,response = tandem.test(X_test, y_test_relate, 'tandem').to(device)
    response_tensor = torch.tensor(response, dtype=torch.float32, device=device)
    err_rms = tandem.loss_fn(response_tensor,y_test_relate.to(device))
    error = tandem.calculate_rse(y_test_relate.to(device),response_tensor)
    min_index, c_hat = lowest_wavelength(y_test_relate.to(device))
    min_index, c = lowest_wavelength(response_tensor)
    label_loss = tandem.loss_fn(c_hat, c)
    lowest_intensity_x = response_tensor[torch.arange(min_index.shape[0], device=device), min_index[:, 0]]
    lowest_intensity_recon_x = y_test_relate.to(device)[torch.arange(min_index.shape[0], device=device), min_index[:, 0]]
    lowest_wavelength_loss = tandem.loss_fn(lowest_intensity_x, lowest_intensity_recon_x)

    print(f'Test set loss1:{err_rms},label_loss:{label_loss},error:{(1-error)*100}%,lowest_wavelength:{lowest_wavelength_loss}')
    #loss2 = tandem.loss2(response_tensor,y_test_relate)
    #print(f'Test set loss1:{err_rms},loss2:{loss2}')
#print(tandem.show_loss(test_x,test_y,'tandem'))


# Make predictions
# input_values = [
#     {'R': 4, 'theta': 0, 'n_host': 1.6},
#     {'R': 5, 'theta': 10, 'n_host': 1.8},
#     {'R': 6, 'theta': 20, 'n_host': 2.0}
# ]
#
# input_array = []
# for data in input_values:
#     values = [data['R'], data['theta'], data['n_host']]
#     input_array.append(values)
#
# input_tensor = torch.tensor(input_array, dtype=torch.float32)
#
#
# with torch.no_grad():
#     predictions = tandem_net(input_tensor,'FNN')
#
# wavelengths = np.array(data_b_list[0])[:, 0]  # Wavelengths from the first data_b_list item
#
# # Plotting
# for i, data in enumerate(input_values):
#     print(f"Input: R={data['R']}, theta={data['theta']}, n_host={data['n_host']}")
#     print("Prediction:", predictions[i])
#     plt.figure()
#     plt.plot(wavelengths, predictions[i][:len(wavelengths)], label='Predicted')  # Use only available wavelengths
#     plt.xlabel('Wavelength')
#     plt.ylabel('Intensity')
#     plt.legend()
#     plt.show()
#     print()
