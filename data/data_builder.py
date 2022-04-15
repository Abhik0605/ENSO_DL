from netCDF4 import Dataset
import numpy as np


def dataset(data_path, save_path):
	f = Dataset(f'{data_path}/cmip5_tr.input.1861_2001.nc','r')
	inp1 = f.variables['sst'][:,:,:,:]
	inp2 = f.variables['t300'][:,:,:,:]
	inp1 = np.array(inp1)
	inp2 = np.array(inp2)
	train_data = np.concatenate((inp1, inp2), axis = 1)
	train_data = np.swapaxes(train_data, 1, 3)
	lab1 = Dataset(f'{data_path}/cmip5_tr.label.1861_2001.nc','r')
	inpv2 = lab1.variables['pr'][:,:]
	train_label = np.squeeze(np.array(inpv2))

	val = Dataset(f'{data_path}/cmip5_val.input.1861_2001.nc','r')
	val1 = val.variables['sst'][:,:,:,:]
	val2 = val.variables['t300'][:,:,:,:]
	val1 = np.array(val1)
	val2 = np.array(val2)

	val_data = np.concatenate((val1, val2), axis = 1)
	val_data = np.swapaxes(val_data, 1, 3)

	foo = Dataset(f'{data_path}/cmip5_val.label.1861_2001.nc','r')
	val_label = foo.variables['pr'][:,:]
	val_label = np.squeeze(np.array(val_label))
	
	np.save(f'{save_path}/tr_data.npy', train_data)
	np.save(f'{save_path}/tr_label.npy', train_label)
	np.save(f'{save_path}/val_data.npy', val_data)
	np.save(f'{save_path}/val_label.npy', val_label)
