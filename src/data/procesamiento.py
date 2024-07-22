import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def minmax(signal, feature_range):
    scaler = MinMaxScaler(feature_range=feature_range)
    signal = signal.reshape(-1,1)
    scaler.fit(signal)

    #save the necessary values
    min_  = scaler.min_
    scale_ = scaler.scale_

    #compute the transformation of the signal
    sig_processed = scaler.transform(signal).squeeze()
    
    return sig_processed, min_, scale_

def norm(signal):
    scaler = StandardScaler()
    signal = signal.reshape(-1,1)
    scaler.fit(signal)
    sig_processed = scaler.transform(signal).squeeze()

    mean = scaler.mean_
    var = scaler.var_
    return sig_processed, mean, var

def preprocessing(n_signals, data, feature_range = (-1, 1), folder_name="data"):
    data_new = []
    inform_signal = []
    print('Preprocessing signals...', end='')
    for i in range(n_signals):
        sig , min_, scale_ = minmax(data[i,:], feature_range=feature_range)
        data_new.append(sig)
        inform_signal.append((min_, scale_))
    np.save(f'src/{folder_name}/min_max_scaler.npy', inform_signal)

    print(' done.')
    return data_new 


# def postprocesing(data, idx, folder_name='data'):
#     data = data.detach().cpu()
#     data_old = np.zeros_like(data)
#     inform_signal = np.load(f'{folder_name}/min_max_scaler.npy')
#     for i in range(len(idx)): #by batch
#         min_, scale_ = inform_signal[idx[i]]
#         signal = data[:,i, :] #take signal by batch

#         data_old[:, i, :] = (signal + min_) / scale_
#     return data_old


