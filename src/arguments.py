
import argparse

import numpy as np
from utils.myTools import dotdict
import ast
import torch
import os

def parse_args():
    args = dotdict()

    parser = argparse.ArgumentParser(description='Model Argument Parser')
    
    parser.add_argument('--resume_run', type=str, default=None, help='Name of the run to resume')

    parser.add_argument('--pe', type=int, help='positional embeddings')

    # Arguments related to transformer heads
    parser.add_argument('--heads', type=int, default=3, help='number of heads of the transformer that focus on high frequencies')
    parser.add_argument('--heads_low', type=int, default=2, help='number of heads of the transformer that focus on low frequencies')

    # Arguments related to encoder layers
    parser.add_argument('--nencoder', type=int, default=2, help='number of layers in the encoder')

    # Arguments related to decoder layers
    parser.add_argument('--prediction_len', type=int, default=1, help='prediction length (the decoder predicts next N values)')

    # Other general arguments
    parser.add_argument('--dropout', type=float, default=0.6, help='dropout rate')
    parser.add_argument('--train_epochs', type=int, default=300, help='number of max epochs to train the model')

    # Arguments related to learning
    parser.add_argument('--learning_rate', type=float, default=0.005, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')

    parser.add_argument('--mean_kernel_size', type=int, default=7, help='mean_kernel_size')
    parser.add_argument('--var_kernel_size', type=int, default=5, help='var_kernel_size')
    parser.add_argument('--filterk', type=int, default=9, help='filter_kernel_size')
    # parser.add_argument('--sigma', type=float, default=0.05, help='noise added to the predicted variance to ensure numerical stability')

    # Arguments related to spectrogram conversion and filtering
    parser.add_argument('--nfft_k', type=int, default=5, help='number of fourier transforms in the spectrogram')
    parser.add_argument('--hop', type=int, default=2, help='number of fourier transforms in the spectrogram')
    parser.add_argument('--kernel', type=int, default=5, help='width of the 1-D sliding kernel')

    # Arguments related to HiLo filters
    parser.add_argument('--kernel_low', type=int, default=21, help='kernel size for low-frequency transformer')
    parser.add_argument('--kernel_high', type=int, default=5, help='kernel size for high-frequency transformer')
    parser.add_argument('--dilation_low', type=int, default=1, help='dilation rate for low-frequency transformer')
    parser.add_argument('--dilation_high', type=int, default=5, help='dilation rate for high-frequency transformer')
    parser.add_argument('--time_compression', type=int, default=1, help='lofi part time compression')
    parser.add_argument('--avgPoolK', type=int, default=5, help='lofi part filter size')
    parser.add_argument('--avgPoolS', type=int, default=1, help='lofi part stride')

    # Arguments related to features
    parser.add_argument('--inchannels_conf', type=int, default=5, help='number of features in the configuration')
    parser.add_argument('--feature_range', type=float, default=1, help='limit of the range of features')
    parser.add_argument('--conf_cnn', action='store_true', help='consider configurations')

    parser.add_argument('--recalculate_memory_every', type=int, default=5, help='when testing autorregresively, recalculate the encoder memory every n iterations')
    parser.add_argument('--p', type=float, default=0.01, help='probability of having no teacher forcing when training')
    parser.add_argument('--test_teacher_forcing_every', type=int, default=10, help='when testing autorregresively, provide a true value every n iterations')

    args = parser.parse_args()

    # Handle argument incompatibilities:
    if args.filterk % 2 == 0:
        args.filterk += 1
    if args.mean_kernel_size % 2 == 0:
        args.mean_kernel_size += 1
    if args.var_kernel_size % 2 == 0:
        args.var_kernel_size += 1
    if args.kernel % 2 == 0:
        args.kernel += 1
    if args.kernel_low % 2 == 0:
        args.kernel_low += 1
    if args.kernel_high % 2 == 0:
        args.kernel_high += 1
    if args.avgPoolK % 2 == 0:
        args.avgPoolK += 1



    args.nfft = int(2*((args.nfft_k * np.lcm(args.heads, args.heads_low) - 1)/2))

    # Additional argument handling
    args.use_gpu = True if torch.cuda.is_available() else False
    args.gpu = 3
    if args.conf_cnn:
        args.outchannels_conf = 2 ** args.inchannels_conf
    args.feature_range = (-1 * args.feature_range, args.feature_range)
    args.num_devices = 1
    args = dotdict(vars(args))

    if args.dilation_high == 0:
        args.dilation_high = 1
    
    args.name_folder = 'Exp1'

    if args.train_epochs != 1: # debug
        try:
            sweep = os.environ['SWEEP']
        except KeyError:
            sweep = False
    else: sweep = False
        
    if sweep:
        args.train = True
        args.test = True
        try: 
            args.gpu = os.environ['GPU']
        except Exception: 
            print('You must set global var "GPU" when running sweeps!')
            raise KeyError


    data_args = dotdict()

    data_args.path = './DATASETS/WS4_preprocessed_multiclass.pkl'
    data_args.get_class = True #if we want to also get the class of the data
    data_args.several_conf = True

    return args, data_args


def load_args(args):
    file = open(f'./results/{args.resume_run}/arguments.txt', "r")
    for line in file:
        s1 = line.split(': ')
        if s1[0] != 'devices':
            try:
                args[s1[0]] = ast.literal_eval(s1[1].split(' \n')[0])
            except ValueError:
                args[s1[0]] = s1[1].split(' \n')[0]
        else:
            args[s1[0]] = s1[1].split(' \n')[0]