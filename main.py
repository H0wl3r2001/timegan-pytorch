# -*- coding: UTF-8 -*-
# Local modules
import argparse
import logging
import os
import pickle
import random
import shutil
import time

# 3rd-Party Modules
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA

# Self-Written Modules
from data.data_preprocess import data_preprocess
from metrics.metric_utils import (
    feature_prediction, one_step_ahead_prediction, reidentify_score
)
from metrics.arima import find_best_arima_model, generate_arima_models, prepare_data1
from metrics.rnn_confidence import RNNPredictor, generate_residuals, bootstrap_predictions_with_sliding_window, prepare_data_tensor

from models.timegan import TimeGAN
from models.utils import timegan_trainer, timegan_generator

def main(args):
    ##############################################
    # Initialize output directories
    ##############################################

    ## Runtime directory
    code_dir = os.path.abspath(".")
    if not os.path.exists(code_dir):
        raise ValueError(f"Code directory not found at {code_dir}.")

    ## Data directory
    data_path = os.path.abspath("./data")
    if not os.path.exists(data_path):
        raise ValueError(f"Data file not found at {data_path}.")
    data_dir = os.path.dirname(data_path)
    data_file_name = os.path.basename(data_path)

    ## Output directories
    args.model_path = os.path.abspath(f"./output/{args.exp}/")
    out_dir = os.path.abspath(args.model_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    # TensorBoard directory
    tensorboard_path = os.path.abspath("./tensorboard")
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path, exist_ok=True)

    print(f"\nCode directory:\t\t\t{code_dir}")
    print(f"Data directory:\t\t\t{data_path}")
    print(f"Output directory:\t\t{out_dir}")
    print(f"TensorBoard directory:\t\t{tensorboard_path}\n")

    ##############################################
    # Initialize random seed and CUDA
    ##############################################

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda" and torch.cuda.is_available():
        print("Using CUDA\n")
        args.device = torch.device("cuda:0")
        # torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("Using CPU\n")
        args.device = torch.device("cpu")

    #########################
    # Load and preprocess data for model
    #########################

    if not os.path.exists("data/sin_func1.csv"):
        idx = np.arange(0, 99+1)
        sin_values = np.sin(idx * (2 * np.pi/100))

        df = pd.DataFrame({
            "Idx": idx,
            "sin_value": sin_values
        })
        df.to_csv("data/sin_func.csv", index=False)

    
    data_path = "data/sin_func.csv"
    X, T, _, args.max_seq_len, args.padding_value = data_preprocess(
        data_path, args.max_seq_len
    )

    print(f"Processed data shape: {X.shape} (Idx x MaxSeqLen x Features)\n")
    print(f"Processed data: {len(X[0])}\n")
    print(f"Original data preview:\n{X[:2, :10, :]}\n")
    print(f"Time preview:\n{T}\n")

    args.feature_dim = X.shape[-1]
    args.Z_dim = X.shape[-1]

    # Train-Test Split data and time
    train_data, test_data, train_time, test_time = train_test_split(
        X, T, test_size=args.train_rate, random_state=args.seed
    )

    print(f"Train data: {train_data} \n")
    print(f"Test data: {test_data} \n")
    print(f"Train time: {train_time} \n")
    print(f"Test time: {test_time} \n")

    #########################
    # Initialize arima model (p,q,d) orders
    #########################
    o1, o2, o3, o4 = generate_arima_models(train_data, test_data)

    #########################
    # Initialize rnn model
    #########################
    #t2 = prepare_data2(train_data)

    #x2 = t2['idx'].values
    #x2 = torch.FloatTensor(x2).unsqueeze(-1).unsqueeze(1)
    rnn_model = RNNPredictor(input_size=1, hidden_size=50, num_layers=1, output_size=1, model='rnn')#.to(x2.device)
    #train_model(rnn_model, x2, y2, num_epochs=100, learning_rate=0.01)

    #########################
    # Initialize and Run model
    #########################

    # Log start time
    start = time.time()

    T1 = T[:int(len(train_time)*1.5)]
    
    #########################
    # Generate confidence intervals on original data
    #########################
    new_train_data, new_test_data = prepare_data1(train_data, test_data)

    new_train_data = pd.concat([new_train_data, new_test_data.iloc[:len(new_test_data)//2]])
    new_test_data = new_test_data.iloc[len(new_test_data)//2:]

    flag = True

    if flag:
        print("Generating ACIW from ARIMA...\n")
        arima_model = ARIMA(new_train_data['val'].values, order=o1)
        forecast = arima_model.fit().get_forecast(len(new_test_data))
        intrv = forecast.conf_int(alpha=0.05)
        o_ACIW = np.mean(intrv[:,1] - intrv[:,0])

    else: #RNN BOOTSTRAPPING A CONFIDENCE INTERVAL
        print("Generating residuals for original data...\n")
        residuals = generate_residuals(rnn_model, prepare_data_tensor(new_train_data), len(train_data))

        print("Bootstrapping predictions with sliding window...\n")
        all_predictions = bootstrap_predictions_with_sliding_window(rnn_model, prepare_data_tensor(new_train_data), prepare_data_tensor(new_test_data), residuals)
        
        differences = [upper - lower for _, lower, upper in all_predictions]

        o_ACIW = np.mean(differences)

    print(f"Average confidence interval in original data: {o_ACIW}\n")

    #########################
    # TimeGAN model
    #########################    

    rnn_model2 = RNNPredictor(input_size=1, hidden_size=50, num_layers=1, output_size=1, model='rnn')
    model = TimeGAN(args, T1, train_data, test_data, o_ACIW, o1) 
    if args.is_train == True:
        timegan_trainer(model, train_data, train_time, args)
    generated_data = timegan_generator(model, T, args)
    generated_data = generated_data[len(test_data):] #TODO: splice this data from what corresponds to the train data, keeping the "test_data" for evaluation
    generated_time = test_time

    # Log end time
    end = time.time()

    print(f"Generated data preview:\n{generated_data[:2, -10:, :2]}\n")
    print(f"Model Runtime: {(end - start)/60} mins\n")

    #########################
    # Save train and generated data for visualization
    #########################
    
    # Save splitted data and generated data
    with open(f"{args.model_path}/sinfunc/ARIMA_ACIW_O.pickle", "wb") as fb:
        pickle.dump(o_ACIW, fb)
    with open(f"{args.model_path}/train_data.pickle", "wb") as fb:
        pickle.dump(train_data, fb)
    with open(f"{args.model_path}/train_time.pickle", "wb") as fb:
        pickle.dump(train_time, fb)
    with open(f"{args.model_path}/test_data.pickle", "wb") as fb:
        pickle.dump(test_data, fb)
    with open(f"{args.model_path}/test_time.pickle", "wb") as fb:
        pickle.dump(test_time, fb)
    with open(f"{args.model_path}/fake_data.pickle", "wb") as fb:
        pickle.dump(generated_data, fb)
    with open(f"{args.model_path}/fake_time.pickle", "wb") as fb:
        pickle.dump(generated_time, fb)

    #########################
    # Preprocess data for seeker
    #########################

    # Define enlarge data and its labels
    enlarge_data = np.concatenate((train_data, test_data), axis=0)
    enlarge_time = np.concatenate((train_time, test_time), axis=0)
    enlarge_data_label = np.concatenate((np.ones([train_data.shape[0], 1]), np.zeros([test_data.shape[0], 1])), axis=0)

    # Mix the order
    idx = np.random.permutation(enlarge_data.shape[0])
    enlarge_data = enlarge_data[idx]
    enlarge_data_label = enlarge_data_label[idx]

    #########################
    # Evaluate the performance
    #########################

#    # 1. Feature prediction
#    if X.shape[2] == 1:
#        feat_idx = [0]
#        flag = False
#    else:
#        feat_idx = np.random.permutation(train_data.shape[2])[:args.feat_pred_no]
#        flag = True
#    
#    print("Running feature prediction using original data...")
#    ori_feat_pred_perf = feature_prediction(
#        (train_data, train_time), 
#        (test_data, test_time),
#        feat_idx,   
#        flag
#    )
#    
#    print("Running feature prediction using generated data: TRTS (Train on real, test on synthetic)...")
#    new_feat_pred_perf = feature_prediction(
#        (train_data, train_time),
#        (generated_data, generated_time),
#        feat_idx,
#        flag
#    )
#
#    feat_pred = [ori_feat_pred_perf, new_feat_pred_perf]
#
#    print('Feature prediction results:\n' +
#          f'(1) Ori: {str(np.round(ori_feat_pred_perf, 4))}\n' +
#          f'(2) New: {str(np.round(new_feat_pred_perf, 4))}\n')
#
#    # 2. One step ahead prediction
#    if(flag):
#        print("Running one step ahead prediction using original data...")
#        ori_step_ahead_pred_perf = one_step_ahead_prediction(
#            (train_data, train_time), 
#            (test_data, test_time),
#            flag
#        )
#        print("Running one step ahead prediction using generated data...")
#        new_step_ahead_pred_perf = one_step_ahead_prediction(
#            (train_data, train_time),
#            (generated_data, generated_time),
#            flag
#        )
#
#        step_ahead_pred = [ori_step_ahead_pred_perf, new_step_ahead_pred_perf]
#
#        print('One step ahead prediction results:\n' +
#              f'(1) Ori: {str(np.round(ori_step_ahead_pred_perf, 4))}\n' +
#              f'(2) New: {str(np.round(new_step_ahead_pred_perf, 4))}\n')
#
#    # 3. Arima prediction (univariate):
#    #TODO: complete this part
#    if(flag == False):
#
#        #p = [0, 1, 2]
#        #q = range(0, 3)
#        #d = range(0, 3)
#
#        print("Running Arima prediction using original data...")
#        #ori_arima_pred_perf = evaluate_models(
#        #    train_data, 
#        #    test_data,
#        #    p,
#        #    q,
#        #    d
#        #)
#        order_og, ori_arima_pred_perf = find_best_arima_model(train_data, test_data)
#
#        print("Running Arima prediction using generated data...")
#        #new_arima_pred_perf = evaluate_models(
#        #    X,
#        #    generated_data,
#        #    p,
#        #    q,
#        #    d
#        #)
#        order_synth, new_arima_pred_perf = find_best_arima_model(train_data, generated_data)
#        step_ahead_pred = [ori_arima_pred_perf, new_arima_pred_perf]
#
#        print('Arima prediction results:\n' +
#            f'(1) Ori: {str(np.round(ori_arima_pred_perf, 4))}\n' +
#            f'(2) New: {str(np.round(new_arima_pred_perf, 4))}\n')
#        
#        print('Arima order results:\n' +
#            f'(1) Ori: {order_og}\n' +
#            f'(2) New: {order_synth}\n')

    print(f"Total Runtime: {(time.time() - start)/60} mins\n")

    return None

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    # Inputs for the main function
    parser = argparse.ArgumentParser()

    # Experiment Arguments
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        default='cpu',
        type=str)
    parser.add_argument(
        '--exp',
        default='test',
        type=str)
    parser.add_argument(
        "--is_train",
        type=str2bool,
        default=True)
    parser.add_argument(
        '--seed',
        default=0,
        type=int)
    parser.add_argument(
        '--feat_pred_no',
        default=2,
        type=int)

    # Data Arguments
    parser.add_argument(
        '--max_seq_len',
        default=2,
        type=int)
    parser.add_argument(
        '--train_rate',
        default=0.5,
        type=float)

    # Model Arguments
    parser.add_argument(
        '--emb_epochs',
        default=500,
        type=int)
    parser.add_argument(
        '--sup_epochs',
        default=500,
        type=int)
    parser.add_argument(
        '--gan_epochs',
        default=1020,
        type=int)
    parser.add_argument(
        '--batch_size',
        default=128,
        type=int)
    parser.add_argument(
        '--hidden_dim',
        default=20,
        type=int)
    parser.add_argument(
        '--num_layers',
        default=3,
        type=int)
    parser.add_argument(
        '--dis_thresh',
        default=0.15,
        type=float)
    parser.add_argument(
        '--optimizer',
        choices=['adam'],
        default='adam',
        type=str)
    parser.add_argument(
        '--learning_rate',
        default=1e-3,
        type=float)

    args = parser.parse_args()

    # Call main function
    main(args)
