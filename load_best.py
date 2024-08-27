import argparse
import copy
import math
import os
import random
import typing

import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import scipy.stats as stats
from scipy.stats import gmean

from dataset import get_dataloader, get_dataset
from EMA import WeightExponentialMovingAverage
from model import DeePMOS

# Define the valid function
def valid(model,
          dataset,
          dataloader, 
          systems,
          steps,
          prefix,
          device,
          MSE_list,
          LCC_list,
          SRCC_list):
    model.eval()

    mos_mus = []
    mos_targets = []
    mos_vars = []
    mos_mus_sys = {system:[] for system in systems}
    mos_vars_sys = {system:[] for system in systems}
    true_sys_mean_scores = {system:[] for system in systems}

    for i, batch in enumerate(tqdm(dataloader, ncols=0, desc=prefix, unit=" step")):
        if dataset == 'vcc2018':
            wav, filename, _, mos, _ = batch
            sys_names = list(set([name.split("_")[0] for name in filename])) # system name, e.g. 'D03'
        elif dataset == 'bvcc':
            wav, mos, sys_names = batch
        wav = wav.to(device)
        wav = wav.unsqueeze(1) # shape (batch, 1, seq_len, 257)

        with torch.no_grad():
            try:
                mos_mu, mos_var = model(speech_spectrum = wav) # shape (batch, seq_len, 1)
                mos_mu = mos_mu.squeeze(-1) # shape (batch, seq_len)
                mos_var = mos_var.squeeze(-1)
                mos_mu = torch.mean(mos_mu, dim = -1) # torch.Size([1])
                mos_var = torch.mean(mos_var, dim = -1)

                mos_mu = mos_mu.cpu().detach().numpy()
                mos_var = mos_var.cpu().detach().numpy()
                mos_mus.extend(mos_mu.tolist())
                mos_targets.extend(mos.tolist())
                mos_vars.extend(mos_var.tolist())

                for j, sys_name in enumerate(sys_names):
                    mos_mus_sys[sys_name].append(mos_mu[j])
                    mos_vars_sys[sys_name].append(mos_var[j])
                    true_sys_mean_scores[sys_name].append(mos.tolist()[j])

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"[Runner] - CUDA out of memory at step {steps}")
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise
    
    mos_mus = np.array(mos_mus)
    mos_vars = np.array(mos_vars)
    mos_targets = np.array(mos_targets)

    mos_mus_sys = np.array([np.mean(scores) for scores in mos_mus_sys.values()])
    mos_vars_sys = np.array([np.mean(scores)/len(scores) for scores in mos_vars_sys.values()])
    true_sys_mean_scores = np.array([np.mean(scores) for scores in true_sys_mean_scores.values()])
    
    utt_MSE=np.mean((mos_targets-mos_mus)**2)
    utt_LCC=np.corrcoef(mos_targets, mos_mus)[0][1]
    utt_SRCC=scipy.stats.spearmanr(mos_targets, mos_mus)[0]
    
    sys_MSE=np.mean((true_sys_mean_scores-mos_mus_sys)**2)
    sys_LCC=np.corrcoef(true_sys_mean_scores, mos_mus_sys)[0][1]
    sys_SRCC=scipy.stats.spearmanr(true_sys_mean_scores, mos_mus_sys)[0]

    Likelihoods = []
    for i in range(len(mos_targets)):
        Likelihoods.append(stats.norm.pdf(mos_targets[i], mos_mus[i], math.sqrt(mos_vars[i])))

    #utt_GML=gmean(Likelihoods) # geometric mean of likelihood
    utt_AML=np.mean(Likelihoods) # arithemic mean of likelihood
    utt_MoV=np.mean(mos_vars) # mean of variance
    utt_VoV=np.var(mos_vars) # variance of variance

    # Likelihoods_sys = []
    # for i in range(len(true_sys_mean_scores)):
    #     Likelihoods_sys.append(stats.norm.pdf(true_sys_mean_scores[i], mos_predictions_sys[i], math.sqrt(mos_vars_sys[i])))
    # sys_GML=gmean(Likelihoods_sys)
    # sys_AML=np.mean(Likelihoods_sys)
    # sys_MoV=np.mean(mos_vars_sys)
    # sys_VoV=np.var(mos_vars_sys)
    
    MSE_list.append(utt_MSE)
    LCC_list.append(utt_LCC) 
    SRCC_list.append(utt_SRCC)

    print(
        f"\n[{prefix}][{steps}][UTT][ MSE = {utt_MSE:.4f} | LCC = {utt_LCC:.4f} | SRCC = {utt_SRCC:.4f} ] [SYS][ MSE = {sys_MSE:.4f} | LCC = {sys_LCC:.4f} | SRCC = {sys_SRCC:.4f} ]"
    )
    print(f"[{prefix}][{steps}][UTT][ AML = {utt_AML:.6f} | MoV = {utt_MoV:.6f} | VoV = {utt_VoV:.6f} ]" )

    model.train()
    return MSE_list, LCC_list, SRCC_list, sys_SRCC

# Arguments
data_path = '/home/hbml/demo/Nikhil/DeePMOS/BVCC/DATA/'
id_table = '/home/hbml/demo/Nikhil/DeePMOS/BVCC/DATA/id_table/'
dataset = 'bvcc'
batch_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the saved model
model_path = '/home/hbml/demo/Nikhil/DeePMOS/best.pt'
best_model = torch.jit.load(model_path)
best_model.to(device)

# Set up the dataset and dataloader
if dataset == 'vcc2018':
    test_set = get_dataset(data_path, "testing_data.csv", vcc18=True, valid=True, idtable=os.path.join(id_table, 'idtable.pkl'))
elif dataset == 'bvcc':
    test_set = get_dataset(data_path, "test", bvcc=True, valid=True, idtable=os.path.join(id_table, 'idtable.pkl'))

test_loader = get_dataloader(test_set, batch_size=batch_size, num_workers=1)

# Prepare lists to store results
MSE_list = []
LCC_list = []
SRCC_list = []

# Run validation on the test set
best_model.eval()
MSE_list, LCC_list, SRCC_list, sys_SRCC = valid(
    best_model, dataset, test_loader, test_set.systems, 0, 'Test(best)', device, MSE_list, LCC_list, SRCC_list)

print(f"Test Results: MSE = {MSE_list[-1]:.4f}, LCC = {LCC_list[-1]:.4f}, SRCC = {SRCC_list[-1]:.4f}")
