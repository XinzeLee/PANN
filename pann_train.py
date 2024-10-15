"""
Created on Thu Mar 21 07:50:04 2024

@author: XinzeLee
@github: https://github.com/XinzeLee/PANN

@reference:
    1: Temporal Modeling for Power Converters With Physics-in-Architecture Recurrent Neural Network
        Authors: Xinze Li, Fanfan Lin (corresponding and co-first author), Huai Wang, Xin Zhang, Hao Ma, Changyun Wen and Frede Blaabjerg
        Paper DOI: 10.1109/TIE.2024.3352119
    2: Data-Light Physics-Informed Modeling for the Modulation Optimization of a Dual-Active-Bridge Converter
        Authors: Xinze Li, Fanfan Lin (corresponding and co-first author), Xin Zhang, Hao Ma and Frede Blaabjerg
        Paper DOI: 10.1109/TPEL.2024.3378184

"""

import torch
import copy
import numpy as np
import torch.nn as nn

from torch.utils.data import Dataset
from pann_utils import evaluate, transform




class CustomDataset(Dataset):
    def __init__(self, states, inputs, targets):
        super(CustomDataset, self).__init__()
        self.states = states
        self.inputs = inputs
        self.targets = targets
        
    def __getitem__(self, index):
        return self.states[index], self.inputs[index], self.targets[index]
        
    def __len__(self):
        return len(self.states)
    

def train(model_pann, clamper, optimizer_pann, data_loader, test_data, val_data,
          Tslen, convert_to_mean=True, epoch=200, verbose=False):
    
    test_inputs, test_states = test_data
    val_inputs, val_states = val_data
    
    loss_pann = nn.MSELoss()
    device = "cpu" # it is a waste to use gpu for this efficient and comparct network
    
    
    loss_best_pann = np.inf
    best_pann_states = None
    circuit_estimation_history = []
    loss_history = []
    val_waveforms = []
    
    model_pann = model_pann.to(device)
    for epoch in range(epoch):
        model_pann.train()
        
        #Forward pass
        total_loss = 0.
        for data in data_loader:
            """ 
                Logic is:
                input_ (full length) -> smooth_all -> PANN pred -> segment final Tslen*2 points -> sync
            """
            
            state, input_, target = data
            state, input_, target = state.to(device), input_.to(device), target.to(device)
            # state0 = state[:, :1] # should be zero to avoid learning the initial state
            state0 = torch.zeros(state.shape).to(device) # should be zero to avoid learning the initial state
            pred = model_pann.forward(input_, state0)
            Vin = 200
            pred, _ = transform(input_[:, -2*Tslen:], pred[:, -2*Tslen:], Vin, 
                                Tslen, convert_to_mean=convert_to_mean)
            
            loss_train = loss_pann(pred, target)
            optimizer_pann.zero_grad()
            loss_train.backward()
            optimizer_pann.step()
            clamper(model_pann) # comment out this line if using pure data-driven model for dk
            total_loss += loss_train.item()
        estimated_circuit = list(map(lambda x: round(x.item(), 7), model_pann.parameters()))
        print("Estimations for circuit parameters: ", estimated_circuit)
        if verbose: 
            circuit_estimation_history.append(estimated_circuit)
        print(f"Epoch {epoch}, Training loss {total_loss/len(data_loader):.3f}")  
        
        if epoch % 1 == 0:
            train_loss = (target-pred).abs().mean().item()
            *_, test_loss = evaluate(test_inputs[:, 1:], test_states, model_pann, Tslen,
                                     Vin, convert_to_mean=convert_to_mean)
            train_test_loss = (train_loss*len(state)+
                       test_loss*len(test_states))/(len(state)+len(test_states))
            if train_test_loss < loss_best_pann:
                # select the best pann checkpoint based on the training and test losses
                loss_best_pann, best_pann_states = train_test_loss, copy.deepcopy(model_pann.state_dict())
                print(f"New loss is {loss_best_pann}.")
                print('-'*81)
            if verbose:
                val_pred, _, val_loss = evaluate(val_inputs[:, 1:], val_states, model_pann, Tslen,
                                         Vin, convert_to_mean=convert_to_mean)
                loss_history.append([train_loss, test_loss, val_loss])
                val_waveforms.append(val_pred[[0, 3]]) # save the first two waveform predictions
                
    if verbose:
        return best_pann_states, circuit_estimation_history, loss_history, val_waveforms
    return best_pann_states



