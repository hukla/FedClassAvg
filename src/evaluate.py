
import numpy as np
from ast import literal_eval
import argparse

def fedavg_acc(log_path, eval_round):
    accs = []
    with open(log_path, 'r') as f:
        for line in f:
            if f'ROUND {eval_round}' in line and 'Accuracy/val' in line:
                accs.append(float(line.split(':')[1][:-1]))
    return np.mean(accs), np.std(accs)

def ktpfl_acc(log_path, eval_round):
    accs = []
    with open(log_path, 'r') as f:
        for line in f:
            if '{' in line:
                acc_dict = literal_eval(line)
    for client_id in acc_dict.keys():
        accs.append(acc_dict[client_id][eval_round])
    return np.mean(accs), np.std(accs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, default='results/debug/log.txt')
    parser.add_argument('--eval_round', type=int, default=516)
    parser.add_argument('--ktpfl', action='store_true')
    args = parser.parse_args()
    
    if args.ktpfl:
        party_mean, party_std = ktpfl_acc(args.log_path, args.eval_round)
    else:
        party_mean, party_std = fedavg_acc(args.log_path, args.eval_round)
    
    print(f'Average accuracy: {party_mean:.4f} (std: {party_std:.4f})')
