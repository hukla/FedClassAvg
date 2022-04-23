import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import numpy as np
import utils
import wandb
from tqdm import tqdm
from utils import EarlyStopping
import misc_ktpfl

# TODO uncomment for debugging
# wandb.login(key="6ce7c21067c5213d01777e0a4527fda5597774a3")

def move_state_dict(source, device):
    for param_tensor in source:
        source[param_tensor] = source[param_tensor].to(device)

def clone_state_dict(source):
    dest = {}
    for param_tensor in source:
        dest[param_tensor] = source[param_tensor].clone()
    return dest

def cmult_state_dict(constant, source):
    dest = {}
    for param_tensor in source:
        dest[param_tensor] = source[param_tensor] * constant
    return dest

def _cmult_state_dict(constant, source):
    for param_tensor in source:
        source[param_tensor] = source[param_tensor] * constant

def add_state_dict(source1, source2):
    dest = {}
    for param_tensor in source1:
        dest[param_tensor] = source1[param_tensor] + source2[param_tensor]
    return dest

def _add_state_dict(source1, source2):
    for param_tensor in source1:
        source1[param_tensor] = source1[param_tensor] + source2[param_tensor]

def zero_state_dict_like(source):
    dest = {}
    for param_tensor in source:
        dest[param_tensor] = torch.zeros_like(source[param_tensor])
    return dest

class KT_pFL():
    def __init__(self, parties,
                 public_dataset,
                 client_train_datasets,
                 client_test_datasets,
                 num_classes,
                 configs,
                 device='cuda'):

        self.parties = parties
        self.n_parties = len(parties)
        self.public_dataset = public_dataset
        self.client_train_datasets = client_train_datasets
        self.client_test_datasets = client_test_datasets
        self.configs = configs
        self.num_classes = num_classes

        # kcm: knowledge coefficient matrix
        kcm = torch.ones(self.n_parties, self.n_parties, requires_grad=True)
        kcm = kcm.float() / (self.n_parties)
        self.kcm = kcm

        self.device = device

        self.init_result = []

        self.num_workers = configs['num_workers']
        self.Temp = 10.0
        self.penalty_ratio = configs['penalty_ratio']

        # self.init()
    
    def init(self, init_epochs=1):
        """ initialize client models 

        Args:
            init_epochs (int): number of epochs to init models (default: 1)
        """        
        #TODO init + kcm update
        print('=====Initialize client models=====')
        for client_id in range(self.n_parties):
            client_train_dataset = self.client_train_datasets[client_id]
            client_test_dataset = self.client_test_datasets[client_id]
            train_loader = DataLoader(client_train_dataset, batch_size=self.configs['private_batchsize'], shuffle=True, num_workers=self.num_workers, drop_last=True)
            test_loader = DataLoader(client_test_dataset, batch_size=self.configs['pivate_batchsize'], shuffle=False, num_workers=self.num_workers)

            model = self.parties[client_id].to(self.device)
            optimizer = torch.optim.SGD(model.parameters(), lr=self.configs['learning_rate'])
            criterion = nn.CrossEntropyLoss()
            
            train_loss, train_acc = misc_ktpfl.train_one_model(model=model, max_epochs=init_epochs, device=self.device, train_dataloader=train_loader, optimizer=optimizer, criterion=criterion, client_id=client_id)
            test_loss, test_acc = misc_ktpfl.evaluate_one_model(model=model, loader=test_loader, criterion=criterion, device=self.device)

            print(f'[ROUND init CLIENT {client_id}] Loss/train:{train_loss:.3f}, Accuracy/train:{train_acc:.3f}')
            print(f'[ROUND init CLIENT {client_id}] Loss/test:{test_loss:.3f}, Accuracy/test:{test_acc:.3f}')

            self.parties[client_id] = model

        print('=====Initialize client models done=====')

    def collaborative_training(self, client_sampling_rate, max_rounds, max_local_epochs, distillation_steps, output_path):
        # start collaborating training
        device = self.device
        collaboration_performance = {i: [] for i in range(self.n_parties)}
        round_idx = 0
        kcm = self.kcm
        best_val_loss = np.inf
        best_val_acc = 0
        early_stopping = EarlyStopping(patience=100, verbose=True)

        n_parties_round = int(self.n_parties * client_sampling_rate)
        
        for round_idx in range(max_rounds):
            global_parties = copy.deepcopy(self.parties)
            sampled_clients = np.random.choice(np.arange(self.n_parties), n_parties_round, replace=False)

            # At beginning of each round, generate new alignment dataset; public
            # public_dataset = misc_ktpfl.load_public_dataset(self.configs['data'], datasize=self.configs['public_datasize'])
            # public_dataloader = DataLoader(public_dataset, batch_size=self.configs['public_batchsize'], num_workers=self.num_workers, shuffle=False, drop_last=True)

            print(f"[ROUND {round_idx}] START")

            '''local training'''
            print(f"[ROUND {round_idx}] Local update")

            # gradually increase local epochs; also not mentioned in the paper
            # local_epoch = 1
            if round_idx == 0:
                local_epoch = 1 - 1
            if local_epoch < max_local_epochs:
                local_epoch += 1
            else:
                local_epoch = max_local_epochs

            # sampled_public_dataset = self.public_dataset

            for client_id in sampled_clients:
                ### update local weights
                print(f"[ROUND {round_idx} model {client_id}] Start local training")
                client_train_dataset = self.client_train_datasets[client_id]
                client_test_dataset = self.client_test_datasets[client_id]
                train_loader = DataLoader(client_train_dataset, batch_size=self.configs['private_batchsize'], shuffle=True, num_workers=self.num_workers, drop_last=True)
                # test_loader = DataLoader(client_test_dataset, batch_size=self.configs['private_batchsize'], shuffle=False, num_workers=self.num_workers)
                model = self.parties[client_id]

                optimizer = torch.optim.SGD(model.parameters(), lr=self.configs['learning_rate']) #mu1
                criterion = nn.CrossEntropyLoss()

                train_loss, train_acc = misc_ktpfl.train_one_model(model=model, max_epochs=local_epoch, device=self.device, train_dataloader=train_loader, optimizer=optimizer, criterion=criterion, client_id=client_id, csr=round_idx)

                print(f"[ROUND {round_idx} model {client_id}] Done local training (loss/train: {train_loss:.4f}, acc/train: {train_acc:.4f})")
                self.parties[client_id] = model
                ### update local weights done

            '''parameterized knowledge transfer'''
            # distill_models = self.parameterized_knowledge_transfer(global_parties)
            for client_id in sampled_clients:
                distill_model = self.parameterized_knowledge_transfer(global_parties, client_id)
                optimizer = torch.optim.SGD(model.parameters(), lr=self.configs['learning_rate']) 
                criterion = nn.MSELoss()

                dist_loss = 0
                for step in range(distillation_steps):
                    print(f"[ROUND {round_idx} model {client_id}] Parameterized knowledge transfer")
                    model = self.parties[client_id]
                    # distill_model = distill_models[client_id]
                    distill_loss = 0

                    optimizer.zero_grad()
                    for name, param in model.named_parameters():
                        distill_param = distill_model[name]
                        distill_loss += criterion(param, distill_param)
                    

                    distill_loss.backward(retain_graph=True)
                    optimizer.step()

                    dist_loss += distill_loss.item()
                del distill_loss, distill_model
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

                dist_loss /= distillation_steps

                print(f"[ROUND {round_idx} model {client_id}] Done parameterized knowledge transfer (loss: {dist_loss:.4f})")
                ### distillation step done

                self.parties[client_id] = model

            '''update knowledge coefficient matrix'''
            print(f"[ROUND {round_idx}] Update knowledge coefficients")
            #TODO: update weights
            kcm = self.update_coefficients(global_parties, self.penalty_ratio)
            self.kcm = kcm
            del kcm
            torch.cuda.empty_cache()

            # balanced test
            print(f"[ROUND {round_idx}] Test performance")
            round_test_acc = 0
            round_test_loss = 0
            for client_id in tqdm(range(self.n_parties)):
                # client id of virutal client
                model = self.parties[client_id]
                criterion = nn.CrossEntropyLoss()

                client_test_dataset = self.client_test_datasets[client_id]
                client_test_loader = DataLoader(client_test_dataset, batch_size=self.configs['private_batchsize'], shuffle=True, num_workers=self.num_workers, drop_last=False)

                val_loss, val_acc = misc_ktpfl.evaluate_one_model(model, client_test_loader, criterion, device)
                collaboration_performance[client_id].append(val_acc)
                round_test_acc += val_acc
                round_test_loss += val_loss

            round_test_acc /= self.n_parties
            round_test_loss /= self.n_parties

            wandb.log({'global/loss': round_test_loss, 'global/acc': round_test_acc}, step=round_idx)
            print(f'[ROUND {round_idx}] Average test accuracy: {round_test_acc:.4f} (loss: {round_test_loss:.4f})')

            if best_val_loss > round_test_loss:
                best_val_acc = round_test_acc
                best_val_loss = round_test_loss
                # save coefficient matrix
                print(f'[ROUND {round_idx}] Saving kcm')
                torch.save(self.kcm, os.path.join(output_path, f'round{round_idx}_kcm.npz'))

            wandb.log({'best_global/loss': best_val_loss, 'best_global/acc': best_val_acc}, step=round_idx)
            early_stopping(round_test_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                for client_id in range(self.n_parties):
                    model = self.parties[client_id]
                    torch.save(model, os.path.join(output_path, f'round{round_idx}_client{client_id}.pt'))
                break

            del global_parties
            torch.cuda.empty_cache()
            # END FOR LOOP

        # save models
        for client_id in range(self.n_parties):
            model = self.parties[client_id]
            torch.save(model, os.path.join(output_path, f'round{round_idx}_client{client_id}.pt'))
        # END WHILE LOOP
        return collaboration_performance


    def parameterized_knowledge_transfer(self, global_parties, self_idx):
        # distil_models = [zero_state_dict_like(global_party.state_dict()) for global_party in global_parties]
        distil_model = zero_state_dict_like(global_parties[self_idx].state_dict())
        weight = self.kcm
        # update raw_logits
        # for self_idx in range(self.n_parties): # Calculate the weighted average of its teacher's logits for each model
        weight_tmp = torch.zeros(self.n_parties) # weight to current client
        for teacher_idx in range(self.n_parties): 
            # For a model, calculate the weight after its softmax (why???)
            weight_tmp[teacher_idx] = self.kcm[self_idx][teacher_idx]
        # weight_local = nn.functional.softmax(weight_tmp*5.0)
        weight_local = weight_tmp

        idx_count = 0
        for teacher_idx in range(self.n_parties): # For a model, calculate the logits of all other models
            teacher_state_dict = cmult_state_dict(weight_local[idx_count], global_parties[teacher_idx].state_dict())
            # distil_models[self_idx] = add_state_dict(distil_models[self_idx], teacher_state_dict)
            _add_state_dict(distil_model, teacher_state_dict)
            with torch.no_grad():
                weight[self_idx][teacher_idx] = weight_local[idx_count]
            idx_count += 1             
        # return distil_models
        return distil_model

        
    def update_coefficients(self, global_parties, penalty_ratio):
        weight_mean = torch.ones(self.n_parties, self.n_parties, requires_grad=True)
        weight_mean = weight_mean.float()/(self.n_parties)
        loss_fn = nn.MSELoss(reduce=True, size_average=True)
        teacher_weights = global_parties
        model_weights = self.parties

        weight = self.kcm
        for self_idx in range(self.n_parties): #Calculate the weighted average of its teacher's logits for each model
            teacher_weights_local = zero_state_dict_like(teacher_weights[self_idx].state_dict())
            for teacher_idx in range(self.n_parties): 
                teacher_state_dict = cmult_state_dict(weight[self_idx][teacher_idx], teacher_weights[teacher_idx].state_dict())
                _add_state_dict(teacher_weights_local, teacher_state_dict)
                del teacher_state_dict
                # teacher_weight = torch.add(teacher_logits_local, weight[self_idx][teacher_idx] * torch.from_numpy(raw_logits[teacher_idx])) 
                # A pixel in tensor, intrinsic scalar * teacher's complete logits
                
            loss_input = model_weights[self_idx]

            loss = 0
            for name, input_param in loss_input.named_parameters():
                loss += loss_fn(input_param, teacher_weights_local[name])
            
            loss_penalty = loss_fn(weight[self_idx], weight_mean[self_idx])
            loss += loss_penalty * penalty_ratio

            weight.retain_grad() 
            loss.backward(retain_graph=True)

            with torch.no_grad():
                weight.sub_(weight.grad*self.configs['distill_lr'])
                weight.grad = None
            del teacher_weights_local, loss_penalty, loss
        del teacher_weights, model_weights
        torch.cuda.empty_cache()


        return weight


def main(args):
    wandb.init(
        project=f"KTpFL-{args.data}",
        notes=f"models:{args.models}",
        tags=[f"{args.data}"],
        config=args
    )
    wandb.run.name = args.runfile
    args = wandb.config

    SEED = args.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    models = args.models.split(',')

    '''gpu'''
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    '''Import Data'''
    train_dataset, test_dataset, _, num_classes = utils.load_dataset(data_split=args.data, target_label_idx=0)
    # public_dataset = misc_ktpfl.load_public_dataset(args.data, datasize=args.public_datasize)
    y_train = np.array(train_dataset.targets)
    net_index_map = utils.partition_data(option=args.data_partition, num_labels=num_classes, y_train=y_train, n_parties=args.n_parties)
    y_test = np.array(test_dataset.targets)
    
    # get client test
    test_indices_k = []
    for k in range(num_classes):
        idx_k = np.where(y_test == k)[0]
        test_indices_k.append(idx_k)

    '''Create client models'''
    print('Creating client models and datasets')
    client_models = []
    client_train_datasets = []
    client_test_datasets = []
    for client_id in range(args.n_parties):  # models including global
        # create client model
        client_model = utils.define_model(
            modelname=models[client_id % len(np.unique(models))], num_classes=num_classes, device=device, dataname=args.data)
        client_models.append(client_model)

        client_train_dataset = torch.utils.data.Subset(train_dataset, net_index_map[client_id])#  balance test dataset
        client_train_datasets.append(client_train_dataset)

        # sample test dataset
        uniques, counts = np.unique(y_train[net_index_map[client_id]], return_counts=True)
        if args.data == 'emnist':
            counts = np.append(counts, 0)
            uniques = np.append(uniques, 0)
        datasize = np.sum(counts)
        label_dist = np.array([0] * num_classes)
        for j, l in enumerate(uniques):
            label_dist[l] = counts[j]
        label_dist = label_dist / datasize
        test_size = datasize / 3

        sampled_test_indices = []
        for k, p in enumerate(label_dist):
            sampled_idx_k = np.random.choice(test_indices_k[k], int(test_size * p))
            sampled_test_indices.extend(sampled_idx_k)
        
        client_test_dataset = torch.utils.data.Subset(test_dataset, sampled_test_indices)
        client_test_datasets.append(client_test_dataset)
    print('DONE creating client models and datasets')

    '''learning configs (for hetero)'''
    configs = {
        'data': args.data,
        'public_datasize': args.public_datasize,
        'num_workers': 4,
        'private_batchsize': 32,
        'public_batchsize': 64
    }
    if 'mnist' in args.data:
        configs['learning_rate'] = 0.01
        configs['distill_lr'] = 0.005
        configs['penalty_ratio'] = 0.6
    else:
        # if cifar
        configs['learning_rate'] = 0.01
        configs['distill_lr'] = 0.01
        configs['penalty_ratio'] = 0.7
    
    '''federated distillation learning'''
    kt_pfl = KT_pFL(client_models,
                    public_dataset=_,
                    client_train_datasets=client_train_datasets,
                    client_test_datasets=client_test_datasets,
                    num_classes=num_classes,
                    configs=configs,
                    device=device)

    ''' collaborative training '''
    collaborative_performance = kt_pfl.collaborative_training(client_sampling_rate=args.c, max_rounds=args.max_rounds, max_local_epochs=args.local_epochs, distillation_steps=args.num_distill, output_path=args.output_path)
    print(collaborative_performance)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='fashion-mnist')
    parser.add_argument('--models', type=str,
                        default='resnet')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--n_parties', type=int, default=100)
    parser.add_argument('--c', type=float, default=0.1)
    parser.add_argument('--data_partition', type=str, default='noniid-labeldir')
    parser.add_argument('--runfile', type=str, default='cifar10_KTpFL-r_noniid-labeldir_20clients_C1_E20')
    # KT-pFL training params
    parser.add_argument('--public_datasize', type=int, default=3000)
    parser.add_argument('--local_epochs', type=int, default=20)
    parser.add_argument('--num_distill', type=int, default=1)
    parser.add_argument('--max_rounds', type=int, default=1000)
    # parser.add_argument('--learning_rate', type=float, default=0.01)
    # parser.add_argument('--distill_lr', type=float, default=0.01, help='mu3 to update c')
    parser.add_argument('--output_path', type=str, default='results/debug')
    parser.add_argument('--seed', type=int, default=2021)
    args = parser.parse_args()
    main(args)
