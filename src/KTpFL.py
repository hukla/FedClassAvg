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

# TODO uncomment for debugging
# os.environ["WANDB_MODE"] = "dryrun"
wandb.login(key="6ce7c21067c5213d01777e0a4527fda5597774a3")

#　Iterate over data classes
class data(torch.utils.data.Dataset):
    def __init__(self,X, y):
        self.X = np.array(X[:len(y)]) #due to drop last
        self.y = np.array(y)
    def __getitem__(self, item):
        X = self.X[item]
        if len(X.shape) == 2:
            # X = np.repeat(X[None],3,axis=0).astype(np.float32)
            X = np.expand_dims(X, axis=0).astype(np.float32)
        else:
            X = np.transpose(X, (2,0,1)).astype(np.float32)
        return X, self.y[item]

    def __len__(self):
        return len(self.X)

def generate_partial_data(dataset, classes, datasize=5000):
    targets = dataset.targets
    if isinstance(targets, list):
        targets = np.array(targets)
    data_indices = []
    for c in classes:
        idx_c = list(np.where(targets == c)[0])
        data_indices.extend(idx_c)
    data_indices = np.random.choice(data_indices, size=datasize, replace=False)
    partial_dataset = copy.deepcopy(dataset)
    partial_dataset.data = partial_dataset.data[data_indices]
    if isinstance(partial_dataset.targets, list):
        partial_dataset.targets = np.array(partial_dataset.targets)

    partial_dataset.targets = partial_dataset.targets[data_indices]
    return partial_dataset


def load_public_dataset(dataname, datasize=5000):
    if 'cifar' in dataname:
        # load cifar100 0: apple, 2: baby, 20: chair, 63:porcupine, 71: sea, 82: sunflower
        # cifar10 0: airplane, 1: automobile, 3: bird, 4: cat, 5: deer, 6:dog, 7: frog, 8: horse, 9: ship, 10: truck
        public_classes = [0, 2, 20, 63, 71, 82]
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406, ],
                                         std=[0.229, 0.224, 0.225])

        trsfm_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize
        ])

        trsfm_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = datasets.CIFAR100(
            './data', train=True, download=True, transform=trsfm_train)
        # test_dataset = datasets.CIFAR100('./data', test=True, download=True, transform=trsfm_test)

        public_dataset = generate_partial_data(train_dataset, public_classes, datasize)

    elif 'mnist' in dataname:
        # TODO load mnist; emnist should have 10 classes 굳이?
        # "public_classes": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        public_dataset = datasets.MNIST(root='./data',
                                       train=True,
                                       transform=transforms.ToTensor(),
                                       download=True)

        # test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
        public_dataset = generate_partial_data(public_dataset, np.arange(10), datasize)

    return public_dataset

def distill_one_model(model, max_epochs, device, train_dataloader, optimizer, criterion, client_id, csr=-1):
    model.train()
    n_data = 0

    for epoch_idx in range(max_epochs):
        epoch_loss = 0
        for batch_idx, (data, labels) in enumerate(train_dataloader):
            data, labels = data.to(device), labels.to(device)
            target_label = labels
            n_data += len(labels)

            optimizer.zero_grad()
            y_hat = model(data)
            if y_hat.__class__.__name__ == 'GoogLeNetOutputs':
                y_hat = y_hat.logits
                # y_hat = torch.nn.functional.softmax(y_hat, dim=1) # distillation target is softmax
            y_hat = torch.nn.functional.softmax(y_hat, dim=1) # distillation target is softmax
            loss = criterion(y_hat, target_label)

            loss.backward()
            optimizer.step()

            # calculate train accuracy
            epoch_loss += loss.item() * len(labels)

        train_loss = epoch_loss / (batch_idx + 1)
    return train_loss

def train_one_model(model, max_epochs, device, train_dataloader, optimizer, criterion, client_id, csr=-1):
    model.train()
    n_iter = 0
    for epoch_idx in range(max_epochs):
        epoch_loss = 0
        correct = 0
        n_data = 0
        for batch_idx, (data, labels) in enumerate(train_dataloader):
            data, labels = data.to(device), labels.to(device)
            target_label = labels
            n_data += len(labels)

            optimizer.zero_grad()
            y_hat = model(data)
            if y_hat.__class__.__name__ == 'GoogLeNetOutputs':
                y_hat = y_hat.logits
            loss = criterion(y_hat, target_label)

            loss.backward()
            optimizer.step()

            # calculate train accuracy
            _, predicted = torch.max(y_hat, 1)
            correct += (predicted == target_label).sum().item()
            epoch_loss += loss.item() * len(labels)

            n_iter += 1
        train_loss = epoch_loss / (batch_idx + 1)
        train_acc = correct / n_data
    return train_loss, train_acc

def evaluate_one_model(model, loader, criterion, device):
    """ evluate one model

    Args:
        model (model): model to evaluate
        loader (torch.utils.data.Dataloader): data to evaluate
        criterion (torch.nn.CrossEntropy()): loss function
        device (str): device

    Returns:
        loss (float): evaluation loss
        accuracy (float): evaluation accuracy
    """
    model.eval()
    epoch_loss = 0
    correct = 0
    n_data = 0

    for batch_idx, (data, labels), in enumerate(loader):
        data, labels = data.to(device), labels.to(device)
        target_label = labels
        n_data += len(labels)

        y_pred = model(data)
        if y_pred.__class__.__name__ == 'GoogLeNetOutputs':
            y_pred = y_pred.logits
        loss = criterion(y_pred, target_label)
        epoch_loss += loss.item() * y_pred.shape[0]

        # calculate train accuracy
        _, predicted = torch.max(y_pred, 1)
        correct += (predicted == target_label).sum().item()

    return epoch_loss / (batch_idx+1), correct / n_data

def predict(model, dataloader, device, T):
    model.eval()
    out= []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            logit = model(images)

            Tsoftmax = nn.Softmax(dim=1)
            # Add temperature coefficient T
            output_logit = Tsoftmax(logit.float()/T)

            out.append(output_logit.cpu().numpy())
    out = np.concatenate(out)
    return out


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
        # self.public_dataset = public_dataset
        self.public_dataset = public_dataset
        self.client_train_datasets = client_train_datasets
        self.client_test_datasets = client_test_datasets
        # self.net_index_map = net_index_map
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

        self.init()

    def init(self, init_epochs=1):
        """ initialize client models

        Args:
            init_epochs (int): number of epochs to init models (default: 1)
        """
        print('=====Initialize client models=====')
        for client_id in range(self.n_parties):
            client_train_dataset = self.client_train_datasets[client_id]
            client_test_dataset = self.client_test_datasets[client_id]
            train_loader = DataLoader(client_train_dataset, batch_size=128, shuffle=True, num_workers=self.num_workers, drop_last=True)
            test_loader = DataLoader(client_test_dataset, batch_size=128, shuffle=False, num_workers=self.num_workers)

            model = self.parties[client_id].to(self.device)
            optimizer = torch.optim.SGD(model.parameters(), lr=self.configs['learning_rate'])
            criterion = nn.CrossEntropyLoss()

            train_loss, train_acc = train_one_model(model=model, max_epochs=init_epochs, device=self.device, train_dataloader=train_loader, optimizer=optimizer, criterion=criterion, client_id=client_id)
            test_loss, test_acc = evaluate_one_model(model=model, loader=test_loader, criterion=criterion, device=self.device)

            print(f'[ROUND init CLIENT {client_id}] Loss/train:{train_loss:.3f}, Accuracy/train:{train_acc:.3f}')
            print(f'[ROUND init CLIENT {client_id}] Loss/test:{test_loss:.3f}, Accuracy/test:{test_acc:.3f}')

            self.parties[client_id] = model

        print('=====Initialize client models done=====')

        # print("=====Initialize logits====")
        # public_dataset = load_public_dataset(self.configs['data'], datasize=self.configs['public_datasize'])
        # public_dataloader = DataLoader(public_dataset, batch_size=256, num_workers=self.num_workers, shuffle=False, drop_last=True)
        # logits = []
        # for client_id in range(self.n_parties):
        #     model = self.parties[client_id].to(self.device)
        #     logits.append(predict(model, public_dataloader, self.device, self.Temp))
        # kcm = self.kcm
        # # print('before get logits:', kcm)
        # # logits_models, kcm = self.get_models_logits(logits, kcm, self.n_parties, self.penalty_ratio)
        # logits_models, kcm = self.get_models_logits(logits, self.penalty_ratio)
        # logits_models = logits_models.detach().numpy()
        # self.kcm = kcm
        # print("=====Initialize logits done====")


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
            sampled_clients = np.random.choice(np.arange(self.n_parties), n_parties_round, replace=False)
        # while True:
            print(f"[ROUND {round_idx}] START")
            # At beginning of each round, generate new alignment dataset; public
            public_dataset = load_public_dataset(self.configs['data'], datasize=self.configs['public_datasize'])
            public_dataloader = DataLoader(public_dataset, batch_size=256, num_workers=self.num_workers, shuffle=False, drop_last=True)

            print(f"[ROUND {round_idx}] Update logits")
            # update logits
            logits = []
            for i in tqdm(range(self.n_parties)):
                model = self.parties[i].to(device)
                logits.append(predict(model, public_dataloader, device, self.Temp))
            # print('before get logits:', kcm)
            logits_models, kcm = self.get_models_logits(logits, self.penalty_ratio)
            logits_models = logits_models.detach().numpy()
            self.kcm = kcm

            # balanced test
            print(f"[ROUND {round_idx}] Test performance")
            round_test_acc = 0
            round_test_loss = 0
            for client_id in tqdm(range(self.n_parties)):
                # client id of virutal client
                model = self.parties[client_id].to(device)
                criterion = nn.CrossEntropyLoss()

                client_test_dataset = self.client_test_datasets[client_id]
                client_test_loader = DataLoader(client_test_dataset, batch_size=128, shuffle=True, num_workers=self.num_workers, drop_last=False)

                val_loss, val_acc = evaluate_one_model(model, client_test_loader, criterion, device)
                collaboration_performance[client_id].append(val_acc)
                round_test_acc += val_acc
                round_test_loss += val_loss

            round_test_acc /= self.n_parties
            round_test_loss /= self.n_parties
            # print(f'[ROUND {round_idx}] Loss/val:{round_test_loss:.3f} Accuracy/val:{round_test_acc:.3f}')
            wandb.log({'global/loss': round_test_loss, 'global/acc': round_test_acc}, step=round_idx)
            print(f'[ROUND {round_idx}] Average test accuracy: {round_test_acc:.4f} (loss: {round_test_loss:.4f})')

            if best_val_loss > round_test_loss:
                best_val_acc = round_test_acc
                best_val_loss = round_test_loss
                # save models
                for client_id in range(self.n_parties):
                    model = self.parties[client_id]
                    # torch.save(model, os.path.join(output_path, f'round{round_idx}_client{client_id}.pt'))
                # save coefficient matrix
                torch.save(self.kcm, os.path.join(output_path, f'round{round_idx}_kcm.npz'))
                # with open(os.path.join(output_path, f'round{round_idx}_kcm.npz'), 'w') as npz_file:
                    # np.save(npz_file, self.kcm)

            wandb.log({'best_global/loss': best_val_loss, 'best_global/acc': best_val_acc}, step=round_idx)
            early_stopping(round_test_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            # local update
            print(f"[ROUND {round_idx}] Local update")

            # gradually increase local epochs; also not mentioned in the paper
            # local_epoch = 1
            if round_idx == 0:
                local_epoch = 1 - 1
            if local_epoch < max_local_epochs:
                local_epoch += 1
            else:
                local_epoch = max_local_epochs

            sampled_public_dataset = self.public_dataset

            for client_id in sampled_clients:
                model = self.parties[client_id]
                ### distillation step : update c
                print(f"[ROUND {round_idx} model {client_id}] Starting distillation with public logits")
                distill_dataset = copy.deepcopy(self.public_dataset)

                distill_loader = DataLoader(data(distill_dataset.data, logits_models[client_id]),
                                              batch_size=256,
                                              shuffle=True,
                                              num_workers=self.num_workers,
                                              drop_last=True)
                # test_dataloader = None
                optimizer = torch.optim.SGD(model.parameters(), lr=self.configs['learning_rate']) #TODO mu2
                # criterion = nn.KLDivLoss()
                criterion = nn.MSELoss()
                epoch = distillation_steps

                dist_loss = distill_one_model(model=model, max_epochs=distillation_steps, device=self.device, train_dataloader=distill_loader, optimizer=optimizer, criterion=criterion, client_id=client_id, csr=round_idx)

                print(f"[ROUND {round_idx} model {client_id}] Done distillation (loss: {dist_loss:.4f})")
                ### distillation step done

                ### update local weights
                print(f"[ROUND {round_idx} model {client_id}] Start local training")
                client_train_dataset = self.client_train_datasets[client_id]
                client_test_dataset = self.client_test_datasets[client_id]
                train_loader = DataLoader(client_train_dataset, batch_size=128, shuffle=True, num_workers=self.num_workers, drop_last=True)
                test_loader = DataLoader(client_test_dataset, batch_size=128, shuffle=False, num_workers=self.num_workers)

                optimizer = torch.optim.SGD(model.parameters(), lr=self.configs['learning_rate']) #mu1
                criterion = nn.CrossEntropyLoss()

                train_loss, train_acc = train_one_model(model=model, max_epochs=local_epoch, device=self.device, train_dataloader=train_loader, optimizer=optimizer, criterion=criterion, client_id=client_id, csr=round_idx)

                print(f"[ROUND {round_idx} model {client_id}] Done local training (loss/train: {train_loss:.4f}, acc/train: {train_acc:.4f})")
                ### update local weights done
            # END FOR LOOP

        for client_id in range(self.n_parties):
            model = self.parties[client_id]
            torch.save(model, os.path.join(output_path, f'round{round_idx}_client{client_id}.pt'))
        # END WHILE LOOP
        return collaboration_performance


    def get_models_logits(self, raw_logits, penalty_ratio):
        """ update knowledge coefficient matrix c and return logit tagets to distillate

        Args:
            raw_logits (_type_): soft predictions from client models
            penalty_ratio (float): regularization penalty

        Returns:
            models_logits: c * target logits to distillate
            weight: knowledge coefficient matrix
        """
        weight_mean = torch.ones(self.n_parties, self.n_parties, requires_grad=True)
        weight_mean = weight_mean.float()/(self.n_parties)
        # loss_fn = nn.KLDivLoss(reduce=True, size_average=True)
        # L_KL ~= L_MSE when \tau is large
        loss_fn = nn.MSELoss(reduce=True, size_average=True)
        teacher_logits = torch.zeros(self.n_parties, np.size(raw_logits[0],0), np.size(raw_logits[0],1), requires_grad=False) #create logits of teacher  #next false
        models_logits = torch.zeros(self.n_parties, np.size(raw_logits[0],0), np.size(raw_logits[0],1), requires_grad=True) #create logits of teacher
        weight = self.kcm.clone()
        for self_idx in range(self.n_parties): #Calculate the weighted average of its teacher's logits for each model
            teacher_logits_local = teacher_logits[self_idx]
            for teacher_idx in range(self.n_parties): # For a model, calculate the logits of all other models
                teacher_logits_local = torch.add(teacher_logits_local, weight[self_idx][teacher_idx] * torch.from_numpy(raw_logits[teacher_idx]))
                # A pixel in tensor, intrinsic scalar * teacher's complete logits

            loss_input = torch.from_numpy(raw_logits[self_idx])
            loss_target = teacher_logits_local

            loss = loss_fn(loss_input, loss_target)

            loss_penalty = loss_fn(weight[self_idx], weight_mean[self_idx])
            loss += loss_penalty * penalty_ratio

            weight.retain_grad()
            loss.backward(retain_graph=True)
            # print('weight:', weight)

            # with torch.no_grad():
                # gradabs = torch.abs(weight.grad)
                # gradsum = torch.sum(gradabs)
                # gradavg = gradsum.item() / (self.n_parties)
                # grad_lr = 1.0
                # for i in range(5): #0.1
                    # if gradavg > 0.01:
                        # gradavg = gradavg*1.0/5
                        # grad_lr = grad_lr/5
                    # if gradavg < 0.01:
                        # gradavg = gradavg*1.0*5
                        # grad_lr = grad_lr*5
                # print('grad_lr:', grad_lr)
                # weight.sub_(weight.grad*grad_lr)
                # weight.grad.zero_()
            with torch.no_grad():
                weight.sub_(weight.grad*self.configs['learning_rate'])
                weight.grad.zero_()

        # update raw_logits
        for self_idx in range(self.n_parties): # Calculate the weighted average of its teacher's logits for each model
            weight_tmp = torch.zeros(self.n_parties) # weight to current client
            for teacher_idx in range(self.n_parties):
                # For a model, calculate the weight after its softmax (why???)
                weight_tmp[teacher_idx] = weight[self_idx][teacher_idx]
            # weight_local = nn.functional.softmax(weight_tmp*5.0)
            weight_local = weight_tmp

            idx_count = 0
            for teacher_idx in range(self.n_parties): # For a model, calculate the logits of all other models
                models_logits[self_idx].data += weight_local[idx_count] * torch.from_numpy(raw_logits[teacher_idx])
                with torch.no_grad():
                    weight[self_idx][teacher_idx] = weight_local[idx_count]
                idx_count += 1
        # print('weight after softmax:', weight)
        #
        return models_logits, weight

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
    train_dataset, test_dataset, val_dataset, num_classes = utils.load_dataset(
        data_split=args.data, target_label_idx=0)
    public_dataset = load_public_dataset(args.data, datasize=args.public_datasize)
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
        'num_workers': 4
    }
    if 'mnist' in args.data:
        configs['learning_rate'] = 0.01
        configs['penalty_ratio'] = 0.6
    else:
        configs['learning_rate'] = 0.02
        configs['penalty_ratio'] = 0.7

    '''federated distillation learning'''
    kt_pfl = KT_pFL(client_models,
                    public_dataset=public_dataset,
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
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--data', type=str, default='cifar10')
    parser.add_argument('--models', type=str,
                        default='resnet,shufflenet,googlenet,alexnet')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--n_parties', type=int, default=20)
    parser.add_argument('--c', type=float, default=1.)
    parser.add_argument('--data_partition', type=str, default='noniid-twoclass')
    parser.add_argument('--runfile', type=str, default='cifar10_KTpFL-rsga_noniid-labeldir_20clients_C1_E20')
    # KT-pFL training params
    parser.add_argument('--public_datasize', type=int, default=3000)
    parser.add_argument('--local_epochs', type=int, default=20)
    parser.add_argument('--num_distill', type=int, default=1)
    parser.add_argument('--max_rounds', type=int, default=1000)
    # parser.add_argument('--learning_rate', type=float, default=0.01)
    # parser.add_argument('--distill_lr', type=float, default=0.01, help='mu3 to update c')
    parser.add_argument('--output_path', type=str, default='results/debug')
    args = parser.parse_args()
    main(args)
